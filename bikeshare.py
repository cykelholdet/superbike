"""
Created on Mon Feb 22 15:52:51 2021

@author: Mattek Group 3
"""
import os
import time
import pickle
import calendar
import logging
import warnings
import datetime
import zipfile
import io
from requests import get

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rarfile
import pyproj

from workalendar.europe import CommunityofMadrid, Finland, UnitedKingdom, \
    Norway, Edinburgh
from workalendar.usa import NewYork, Massachusetts, ChicagoIllinois, \
    DistrictOfColumbia, Minnesota, CaliforniaSanFrancisco, California
from workalendar.asia import Taiwan
from workalendar.america import Mexico, Argentina, Quebec

import dataframe_key



def download_data(city, year, verbose=True):
    if not os.path.exists(f'data/{city}'):
        os.makedirs(f'data/{city}')
    with open(f'bikeshare_data_sources/{city}/{year}.txt', 'r', encoding='utf-8') as file:
        filenames = file.read().splitlines()
    for url in filenames:
        if verbose:
            print(url)
        r = get(url, stream=True)
        if (city in ['bergen', 'oslo', 'trondheim',]) and year >= 2019:
            with open(f'data/{city}/{year}{r.url.rsplit("/", 1)[-1][:-4]}-{city}.csv', 'wb') as file:
                file.write(r.content)
        elif city == 'london':
            with open(f'data/{city}/{r.url.rsplit("/", 1)[-1]}', 'wb') as file:
                file.write(r.content)
        elif city == 'helsinki':
            z = zipfile.ZipFile(io.BytesIO(r.content))
            zipinfos = z.infolist()
            for zipinfo in zipinfos:
                zipinfo.filename = f'{zipinfo.filename[:-4]}-helsinki.csv'
                z.extract(zipinfo, f'data/{city}/')
        else:
            try:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(f'data/{city}/')
            except zipfile.BadZipFile:
                try:
                    z = rarfile.RarFile(io.BytesIO(r.content))
                    z.extractall(f'data/{city}/')
                except rarfile.NotRarFile:
                    with open(f'data/{city}/{r.url.rsplit("/", 1)[-1]}', 'wb') as file:
                        file.write(r.content)
                except rarfile.RarCannotExec:
                    raise rarfile.RarCannotExec("Please install unrar from your distribution's package manager or install unrar.dll")
    return filenames


def compile_chicago_stations():
    """
    Reads data files containing information about docking stations in Chicago
    and compiles the data into a dataframe. The dataframe is then saved as a
    pickle for further use.

    The relevant files can be found at:
    https://divvy-tripdata.s3.amazonaws.com/index.html
    https://data.cityofchicago.org/Transportation/Divvy-Bicycle-Stations-All-Map/bk89-9dk7

    Raises
    ------
    FileNotFoundError
        Raised if no data files containing station data are found.

    Returns
    -------
    stat_df : pandas DataFrame
        Dataframe of all docking station information.

    """
    try:
        with open('./python_variables/Chicago_stations.pickle', 'rb') as file:
            stat_df = pickle.load(file)

    except FileNotFoundError as exc:
        print('No pickle found. Creating pickle...')

        stat_files = [file for file in os.listdir(
            'data/chicago') if 'Divvy_Stations' in file]

        # Only a subset of the columns are of interest. These are then renamed
        # to a standard format.
        col_list = ['id', 'name', 'latitude', 'longitude']
        key = {'ID': 'id', 'Station Name': 'name',
               'Latitude': 'latitude', 'Longitude': 'longitude'}

        try:
            stat_df = pd.read_csv(
                'data/chicago/Divvy_Bicycle_Stations_-_All_-_Map.csv').rename(columns=key)
            stat_df = stat_df[col_list]
        except FileNotFoundError:
            stat_df = pd.DataFrame(columns=col_list)

        for file in stat_files:
            df = pd.read_csv(f'./data/chicago/{file}')[col_list]
            stat_df = pd.concat([stat_df, df], sort=False)

        if stat_df.size == 0:
            raise FileNotFoundError(
                'No data files containing station data found. Please read the docstring for more information.') from exc

        stat_df.drop_duplicates(subset='name', inplace=True)

        with open('./python_variables/Chicago_stations.pickle', 'wb') as file:
            pickle.dump(stat_df, file)

    print('Pickle loaded')

    return stat_df


def get_data_month(city, year, month, blocklist=None, overwrite=False):
    """
    Read bikeshare data from provider provided files.

    Parameters
    ----------
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.
    blocklist : list, optional
        List of IDs of stations to remove. Default is None.
    overwrite : bool, optional
        If True, create a new pickle regardless of whether there is an existing
        pickle.

    Returns
    -------
    df : pandas DataFrame
        Dataframe containing bikeshare trip data.
    days : dict
        Contains the indices of the first trip per day.

    """
    # Remember to update this list when implementing a new city
    supported_cities = [
        'bergen', 'boston', 'buenos_aires', 'chicago', 'edinburgh', 'guadalajara',
        'helsinki', 'la', 'london', 'madrid', 'mexico', 'minneapolis', 'montreal',
        'nyc', 'oslo', 'sfran', 'sjose', 'taipei', 'trondheim', 'washdc']

    if city not in supported_cities:
        raise ValueError(
            f"{city} is not currently supported. Supported cities are {supported_cities}")

    # Make directory for dataframes if not found
    if not os.path.exists('python_variables'):
        os.makedirs('python_variables')
    if not overwrite:
        print(f'Loading pickle {name_dict[city]} {year:d} {month_dict[month]}... ', end="")
        try:
            with open(f'./python_variables/{city}{year:d}{month:02d}_dataframe_blcklst={blocklist}.pickle', 'rb') as file:
                df = pickle.load(file)
            print("Done")

        except FileNotFoundError:
            print('\n No dataframe pickle found. ', end="")
            overwrite = True

    if overwrite:
        print("Pickling dataframe...")
        if city == "nyc":

            try:
                df = pd.read_csv(
                    f'./data/{city}/{year:d}{month:02d}-citibike-tripdata.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.citibikenyc.com/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            # Remove stations in Jersey City by splitting at the -74.026 degree
            # meridian.
            df = df[(df['start_stat_long'] > -74.026) & (df['end_stat_long'] > -74.026)]

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "washdc":

            if datetime.datetime(year, month, 1) == datetime.datetime(2018, 1, 1):
                systemname = '_capitalbikeshare_'
            else:
                systemname = '-capitalbikeshare-'

            try:
                df = pd.read_csv(
                    f'./data/{city}/{year:d}{month:02d}{systemname}tripdata.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.capitalbikeshare.com/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            stations = pd.read_csv(f'data/{city}/Capital_Bike_Share_Locations.csv')

            long_dict = dict(zip(stations['TERMINAL_NUMBER'], stations['LONGITUDE']))
            lat_dict = dict(zip(stations['TERMINAL_NUMBER'], stations['LATITUDE']))

            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)

            df.dropna(inplace=True)

            # Remove stations in towns outside of the main DC area.
            max_lat = 38.961029
            min_lat = 38.792686
            max_long = -76.909415
            min_long = -77.139396

            df = df.iloc[np.where(
                (df['start_stat_lat'] < max_lat) &
                (df['start_stat_lat'] > min_lat) &
                (df['start_stat_long'] < max_long) &
                (df['start_stat_long'] > min_long))]

            df = df.iloc[np.where(
                (df['end_stat_lat'] < max_lat) &
                (df['end_stat_lat'] > min_lat) &
                (df['end_stat_long'] < max_long) &
                (df['end_stat_long'] > min_long))]

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)
            
            df = df[df['start_dt'].dt.month == month]
            df.reset_index(inplace=True, drop=True)

            df['user_type'] = df['user_type'].map(
                {'Member': 'Subscriber',
                 'Casual': 'Customer'})

        elif city == 'minneapolis':
            try:
                if year == 2019 and month == 6:  # data is misnamed.
                    df = pd.read_csv(
                        f'./data/{city}/20019{month:02d}-niceride-tripdata.csv')
                else:
                    df = pd.read_csv(
                        f'./data/{city}/{year:d}{month:02d}-niceride-tripdata.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.niceridemn.com/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == 'boston':
            if datetime.datetime(year, month, 1) <= datetime.datetime(2018, 3, 1):
                systemname = '_hubway_'
            elif datetime.datetime(year, month, 1) == datetime.datetime(2018, 4, 1):
                systemname = '-hubway-'
            else:
                systemname = '-bluebikes-'
            try:
                df = pd.read_csv(
                    f'./data/{city}/{year:d}{month:02d}{systemname}tripdata.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.bluebikes.com/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            df = df[~df['start_stat_id'].isin([382, 383, 223, 230, 164, 158])] # Filter out virtual stations
            df = df[~df['end_stat_id'].isin([382, 383, 223, 230,  164, 158])]

            # Merge stations which have the same coordinates and harmonise names.
            merge_id_dict = {241: 336, 242: 337, 254: 348, 256: 349, 263: 353}
            merge_name_dict = {
                'Talbot Ave At Blue Hill Ave (former)': 'Talbot Ave At Blue Hill Ave',
                'Washington St at Talbot Ave (former)': 'Washington St at Talbot Ave',
                'Mattapan Library (former)': 'Mattapan Library'}

            df['start_stat_id'] = df['start_stat_id'].replace(merge_id_dict)
            df['end_stat_id'] = df['end_stat_id'].replace(merge_id_dict)

            df['start_stat_name'] = df['start_stat_name'].replace(merge_name_dict)
            df['end_stat_name'] = df['end_stat_name'].replace(merge_name_dict)

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "chicago":

            q = int(np.ceil(month/3))

            try:
                df = pd.read_csv(f'./data/{city}/Divvy_Trips_{year:d}_Q{q}.csv')

            except FileNotFoundError:
                try:
                    df = pd.read_csv(f'./data/{city}/Divvy_Trips_{year:d}_Q{q}')
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://www.divvybikes.com/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            n_days = calendar.monthrange(year, month)[1]

            df = df.iloc[np.where(
                df['start_t'] > f'{year:d}-{month:02d}-01 00:00:00')]
            df = df.iloc[np.where(
                df['start_t'] < f'{year:d}-{month:02d}-{n_days} 23:59:59')]

            df.reset_index(inplace=True, drop=True)

            with open('./python_variables/Chicago_stations.pickle', 'rb') as file:
                stations = pickle.load(file)

            lat_dict = dict(zip(stations['name'], stations['latitude']))
            long_dict = dict(zip(stations['name'], stations['longitude']))

            df['start_stat_lat'] = df['start_stat_name'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_name'].map(long_dict)

            df['end_stat_lat'] = df['end_stat_name'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_name'].map(long_dict)

            df.dropna(subset=['start_stat_lat',
                              'start_stat_long',
                              'end_stat_lat',
                              'end_stat_long'], inplace=True)

            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df['duration'] = df['duration'].str.replace(',', '').astype(float)
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "la":
            q = int(np.ceil(month/3))

            try:
                df = pd.read_csv(f'./data/{city}/metro-bike-share-trips-{year:d}-q{q}.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://bikeshare.metro.net/about/data/') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])

            df = df[df['start_dt'].dt.month == month]
            df.drop(columns=['start_t', 'end_t'], inplace=True)
            df.dropna(inplace=True) # Removes stations 3000	Virtual Station, 4285 Metro Bike Share Free Bikes, 4286 Metro Bike Share Out of Service Area Smart Bike

            files = [file for file in os.listdir(
                f'data/{city}') if 'metro-bike-share-stations' in file]
            try:
                stations = pd.read_csv(f"data/{city}/{files[0]}", names=['stat_id', 'stat_name', 'date', 'authority', 'status'])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No station data found. All relevant files can be found at https://bikeshare.metro.net/about/data/') from exc

            stat_name_dict = dict(zip(stations['stat_id'], stations['stat_name']))
            df['start_stat_name'] = df['start_stat_id'].map(stat_name_dict)
            df['end_stat_name'] = df['end_stat_id'].map(stat_name_dict)

            df['duration'] = df['duration']*60

            df.reset_index(inplace=True, drop=True)


        elif city == "sfran":

            try:
                df = pd.read_csv(
                    f'./data/{city}/{year:d}{month:02d}-baywheels-tripdata.csv')
            except FileNotFoundError:
                try:
                    df = pd.read_csv(
                        f'./data/{city}/{year:d}{month:02d}-fordgobike-tripdata.csv')
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            if 'bike_share_for_all_trip' in df.columns:
                df['bike_share_for_all_trip'].fillna('N', inplace=True)
            if 'rental_access_method' in df.columns:
                df['rental_access_method'].fillna('N', inplace=True)
            df.dropna(inplace=True)

            # Split stations in San Francisco from San Jose
            df = df.iloc[np.where(df['start_stat_lat'] > 37.593220)]
            df = df.iloc[np.where(df['end_stat_lat'] > 37.593220)]
            df = df.iloc[np.where(df['start_stat_long'] < -80)]
            df = df.iloc[np.where(df['end_stat_long'] < -80)]

            df.sort_values(by='start_t', inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "sjose":

            try:
                df = pd.read_csv(
                    f'./data/sfran/{year:d}{month:02d}-baywheels-tripdata.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)

            # Split stations in San Francisco from San Jose
            df = df.iloc[np.where(df['start_stat_lat'] < 37.593220)]
            df = df.iloc[np.where(df['end_stat_lat'] < 37.593220)]

            df.sort_values(by='start_t', inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "london":
            if year == 2018:
                month_london = month_dict.copy()
                month_london[6] = 'June'
                month_london[7] = 'July'
            else:
                month_london = month_dict

            data_files = [file for file in os.listdir(
                'data/london') if 'JourneyDataExtract' in file]
            data_files = [file for file in data_files if f"{month_london[month]}{year}" in file]

            if len(data_files) == 0:
                raise FileNotFoundError(
                    'No London data for {}. {} found. All relevant files can be found at https://cycling.data.tfl.gov.uk/.'.format(month_dict[month], year))

            if isinstance(data_files, str):
                warnings.warn(
                    'Only one data file found. Please check that you have all available data.')

            df = pd.read_csv('./data/london/' + data_files[0])

            for file in data_files[1:]:
                df_temp = pd.read_csv('./data/london/' + file)
                df = pd.concat([df, df_temp], sort=False)

            df.rename(columns=dataframe_key.get_key(city), inplace=True)

            df['start_dt'] = pd.to_datetime(df['start_t'], format='%d/%m/%Y %H:%M')
            df['end_dt'] = pd.to_datetime(df['end_t'], format='%d/%m/%Y %H:%M')
            df.drop(columns=['start_t', 'end_t'], inplace=True)

            df = df[(df['start_dt'].dt.month == month) & (df['start_dt'].dt.year == year)]

            df.sort_values(by='start_dt', inplace=True)

            df.reset_index(inplace=True)

            stat_df = pd.read_csv('./data/london/london_stations.csv')
            stat_df.at[np.where(stat_df['station_id'] == 502)[
                0][0], 'latitude'] = 51.53341

            long_dict = dict(zip(stat_df['station_id'], stat_df['longitude']))
            lat_dict = dict(zip(stat_df['station_id'], stat_df['latitude']))

            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)

            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)

            # Merge stations with identical coordinates
            merge_id_dict = {361: 154, 374: 154,
                              280: 250, 819: 273, 328: 327, 336: 334,
                              421: 420, 816: 812}
            waterloo_dict = {f"Waterloo Station {i}, Waterloo": "Waterloo Station, Waterloo" for i in range(1, 4)}
            royal_dict = {f"Royal Avenue {i}, Chelsea": "Royal Avenue, Chelsea" for i in range(1, 3)}
            belvedere_dict = {f"Belvedere Road {i}, South Bank": "Belvedere Road, South Bank" for i in range(1, 3)}
            north_dict = {f"New North Road {i}, Hoxton": "New North Road, Hoxton" for i in range(1, 3)}
            concert_dict = {f"Concert Hall Approach {i}, South Bank": "Concert Hall Approach, South Bank" for i in range(1, 3)}
            south_dict = {f"Southwark Station {i}, Southwark": "Southwark Station, Southwark" for i in range(1, 3)}
            here_dict = {"Here East North, Queen Elizabeth Olympic Park": "Here East, Queen Elizabeth Olympic Park",
                          "Here East South, Queen Elizabeth Olympic Park": "Here East, Queen Elizabeth Olympic Park"}
            merge_name_dict = {**waterloo_dict, **royal_dict, **belvedere_dict, **north_dict, **concert_dict, **south_dict, **here_dict}

            df['start_stat_id'] = df['start_stat_id'].replace(merge_id_dict)
            df['start_stat_name'] = df['start_stat_name'].replace(merge_name_dict)

            df['end_stat_id'] = df['end_stat_id'].replace(merge_id_dict)
            df['end_stat_name'] = df['end_stat_name'].replace(merge_name_dict)

            df.reset_index(inplace=True, drop=True)

        elif city == "oslo":

            if datetime.datetime(year, month, 1) <= datetime.datetime(2018, 12, 1):
                try:
                    df = pd.read_csv(f'./data/{city}/oslo-trips-{year:d}.{month:d}.csv')
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://oslobysykkel.no/en/open-data/historical') from exc

                df = df.rename(columns=dataframe_key.get_key(city))
                df.dropna(inplace=True)
                df.reset_index(inplace=True, drop=True)

                stat_loc = pd.read_csv(
                    f'./data/{city}/legacy_station_locations.csv')
                stat_ids = pd.read_csv(
                    f'./data/{city}/legacy_new_station_id_mapping.csv')

                long_dict = dict(zip(stat_loc['legacy_id']+156, stat_loc['longitude']))
                lat_dict = dict(zip(stat_loc['legacy_id']+156, stat_loc['latitude']))

                id_dict = dict(zip(stat_ids['legacy_id']+156, stat_ids['new_id']))

                df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
                df['start_stat_long'] = df['start_stat_id'].map(long_dict)

                df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
                df['end_stat_long'] = df['end_stat_id'].map(long_dict)

                df['start_stat_id'] = df['start_stat_id'].map(id_dict)
                df['end_stat_id'] = df['end_stat_id'].map(id_dict)

                df['start_dt'] = pd.to_datetime(df['start_t'])
                df['end_dt'] = pd.to_datetime(df['end_t'])
                df.drop(columns=['start_t', 'end_t'], inplace=True)

            else:
                try:
                    df = pd.read_csv(f'./data/{city}/{year:d}{month:02d}-oslo.csv')
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://oslobysykkel.no/en/open-data/historical') from exc

                df = df.rename(columns=dataframe_key.get_key(city))
                df.dropna(inplace=True)
                df.reset_index(inplace=True, drop=True)

                # Merge stations with identical coordinates
                merge_id_dict = {619: 618}
                merge_name_dict = {f"Bak Niels Treschows hus {i}": "Bak Niels Treschows hus" for i in ['sør', 'nord']}

                df['start_stat_id'] = df['start_stat_id'].replace(merge_id_dict)
                df['start_stat_name'] = df['start_stat_name'].replace(merge_name_dict)

                df['end_stat_id'] = df['end_stat_id'].replace(merge_id_dict)
                df['end_stat_name'] = df['end_stat_name'].replace(merge_name_dict)

                # remove timezone information as data is erroneously tagged as UTC
                # while actually being wall time.
                df['start_dt'] = pd.to_datetime(df['start_t']).dt.tz_convert('Europe/Oslo')
                df['end_dt'] = pd.to_datetime(df['end_t']).dt.tz_convert('Europe/Oslo')
                df.drop(columns=['start_t', 'end_t'], inplace=True)
                df = df[df['start_dt'].dt.month == month]

        elif city == "edinburgh":

            try:
                df = pd.read_csv(f'./data/{city}/{year:d}-{month:02d}-edinburgh.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://edinburghcyclehire.com/open-data/historical') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t']).dt.tz_convert('Europe/Edinburgh')
            df['end_dt'] = pd.to_datetime(df['end_t']).dt.tz_convert('Europe/Edinburgh')
            df.drop(columns=['start_t', 'end_t'], inplace=True)
            df = df[df['start_dt'].dt.month == month]

        elif city == "bergen":

            try:
                df = pd.read_csv(f'./data/{city}/{year:d}{month:02d}-bergen.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://bergenbysykkel.no/en/open-data/historical') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t']).dt.tz_convert('Europe/Oslo')
            df['end_dt'] = pd.to_datetime(df['end_t']).dt.tz_convert('Europe/Oslo')
            df.drop(columns=['start_t', 'end_t'], inplace=True)
            df = df[df['start_dt'].dt.month == month]

        elif city == "trondheim":

            try:
                df = pd.read_csv(f'./data/{city}/{year:d}{month:02d}-trondheim.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://bergenbysykkel.no/en/open-data/historical') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t']).dt.tz_convert('Europe/Oslo')
            df['end_dt'] = pd.to_datetime(df['end_t']).dt.tz_convert('Europe/Oslo')
            df.drop(columns=['start_t', 'end_t'], inplace=True)
            df = df[df['start_dt'].dt.month == month]

        elif city == "helsinki":

            try:
                df = pd.read_csv(f'./data/{city}/{year:d}-{month:02d}-helsinki.csv')
            except FileNotFoundError as exc:
                if month not in range(4, 10+1):
                    warnings.warn(
                        'Data not available in winter.')
                    return pd.DataFrame(columns=[*dataframe_key.get_key(city).values(), 'start_stat_lat', 'start_stat_long', 'end_stat_lat', 'end_stat_long']), []
                else:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://hri.fi/data/en_GB/dataset/helsingin-ja-espoon-kaupunkipyorilla-ajatut-matkat') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

            try:
                stations = pd.read_csv(
                    f'./data/{city}/Helsingin_ja_Espoon_kaupunkipyöräasemat_avoin.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No station data found. All relevant files can be found at https://hri.fi/data/en_GB/dataset/helsingin-ja-espoon-kaupunkipyorilla-ajatut-matkat') from exc

            long_dict = dict(zip(stations['ID'], stations['x'].astype(float)))
            lat_dict = dict(zip(stations['ID'], stations['y'].astype(float)))
            addr_dict = dict(zip(stations['ID'], stations['Osoite']))

            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)
            df['start_stat_desc'] = df['start_stat_id'].map(addr_dict)

            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)
            df['end_stat_desc'] = df['end_stat_id'].map(addr_dict)

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

        elif city == "buenos_aires":

            try:
                df = pd.read_csv(
                    f"./data/{city}/recorridos-realizados-{year:d}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://data.buenosaires.gob.ar/dataset/bicicletas-publicas') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            df['month'] = pd.to_datetime(df['start_t']).dt.month
            df = df[df.month == month]

            # Fix errors regarding station 159 in the data
            mask = df['start_stat_id'] == "159_0"
            df.loc[mask, 'start_stat_id'] = 159
            df.loc[mask, 'start_stat_lat'] = -34.584953
            df.loc[mask, 'start_stat_long'] = -58.437340
            df.loc[mask, 'start_stat_desc'] = ""

            mask = df['end_stat_id'] == "159_0"
            df.loc[mask, 'end_stat_id'] = 159
            df.loc[mask, 'end_stat_lat'] = -34.584953
            df.loc[mask, 'end_stat_long'] = -58.437340
            df.loc[mask, 'end_stat_desc'] = ""

            df['gender'].fillna(0, inplace=True)

            df.dropna(inplace=True)
            df['start_stat_id'] = df['start_stat_id'].astype(int)
            df['end_stat_id'] = df['end_stat_id'].astype(int)

            df.sort_values(by=['start_t'], inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "madrid":
            # Depending on the month, the data has different formats
            if datetime.datetime(year, month, 1) > datetime.datetime(2019, 7, 1):
                try:
                    df = pd.read_json(
                        f"./data/{city}/{year:d}{month:02d}_movements.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc
                try:
                    df_pre = pd.read_json(
                        f"./data/{city}/{year:d}{(month-1):02d}_movements.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc
                df = df.rename(columns=dataframe_key.get_key(city))
                df_pre = df_pre.rename(columns=dataframe_key.get_key(city))

                # Convert from UTC to local time. There is no end time in the
                # data, only start and duration.

                # df['start_dt'] = pd.to_datetime(
                #     df['start_t'], format='%Y-%m-%dT%H:%M:%SZ') + pd.DateOffset(hours=2)
                # df_pre['start_dt'] = pd.to_datetime(
                #     df_pre['start_t'], format='%Y-%m-%dT%H:%M:%SZ') + pd.DateOffset(hours=2)
                df['start_dt'] = pd.to_datetime(
                    df['start_t'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize(tz='UTC', ambiguous=False)

                df['start_dt'] = df['start_dt'].dt.tz_convert(tz='Europe/Madrid')

                df_pre['start_dt'] = pd.to_datetime(
                    df_pre['start_t'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize(tz='UTC', ambiguous=False)

                df_pre['start_dt'] = df_pre['start_dt'].dt.tz_convert(tz='Europe/Madrid')
                # df_pre['start_dt'] = pd.to_datetime(
                #     df_pre['start_t'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize(tz='Europe/Madrid', ambiguous=False)

                df = df[df['start_dt'].dt.month == month]
                df_pre = df_pre[df_pre['start_dt'].dt.month == month]

                df = pd.concat((df_pre, df))

            elif datetime.datetime(year, month, 1) == datetime.datetime(2019, 7, 1):
                try:
                    df = pd.read_json(
                        f"./data/{city}/{year:d}{month:02d}_movements.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc

                df = df.rename(columns=dataframe_key.get_key(city))
                df['start_dt'] = pd.to_datetime(
                    df['start_t'], format='%Y-%m-%dT%H:%M:%SZ').dt.tz_localize(tz='UTC', ambiguous=False)

                df['start_dt'] = df['start_dt'].dt.tz_convert(tz='Europe/Madrid')
            else:
                try:
                    df = pd.DataFrame()
                    with pd.read_json(f'data/madrid/{year:d}{month:02d}_Usage_Bicimad.json', lines=True, chunksize=10000, encoding = "ISO-8859-1") as chunks:
                        for chunk in chunks:
                            df = pd.concat((df, chunk.drop(columns=['track'], errors='ignore')))
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc
                except ValueError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc

                if 'track' in df.columns:
                    df.drop(columns=['track'], inplace=True) # Remember that they exist in some data
                df['unplug_hourTime'] = pd.json_normalize(
                    df['unplug_hourTime'])
                df.rename(columns = dataframe_key.get_key(city), inplace=True)
                df['start_t'] = df['start_t'].str[:-6]

                # Timezone is local in older data.
                df['start_dt'] = pd.to_datetime(
                    df['start_t'], format='%Y-%m-%dT%H:%M:%S').dt.tz_localize(tz='Europe/Madrid', ambiguous=False)

            df.drop(columns=['start_t'], inplace=True)

            df = df[df['start_dt'].dt.month == month]
            
            df['end_dt'] = df['start_dt'] + \
                pd.to_timedelta(df['duration'], unit='s')

            logging.debug('Getting madrid stations')

            if datetime.datetime(year, month, 1) >= datetime.datetime(2019, 7, 1):
                # In the last months of 2019 the station data is UTF-8
                _, stations = pd.read_json(
                    f"./data/{city}/{year:d}{month:02d}_stations_madrid.json",
                    lines=True).iloc[-1]
            elif (datetime.datetime(year, month, 1) <= datetime.datetime(2018, 10, 1)) and (datetime.datetime(year, month, 1) >= datetime.datetime(2018, 8, 1)):
                _, stations = pd.read_json(
                    f"./data/{city}/Bicimad_Estacions_{year:d}{month:02d}.json",
                    lines=True, encoding = "ISO-8859-1").iloc[-1]
            # There is no station information prior to July 2018, so the earliest
            # station information is used. The data from July 2018 is in invalid json
            # and contains the same locations as August, so August is used.
            elif datetime.datetime(year, month, 1) < datetime.datetime(2018, 8, 1):
                _, stations = pd.read_json(
                    f"./data/{city}/Bicimad_Estacions_201808.json",
                    lines=True, encoding = "ISO-8859-1").iloc[-1]
            # Else condition is reached in 2019 Jan-Jun
            else:
                _, stations = pd.read_json(
                    f"./data/{city}/Bicimad_Stations_{year:d}{month:02d}.json",
                    lines=True, encoding = "ISO-8859-1").iloc[-1]

            stations = pd.DataFrame(stations)

            stat_name_dict = dict(zip(stations['id'], stations['name']))
            long_dict = dict(
                zip(stations['id'], stations['longitude'].astype(float)))
            lat_dict = dict(
                zip(stations['id'], stations['latitude'].astype(float)))
            addr_dict = dict(zip(stations['id'], stations['address']))

            df['start_stat_name'] = df['start_stat_id'].map(stat_name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)
            df['start_stat_desc'] = df['start_stat_id'].map(addr_dict)

            df['end_stat_name'] = df['end_stat_id'].map(stat_name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)
            df['end_stat_desc'] = df['end_stat_id'].map(addr_dict)

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['user_type'] = df['user_type'].map(
                {1: 'Subscriber',
                 2: 'Customer',
                 3: 'Company Worker'})

        elif city == "mexico":

            try:
                df = pd.read_csv(f"./data/{city}/{year:d}-{month:02d}-mexico.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.ecobici.cdmx.gob.mx/en/informacion-del-servicio/open-data') from exc

            df.rename(columns=dataframe_key.get_key(city), inplace=True)

            # Remove corrupt data.
            if month == 3 and year == 2019:
                df.drop(index=df.loc[df['end_date'] == '10'].index, inplace=True)
                df.drop(columns="Unnamed: 9", inplace=True)
            if month == 4 and year == 2019:
                df.drop(index=df.loc[df['start_time'] == '18::'].index, inplace=True)

            df['start_dt'] = pd.to_datetime(df['start_date'] + ' ' + df['start_time'],
                                            format='%d/%m/%Y %H:%M:%S')
            df['end_dt'] = pd.to_datetime(df['end_date'] + ' ' + df['end_time'],
                                          format='%d/%m/%Y %H:%M:%S')
            df.drop(['start_date', 'start_time', 'end_date',
                    'end_time'], axis=1, inplace=True)
            df['duration'] = (df['end_dt'] - df['start_dt']).dt.total_seconds()

            df = df[(df['start_dt'].dt.year == year) & (df['start_dt'].dt.month == month)]

            stations = pd.DataFrame(pd.read_json(
                "./data/mexico/stations_mexico.json", lines=True)['stations'][0])

            stat_name_dict = dict(zip(stations['id'], stations['address']))
            locations = stations['location'].apply(pd.Series)
            long_dict = dict(
                zip(stations['id'], locations['lon'].astype(float)))
            lat_dict = dict(
                zip(stations['id'], locations['lat'].astype(float)))
            type_dict = dict(zip(stations['id'], stations['stationType']))

            df['start_stat_name'] = df['start_stat_id'].map(stat_name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_name'] = df['end_stat_id'].map(stat_name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)
            df['station_type'] = df['end_stat_id'].map(type_dict)

            df.dropna(inplace=True)
            df = df[df.start_dt.dt.month == month]
            df.sort_values(by=['start_dt'], inplace=True)
            df.reset_index(inplace=True, drop=True)

        elif city == "guadalajara":

            try:
                df = pd.read_csv(
                    f"./data/{city}/datos_abiertos_{year:d}_{month:02d}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.mibici.net/en/open-data/') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            # Get station information from the nomenclatura file.
            station_files = [file for file in os.listdir(
                'data/guadalajara/') if 'nomenclatura' in file]
            try:
                stations = pd.read_csv(f"data/{city}/{station_files[0]}", encoding = "ISO-8859-1")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.mibici.net/en/open-data/') from exc

            stat_name_dict = dict(zip(stations['id'], stations['name']))
            long_dict = dict(zip(stations['id'], stations['longitude']))
            lat_dict = dict(zip(stations['id'], stations['latitude']))

            df['start_stat_name'] = df['start_stat_id'].map(stat_name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_name'] = df['end_stat_id'].map(stat_name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

            df['duration'] = (df['end_dt'] - df['start_dt']).dt.total_seconds()

        elif city == "montreal":
            try:
                df = pd.read_csv(
                    f"./data/{city}/OD_{year:d}-{month:02d}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://bixi.com/en/open-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            try:
                stations = pd.read_csv(f"data/{city}/Stations_{year}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://bixi.com/en/open-data') from exc

            stat_name_dict = dict(zip(stations['Code'], stations['name']))
            long_dict = dict(zip(stations['Code'], stations['longitude']))
            lat_dict = dict(zip(stations['Code'], stations['latitude']))

            df['start_stat_name'] = df['start_stat_id'].map(stat_name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_name'] = df['end_stat_id'].map(stat_name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)
            df['start_stat_id'] = df['start_stat_id'].astype(int)
            df['end_stat_id'] = df['end_stat_id'].astype(int)

            df['user_type'] = df['user_type'].map(
                {1: 'Subscriber',
                 0: 'Customer'})

        elif city == "taipei":
            colnames = ['start_t', 'start_stat_name_zh',
                        'end_t', 'end_stat_name_zh', 'duration', 'rent_date']

            try:
                df = pd.read_csv(f"./data/{city}/{year:d}{month:02d}-taipei.csv",
                                 usecols=range(5), names=colnames)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant data can be found at https://data.taipei/#/ and https://drive.google.com/drive/folders/1QsROgp8AcER6qkTJDxpuV8Mt1Dy6lGQO') from exc

            # Fix names of stations to match current naming scheme.
            df.replace(to_replace='信義杭州路口(中華電信總公司',
                            value='信義杭州路口(中華電信總公司)', inplace=True)
            df.replace(to_replace='捷運科技大樓站',
                            value='捷運科技大樓站(台北教育大學)', inplace=True)
            df.replace(to_replace='?公公園',
                            value='瑠公公園', inplace=True)
            df.replace(to_replace='饒河夜市',
                            value='饒河夜市(八德路側)', inplace=True)
            df.replace(to_replace='捷運大坪林站(3號出口)',
                            value='捷運大坪林站(1號出口)', inplace=True)
            df.replace(to_replace='新明路321巷口',
                            value='新明路262巷口', inplace=True)

            # Remove header in data containing header.
            if (df.loc[0] == ['rent_time', 'rent_station', 'return_time', 'return_station', 'rent']).all():
                df.drop(0, inplace=True)

            df['start_dt'] = pd.to_datetime(
                df['start_t'], format='%Y-%m-%d %H:%M:%S')
            df['end_dt'] = pd.to_datetime(
                df['end_t'], format='%Y-%m-%d %H:%M:%S')
            df['duration'] = pd.to_timedelta(df.duration).dt.total_seconds()

            try:
                stations = pd.DataFrame.from_dict(
                    list(pd.read_json("./data/taipei/YouBikeTP.json")['retVal']))
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No station data found. The data can be found at https://tcgbusfs.blob.core.windows.net/blobyoubike/YouBikeTP.json') from exc

            stations['sno'] = stations['sno'].astype(int)
            stations['lat'] = stations['lat'].astype(float)
            stations['lng'] = stations['lng'].astype(float)
            id_dict = dict(zip(stations['sna'], stations['sno']))

            # The data includes origin and destination stations in New Taipei.
            # It is possible to include these with the commented code below.

            # stations_ntpc = pd.read_csv("./data/stations_new_taipei.csv")
            # stations_ntpc['sno'] = stations_ntpc['sno'].astype(int)
            # stations_ntpc['lat'] = stations_ntpc['lat'].astype(float)
            # stations_ntpc['lng'] = stations_ntpc['lng'].astype(float)
            # id_dict_ntpc = dict(zip(stations_ntpc['sna'], stations_ntpc['sno']))
            # id_dict = {**id_dict_tp, **id_dict_ntpc}

            df['start_stat_id'] = df['start_stat_name_zh'].map(id_dict)
            df['end_stat_id'] = df['end_stat_name_zh'].map(id_dict)

            stat_name_dict = dict(zip(stations['sno'], stations['snaen']))
            long_dict = dict(zip(stations['sno'], stations['lng']))
            lat_dict = dict(zip(stations['sno'], stations['lat']))
            addr_dict = dict(zip(stations['sno'], stations['aren']))

            # name_dict_ntpc = dict(zip(stations_ntpc['sno'], stations_ntpc['snaen']))
            # long_dict_ntpc = dict(zip(stations_ntpc['sno'], stations_ntpc['lng']))
            # lat_dict_ntpc = dict(zip(stations_ntpc['sno'], stations_ntpc['lat']))
            # addr_dict_ntpc = dict(zip(stations_ntpc['sno'], stations_ntpc['aren']))

            # name_dict = {**name_dict_tp, **name_dict_ntpc}
            # long_dict = {**long_dict_tp, **long_dict_ntpc}
            # lat_dict = {**lat_dict_tp, **lat_dict_ntpc}
            # addr_dict = {**addr_dict_tp, **addr_dict_ntpc}

            df['start_stat_name'] = df['start_stat_id'].map(stat_name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)
            df['start_stat_desc'] = df['start_stat_id'].map(addr_dict)

            df['end_stat_name'] = df['end_stat_id'].map(stat_name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)
            df['end_stat_desc'] = df['end_stat_id'].map(addr_dict)

            #df_nan = df[df.isna().any(axis=1)]
            df.drop(columns=['start_t', 'end_t'], inplace=True)

            df.dropna(inplace=True)
            df.sort_values(by=['start_dt'], inplace=True)
            df.reset_index(inplace=True, drop=True)
        
        # Remove trips with duration less than 1 minute
        df = df[df['duration'] > 60] 
        df = df.reset_index(drop=True)
        
        if blocklist:
            df = df[~df['start_stat_id'].isin(blocklist)]
            df = df[~df['end_stat_id'].isin(blocklist)]

        with open(f'./python_variables/{city}{year:d}{month:02d}_dataframe_blcklst={blocklist}.pickle', 'wb') as file:
            pickle.dump(df, file)

        print('Pickling done.')

    print(f"Data loaded: {city}{year:d}{month:02d}")

    return df


def get_data_year(city, year, blocklist=None, overwrite=False):
    """
    Read bikeshare data for a whole year from provider provided files.

    Parameters
    ----------
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.
    year : int
        The year of interest in YYYY format.
    blocklist : list, optional
        List of IDs of stations to remove. Default is None.
    overwrite : bool, optional
        If True, create a new pickle regardless of whether there is an existing
        pickle.

    Returns
    -------
    df : pandas DataFrame
        Dataframe containing bikeshare trip data.
    days : dict
        Contains the indices of the first trip per day.

    """
    # Remember to update this list when ading support for new cities
    supported_cities = [
        'bergen', 'boston', 'buenos_aires', 'chicago', 'edinburgh', 'guadalajara',
        'helsinki', 'la', 'london', 'madrid', 'mexico', 'minneapolis', 'montreal',
        'nyc', 'oslo', 'sfran', 'taipei', 'trondheim', 'washdc']

    if city not in supported_cities:
        raise ValueError(
            f"This city is not currently supported. Supported cities are {supported_cities}")

    # Make folder for dataframes if not found
    if not os.path.exists('python_variables'):
        os.makedirs('python_variables')

    if not overwrite:
        try:
            print(f'Loading pickle {name_dict[city]} {year:d}... ', end="")
            with open(f'./python_variables/{city}{year:d}_dataframe.pickle', 'rb') as file:
                df = pickle.load(file)
            print('Done')

        except FileNotFoundError:
            print('\n Pickle not found. ', end="")
            overwrite = True

    if overwrite:
        print("Pickling dataframe", end="")
        if city == "nyc":

            files = [file for file in os.listdir(
                'data/nyc') if f'{year:d}' in file[:4] and 'citibike' in file]
            files.sort()

            if len(files) < 12:
                raise FileNotFoundError(
                    "Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")

            df = pd.read_csv('data/nyc/' + files[0])

            for file in files[1:]:
                df_temp = pd.read_csv('data/nyc/' + file)
                df = pd.concat([df, df_temp], sort=False)
                print(".", end="")

            df = df.rename(columns=dataframe_key.get_key(city))

            # Remove stations in Jersey City by splitting at the 74.026 degree
            # meridian.
            df = df[(df['start_stat_long'] > -74.026) & (df['end_stat_long'] > -74.026)]

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "washdc":

            files = [file for file in os.listdir(
                f'data/{city}') if f'{year:d}' in file[:4] and 'capitalbikeshare' in file]
            files.sort()

            if len(files) < 12:
                raise FileNotFoundError(
                    "Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")

            df = pd.read_csv(f'data/{city}/' + files[0])

            for file in files[1:]:
                df_temp = pd.read_csv(f'data/{city}/' + file)
                df = pd.concat([df, df_temp], sort=False)
                print(".", end="")

            df.reset_index(inplace=True, drop=True)

            df = df.rename(columns=dataframe_key.get_key(city))

            df['start_stat_lat'] = ''
            df['start_stat_long'] = ''
            df['end_stat_lat'] = ''
            df['end_stat_long'] = ''

            stations = pd.read_csv('data/washdc/Capital_Bike_Share_Locations.csv')

            long_dict = dict(zip(stations['TERMINAL_NUMBER'], stations['LONGITUDE']))
            lat_dict = dict(zip(stations['TERMINAL_NUMBER'], stations['LATITUDE']))

            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)

            df.dropna(inplace=True)

            max_lat = 38.961029
            min_lat = 38.792686
            max_long = -76.909415
            min_long = -77.139396

            df = df.iloc[np.where(
                (df['start_stat_lat'] < max_lat) &
                (df['start_stat_lat'] > min_lat) &
                (df['start_stat_long'] < max_long) &
                (df['start_stat_long'] > min_long))]

            df = df.iloc[np.where(
                (df['end_stat_lat'] < max_lat) &
                (df['end_stat_lat'] > min_lat) &
                (df['end_stat_long'] < max_long) &
                (df['end_stat_long'] > min_long))]

            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])

            df.drop(columns=['start_t', 'end_t'], inplace=True)

            df['user_type'] = df['user_type'].map(
                {'Member': 'Subscriber',
                 'Casual': 'Customer'})

        elif city == "chicago":

            files = [file for file in os.listdir(
                'data/chicago') if f'Divvy_Trips_{year:d}' in file]
            files.sort()

            if len(files) < 4:
                raise FileNotFoundError(
                    "Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")

            df = pd.read_csv(f'data/{city}/' + files[0])
            df = df.rename(columns=dataframe_key.get_key(city))

            for file in files[1:]:
                df_temp = pd.read_csv(f'data/{city}/' + file)
                df_temp = df_temp.rename(columns=dataframe_key.get_key(city))

                df = pd.concat([df, df_temp], sort=False, axis=0)
                print(".", end="")

            df.reset_index(inplace=True, drop=True)

            with open('./python_variables/Chicago_stations.pickle', 'rb') as file:
                stations = pickle.load(file)

            lat_dict = dict(zip(stations['name'], stations['latitude']))
            long_dict = dict(zip(stations['name'], stations['longitude']))

            df['start_stat_lat'] = df['start_stat_name'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_name'].map(long_dict)

            df['end_stat_lat'] = df['end_stat_name'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_name'].map(long_dict)

            df.replace('', np.nan, inplace=True)
            df.dropna(subset=['start_stat_lat',
                              'start_stat_long',
                              'end_stat_lat',
                              'end_stat_long'], inplace=True)

            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df['duration'] = df['duration'].str.replace(',', '').astype(float)

        elif city == "la":

            df = []
            for q in range(1, 4+1):
                try:
                    df.append(pd.read_csv(f'./data/{city}/metro-bike-share-trips-{year:d}-q{q}.csv'))
                    print(".", end="")

                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f'No trip data found for q{q}. All relevant files can be found at https://bikeshare.metro.net/about/data/') from exc
            df = pd.concat(df)

            df = df.rename(columns=dataframe_key.get_key(city))
            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])

            df = df[df['start_dt'].dt.year == year]
            df.drop(columns=['start_t', 'end_t'], inplace=True)
            df.dropna(inplace=True) # Removes stations 3000	Virtual Station, 4285 Metro Bike Share Free Bikes, 4286 Metro Bike Share Out of Service Area Smart Bike

            files = [file for file in os.listdir(
                f'data/{city}') if 'metro-bike-share-stations' in file]
            try:
                stations = pd.read_csv(f"data/{city}/{files[0]}", names=['stat_id', 'stat_name', 'date', 'authority', 'status'])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No station data found. All relevant files can be found at https://bikeshare.metro.net/about/data/') from exc

            stat_name_dict = dict(zip(stations['stat_id'], stations['stat_name']))
            df['start_stat_name'] = df['start_stat_id'].map(stat_name_dict)
            df['end_stat_name'] = df['end_stat_id'].map(stat_name_dict)

            df['duration'] = df['duration']*60

            df.reset_index(inplace=True, drop=True)

        elif city == "sfran":

            col_list = ['duration_sec', 'start_time', 'end_time', 'start_station_id',
                        'start_station_name', 'start_station_latitude',
                        'start_station_longitude', 'end_station_id', 'end_station_name',
                        'end_station_latitude', 'end_station_longitude', 'bike_id', 'user_type']

            files = []
            for file in os.listdir(f'data/{city}'):
                if f'{year:d}' and 'baywheels-tripdata' in file:
                    files.append(file)
                elif f'{year:d}' and 'fordgobike-tripdata' in file:
                    files.append(file)
            files.sort()

            if len(files) < 12:
                raise FileNotFoundError(
                    "Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")

            df = pd.DataFrame(columns=col_list)

            for file in files:
                df_temp = pd.read_csv(f'data/{city}/' + file)[col_list]
                df = pd.concat([df, df_temp], sort=False)
                print(".", end="")

            df = df.rename(columns=dataframe_key.get_key(city))
            if 'bike_share_for_all_trip' in df.columns:
                df['bike_share_for_all_trip'].fillna('N', inplace=True)
            if 'rental_access_method' in df.columns:
                df['rental_access_method'].fillna('N', inplace=True)
            df.dropna(inplace=True)

            df = df.iloc[np.where(df['start_stat_lat'] > 37.593220)]
            df = df.iloc[np.where(df['end_stat_lat'] > 37.593220)]

            df.sort_values(by='start_t', inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])

        elif city == "london":

            data_files = [file for file in os.listdir(
                f'data/{city}') if 'JourneyDataExtract' in file]
            data_files = [file for file in data_files if f'{year:d}' in file]
            data_files.sort()

            if len(data_files) == 0:
                raise FileNotFoundError(
                    f'No London data for {year:d} found. All relevant files can be found at https://cycling.data.tfl.gov.uk/.')

            if isinstance(data_files, str):
                warnings.warn(
                    'Only one data file found. Please check that you have all available data.')

            df = pd.read_csv(f'data/{city}/' + data_files[0])

            for file in data_files[1:]:
                df_temp = pd.read_csv(f'data/{city}/' + file)
                df = pd.concat([df, df_temp], sort=False)
                print(".", end="")

            df.rename(columns=dataframe_key.get_key(city), inplace=True)

            df['start_dt'] = pd.to_datetime(df['start_t'], format='%d/%m/%Y %H:%M')
            df['end_dt'] = pd.to_datetime(df['end_t'], format='%d/%m/%Y %H:%M')
            df.drop(columns=['start_t', 'end_t'], inplace=True)

            df = df[df['start_dt'].dt.year == year]

            df.sort_values(by='start_dt', inplace=True)
            df.reset_index(inplace=True, drop=True)

            stat_df = pd.read_csv(f'./data/{city}/london_stations.csv')
            stat_df.at[np.where(stat_df['station_id'] == 502)
                       [0][0], 'latitude'] = 51.53341

            long_dict = dict(zip(stat_df['station_id'], stat_df['longitude']))
            lat_dict = dict(zip(stat_df['station_id'], stat_df['latitude']))

            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)

            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)

            merge_id_dict = {361: 154, 374: 154,
                             280: 250, 819: 273, 328: 327, 336: 334,
                             421: 420, 816: 812}
            waterloo_dict = {f"Waterloo Station {i}, Waterloo": "Waterloo Station, Waterloo" for i in range(1, 4)}
            royal_dict = {f"Royal Avenue {i}, Chelsea": "Royal Avenue, Chelsea" for i in range(1, 3)}
            belvedere_dict = {f"Belvedere Road {i}, South Bank": "Belvedere Road, South Bank" for i in range(1, 3)}
            north_dict = {f"New North Road {i}, Hoxton": "New North Road, Hoxton" for i in range(1, 3)}
            concert_dict = {f"Concert Hall Approach {i}, South Bank": "Concert Hall Approach, South Bank" for i in range(1, 3)}
            south_dict = {f"Southwark Station {i}, Southwark": "Southwark Station, Southwark" for i in range(1, 3)}
            here_dict = {"Here East North, Queen Elizabeth Olympic Park": "Here East, Queen Elizabeth Olympic Park",
                          "Here East South, Queen Elizabeth Olympic Park": "Here East, Queen Elizabeth Olympic Park"}
            merge_name_dict = {**waterloo_dict, **royal_dict, **belvedere_dict, **north_dict, **concert_dict, **south_dict, **here_dict}

            df['start_stat_id'] = df['start_stat_id'].replace(merge_id_dict)
            df['start_stat_name'] = df['start_stat_name'].replace(merge_name_dict)

            df['end_stat_id'] = df['end_stat_id'].replace(merge_id_dict)
            df['end_stat_name'] = df['end_stat_name'].replace(merge_name_dict)

            df.reset_index(inplace=True, drop=True)


        elif city == 'mexico':
            files = [file for file in os.listdir(
                f'data/{city}') if 'mexico' in file]
            files = [file for file in files if f'{year:d}' in file]
            files.sort()

            if len(files) < 12:
                raise FileNotFoundError(
                    "Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")

            dfs = []

            for file in files:
                dfs.append(pd.read_csv(f'data/{city}/' + file))
                print(".", end="")

            if year == 2019:
                dfs[2].drop(index=dfs[2].loc[dfs[2]['Fecha_Arribo'] == '10'].index, inplace=True) # Remove datapoint from march
                dfs[2].drop(columns="Unnamed: 9", inplace=True)
                dfs[3].drop(index=dfs[3].loc[dfs[3]['Hora_Retiro'] == '18::'].index, inplace=True)
            df = pd.concat(dfs)

            df.rename(columns=dataframe_key.get_key(city), inplace=True)

            df['start_dt'] = pd.to_datetime(df['start_date'] + ' ' + df['start_time'],
                                            format='%d/%m/%Y %H:%M:%S')
            df['end_dt'] = pd.to_datetime(df['end_date'] + ' ' + df['end_time'],
                                          format='%d/%m/%Y %H:%M:%S')
            df.drop(['start_date', 'start_time', 'end_date',
                    'end_time'], axis=1, inplace=True)
            df['duration'] = (df['end_dt'] - df['start_dt']).dt.total_seconds()

            # purged = df[~(df['start_dt'].dt.year == year)] # Look at this for some good 818 day long trips :)
            df = df[df['start_dt'].dt.year == year]

            stations = pd.DataFrame(pd.read_json("./data/mexico/stations_mexico.json",
                                                 lines=True)['stations'][0])

            stat_name_dict = dict(zip(stations['id'], stations['address']))
            locations = stations['location'].apply(pd.Series)
            long_dict = dict(
                zip(stations['id'], locations['lon'].astype(float)))
            lat_dict = dict(
                zip(stations['id'], locations['lat'].astype(float)))
            type_dict = dict(zip(stations['id'], stations['stationType']))

            df['start_stat_name'] = df['start_stat_id'].map(stat_name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_name'] = df['end_stat_id'].map(stat_name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)
            df['station_type'] = df['end_stat_id'].map(type_dict)

            df.dropna(inplace=True)
            df.sort_values(by=['start_dt'], inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['user_type'] = df['user_type'].map(
                {1: 'Subscriber',
                 2: 'Customer',
                 3: 'Company Worker'})

        elif city == "buenos_aires":

            try:
                df = pd.read_csv(
                    f"./data/{city}/recorridos-realizados-{year:d}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://data.buenosaires.gob.ar/dataset/bicicletas-publicas') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            mask = df['start_stat_id'] == "159_0"
            df.loc[mask, 'start_stat_id'] = 159
            df.loc[mask, 'start_stat_lat'] = -34.584953
            df.loc[mask, 'start_stat_long'] = -58.437340
            df.loc[mask, 'start_stat_desc'] = ""

            mask = df['end_stat_id'] == "159_0"
            df.loc[mask, 'end_stat_id'] = 159
            df.loc[mask, 'end_stat_lat'] = -34.584953
            df.loc[mask, 'end_stat_long'] = -58.437340
            df.loc[mask, 'end_stat_desc'] = ""

            df['gender'].fillna(0, inplace=True)

            df.dropna(inplace=True)
            df['start_stat_id'] = df['start_stat_id'].astype(int)
            df['end_stat_id'] = df['end_stat_id'].astype(int)

            df.sort_values(by=['start_t'], inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city in ['edinburgh', 'bergen', 'trondheim', 'oslo']:
            dfs = []
            for month in get_valid_months(city, year):
                if city == 'edinburgh':
                    timestamp = f'{year:d}-{month:02d}'
                else:
                    timestamp = f'{year:d}{month:02d}'
                try:
                    df = pd.read_csv(f'./data/{city}/{timestamp}-{city}.csv')
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at') from exc

                df = df.rename(columns=dataframe_key.get_key(city))
                df.dropna(inplace=True)
                df.reset_index(inplace=True, drop=True)

                if city == 'oslo':
                    # Merge stations with identical coordinates
                    merge_id_dict = {619: 618}
                    merge_name_dict = {f"Bak Niels Treschows hus {i}": "Bak Niels Treschows hus" for i in ['sør', 'nord']}

                    df['start_stat_id'] = df['start_stat_id'].replace(merge_id_dict)
                    df['start_stat_name'] = df['start_stat_name'].replace(merge_name_dict)

                    df['end_stat_id'] = df['end_stat_id'].replace(merge_id_dict)
                    df['end_stat_name'] = df['end_stat_name'].replace(merge_name_dict)

                # Change timezone from UTC to wall time
                if city == 'edinburgh':
                    df['start_dt'] = pd.to_datetime(df['start_t']).dt.tz_convert('Europe/London')
                    df['end_dt'] = pd.to_datetime(df['end_t']).dt.tz_convert('Europe/London')
                else:
                    df['start_dt'] = pd.to_datetime(df['start_t']).dt.tz_convert('Europe/Oslo')
                    df['end_dt'] = pd.to_datetime(df['end_t']).dt.tz_convert('Europe/Oslo')
                df.drop(columns=['start_t', 'end_t'], inplace=True)
                dfs.append(df)
                print(".", end="")

            df = pd.concat(dfs)
            df.reset_index(inplace=True, drop=True)


        # For the other cities, the data is more nicely split into months, so
        # we can just import the data month by month and concatenate.
        elif city in [
                'madrid', 'edinburgh', 'taipei', 'bergen', 'boston',
                'guadalajara', 'trondheim', 'minneapolis', 'oslo', 'helsinki',
                'montreal']:
            dfs = []
            for month in get_valid_months(city, year):
                dfs.append(get_data_month(city, year, month, overwrite=overwrite))
            df = pd.concat(dfs)
            df.sort_values(by=['start_dt'], inplace=True)
            df.reset_index(inplace=True, drop=True)

        with open(f'./python_variables/{city}{year:d}_dataframe.pickle', 'wb') as file:
            pickle.dump(df, file)

        print(' Pickling done.')

    if blocklist:
        df = df[~df['start_stat_id'].isin(blocklist)]
        df = df[~df['end_stat_id'].isin(blocklist)]

    return df


def get_data_day(city, year, month, day, blocklist=None):
    """
    Read bikeshare data for a single day from provider provided files.

    Parameters
    ----------
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.
    year : int
        The year of interest in YYYY format.
    month : int
        The month in MM format.
    day : int
        The day in DD format.
    blocklist : list, optional
        List of IDs of stations to remove. Default is None.

    Returns
    -------
    df : pandas DataFrame
        Dataframe containing bikeshare trip data.

    """
    df = get_data_month(city, year, month, blocklist=blocklist)

    # either: Start day and start month as specified
    # or: End day and end month as specified
    df = df[((df['start_dt'].dt.day == day) & (df['start_dt'].dt.month == month)) | ((df['end_dt'].dt.day == day) & (df['end_dt'].dt.month == month))]

    return df


def station_locations(df, id_index):
    """
    Creates a dictionary with station IDs as keys and locations as values.

    Parameters
    ----------
    df : pandas DataFrame
        Bikeshare trip data.
    id_index : dict
        Maps station ID (arbitrary integer) to the range from 0 to number of stations

    Returns
    -------
    locations : dict
        key : station index (returned from id_index)
        value : tuple (longitude, latitude)

    """
    start_loc = df[['start_stat_id', 'start_stat_lat',
                    'start_stat_long']].drop_duplicates()
    end_loc = df[['end_stat_id', 'end_stat_lat',
                  'end_stat_long']].drop_duplicates()

    rename_dict = {
        'end_stat_id': 'stat_id',
        'end_stat_lat': 'lat',
        'end_stat_long': 'long',
        'start_stat_id': 'stat_id',
        'start_stat_lat': 'lat',
        'start_stat_long': 'long'}

    locs = pd.concat([start_loc.rename(columns=rename_dict),
                      end_loc.rename(columns=rename_dict)],
        axis=0, ignore_index=True)
    locs.drop_duplicates(inplace=True)

    duplicates = locs['stat_id'].duplicated()
    if sum(duplicates) > 0:
        print(f'There are {sum(duplicates)} stations which have moved.'
              ' Keeping first instance of the coordinates.')

        locs = locs[~duplicates]

    duplicates = locs[['lat', 'long']].duplicated()
    if sum(duplicates) > 0:
        print(f'There are {sum(duplicates)} stations'
              ' which have the same coordinates as other stations. This may lead to errors.')

        locs = locs[~duplicates]

    locs['stat_id'] = locs['stat_id'].map(id_index).drop_duplicates()
    locations = locs.set_index('stat_id').sort_index()[['long', 'lat']]
    locations.index.name = None
    return locations


def station_names(df, id_index):
    """
    Creates a dictionary with station IDs as keys and station names as values.

    Parameters
    ----------
    df : pandas DataFrame
        Bikeshare trip data.
    id_index : dict
        Maps station ID (arbitrary integer) to the range from 0 to number of stations

    Returns
    -------
    names : dict
        key : station index (returned from id_index)
        value : string containing station name

    """

    start_name = df[['start_stat_id', 'start_stat_name']].drop_duplicates()
    end_name = df[['end_stat_id', 'end_stat_name']].drop_duplicates()

    nams = pd.concat([start_name, end_name.rename(
        columns={'end_stat_id': 'start_stat_id',
                 'end_stat_name': 'start_stat_name'})],
        axis=0, ignore_index=True)
    nams.drop_duplicates(inplace=True)
    nams['start_stat_id'] = nams['start_stat_id'].map(id_index)
    names = nams.set_index('start_stat_id').sort_index().to_dict()['start_stat_name']
    return names


def get_elevation(lat, long, dataset="aster30m"):
    """
    Finds the elevation for a specific coordinate or list of coordinates.

    Elevation data is taken from https://www.opentopodata.org/

    Parameters
    ----------
    lat : float or iterable
        Latitude or iterable containing latitudes.
    long : float or iterable
        Longitude or iterable containing longitudes.
    dataset : str, optional
        Dataset used for elevation data. The default is "aster30m".

    Returns
    -------
    elevation : ndarray
        Array containing elevations.

    """

    if (lat is None) or (long is None):
        return None

    elevation = np.array([])

    if isinstance(long, float):
        query = (f'https://api.opentopodata.org/v1/{dataset}?locations='
                 f'{lat},{long}')
        print(query)
        # Request with a timeout for slow responses
        r = get(query, timeout=60)
        # Only get the json response in case of 200 or 201
        if r.status_code == 200 or r.status_code == 201:
            elevation = pd.json_normalize(r.json(), 'results')['elevation']

    else:
        # if it is a list or iterable
        i = 100

        for n in range(0, len(long), i):
            lo = long[n:n+i]
            la = lat[n:n+i]
            loc_string = f'https://api.opentopodata.org/v1/{dataset}?locations='

            for at, ong in zip(la, lo):
                loc_string = loc_string + f"{at},{ong}|"
            loc_string = loc_string[:-1]

            query = (loc_string)
            
            response = False
            
            while response == False:
                
                r = get(query, timeout=60)
                
                # Only get the json response in case of 200 or 201
                if r.status_code == 200 or r.status_code == 201:
                    elevation = np.append(
                        elevation, np.array(pd.json_normalize(
                            r.json(), 'results')['elevation']))
                    response = True
                else:
                    sleeptime = np.random.randint(1, 6)
                    warnings.warn(f"No json response. Status code = {r.status_code}, waiting {sleeptime}s")
                    time.sleep(sleeptime)
            
            
            if n == 0:
                print('Waiting for elevation API', end='')
                time.sleep(1)
            elif n < len(long)-i:
                print('.', end='')
                time.sleep(1)
            else:
                print(' Done')
    return elevation


def get_weather(city, year, month, key):
    """
    Get weather data for the given city, year and month. Uses
    api.worldweatheronline.com. An API key is necessary to use the API. There
    is a free trial available.

    Parameters
    ----------
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.
    key : str
        The API key for api.worldweatheronline.com

    Returns
    -------
    request : str
        Metadata about the requested weather, including city name.
    rain : pandas DataFrame
        Has columns 'month', 'day', 'hour', 'time_dt', 'precipMM', 'tempC',
        'windspeedKmph', 'desc'.

    """

    city_name = name_dict[city]

    n_days = calendar.monthrange(year, month)[1]
    tp = 1 # Time period.
    query = (f"http://api.worldweatheronline.com/premium/v1/past-weather.ashx?"
             f"key={key}&q={city_name}&format=json&date={year}-{month}-01&enddate={year}-{month}-{n_days}&tp={tp}")  # Request with a timeout for slow responses
    r = get(query, timeout=60)
    # Only get the json response in case of 200 or 201

    if r.status_code == 200 or r.status_code == 201:
        result = r.json()['data']

    request = result['request']
    weather = pd.DataFrame(result['weather'])

    hourweather = [pd.DataFrame(i) for i in weather['hourly']]

    for day, wea in zip(weather['date'], hourweather):
        wea['day'] = day[-2:]

    hweather = pd.concat(hourweather)

    hweather.reset_index(inplace=True)
    hweather['hour'] = (hweather['time'].astype(int)//100).astype(str)
    hweather['precipMM'] = hweather['precipMM'].astype(float)

    hweather['time_dt'] = pd.to_datetime(
        str(year) + f'{month:02n}' + hweather['day'] + hweather['hour'],
        format='%Y%m%d%H')

    hweather['day'] = hweather['day'].astype(int)
    hweather['hour'] = hweather['hour'].astype(int)
    hweather['month'] = month
    hweather['desc'] = pd.DataFrame(hweather['weatherDesc'].explode().tolist())

    rain = hweather[['month', 'day', 'hour', 'time_dt',
                     'precipMM', 'tempC', 'windspeedKmph', 'desc']]

    return request, rain


def get_weather_year(city, year, key):
    """
    Get weather data for the given city, year and month.

    Parameters
    ----------
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.
    key : str
        The API key for api.worldweatheronline.com

    Returns
    -------
    request : str
        Metadata about the requested weather, including city name.
    rain : pandas DataFrame
        Has columns 'month', 'day', 'hour', 'time_dt', 'precipMM', 'tempC',
        'windspeedKmph', 'desc'.

    """

    rain = []
    for month in range(1, 13):
        rain.append(get_weather(city, year, month, key)[1])

    return pd.concat(rain)


def purge_pickles(city, year, month):
    """
    Delete pickles regarding a specific city, year and month.

    Parameters
    ----------
    city : str
        name of city.
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.

    Returns
    -------
    None.

    """

    lookfor = f'{city}{year:d}{month:02d}'

    print("Purging in 'python_variables'...")

    for file in os.listdir('python_variables'):
        if lookfor in file:
            os.remove('python_variables/' + file)

    print('Purging done')


def nuke_pickles(cities):
    """
    Delete all pickles for a city, except for average station dataframes.

    Parameters
    ----------
    city : str
        name of city.
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.

    Returns
    -------
    None.

    """

    if isinstance(cities, str):

        if cities == 'all':

            cities = [ 'bergen', 'boston', 'buenos_aires', 'chicago',
                      'edinburgh', 'guadalajara', 'helsinki', 'la', 'london',
                      'madrid', 'mexico', 'minneapolis', 'montreal','nyc',
                      'oslo', 'sfran', 'sjose', 'taipei', 'trondheim', 'washdc']

            for city in cities:

                lookfor = f'{city}'

                print(f"Nuking {city} in 'python_variables'...")

                for file in os.listdir('python_variables'):
                    if lookfor in file:
                        if 'avg_stat_df' not in file:
                            os.remove('python_variables/' + file)

            print('Nuke succesful. What have we done...')

        else:
            lookfor = f'{cities}'

            print(f"Nuking {cities} in 'python_variables'...")

            for file in os.listdir('python_variables'):
                if lookfor in file:
                    if 'avg_stat_df' not in file:
                        os.remove('python_variables/' + file)

            print('Nuke succesful. What have we done...')

    else:
        for city in cities:

            lookfor = f'{city}'

            print(f"Nuking in {city} 'python_variables'...")

            for file in os.listdir('python_variables'):
                if lookfor in file:
                    if 'avg_stat_df' not in file:
                        os.remove('python_variables/' + file)

        print('Nuke succesful. What have we done...')


class Stations:
    """
    Represents the stations in the bike-share data.

    Attributes
    ----------
    n_start : int
        Number of stations at which trips start
    n_end : int
        Number of stations at which trips end
    n_tot : int
        Total number of stations
    id_index : dict
        Maps station id (arbitrary integer) to the range from 0 to number of stations
    inverse : dict
        Maps the range from 0 to number of stations to station id (arbitrary integer)
    locations : dict
        Key : station index (returned from id_index)
        Value : tuple (longitude, latitude)
    loc : np array
        Locations as numpy array
    loc_merc : np array
        Locations in web mercator projection
    names : dict
        Key: station number (in range from 0 to number of stations)
        Value: Name of station

    Methods
    -------
    None
    """

    def __init__(self, df):
        """
        Initialises the class.

        Parameters
        ----------
        df : pandas DataFrame
            Bikeshare trip data.

        """
        print("Loading Stations", end="")
        self.n_start = len(df['start_stat_id'].unique())
        self.n_end = len(df['end_stat_id'].unique())
        total_station_id = set(df['start_stat_id']).union(
            set(df['end_stat_id']))
        self.n_tot = len(total_station_id)
        print(".", end="")

        self.id_index = dict(
            zip(sorted(total_station_id), np.arange(self.n_tot)))

        self.inverse = dict(
            zip(np.arange(self.n_tot), sorted(total_station_id)))
        print(".", end="")
        self.locations = station_locations(df, self.id_index)
        print(".", end="")
        self.names = station_names(df, self.id_index)
        
        print(" Done")


class Data:
    """
    Class containing relevant data of a month for a city.

    The cities which are currently supported by the Data class and their
    identification are:

    New York City: nyc
        Relevant data can be found at https://www.citibikenyc.com/system-data
    Washington DC: washdc
        Relevant data can be found at https://www.capitalbikeshare.com/system-data
    Chicago: chicago
        Relevant data can be found at https://www.divvybikes.com/system-data
    San Francisco: sfran
        Relevant data can be found at https://www.lyft.com/bikes/bay-wheels/system-data
    San Jose: sjose
        Relevant data can be found at https://www.lyft.com/bikes/bay-wheels/system-data
    London: london
        Relevant data can be found at https://cycling.data.tfl.gov.uk/
    Oslo: oslo
        Relevant data can be found at https://oslobysykkel.no/en/open-data/historical
    Edinburgh: edinburgh
        Relevant data can be found at https://edinburghcyclehire.com/open-data/historical
    Bergen: bergen
        Relevant data can be found at https://bergenbysykkel.no/en/open-data/historical
    Buenos Aires: buenos_aires
        Relevant data can be found at https://data.buenosaires.gob.ar/dataset/bicicletas-publicas
    Madrid: madrid
        Relevant data can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)
    Mexico City: mexico
        Relevant data can be found at https://www.ecobici.cdmx.gob.mx/en/informacion-del-servicio/open-data
    Taipei: taipei
        Relevant data can be found at https://data.taipei/#/ and
        https://drive.google.com/drive/folders/1QsROgp8AcER6qkTJDxpuV8Mt1Dy6lGQO

    Attributes
    ----------
    city : str
        The identification of the city
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.
    num_days : int
        Number of days in the month
    Weekdays : list
        Weekdays of days in the month.
        Index i is day no. i+1, that is index 0 is day 1. Weekday 0 is monday.
    df : pandas DataFrame
        Bikeshare trip data.
    d_index : dict
        Index of the first trip starting on the given day
    stat : object of Stations class
        Contains station information
    adj : dict
        Adjacency matrices are stored in this dict
    deg : dict
        Degree matrices are stored in this dict
    lap : dict
        Laplace matrices are stored in this dict

    Methods
    -------
    daily_traffic_average(self, stat_index, period='b', normalise=True, plot=False, return_all=False, return_fig=False, return_std=False, user_type='all'):
        Computes the average daily traffic of a station over either business
        days or weekends. Both average number of departures and arrivals are
        computed for each hour.

    daily_traffic_average_all(self, period='b', normalise=True, plot=False, return_all=False, holidays=True, user_type='all'):
        Computes the average daily traffic of a station over either business
        days or weekends. Both average number of departures and arrivals are
        computed for each hour.

    pickle_daily_traffic(self, normalise=True, plot=False, overwrite=False, holidays=True, user_type='all'):
        Pickles matrices containing the average number of departures and
        arrivals to and from each station for every hour. One matrix
        contains the average traffic on business days while the other contains
        the average traffic for weekends.

    pickle_dump(self):
        dumps Data object as pickle

    """

    def __init__(self, city, year, month=None, day=None, blocklist=None, overwrite=False, day_type=None, user_type=None, remove_loops=False):
        """
        Parameters
        ----------
        city : str
            the identification of a city either: "nyc" or "sf"
        year : int
            the year of interest
        month : int
            the month of interest

        """

        # Make folder for python variables if not created
        if not os.path.exists('python_variables'):
            os.makedirs('python_variables')

        self.city = city
        self.year = year
        self.month = month
        self.day = day

        if self.day is not None:
            self.num_days = 1
            self.weekdays = [calendar.weekday(year, month, day)]

            self.df = get_data_day(city, year, month, day, blocklist)

        elif self.month is not None:
            first_weekday, self.num_days = calendar.monthrange(year, month)

            self.weekdays = [(i+(first_weekday)) %
                             7 for i in range(self.num_days)]

            self.df = get_data_month(
                city, year, month, blocklist, overwrite=overwrite)

        else:
            self.num_days = 365 + 1*calendar.isleap(year)
            first_weekday = calendar.weekday(year, 1, 1)

            self.weekdays = [(i+(first_weekday)) %
                             7 for i in range(self.num_days)]

            self.df = get_data_year(city, year, blocklist, overwrite=overwrite)

        if day_type == 'business_days':
            self.df = self.df[self.df['start_dt'].dt.weekday <= 4] # business days minus holidays
            self.df = self.df[self.df['end_dt'].dt.weekday <= 4] # business days minus holidays
            holiday_year = pd.DataFrame(
                get_cal(city).get_calendar_holidays(year), columns=['day', 'name'])
            holiday_list = holiday_year['day'].tolist()
            self.df = self.df[~self.df['start_dt'].dt.date.isin(holiday_list)] # Rows which are not in holiday list
            self.df = self.df[~self.df['end_dt'].dt.date.isin(holiday_list)] # Rows which are not in holiday list
            
        elif day_type == 'weekend':
            self.df = self.df[self.df['start_dt'].dt.weekday > 4] # weekend
            self.df = self.df[self.df['end_dt'].dt.weekday > 4] # weekend
        # else: Keep dataframe as is.

        if (user_type == 'Subscriber') and ('user_type' in self.df.columns):
            self.df = self.df[self.df['user_type'] == 'Subscriber'] # weekend
        elif (user_type == 'Customer') and ('user_type' in self.df.columns):
            self.df = self.df[self.df['user_type'] == 'Customer'] # weekend
        # else: Keep dataframe as is
        
        # Loops have same start and end station.
        if remove_loops is True:
            self.df = self.df[self.df['start_stat_id'] != self.df['end_stat_id']]

        self.stat = Stations(self.df)
        
        if self.city in system_center_dict.keys():
            self.laea_crs = pyproj.crs.CRS(f"+proj=laea +lat_0={system_center_dict[self.city]['lat']} +lon_0={system_center_dict[self.city]['long']}")
        


    def daily_traffic_average(self, stat_index, period='b', normalise=True, plot=False, return_all=False, return_fig=False, return_std=False, user_type='all'):
        """
        Computes the average daily traffic of a station over either business
        days or weekends. Both average number of departures and arrivals are
        computed for each hour.

        Parameters
        ----------
        stat_index : int
            Station index.
        period : str, optional
            Period to average over. Either 'b' = business days or 'w' = weekends.
            The default is 'b'.
        plot : bool, optional
            Plots the average daily traffic if set to True. The default is False.

        Raises
        ------
        ValueError
            Raised if period is not given as 'b' or 'w'.

        Returns
        -------
        trips_departures_average : ndarray
            24-dimensional array containing average number of departures for
            each hour.
        trips_arrivals_average : ndarray
            24-dimensional array containing average number of arrivals for
            each hour.

        """
        weekdays = self.weekdays

        if self.day != None:
            days = [self.day]
        # In case of a month, days will hold the day in the month. In case of a
        # year, days will hold the day of year.
        elif period == 'b':
            days = [date+1 for date, day in enumerate(weekdays) if day <= 4]
        elif period == 'w':
            days = [date+1 for date, day in enumerate(weekdays) if day > 4]
        else:
            raise ValueError(
                "Please provide the period as either 'b' = business days or 'w' = weekends")

        if user_type != 'all':
            if 'user_type' in self.df.columns:
                df = self.df.loc[self.df['user_type'] == user_type, ['start_stat_id', 'start_dt', 'end_stat_id', 'end_dt']]
            else:
                warnings.warn('dataframe contains no "user_type". Continuing with all users.')
                df = self.df[['start_stat_id', 'start_dt', 'end_stat_id', 'end_dt']]
            if len(df) == 0:
                raise NameError(f'user_type {user_type} not found in user_type column in DataFrame.')
        else:
            df = self.df[['start_stat_id', 'start_dt', 'end_stat_id', 'end_dt']]

        df_start = df[df['start_stat_id'] ==
                           self.stat.inverse[stat_index]]['start_dt']
        df_end = df[df['end_stat_id'] ==
                         self.stat.inverse[stat_index]]['end_dt']

        trips_arrivals = np.zeros(shape=(len(days), 24))
        trips_departures = np.zeros(shape=(len(days), 24))

        if self.month == None:  # If data is from whole year
            start_day = df_start.dt.dayofyear
            end_day = df_end.dt.dayofyear
        else:
            start_day = df_start.dt.day
            end_day = df_end.dt.day

        start_hour = df_start.dt.hour
        end_hour = df_end.dt.hour

        for i, day in enumerate(days):
            for hour in range(24):
                trips_departures[i, hour] = np.sum(
                    (start_day == day) & (start_hour == hour))
                trips_arrivals[i, hour] = np.sum(
                    (end_day == day) & (end_hour == hour))

        trips_departures_average = np.mean(trips_departures, axis=0)
        trips_arrivals_average = np.mean(trips_arrivals, axis=0)

        trips_departures_std = np.std(trips_departures, axis=0)
        trips_arrivals_std = np.std(trips_arrivals, axis=0)

        if normalise:
            divisor = trips_arrivals.sum(axis=1) + trips_departures.sum(axis=1)

            trips_arrivals = np.divide(trips_arrivals.T, divisor, out=np.zeros_like(
                trips_arrivals.T), where=divisor != 0).T
            trips_departures = np.divide(trips_departures.T, divisor, out=np.zeros_like(
                trips_arrivals.T), where=divisor != 0).T

        trips_departures_average = np.mean(trips_departures, axis=0)
        trips_arrivals_average = np.mean(trips_arrivals, axis=0)

        trips_departures_std = np.std(trips_departures, axis=0)
        trips_arrivals_std = np.std(trips_arrivals, axis=0)

        if plot:

            fig, ax = plt.subplots()

            if normalise:
                ax.plot(np.arange(24), trips_arrivals_average*100)
                ax.plot(np.arange(24), trips_departures_average*100)

                ax.fill_between(np.arange(24), trips_arrivals_average*100-trips_arrivals_std*100,
                                trips_arrivals_average*100+trips_arrivals_std*100,
                                facecolor='b', alpha=0.2)
                ax.fill_between(np.arange(24), trips_departures_average*100-trips_departures_std*100,
                                trips_departures_average*100+trips_departures_std*100,
                                facecolor='orange', alpha=0.2)
                ax.set_ylabel('% of total trips')

            else:
                ax.plot(np.arange(24), trips_arrivals_average)
                ax.plot(np.arange(24), trips_departures_average)

                ax.fill_between(np.arange(24), trips_arrivals_average-trips_arrivals_std,
                                trips_arrivals_average+trips_arrivals_std,
                                facecolor='b', alpha=0.2)
                ax.fill_between(np.arange(24), trips_departures_average-trips_departures_std,
                                trips_departures_average+trips_departures_std,
                                facecolor='orange', alpha=0.2)
                ax.set_ylabel('# trips')

            ax.set_xticks(np.arange(24))
            # plt.legend(['Arrivals','Departures','$\pm$std - arrivals','$\pm$std - departures'])
            ax.set_xlabel('Hour')

            if period == 'b':
                ax.set_title(
                    f'Average hourly traffic for {self.stat.names[stat_index]} \n in {month_dict[self.month]} {self.year} on business days')

            elif period == 'w':
                ax.set_title(
                    f'Average hourly traffic for {self.stat.names[stat_index]} \n in {month_dict[self.month]} {self.year} on weekends')
            if not return_fig:
                plt.show()

        if return_fig:
            return fig
        elif return_std:
            return trips_departures_average, trips_arrivals_average, trips_departures_std, trips_arrivals_std
        elif return_all:
            return trips_departures_average, trips_arrivals_average, trips_departures, trips_arrivals
        else:
            return trips_departures_average, trips_arrivals_average


    def daily_traffic_average_all(self, period='b', normalise=True, plot=False, return_all=False, holidays=True, user_type='all'):
        """
        Computes the average daily traffic of a station over either business
        days or weekends. Both average number of departures and arrivals are
        computed for each hour.

        Parameters
        ----------
        stat_index : int
            Station index.
        period : str, optional
            Period to average over. Either 'b' = business days or 'w' = weekends.
            The default is 'b'.
        plot : bool, optional
            Plots the average daily traffic if set to True. The default is False.

        Raises
        ------
        ValueError
            Raised if period is not given as 'b' or 'w'.

        Returns
        -------
        trips_departures_average : ndarray
            24-dimensional array containing average number of departures for
            each hour.
        trips_arrivals_average : ndarray
            24-dimensional array containing average number of arrivals for
            each hour.

        """

        if user_type != 'all':
            if 'user_type' in self.df.columns:
                df = self.df.loc[self.df['user_type'] == user_type, ['start_stat_id', 'start_dt']]
            else:
                warnings.warn('dataframe contains no "user_type". Continuing with all users.')
                df = self.df[['start_stat_id', 'start_dt']]
            if len(df) == 0:
                raise NameError(f'user_type {user_type} not found in user_type column in DataFrame.')
        else:
            df = self.df[['start_stat_id', 'start_dt']]
        # Departures

        if period == 'b':
            df = df[df['start_dt'].dt.weekday <= 4]
            if not holidays:
                holiday_year = pd.DataFrame(
                    get_cal(self.city).get_calendar_holidays(self.year), columns=['day', 'name'])
                holiday_list = holiday_year['day'].tolist()
                df = df[~df['start_dt'].dt.date.isin(holiday_list)] # Rows which are not in holiday list
            else:
                holiday_list = []

            weekmask = '1111100'
        elif period == 'w':
            df = df[df['start_dt'].dt.weekday > 4]
            holiday_list = []
            weekmask = '0000011'

        if self.month == None:
            n_days = np.busday_count(datetime.date(self.year, 1, 1), datetime.date(self.year+1, 1, 1), weekmask=weekmask, holidays=holiday_list)

        else:
            if self.month != 12:
                n_days = np.busday_count(datetime.date(self.year, self.month, 1), datetime.date(self.year, self.month+1, 1), weekmask=weekmask, holidays=holiday_list)
            else:
                n_days = np.busday_count(datetime.date(self.year, self.month, 1), datetime.date(self.year+1, 1, 1), weekmask=weekmask, holidays=holiday_list)

        print(f'Departures {period}...')


        stations_start = df['start_stat_id'].unique()

        start_mean = []
        start_std = []

        for station in stations_start:
            subset = df['start_dt'][df['start_stat_id'] == station]
            day_hour_count = pd.concat({'date': subset.dt.date, 'hour': subset.dt.hour}, axis=1).value_counts().unstack(fill_value=0)
            
            start_mean.append(day_hour_count.mean().rename(station))
            start_std.append(day_hour_count.std().rename(station))
            # shap = day_hour_count.shape
            # If shap[0] > n_days, there are too many days in the data.
            # start_mean.append(day_hour_count.sum(axis=0).rename(station) / n_days)
            # start_std.append(
            #     pd.concat(
            #         (day_hour_count, pd.DataFrame(np.zeros((n_days - shap[0], shap[1])),columns = day_hour_count.columns))
            #               ).std(axis=0).rename(station))

        departures_mean = pd.concat(start_mean, axis=1).fillna(0)
        departures_std = pd.concat(start_std, axis=1).fillna(0)


        # Arrivals
        if user_type != 'all':
            if 'user_type' in self.df.columns:
                df = self.df.loc[self.df['user_type'] == user_type, ['end_stat_id', 'end_dt']]
            else:
                warnings.warn('dataframe contains no "user_type". Continuing with all users.')
                df = self.df[['end_stat_id', 'end_dt']]
            if len(df) == 0:
                raise NameError(f'user_type {user_type} not found in user_type column in DataFrame.')
        else:
            df = self.df[['end_stat_id', 'end_dt']]

        if period == 'b':
            df = df[df['end_dt'].dt.weekday <= 4]
            df = df[~df['end_dt'].dt.date.isin(holiday_list)]
        elif period == 'w':
            df = df[['end_stat_id', 'end_dt']][df['end_dt'].dt.weekday > 4]


        print(f'Arrivals {period}...')

        stations_end = df['end_stat_id'].unique()

        end_mean = []
        end_std = []

        for station in stations_end:
            subset = df['end_dt'][df['end_stat_id'] == station]
            day_hour_count = pd.concat({'date': subset.dt.date, 'hour': subset.dt.hour}, axis=1).value_counts().unstack(fill_value=0)
            day_hour_count.index = pd.to_datetime(day_hour_count.index)
            if self.month == None:
                day_hour_count = day_hour_count.loc[day_hour_count.index.year == self.year]
            else:
                day_hour_count = day_hour_count.loc[(day_hour_count.index.month == self.month) & (day_hour_count.index.year == self.year)]
            
            end_mean.append(day_hour_count.mean().rename(station))
            end_std.append(day_hour_count.std().rename(station))
            # shap = day_hour_count.shape
            # end_mean.append(day_hour_count.sum(axis=0).rename(station) / n_days)
            # end_std.append(
            #     pd.concat(
            #         (day_hour_count, pd.DataFrame(np.zeros((n_days - shap[0], shap[1])),columns = day_hour_count.columns))
            #               ).std(axis=0).rename(station))

        arrivals_mean = pd.concat(end_mean, axis=1).fillna(0)
        arrivals_std = pd.concat(end_std, axis=1).fillna(0)

        if normalise:

            # normalise with respect to the sum of trips
            divisor = arrivals_mean.sum(axis=0).add(departures_mean.sum(axis=0), fill_value=0)

            arrivals_std = arrivals_std / divisor
            arrivals_mean = arrivals_mean / divisor

            departures_std = departures_std / divisor
            departures_mean = departures_mean / divisor

        arrivals_std = arrivals_std.T
        arrivals_mean = arrivals_mean.T

        departures_std = departures_std.T
        departures_mean = departures_mean.T

        if plot:
            for station in self.stat.id_index.keys():
                print(station)
                try:
                    tda = departures_mean.loc[station]
                    tds = departures_std.loc[station]
                except KeyError:
                    tda = pd.Series(np.zeros(24))
                    tds = pd.Series(np.zeros(24))
                try:
                    taa = arrivals_mean.loc[station]
                    tas = arrivals_std.loc[station]
                except KeyError:
                    taa = pd.Series(np.zeros(24))
                    tas = pd.Series(np.zeros(24))
                if normalise:

                    plt.plot(np.arange(24), taa*100)
                    plt.plot(np.arange(24), tda*100)

                    plt.fill_between(np.arange(24), taa*100-tas*100,
                                     taa*100+tas*100,
                                     facecolor='b', alpha=0.2)
                    plt.fill_between(np.arange(24), tda*100-tds*100,
                                     tda*100+tds*100,
                                     facecolor='orange', alpha=0.2)
                    plt.ylabel('% of total trips')

                else:
                    plt.plot(np.arange(24), taa)
                    plt.plot(np.arange(24), tda)

                    plt.fill_between(np.arange(24), taa-tas,
                                     taa+tas,
                                     facecolor='b', alpha=0.2)
                    plt.fill_between(np.arange(24), tda-tds,
                                     tda+tds,
                                     facecolor='orange', alpha=0.2)
                    plt.ylabel('# trips')

                plt.xticks(np.arange(24))
                # plt.legend(['Arrivals','Departures','$\pm$std - arrivals','$\pm$std - departures'])
                plt.xlabel('Hour')

                if self.month == None:
                    monstr = ""
                else:
                    monstr = f"in {month_dict[self.month]}"
                if period == 'b':
                    plt.title(
                        f'Average hourly traffic for {self.stat.names[self.stat.id_index[station]]} \n {monstr} {self.year} on business days')

                elif period == 'w':
                    plt.title(
                        f'Average hourly traffic for {self.stat.names[self.stat.id_index[station]]} \n {monstr} {self.year} on weekends')

                plt.show()

        if return_all:
            return departures_mean, arrivals_mean, departures_std, arrivals_std
        else:
            return departures_mean, arrivals_mean


    def pickle_daily_traffic(self, normalise=True, plot=False, overwrite=False, 
                             holidays=True, user_type='all', day_type='business_days',
                             return_std=False):
        """
        Pickles matrices containing the average number of departures and
        arrivals to and from each station for every hour. One matrix
        contains the average traffic on business days while the other contains
        the average traffic for weekends.

        The matrices are of shape (n,48) where n is the number of stations.
        The first 24 entries in each row contains the average traffic for
        business days and the last 24 entries contain the same for weekends.


        Returns
        -------
        None.

        """
        if self.month == None:
            monstr = ""
        else:
            monstr = f"{self.month:02d}"
        if not holidays:
            monstr = monstr + "_no_holidays"
        if user_type != 'all' and 'user_type' in self.df.columns:
            monstr = monstr + f"_{user_type}"
        if normalise:
            monstr = monstr + "_normalised"
        if not overwrite:
            
            if return_std:
                try:
                    with open(f'./python_variables/daily_traffic_{self.city}{self.year:d}{monstr}_std.pickle', 'rb') as file:
                        matrix_b, matrix_w, matrix_b_std, matrix_w_std = pickle.load(file)
                    
                    if day_type == 'business_days':
                        return matrix_b, matrix_b_std
                    elif day_type == 'weekend':
                        return matrix_w, matrix_w_std                
                    else:
                        raise ValueError("Please provide either 'business_days' or 'weekend' as day_type")
                    
                except FileNotFoundError:
                    print("Daily traffic pickle not found with stds")
            
            else:
            
                try:
                    with open(f'./python_variables/daily_traffic_{self.city}{self.year:d}{monstr}.pickle', 'rb') as file:
                        matrix_b, matrix_w = pickle.load(file)
                    if day_type == 'business_days':
                        return matrix_b
                    elif day_type == 'weekend':
                        return matrix_w                
                    else:
                        raise ValueError("Please provide either 'business_days' or 'weekend' as day_type")
                    
                except FileNotFoundError:
                    print("Daily traffic pickle not found")
        
        print('Pickling average daily traffic for all stations...')
        pre = time.time()
        
        if return_std:
            departures_b, arrivals_b, departures_b_std, arrivals_b_std = self.daily_traffic_average_all(
                'b', normalise=normalise, plot=plot, holidays=holidays, 
                user_type=user_type, return_all=True)

            departures_w, arrivals_w, departures_w_std, arrivals_w_std = self.daily_traffic_average_all(
                'w', normalise=normalise, plot=plot, holidays=holidays, 
                user_type=user_type, return_all=True)
        
        else:
            departures_b, arrivals_b = self.daily_traffic_average_all(
                'b', normalise=normalise, plot=plot, holidays=holidays, 
                user_type=user_type)
    
            departures_w, arrivals_w = self.daily_traffic_average_all(
                'w', normalise=normalise, plot=plot, holidays=holidays, 
                user_type=user_type)
        
        
        zeroseries = pd.Series(np.zeros((24,)))
        departures_b = departures_b.add(zeroseries).fillna(0)
        arrivals_b = arrivals_b.add(zeroseries).fillna(0)

        departures_w = departures_w.add(zeroseries).fillna(0)
        arrivals_w = arrivals_w.add(zeroseries).fillna(0)
        
        if return_std:
            departures_b_std = departures_b_std.add(zeroseries).fillna(0)
            arrivals_b_std = arrivals_b_std.add(zeroseries).fillna(0)
    
            departures_w_std = departures_w_std.add(zeroseries).fillna(0)
            arrivals_w_std = arrivals_w_std.add(zeroseries).fillna(0)
        
        id_index = self.stat.id_index
        matrix_b = np.zeros((len(id_index.keys()), 48))
        matrix_w = np.zeros((len(id_index.keys()), 48))
        
        if return_std:
            matrix_b_std = np.zeros((len(id_index.keys()), 48))
            matrix_w_std = np.zeros((len(id_index.keys()), 48))
            
        
        for id_, index in zip(id_index.keys(), id_index.values()):
            try:
                matrix_b[index, :24] = departures_b.loc[id_]
                if return_std: 
                    matrix_b_std[index, :24] = departures_b_std.loc[id_]
            except KeyError:
                print(f"Key {id_} not found in departures weekdays.")
            try:
                matrix_b[index, 24:] = arrivals_b.loc[id_]
                if return_std: 
                    matrix_b_std[index, 24:] = arrivals_b_std.loc[id_]
            except KeyError:
                print(f"Key {id_} not found in arrivals weekdays.")
            try:
                matrix_w[index, :24] = departures_w.loc[id_]
                if return_std:
                    matrix_w_std[index, :24] = departures_w_std.loc[id_]
            except KeyError:
                print(f"Key {id_} not found in departures weekend.")
            try:
                matrix_w[index, 24:] = arrivals_w.loc[id_]
                if return_std:
                    matrix_w_std[index, 24:] = arrivals_w_std.loc[id_]
            except KeyError:
                print(f"Key {id_} not found in arrivals weekdays.")

        matrix_b = np.nan_to_num(matrix_b, copy=False, nan=0.0)
        matrix_w = np.nan_to_num(matrix_w, copy=False, nan=0.0)
        
        if return_std:
            matrix_b_std = np.nan_to_num(matrix_b_std, copy=False, nan=0.0)
            matrix_w_std = np.nan_to_num(matrix_w_std, copy=False, nan=0.0)

        if return_std:
            with open(f'./python_variables/daily_traffic_{self.city}{self.year:d}{monstr}_std.pickle', 'wb') as file:
                pickle.dump((matrix_b, matrix_w, matrix_b_std, matrix_w_std), file)
        else:
            with open(f'./python_variables/daily_traffic_{self.city}{self.year:d}{monstr}.pickle', 'wb') as file:
                pickle.dump((matrix_b, matrix_w), file)

        print(f'Pickling daily traffic done. Time taken: {(time.time()-pre):.1f} s')
        
        if return_std:
            
            if day_type == 'business_days':
                return matrix_b, matrix_b_std
            elif day_type == 'weekend':
                return matrix_w, matrix_w_std                
            else:
                raise ValueError("Please provide either 'business_days' or 'weekend' as day_type")
            
        else:
            if day_type == 'business_days':
                return matrix_b
            elif day_type == 'weekend':
                return matrix_w                
            else:
                raise ValueError("Please provide either 'business_days' or 'weekend' as day_type")
            

    def df_subset(self, days='all', hours='all', minutes='all', activity_type='all'):
        """
        Get subset of dataframe df.

        Parameters
        ----------
        days : 'all' or 'weekdays' or 'weekend' or list or np array, optional
            the days to choose. The default is 'all'.
        hours : 'all' or list or numpy array, optional
            the hours to choose. The default is 'all'.
        minutes : 'all' or list or numpy array, optional
            the minutes to choose. The default is 'all'.
        activity_type : 'departures' or 'arrivals' or 'all', optional
            The type of traffic which should be in the given timeframe. The default is 'all'.

        Returns
        -------
        subset : pandas dataframe
            The chosen subset.

        """
        df = self.df
        weekdays = self.weekdays

        activity_time_dict = {'departures': 'start_dt', 'arrivals': 'end_dt',
                              'd': 'start_dt', 'a': 'end_dt', 'start_dt': 'start_dt', 'end_dt': 'end_dt'}

        if activity_type == 'all':
            if (days == 'business_days') or (days == 'b'):
                subset = df[df['start_dt'].dt.day.isin(np.where(np.array(weekdays) < 5)[
                                                       0] + 1) | df['end_dt'].dt.day.isin(np.where(np.array(weekdays) < 5)[0] + 1)]
            elif (days == 'weekend') or (days == 'w'):
                subset = df[df['start_dt'].dt.day.isin(np.where(np.array(weekdays) >= 5)[
                                                       0] + 1) | df['end_dt'].dt.day.isin(np.where(np.array(weekdays) >= 5)[0] + 1)]
            elif type(days) == int:
                subset = df[(df['start_dt'].dt.day == days)
                            | (df['end_dt'].dt.day == days)]
            elif (type(days) == list) or (type(days) == np.ndarray):
                subset = df[df['start_dt'].dt.day.isin(
                    np.array(days)) | df['end_dt'].dt.day.isin(np.array(days))]
            elif days == 'all':
                subset = df
            else:
                raise TypeError(
                    'days should be "business_days", "weekend", "all", an int, a list, or a numpy array')

            if hours != 'all':
                if type(hours) == int:
                    subset = subset[(subset['start_dt'].dt.hour == hours) | (
                        subset['end_dt'].dt.hour == hours)]
                elif (type(hours) == list) or (type(hours) == np.ndarray):
                    subset = subset[subset['start_dt'].dt.hour.isin(
                        np.array(hours)) | subset['end_dt'].dt.hour.isin(np.array(hours))]
            if minutes != 'all':
                if type(minutes) == int:
                    subset = subset[(subset['start_dt'].dt.minute == minutes) | (
                        subset['end_dt'].dt.minute == minutes)]
                elif (type(minutes) == list) or (type(minutes) == np.ndarray):
                    subset = subset[subset['start_dt'].dt.minute.isin(
                        np.array(minutes)) | subset['end_dt'].dt.minute.isin(np.array(minutes))]

        elif activity_type in activity_time_dict.keys():
            act_time = activity_time_dict[activity_type]
            if (days == 'business_days') or (days == 'b'):
                subset = df[df[act_time].dt.day.isin(
                    np.where(np.array(weekdays) < 5)[0] + 1)]
            elif (days == 'weekend') or (days == 'b'):
                subset = df[df[act_time].dt.day.isin(
                    np.where(np.array(weekdays) >= 5)[0] + 1)]
            elif type(days) == int:
                subset = df[df[act_time].dt.day == days]
            elif (type(days) == list) or (type(days) == np.ndarray):
                subset = df[df[act_time].dt.day.isin(np.array(days))]
            elif days == 'all':
                subset = df
            else:
                raise TypeError(
                    'days should be "business_days", "weekend", "all", an int, a list, or a numpy array')

            if hours != 'all':
                if type(hours) == int:
                    subset = subset[subset[act_time].dt.hour == hours]
                elif (type(hours) == list) or (type(hours) == np.ndarray):
                    subset = subset[subset[act_time].dt.hour.isin(np.array(hours))]
            if minutes != 'all':
                if type(minutes) == int:
                    subset = subset[subset[act_time].dt.minute == minutes]
                elif (type(minutes) == list) or (type(minutes) == np.ndarray):
                    subset = subset[subset[act_time].dt.minute.isin(
                        np.array(minutes))]
        else:
            raise ValueError(
                'activity_type should be "arrivals", "departures", or "all"')
        return subset


def get_cal(city):
    """
    Holiday calendar from workalendar for a specific city. Holidays for a year
    can be obtained by

    get_cal(city).get_calendar_holidays(year)

    Parameters
    ----------
    city : str
        name of city.

    Returns
    -------
    cal : workalendar calendar.
        contains holiday information for the city.

    """
    if city == 'bergen':
        cal = Norway()
    elif city == 'boston':
        cal = Massachusetts()
    elif city == 'buenos_aires':
        cal = Argentina()
    elif city == 'chicago':
        cal = ChicagoIllinois()
    elif city == 'edinburgh':
        cal = Edinburgh()
    elif city == 'guadalajara':
        cal = Mexico()
        cal.include_holy_thursday = True
        cal.include_good_friday = True
    elif city == 'helsinki':
        cal = Finland()
    elif city == 'la':
        cal = California()
    elif city == 'london':
        cal = UnitedKingdom()
    elif city == 'madrid':
        cal = CommunityofMadrid()
    elif city == 'mexico':
        cal = Mexico()
        cal.include_holy_thursday = True
        cal.include_good_friday = True
    elif city == 'minneapolis':
        cal = Minnesota()
    elif city == 'montreal':
        cal = Quebec()
    elif city == 'nyc':
        cal = NewYork()
    elif city == 'oslo':
        cal = Norway()
    elif city == 'sfran':
        cal = CaliforniaSanFrancisco()
    elif city == 'taipei':
        cal = Taiwan()
    elif city == 'trondheim':
        cal = Norway()
    elif city == 'washdc':
        cal = DistrictOfColumbia()
    else:
        raise KeyError('Calendar key not found')
    return cal


def get_valid_months(city, year):
    """
    The months in which the different bikeshare systems operate. Is used to
    determine whether data is missing or simply does not exist.

    Parameters
    ----------
    city : str
        name of city.
    year : int
        year in YYYY format.

    Returns
    -------
    range
        iterable containing the operating months as int.

    """
    if city in ['trondheim', 'minneapolis']:
        return range(4, 11+1)
    elif city == 'oslo':
        return range(4, 12+1)
    elif city in ['helsinki', 'montreal']:
        return range(4, 10+1)
    elif city == 'minneapolis':
        return range(4, 11+1)
    else:
        return range(1, 12+1)


name_dict = {
    'bergen': 'Bergen',
    'boston': 'Boston',
    'buenos_aires': 'Buenos Aires',
    'chicago': 'Chicago',
    'edinburgh': 'Edinburgh',
    'guadalajara': 'Guadalajara',
    'helsinki': 'Helsinki',
    'london': 'London',
    'la': 'Los Angeles',
    'madrid': 'Madrid',
    'mexico': 'Mexico City',
    'minneapolis': 'Minneapolis',
    'montreal': 'Montreal',
    'nyc': 'New York City',
    'oslo': 'Oslo',
    'sfran': 'San Francisco',
    'taipei': 'Taipei',
    'trondheim': 'Trondheim',
    'washdc': 'Washington DC'}

month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None'}

system_center_dict = {  # Average of 2019 data station locations
    'bergen': {'long': 5.38771445313615, 'lat': 60.383577733994045},
    'boston': {'long': -71.08758305223354, 'lat': 42.35019750316253},
    'buenos_aires': {'long': -58.42347574860326, 'lat': -34.60310703308861},
    'chicago': {'long': -87.65545296053962, 'lat': 41.888300359494096},
    'edinburgh': {'long': -3.20581957820512, 'lat': 55.93236528347738},
    'guadalajara': {'long': -103.36447705145984, 'lat': 20.679296934525546},
    'helsinki': {'long': 24.9017870418595, 'lat': 60.190035541236846},
    'london': {'long': -0.1285233400760456, 'lat': 51.505906433548795},
    'la': {'long': -118.32312902690583, 'lat': 34.03107351121076},
    'madrid': {'long': -3.693598130516432, 'lat': 40.42447626338028},
    'minneapolis': {'long': -93.26299728704082, 'lat': 44.97102880906168},
    'montreal': {'long': -73.58646867352535, 'lat': 45.52028355707451},
    'nyc': {'long': -73.9659602859011, 'lat': 40.72974815433999},
    'oslo': {'long': 10.74278096032806, 'lat': 59.92224401048971},
    'sfran': {'long': -122.08716554792483, 'lat': 37.82120403828431},
    'taipei': {'long': 121.54354860523308, 'lat': 25.054554026691726},
    'trondheim': {'long': 10.405688671739478, 'lat': 63.42914081309114},
    'washdc': {'long': -77.03355484615385, 'lat': 38.89052952680653},
}

city_center_dict = {  # From OpenStreetMap May 2022
    'bergen': {'long': 5.3259192, 'lat': 60.3943055},
    'boston': {'long': -71.0582912, 'lat': 42.3602534},
    'buenos_aires': {'long': -58.4370894, 'lat': -34.6075682},
    'chicago': {'long': -87.6244212, 'lat': 41.8755616},
    'edinburgh': {'long': -3.1883749, 'lat': 55.9533456},
    'guadalajara': {'long': -103.338396, 'lat': 20.6720375},
    'helsinki': {'long': 24.9427473, 'lat': 60.1674881},
    'london': {'long': -0.1276474, 'lat': 51.5073219},
    'la': {'long': -118.242766, 'lat': 34.0536909},
    'madrid': {'long': -3.7035825, 'lat': 40.4167047},
    'mexico': {'long': -99.1331785, 'lat': 19.4326296},
    'minneapolis': {'long': -93.2654692, 'lat': 44.9772995},
    'montreal': {'long': -73.5698065, 'lat': 45.5031824},
    'nyc': {'long': -74.0060152, 'lat': 40.7127281},
    'oslo': {'long': 10.7389701, 'lat': 59.9133301},
    'sfran': {'long': -122.419906, 'lat': 37.7790262},
    'taipei': {'long': 121.5636796, 'lat': 25.0375198},
    'trondheim': {'long': 10.3951929, 'lat': 63.4305658},
    'washdc': {'long': -77.0365427, 'lat': 38.8950368},
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    pre = time.time()
    data = Data('nyc', 2019, None, overwrite=True, user_type='Subscriber', remove_loops=True)
    print(f"time taken: {time.time() - pre:.2f}s")
    traf_mats = data.pickle_daily_traffic(overwrite=True, holidays=False, 
                                          user_type='Subscriber')
    
    #traffic_arr, traffic_dep = data.daily_traffic_average_all()

