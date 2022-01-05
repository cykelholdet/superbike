"""
Created on Mon Feb 22 15:52:51 2021

@author: Mattek Group 3
"""
import os
import time
import pickle
import calendar
import warnings
import datetime

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import simpledtw as dtw
from pyproj import Transformer
from requests import get

import dataframe_key

from workalendar.europe import CommunityofMadrid, Finland, UnitedKingdom, Norway, Edinburgh
from workalendar.usa import NewYork, Massachusetts, ChicagoIllinois, DistrictOfColumbia, Minnesota, CaliforniaSanFrancisco, California
from workalendar.asia import Taiwan
from workalendar.america import Mexico, Argentina, Quebec


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
            'data') if 'Divvy_Stations' in file]

        col_list = ['id', 'name', 'latitude', 'longitude']
        key = {'ID': 'id', 'Station Name': 'name',
               'Latitude': 'latitude', 'Longitude': 'longitude'}

        try:
            stat_df = pd.read_csv(
                'data/Divvy_Bicycle_Stations_-_All_-_Map.csv').rename(columns=key)
            stat_df = stat_df[col_list]
        except FileNotFoundError:
            stat_df = pd.DataFrame(columns=col_list)

        for file in stat_files:
            df = pd.read_csv(f'./data/{file}')[col_list]
            stat_df = pd.concat([stat_df, df], sort=False)

        if stat_df.size == 0:
            raise FileNotFoundError(
                'No data files containing station data found. Please read the docstring for more information.') from exc

        stat_df.drop_duplicates(subset='name', inplace=True)

        with open('./python_variables/Chicago_stations.pickle', 'wb') as file:
            pickle.dump(stat_df, file)

    print('Pickle loaded')

    return stat_df


def get_JC_blacklist():
    """
    Constructs/updates a blacklist of stations in Jersey City area. The
    blacklist is created using historical biketrip datasets for the area.
    Use only if you know what you are doing.

    The relevant files can be found at:
    https://www.citibikenyc.com/system-data

    Raises
    ------
    FileNotFoundError
        Raised if no Jersey City dataset is found.

    Returns
    -------
    blacklist : list
        List of IDs of the Jersey City docking stations.

    """

    try:
        with open('./python_variables/JC_blacklist', 'rb') as file:
            blacklist = pickle.load(file)

    except FileNotFoundError:
        print('No previous blacklist found. Creating blacklist...')
        blacklist = set()

    JC_files = [file for file in os.listdir('data') if 'JC' in file]

    if len(JC_files) == 0:
        raise FileNotFoundError(
            'No JC files found. Please have a JC file in the data directory to create/update blacklist.')

    for file in JC_files:
        df = pd.read_csv('data/' + file)
        df = df.rename(columns=dataframe_key.get_key('nyc'))

        JC_start_stat_indices = np.where(df['start_stat_long'] < -74.02)
        JC_end_stat_indices = np.where(df['end_stat_long'] < -74.02)

        stat_IDs = set(
            df['start_stat_id'].iloc[JC_start_stat_indices]) | set(df['end_stat_id'].iloc[JC_end_stat_indices])

        blacklist = blacklist | stat_IDs

    with open('./python_variables/JC_blacklist', 'wb') as file:
        pickle.dump(blacklist, file)

    print('Blacklist updated')

    return blacklist


def days_index(df):
    """
    Find indices of daily trips.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe containing bikeshare trip data with columns that have been
        renamed to the common key.

    Returns
    -------
    d_i : dict
        Contains the indices of the first trip per day.

    """

    days = df['start_dt'].dt.day
    d_i = [(days == i).idxmax() for i in range(1, max(days)+1)]

    return dict(zip(range(1, max(days)+1), d_i))


def pickle_data(df, city, year, month):
    """
    Generate pickle of days' starting indices.

    Parameters
    ----------
    df : pandas DataFrame
        bikeshare trip data with columns that have been renamed to the common
        key.
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.

    Returns
    -------
    d : dict
        Contains the indices of the first trip per day.

    """

    d = days_index(df)

    with open(f'./python_variables/day_index_{city}{year:d}{month:02d}.pickle', 'wb') as file:
        pickle.dump(d, file)
    return d


def get_data_month(city, year, month, blacklist=None, overwrite=False):
    """
    Read data from csv files.

    Parameters
    ----------
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.
    blacklist : list, optional
        List of IDs of stations to remove. Default is None.

    Returns
    -------
    df : pandas DataFrame
        Dataframe containing bikeshare trip data.
    days : dict
        Contains the indices of the first trip per day.

    """

    supported_cities = ['nyc', 'sfran', 'sjose',
                        'washDC', 'chic', 'london',
                        'oslo', 'edinburgh', 'bergen',
                        'trondheim', 'buenos_aires', 'madrid',
                        'mexico', 'taipei', 'helsinki',
                        'minn', 'boston', 'guadalajara', 'montreal', 'la']  # Remember to update this list

    if city not in supported_cities:
        raise ValueError(
            "This city is not currently supported. Supported cities are {}".format(supported_cities))

    # Make folder for dataframes if not found
    if not os.path.exists('python_variables/big_data'):
        os.makedirs('python_variables/big_data')
    if not overwrite:
        try:
            with open(f'./python_variables/big_data/{city}{year:d}{month:02d}_dataframe_blcklst={blacklist}.pickle', 'rb') as file:
                df = pickle.load(file)
            print('Pickle loaded')

        except FileNotFoundError:
            print('No dataframe pickle found. ', end="")
            overwrite = True

    if overwrite:
        print("Pickling dataframe...")
        if city == "nyc":

            try:
                df = pd.read_csv(
                    f'./data/{year:d}{month:02d}-citibike-tripdata.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.citibikenyc.com/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            try:
                with open('./python_variables/JC_blacklist', 'rb') as file:
                    JC_blacklist = pickle.load(file)

                df = df[~df['start_stat_id'].isin(JC_blacklist)]
                df = df[~df['end_stat_id'].isin(JC_blacklist)]

            except FileNotFoundError:
                print('No JC blacklist found. Continuing...')

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "washDC":

            try:
                df = pd.read_csv(
                    f'./data/{year:d}{month:02d}-capitalbikeshare-tripdata.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.capitalbikeshare.com/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            df['start_stat_lat'] = ''
            df['start_stat_long'] = ''
            df['end_stat_lat'] = ''
            df['end_stat_long'] = ''

            stat_df = pd.read_csv('data/Capital_Bike_Share_Locations.csv')

            for _, stat in stat_df.iterrows():
                start_matches = np.where(
                    df['start_stat_id'] == stat['TERMINAL_NUMBER'])
                end_matches = np.where(
                    df['end_stat_id'] == stat['TERMINAL_NUMBER'])

                df.at[start_matches[0], 'start_stat_lat'] = stat['LATITUDE']
                df.at[start_matches[0], 'start_stat_long'] = stat['LONGITUDE']
                df.at[end_matches[0], 'end_stat_lat'] = stat['LATITUDE']
                df.at[end_matches[0], 'end_stat_long'] = stat['LONGITUDE']

            df.replace('', np.nan, inplace=True)
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

        elif city == 'minn':
            try:
                if year == 2019 and month == 6:
                    df = pd.read_csv(
                        f'./data/20019{month:02d}-niceride-tripdata.csv')
                else:
                    df = pd.read_csv(
                        f'./data/{year:d}{month:02d}-niceride-tripdata.csv')

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
            try:
                df = pd.read_csv(
                    f'./data/{year:d}{month:02d}-bluebikes-tripdata.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.bluebikes.com/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "chic":

            q = int(np.ceil(month/3))

            try:
                df = pd.read_csv(f'./data/Divvy_Trips_{year:d}_Q{q}.csv')

            except FileNotFoundError:
                try:
                    df = pd.read_csv(f'./data/Divvy_Trips_{year:d}_Q{q}')
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://www.divvybikes.com/system-data') from exc

            if q == 2:
                col_dict = {'01 - Rental Details Rental ID': 'trip_id',
                            '01 - Rental Details Local Start Time': 'start_time',
                            '01 - Rental Details Local End Time': 'end_time',
                            '01 - Rental Details Bike ID': 'bikeid',
                            '01 - Rental Details Duration In Seconds Uncapped': 'tripduration',
                            '03 - Rental Start Station ID': 'from_station_id',
                            '03 - Rental Start Station Name': 'from_station_name',
                            '02 - Rental End Station ID': 'to_station_id',
                            '02 - Rental End Station Name': 'to_station_name',
                            'User Type': 'usertype',
                            'Member Gender': 'gender',
                            '05 - Member Details Member Birthday Year': 'birthyear'}
                df = df.rename(columns=col_dict)

            df = df.rename(columns=dataframe_key.get_key(city))

            n_days = calendar.monthrange(year, month)[1]

            df = df.iloc[np.where(
                df['start_t'] > f'{year:d}-{month:02d}-01 00:00:00')]
            df = df.iloc[np.where(
                df['start_t'] < f'{year:d}-{month:02d}-{n_days} 23:59:59')]

            df.reset_index(inplace=True, drop=True)

            df['start_stat_lat'] = ''
            df['start_stat_long'] = ''
            df['end_stat_lat'] = ''
            df['end_stat_long'] = ''

            with open('./python_variables/Chicago_stations.pickle', 'rb') as file:
                stat_df = pickle.load(file)

            for _, stat in stat_df.iterrows():
                start_matches = np.where(df['start_stat_name'] == stat['name'])
                end_matches = np.where(df['end_stat_name'] == stat['name'])

                df.at[start_matches[0], 'start_stat_lat'] = stat['latitude']
                df.at[start_matches[0], 'start_stat_long'] = stat['longitude']
                df.at[end_matches[0], 'end_stat_lat'] = stat['latitude']
                df.at[end_matches[0], 'end_stat_long'] = stat['longitude']

            df.replace('', np.nan, inplace=True)
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
                df = pd.read_csv(f'./data/metro-bike-share-trips-{year:d}-q{q}.csv')

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
                'data') if 'metro-bike-share-stations' in file]
            try:
                stations = pd.read_csv(f"data/{files[0]}", names=['stat_id', 'stat_name', 'date', 'authority', 'status'])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No station data found. All relevant files can be found at https://bikeshare.metro.net/about/data/') from exc

            name_dict = dict(zip(stations['stat_id'], stations['stat_name']))
            df['start_stat_name'] = df['start_stat_id'].map(name_dict)
            df['end_stat_name'] = df['end_stat_id'].map(name_dict)
            
            df['duration'] = df['duration']*60
            
            df.reset_index(inplace=True, drop=True)
            
        
        elif city == "sfran":

            try:
                df = pd.read_csv(
                    f'./data/{year:d}{month:02d}-baywheels-tripdata.csv')
            except FileNotFoundError:
                try:
                    df = pd.read_csv(
                        f'./data/{year:d}{month:02d}-fordgobike-tripdata.csv')
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)

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
                    f'./data/{year:d}{month:02d}-baywheels-tripdata.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)

            df = df.iloc[np.where(df['start_stat_lat'] < 37.593220)]
            df = df.iloc[np.where(df['end_stat_lat'] < 37.593220)]

            df.sort_values(by='start_t', inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "london":

            month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
                          6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
                          11: 'Nov', 12: 'Dec'}

            data_files = [file for file in os.listdir(
                'data') if 'JourneyDataExtract' in file]
            data_files = [file for file in data_files if f"{month_dict[month]}{year}" in file]

            if len(data_files) == 0:
                raise FileNotFoundError(
                    'No London data for {}. {} found. All relevant files can be found at https://cycling.data.tfl.gov.uk/.'.format(month_dict[month], year))

            if isinstance(data_files, str):
                warnings.warn(
                    'Only one data file found. Please check that you have all available data.')

            df = pd.read_csv('./data/' + data_files[0])

            for file in data_files[1:]:
                df_temp = pd.read_csv('./data/' + file)
                df = pd.concat([df, df_temp], sort=False)

            df.rename(columns=dataframe_key.get_key(city), inplace=True)      

            df['start_dt'] = pd.to_datetime(df['start_t'], format='%d/%m/%Y %H:%M')
            df['end_dt'] = pd.to_datetime(df['end_t'], format='%d/%m/%Y %H:%M')
            df.drop(columns=['start_t', 'end_t'], inplace=True)
            
            df = df[(df['start_dt'].dt.month == month) & (df['start_dt'].dt.year == year)]
            
            df.sort_values(by='start_dt', inplace=True)
            
            df.reset_index(inplace=True)            

            stat_df = pd.read_csv('./data/london_stations.csv')
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

            try:
                df = pd.read_csv(f'./data/{year:d}{month:02d}-oslo.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://oslobysykkel.no/en/open-data/historical') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "edinburgh":

            try:
                df = pd.read_csv(f'./data/{year:d}-{month:02d}-edinburgh.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://edinburghcyclehire.com/open-data/historical') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "bergen":

            try:
                df = pd.read_csv(f'./data/{year:d}{month:02d}-bergen.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://bergenbysykkel.no/en/open-data/historical') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "helsinki":

            try:
                df = pd.read_csv(f'./data/{year:d}-{month:02d}-helsinki.csv')
            except FileNotFoundError as exc:
                if month not in range(4, 10+1):
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
                    './data/Helsingin_ja_Espoon_kaupunkipyöräasemat_avoin.csv')
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
                    f"./data/recorridos-realizados-{year:d}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://data.buenosaires.gob.ar/dataset/bicicletas-publicas') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            #df_year['month'] = pd.to_datetime(df_year['fecha_origen_recorrido']).dt.month
            df['month'] = pd.to_datetime(df['start_t']).dt.month
            df = df[df.month == month]
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
            # df
            if year == 2019 and month > 7:
                try:
                    df = pd.read_json(
                        f"./data/{year:d}{month:02d}_movements.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc
                try:
                    df_pre = pd.read_json(
                        f"./data/{year:d}{(month-1):02d}_movements.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc
                df = df.rename(columns=dataframe_key.get_key(city))
                df_pre = df_pre.rename(columns=dataframe_key.get_key(city))

                df['start_dt'] = pd.to_datetime(
                    df['start_t'], format='%Y-%m-%dT%H:%M:%SZ') + pd.DateOffset(hours=2)
                df_pre['start_dt'] = pd.to_datetime(
                    df_pre['start_t'], format='%Y-%m-%dT%H:%M:%SZ') + pd.DateOffset(hours=2)

                df = df[df['start_dt'].dt.month == month]
                df_pre = df_pre[df_pre['start_dt'].dt.month == month]

                df = pd.concat((df_pre, df))

            elif year == 2019 and month == 7:
                try:
                    df = pd.read_json(
                        f"./data/{year:d}{month:02d}_movements.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc

                df = df.rename(columns=dataframe_key.get_key(city))
                df['start_dt'] = pd.to_datetime(
                    df['start_t'], format='%Y-%m-%dT%H:%M:%SZ') + pd.DateOffset(hours=2)
            else:
                try:
                    df = pd.read_json(
                        f"./data/{year:d}{month:02d}_Usage_Bicimad.json", lines=True, encoding = "ISO-8859-1")
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        'No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc

                df.drop(columns=['track'], inplace=True) # Remember that they exist in some data
                df['unplug_hourTime'] = pd.json_normalize(
                    df['unplug_hourTime'])
                df.rename(columns = dataframe_key.get_key(city), inplace=True)
                df['start_t'] = df['start_t'].str[:-6]
                # Timezone is correct in older data.
                df['start_dt'] = pd.to_datetime(
                    df['start_t'], format='%Y-%m-%dT%H:%M:%S')

            df.drop(columns=['start_t'], inplace=True)
            
            df = df[df['start_dt'].dt.month == month] # Hour adjusted because of timezone... Fix in year data.
            
            df['end_dt'] = df['start_dt'] + \
                pd.to_timedelta(df['duration'], unit='s')
            #df['end_t'] = pd.to_datetime(df['end_dt']).astype(str)
            if year == 2019 and month >= 7:
                _, stations = pd.read_json(
                    f"./data/{year:d}{month:02d}_stations_madrid.json",
                    lines=True).iloc[-1]
            else:
                _, stations = pd.read_json(
                    f"./data/Bicimad_Stations_{year:d}{month:02d}.json",
                    lines=True, encoding = "ISO-8859-1").iloc[-1]

            stations = pd.DataFrame(stations)

            name_dict = dict(zip(stations['id'], stations['name']))
            long_dict = dict(
                zip(stations['id'], stations['longitude'].astype(float)))
            lat_dict = dict(
                zip(stations['id'], stations['latitude'].astype(float)))
            addr_dict = dict(zip(stations['id'], stations['address']))

            df['start_stat_name'] = df['start_stat_id'].map(name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)
            df['start_stat_desc'] = df['start_stat_id'].map(addr_dict)

            df['end_stat_name'] = df['end_stat_id'].map(name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)
            df['end_stat_desc'] = df['end_stat_id'].map(addr_dict)

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

        elif city == "mexico":

            try:
                df = pd.read_csv(f"./data/{year:d}-{month:02d}-mexico.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.ecobici.cdmx.gob.mx/en/informacion-del-servicio/open-data') from exc

            df.rename(columns=dataframe_key.get_key(city), inplace=True)

            if month == 3 and year == 2019:
                df.drop(index=df.loc[df['end_date'] == '10'].index, inplace=True) # March 2019
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

            stations = pd.DataFrame(pd.read_json("./data/stations_mexico.json",
                                                 lines=True)['stations'][0])

            name_dict = dict(zip(stations['id'], stations['address']))
            locations = stations['location'].apply(pd.Series)
            long_dict = dict(
                zip(stations['id'], locations['lon'].astype(float)))
            lat_dict = dict(
                zip(stations['id'], locations['lat'].astype(float)))
            type_dict = dict(zip(stations['id'], stations['stationType']))

            df['start_stat_name'] = df['start_stat_id'].map(name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_name'] = df['end_stat_id'].map(name_dict)
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
                    f"./data/datos_abiertos_{year:d}_{month:02d}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.mibici.net/en/open-data/') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            files = [file for file in os.listdir(
                'data') if 'nomenclatura' in file]
            try:
                stations = pd.read_csv(f"data/{files[0]}", encoding = "ISO-8859-1")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://www.mibici.net/en/open-data/') from exc

            name_dict = dict(zip(stations['id'], stations['name']))
            long_dict = dict(zip(stations['id'], stations['longitude']))
            lat_dict = dict(zip(stations['id'], stations['latitude']))

            df['start_stat_name'] = df['start_stat_id'].map(name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_name'] = df['end_stat_id'].map(name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

            df['duration'] = (df['end_dt'] - df['start_dt']).dt.total_seconds()

        elif city == "montreal":
            try:
                df = pd.read_csv(
                    f"./data/OD_{year:d}-{month:02d}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://bixi.com/en/open-data') from exc

            df = df.rename(columns=dataframe_key.get_key(city))

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            try:
                stations = pd.read_csv(f"data/Stations_{year}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://bixi.com/en/open-data') from exc

            name_dict = dict(zip(stations['Code'], stations['name']))
            long_dict = dict(zip(stations['Code'], stations['longitude']))
            lat_dict = dict(zip(stations['Code'], stations['latitude']))

            df['start_stat_name'] = df['start_stat_id'].map(name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_name'] = df['end_stat_id'].map(name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)
            df['start_stat_id'] = df['start_stat_id'].astype(int)
            df['end_stat_id'] = df['end_stat_id'].astype(int)

        elif city == "taipei":
            colnames = ['start_t', 'start_stat_name_zh',
                        'end_t', 'end_stat_name_zh', 'duration', 'rent_date']

            try:
                df = pd.read_csv(f"./data/{year:d}{month:02d}-taipei.csv",
                                 usecols=range(5), names=colnames)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant data can be found at https://data.taipei/#/ and https://drive.google.com/drive/folders/1QsROgp8AcER6qkTJDxpuV8Mt1Dy6lGQO') from exc

            # Update names of stations
            df.replace(to_replace='信義杭州路口(中華電信總公司',
                       value='信義杭州路口(中華電信總公司)', inplace=True)
            df.replace(to_replace='捷運科技大樓站',
                       value='捷運科技大樓站(台北教育大學)', inplace=True)
            df.replace(to_replace='?公公園', value='瑠公公園', inplace=True)
            df.replace(to_replace='饒河夜市', value='饒河夜市(八德路側)', inplace=True)
            df.replace(to_replace='捷運大坪林站(3號出口)',
                       value='捷運大坪林站(1號出口)', inplace=True)
            df.replace(to_replace='新明路321巷口', value='新明路262巷口', inplace=True)

            df['start_dt'] = pd.to_datetime(
                df['start_t'], format='%Y-%m-%d %H:%M:%S')
            df['end_dt'] = pd.to_datetime(
                df['end_t'], format='%Y-%m-%d %H:%M:%S')
            df['duration'] = pd.to_timedelta(df.duration).dt.total_seconds()

            try:
                stations = pd.DataFrame.from_dict(
                    list(pd.read_json("./data/YouBikeTP.json")['retVal']))
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No station data found. The data can be found at https://tcgbusfs.blob.core.windows.net/blobyoubike/YouBikeTP.json') from exc

            stations['sno'] = stations['sno'].astype(int)
            stations['lat'] = stations['lat'].astype(float)
            stations['lng'] = stations['lng'].astype(float)
            id_dict = dict(zip(stations['sna'], stations['sno']))

            # stations_ntpc = pd.read_csv("./data/stations_new_taipei.csv")
            # stations_ntpc['sno'] = stations_ntpc['sno'].astype(int)
            # stations_ntpc['lat'] = stations_ntpc['lat'].astype(float)
            # stations_ntpc['lng'] = stations_ntpc['lng'].astype(float)
            # id_dict_ntpc = dict(zip(stations_ntpc['sna'], stations_ntpc['sno']))
            # id_dict = {**id_dict_tp, **id_dict_ntpc}

            df['start_stat_id'] = df['start_stat_name_zh'].map(id_dict)
            df['end_stat_id'] = df['end_stat_name_zh'].map(id_dict)

            name_dict = dict(zip(stations['sno'], stations['snaen']))
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

            df['start_stat_name'] = df['start_stat_id'].map(name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)
            df['start_stat_desc'] = df['start_stat_id'].map(addr_dict)

            df['end_stat_name'] = df['end_stat_id'].map(name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)
            df['end_stat_desc'] = df['end_stat_id'].map(addr_dict)

            #df_nan = df[df.isna().any(axis=1)]
            df.drop(columns=['start_t', 'end_t'], inplace=True)

            df.dropna(inplace=True)
            df.sort_values(by=['start_dt'], inplace=True)
            df.reset_index(inplace=True, drop=True)

        if blacklist:
            df = df[~df['start_stat_id'].isin(blacklist)]
            df = df[~df['end_stat_id'].isin(blacklist)]

        with open(f'./python_variables/big_data/{city}{year:d}{month:02d}_dataframe_blcklst={blacklist}.pickle', 'wb') as file:
            pickle.dump(df, file)

        print('Pickling done.')
    
    if not overwrite:
        try:
            with open(f'./python_variables/day_index_{city}{year:d}{month:02d}.pickle', 'rb') as file:
                days = pickle.load(file)
        except FileNotFoundError:
            print("Pickle does not exist. Pickling day indices...")
            days = pickle_data(df, city, year, month)
            print("Pickling done.")
        # days = days_index(df) # adds about 0.2 to 1 second to not pickle
    else:
        days = pickle_data(df, city, year, month)
        print("Pickling day indices done.")
        
    print(f"Data loaded: {city}{year:d}{month:02d}")

    return df, days


def get_data_year(city, year, blacklist=None, day_index=True, overwrite=False):

    supported_cities = ['nyc', 'sfran', 'washDC', 'chic', 'london', 'madrid', 'edinburgh', 'helsinki', 'mexico', 'taipei', 'oslo', 'bergen', 'trondheim', 'boston', 'minn', 'guadalajara', 'montreal', 'buenos_aires', 'la'
                        ]  # Remember to update this list

    if city not in supported_cities:
        raise ValueError(
            "This city is not currently supported. Supported cities are {}".format(supported_cities))

    # Make folder for dataframes if not found
    if not os.path.exists('python_variables/big_data'):
        os.makedirs('python_variables/big_data')
    
    if not overwrite:
        try:
            with open(f'./python_variables/big_data/{city}{year:d}_dataframe.pickle', 'rb') as file:
                df = pickle.load(file)
            print('Pickle loaded')
    
        except FileNotFoundError:
            print('Pickle not found. ', end="")
            overwrite = True
    
    if overwrite:
        print("Pickling dataframe...")
        if city == "nyc":

            files = [file for file in os.listdir(
                'data') if f'{year:d}' in file[:4] and 'citibike' in file]
            files.sort()

            if len(files) < 12:
                raise FileNotFoundError(
                    "Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")

            df = pd.read_csv('data/' + files[0])

            for file in files[1:]:
                df_temp = pd.read_csv('data/' + file)
                df = pd.concat([df, df_temp], sort=False)

            df = df.rename(columns=dataframe_key.get_key(city))

            try:
                with open('./python_variables/JC_blacklist', 'rb') as file:
                    JC_blacklist = pickle.load(file)

                df = df[~df['start_stat_id'].isin(JC_blacklist)]
                df = df[~df['end_stat_id'].isin(JC_blacklist)]

            except FileNotFoundError:
                print('No JC blacklist found. Continuing...')

            df.dropna(inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df.drop(columns=['start_t', 'end_t'], inplace=True)

        elif city == "washDC":

            files = [file for file in os.listdir(
                'data') if f'{year:d}' in file[:4] and 'capitalbikeshare' in file]
            files.sort()

            if len(files) < 12:
                raise FileNotFoundError(
                    "Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")

            df = pd.read_csv('data/' + files[0])

            for file in files[1:]:
                df_temp = pd.read_csv('data/' + file)
                df = pd.concat([df, df_temp], sort=False)

            df.reset_index(inplace=True, drop=True)

            df = df.rename(columns=dataframe_key.get_key(city))

            df['start_stat_lat'] = ''
            df['start_stat_long'] = ''
            df['end_stat_lat'] = ''
            df['end_stat_long'] = ''

            stat_df = pd.read_csv('data/Capital_Bike_Share_Locations.csv')

            for _, stat in stat_df.iterrows():
                start_matches = np.where(
                    df['start_stat_id'] == stat['TERMINAL_NUMBER'])
                end_matches = np.where(
                    df['end_stat_id'] == stat['TERMINAL_NUMBER'])

                df.at[start_matches[0], 'start_stat_lat'] = stat['LATITUDE']
                df.at[start_matches[0], 'start_stat_long'] = stat['LONGITUDE']
                df.at[end_matches[0], 'end_stat_lat'] = stat['LATITUDE']
                df.at[end_matches[0], 'end_stat_long'] = stat['LONGITUDE']

            df.replace('', np.nan, inplace=True)
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

        elif city == "chic":

            files = [file for file in os.listdir(
                'data') if f'Divvy_Trips_{year:d}' in file]
            files.sort()

            if len(files) < 4:
                raise FileNotFoundError(
                    "Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")

            df = pd.read_csv('data/' + files[0])

            for file in files[1:]:
                if file == 'Divvy_Trips_2019_Q2.csv':
                    col_dict = {'01 - Rental Details Rental ID': 'trip_id',
                                '01 - Rental Details Local Start Time': 'start_time',
                                '01 - Rental Details Local End Time': 'end_time',
                                '01 - Rental Details Bike ID': 'bikeid',
                                '01 - Rental Details Duration In Seconds Uncapped': 'tripduration',
                                '03 - Rental Start Station ID': 'from_station_id',
                                '03 - Rental Start Station Name': 'from_station_name',
                                '02 - Rental End Station ID': 'to_station_id',
                                '02 - Rental End Station Name': 'to_station_name',
                                'User Type': 'usertype',
                                'Member Gender': 'gender',
                                '05 - Member Details Member Birthday Year': 'birthyear'}
                    df_temp = pd.read_csv('data/' + file).rename(columns=col_dict)
                    # df_temp = df_temp.rename(columns = col_dict)

                else:
                    df_temp = pd.read_csv('data/' + file)

                df = pd.concat([df, df_temp], sort=False)

            df = df.rename(columns=dataframe_key.get_key(city))

            df.reset_index(inplace=True, drop=True)

            df['start_stat_lat'] = ''
            df['start_stat_long'] = ''
            df['end_stat_lat'] = ''
            df['end_stat_long'] = ''

            with open('./python_variables/Chicago_stations.pickle', 'rb') as file:
                stat_df = pickle.load(file)

            for _, stat in stat_df.iterrows():
                start_matches = np.where(df['start_stat_name'] == stat['name'])
                end_matches = np.where(df['end_stat_name'] == stat['name'])

                df.at[start_matches[0], 'start_stat_lat'] = stat['latitude']
                df.at[start_matches[0], 'start_stat_long'] = stat['longitude']
                df.at[end_matches[0], 'end_stat_lat'] = stat['latitude']
                df.at[end_matches[0], 'end_stat_long'] = stat['longitude']

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
                    df.append(pd.read_csv(f'./data/metro-bike-share-trips-{year:d}-q{q}.csv'))
    
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
                'data') if 'metro-bike-share-stations' in file]
            try:
                stations = pd.read_csv(f"data/{files[0]}", names=['stat_id', 'stat_name', 'date', 'authority', 'status'])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No station data found. All relevant files can be found at https://bikeshare.metro.net/about/data/') from exc

            name_dict = dict(zip(stations['stat_id'], stations['stat_name']))
            df['start_stat_name'] = df['start_stat_id'].map(name_dict)
            df['end_stat_name'] = df['end_stat_id'].map(name_dict)
            
            df['duration'] = df['duration']*60
            
            df.reset_index(inplace=True, drop=True)
            
        elif city == "sfran":

            col_list = ['duration_sec', 'start_time', 'end_time', 'start_station_id',
                        'start_station_name', 'start_station_latitude',
                        'start_station_longitude', 'end_station_id', 'end_station_name',
                        'end_station_latitude', 'end_station_longitude', 'bike_id', 'user_type']

            files = []
            for file in os.listdir('data'):
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
                df_temp = pd.read_csv('data/' + file)[col_list]
                df = pd.concat([df, df_temp], sort=False)

            df = df.rename(columns=dataframe_key.get_key(city))
            df.dropna(inplace=True)

            df = df.iloc[np.where(df['start_stat_lat'] > 37.593220)]
            df = df.iloc[np.where(df['end_stat_lat'] > 37.593220)]

            df.sort_values(by='start_t', inplace=True)
            df.reset_index(inplace=True, drop=True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])

        elif city == "london":

            data_files = [file for file in os.listdir(
                'data') if 'JourneyDataExtract' in file]
            data_files = [file for file in data_files if f'{year:d}' in file]
            data_files.sort()

            if len(data_files) == 0:
                raise FileNotFoundError(
                    f'No London data for {year:d} found. All relevant files can be found at https://cycling.data.tfl.gov.uk/.')

            if isinstance(data_files, str):
                warnings.warn(
                    'Only one data file found. Please check that you have all available data.')

            df = pd.read_csv('./data/' + data_files[0])

            for file in data_files[1:]:
                df_temp = pd.read_csv('./data/' + file)
                df = pd.concat([df, df_temp], sort=False)

            df.rename(columns=dataframe_key.get_key(city), inplace=True)

            df['start_dt'] = pd.to_datetime(df['start_t'], format='%d/%m/%Y %H:%M')
            df['end_dt'] = pd.to_datetime(df['end_t'], format='%d/%m/%Y %H:%M')
            df.drop(columns=['start_t', 'end_t'], inplace=True)
            
            df = df[df['start_dt'].dt.year == year]
            
            df.sort_values(by='start_dt', inplace=True)
            df.reset_index(inplace=True, drop=True)

            stat_df = pd.read_csv('./data/london_stations.csv')
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
                'data') if 'mexico' in file]
            files = [file for file in files if f'{year:d}' in file]
            files.sort()

            if len(files) < 12:
                raise FileNotFoundError(
                    "Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")

            dfs = []

            for file in files:
                dfs.append(pd.read_csv('data/' + file))

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
            
            stations = pd.DataFrame(pd.read_json("./data/stations_mexico.json",
                                                 lines=True)['stations'][0])

            name_dict = dict(zip(stations['id'], stations['address']))
            locations = stations['location'].apply(pd.Series)
            long_dict = dict(
                zip(stations['id'], locations['lon'].astype(float)))
            lat_dict = dict(
                zip(stations['id'], locations['lat'].astype(float)))
            type_dict = dict(zip(stations['id'], stations['stationType']))

            df['start_stat_name'] = df['start_stat_id'].map(name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)

            df['end_stat_name'] = df['end_stat_id'].map(name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)
            df['station_type'] = df['end_stat_id'].map(type_dict)

            df.dropna(inplace=True)
            #df = df[df.start_dt.dt.month == month]
            df.sort_values(by=['start_dt'], inplace=True)
            df.reset_index(inplace=True, drop=True)

        elif city == "buenos_aires":

            try:
                df = pd.read_csv(
                    f"./data/recorridos-realizados-{year:d}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    'No trip data found. All relevant files can be found at https://data.buenosaires.gob.ar/dataset/bicicletas-publicas') from exc

            df = df.rename(columns=dataframe_key.get_key(city))
            #df_year['month'] = pd.to_datetime(df_year['fecha_origen_recorrido']).dt.month

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

        elif city in ['madrid', 'edinburgh', 'taipei', 'bergen', 'boston', 'guadalajara',
                      'trondheim', 'minn',
                      'oslo',
                      'helsinki', 'montreal']:
            dfs = []
            for month in get_valid_months(city, year):
                dfs.append(get_data_month(city, year, month, overwrite=overwrite)[0])
            df = pd.concat(dfs)

        with open(f'./python_variables/big_data/{city}{year:d}_dataframe.pickle', 'wb') as file:
            pickle.dump(df, file)

        print('Pickling done.')

    if blacklist:
        df = df[~df['start_stat_id'].isin(blacklist)]
        df = df[~df['end_stat_id'].isin(blacklist)]

    if day_index:
        days = days_index(df)
        return df, days
    else:
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

    c = pd.concat([start_loc, end_loc.rename(
        columns={'end_stat_id': 'start_stat_id',
                 'end_stat_lat': 'start_stat_lat', 'end_stat_long': 'start_stat_long'}
    )], axis=0, ignore_index=True)
    c.drop_duplicates(inplace=True)
    c['start_stat_id'] = c['start_stat_id'].map(id_index)
    locations = c.set_index('start_stat_id').sort_index(
    )[['start_stat_long', 'start_stat_lat']].apply(tuple, axis=1).to_dict()

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

    c = pd.concat([start_name, end_name.rename(
        columns={'end_stat_id': 'start_stat_id',
                 'end_stat_name': 'start_stat_name'}
    )], axis=0, ignore_index=True)
    c.drop_duplicates(inplace=True)
    c['start_stat_id'] = c['start_stat_id'].map(id_index)
    names = c.set_index('start_stat_id').sort_index().to_dict()[
        'start_stat_name']
    return names


def diradjacency(df, city, year, month, day_index, days, stations,
                 threshold=1, remove_self_loops=True):
    """
    Calculate the directed adjacency matrix for the network.

    Parameters
    ----------
    df : pandas DataFrame
        bikesharing data.
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.
    day_index : list
        Indices of the first trip per day.
    days : iterable
        Days in consideration.
    stations : Stat class
        Station class containing station information.
    threshold : int, optional
        Threshold for weights. If an edge has a weight below the threshold
        then the weight is set to zero. The default threshold is 1.
    remove_self_loops : bool, optional
        Does not count trips which start and end at the same station if
        True. The default is True.

    Returns
    -------
    d_adj : ndarray
        Array containing the directed adjacency matrix.

    """

    try:
        # If Pickle exists, load it
        with open(f'./python_variables/directedadjacency_{city}{year:d}{month:02d}{tuple(days)}thr_{threshold:d}.pickle', 'rb') as file:
            d_adj = pickle.load(file)
        print("Pickle loaded")

    except FileNotFoundError:
        # If not, calculate weighted adjacency matrix and create Pickle
        print(
            f"Pickle does not exist. Pickling directed adjacency matrix: directedadjacency_{city}{year:d}{month:02d}{tuple(days)}thr_{threshold:d}.pickle...")
        d_adj = np.zeros((stations.n_tot, stations.n_tot))

        for day in days:

            if day is max(days):
                for _, row in df.iloc[day_index[day]:].iterrows():
                    d_adj[stations.id_index[row['start_stat_id']],
                          stations.id_index[row['end_stat_id']]] += 1
                print('Day {} loaded...'.format(day))

            else:
                for _, row in df.iloc[day_index[day]:day_index[day+1]].iterrows():
                    d_adj[stations.id_index[row['start_stat_id']],
                          stations.id_index[row['end_stat_id']]] += 1
                print('Day {} loaded...'.format(day))

        d_adj[d_adj <= threshold] = 0

        if remove_self_loops:
            for i in range(stations.n_tot):
                d_adj[i, i] = 0

        with open(f'./python_variables/directedadjacency_{city}{year:d}{month:02d}{tuple(days)}thr_{threshold:d}.pickle', 'wb') as file:
            pickle.dump(d_adj, file)
        print("Pickling done.")

    return d_adj


def diradjacency_hour(data, day, hour, threshold=1, remove_self_loops=True):

    d_adj = np.zeros((data.stat.n_tot, data.stat.n_tot))

    if day == data.num_days:
        df_slice = data.df.iloc[data.d_index[day]:]
    else:
        df_slice = data.df.iloc[data.d_index[day]:data.d_index[day+1]]

    df_slice = df_slice.loc[data.df['start_dt'].dt.hour == hour]

    si = df_slice['start_stat_id'].map(data.stat.id_index)
    ei = df_slice['end_stat_id'].map(data.stat.id_index)
    #start_stat_index = id_index(df['start_stat_id'])

    for i, j in zip(si, ei):
        d_adj[i, j] += 1

    d_adj[d_adj <= threshold] = 0

    if remove_self_loops:
        for i in range(data.stat.n_tot):
            d_adj[i, i] = 0

    return d_adj


def get_degree_matrix(adj):
    """
    Computes the degree matrix of the network.

    Parameters
    ----------
    adj : ndarray
        Adjacency matrix.

    Returns
    -------
    deg_matrix: ndarray
        The degree matrix.

    """

    deg_matrix = np.zeros_like(adj)

    for i in range(len(adj)):
        deg_matrix[i, i] = np.sum(adj[[i], :])

    return deg_matrix


def data_pickle_load(city, year, month):
    """
    Load data from a Data class object pickle. See Data.pickle_dump

    Parameters
    ----------
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.
    year : int
        The year of interest in YYYY format.
    month : int
        The month of interest in MM format.

    Returns
    -------
    object of Data class
    """

    with open(f'./python_variables/big_data/data_{city}{year:d}{month:02d}.pickle', 'rb') as file:
        return pickle.load(file)


def adjacency(df, n_tot, id_index, threshold=1, remove_self_loops=True):
    """
    Calculate the weighted adjacency matrix for the network assuming an
    undirected graph.

    Parameters
    ----------
    df : pandas DataFrame
        Contains the data over which the adjacency matrix is calculated.
    n_tot : int
        Number of stations.
    id_index : dict
        Translates station id to an index starting from 0.
    threshold : int, optional
        Threshold for weights. If an edge has a weight below the threshold
        then the weight is set to zero. The default threshold is 1.
    remove_self_loops : bool, optional
        Does not count trips which start and end at the same station if
        True. The default is True.

    Returns
    -------
    adj : ndarray
        Adjacency matrix of the network.

    """

    adj = np.zeros((n_tot, n_tot))
    si = df['start_stat_id'].map(id_index)
    ei = df['end_stat_id'].map(id_index)
    #start_stat_index = id_index(df['start_stat_id'])

    for i, j in zip(si, ei):
        adj[i, j] += 1

    adj = adj + adj.T

    adj[adj <= threshold] = 0

    if remove_self_loops:
        for i in range(n_tot):
            adj[i, i] = 0

    return adj


def PageRank(adj, d=0.85, iterations=50, initialisation="uniform"):
    """
    Calculates the PageRank of each vertex in a graph.

    Parameters
    ----------
    adj : ndarray
        Directed and weighted adjacency matrix.
    d : float, optional
        Dampening factor. The default is 0.85.
    iterations : int, optional
        The amount of iterations we run the PageRank algortihm. The default is
        100.
    initialisation : str, optional
        Determines if we have random initialisation or 1/n initialisation. The
        default is "rdm".

    Returns
    -------
    v : ndarray
        contains the PageRank of each vertex.

    """

    N = adj.shape[0]
    weightlist = []

    for i in range(N):
        weight = 0

        for n in range(N):
            weight += adj[i, n]

        weightlist.append(weight)

    if initialisation == "rdm":
        v = np.random.rand(N, 1)
        v = v / np.linalg.norm(v, ord=1)

    else:  # Uniform initialisation
        v = np.linspace(1/N, 1/N, N)

    for i in range(iterations):

        for n in range(N):
            if weightlist[n] != 0:
                v[n] = v[n]/weightlist[n]

        v = (1 - d)/(N+1) + d * adj @ v

    return v


def get_elevation(lat, long, dataset="mapzen"):
    """
    Finds the elevation for a specific coordinate.

    Elevation data is taken from https://www.opentopodata.org/

    Parameters
    ----------
    lat : float or iterable
        Latitude or iterable containing latitudes.
    long : float or iterable
        Longitude or iterable containing longitudes.
    dataset : str, optional
        Dataset used for elevation data. The default is "mapzen".

    Returns
    -------
    elevation : ndarray
        Array containing elevations.

    """

    if lat is None or long is None:
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

            r = get(query, timeout=60)

            # Only get the json response in case of 200 or 201
            if r.status_code == 200 or r.status_code == 201:
                elevation = np.append(elevation,
                                      np.array(pd.json_normalize(
                                          r.json(), 'results')['elevation'])
                                      )

    return elevation


def get_weather(city, year, month):
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

    Returns
    -------
    request : str
        DESCRIPTION.
    rain :
        DESCRIPTION.

    """

    cities = ['chic', 'london', 'madrid', 'mexico',
              'nyc', 'sfran', 'taipei', 'washDC']

    if city in cities:
        name_dict = {'chic': 'Chicago', 'london': 'London', 'madrid': 'Madrid',
                     'mexico': 'Mexico City', 'nyc': 'New York City',
                     'sfran': 'San Francisco', 'taipei': 'Taipei',
                     'washDC': 'Washington DC'}
        city = name_dict[city]

    n_days = calendar.monthrange(year, month)[1]
    tp = 1
    query = (f"http://api.worldweatheronline.com/premium/v1/past-weather.ashx?"
             f"key=798ccdae8e3b4238a72124749212610&q={city}&format=json&date={year}-{month}-01&enddate={year}-{month}-{n_days}&tp={tp}")  # Request with a timeout for slow responses
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


def get_weather_year(city, year):
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

    Returns
    -------
    request : str
        DESCRIPTION.
    rain :
        DESCRIPTION.

    """

    rain = []
    for month in range(1, 13):
        rain.append(get_weather(city, year, month)[1])

    return pd.concat(rain)


def TotalVariation(adj, cutoff):
    """
    Calculates the total variation of given graph with the degree as the signal.

    Parameters
    ----------
    adj : ndarray
        Adjacency matrix.
    threshold : float
        Threshold for when the signal is high or low frequency.

    Returns
    -------
    filterarray : ndarray
        Binary array. 1 indicates low frequency and 0 indicates high frequency.

    """

    n = len(adj)
    Lambda, u = np.linalg.eig(adj)
    Lambda_max = np.max(abs(Lambda))
    W_tilde = adj * 1/Lambda_max
    T = np.zeros(n)
    filterarray = np.zeros(n)
    for m in range(n):
        T[m] = np.linalg.norm(u[:, m] - (W_tilde @ u[:, m]), ord=1)
        if T[m] < cutoff:
            filterarray[m] = 1
    return filterarray


def subframe(filterarray, df, id_index, low):
    """
    Creates a lowpass- or highpass-filtered dataframe

    Parameters
    ----------
    filterarray : ndarray
        Binary array. 1 indicates low frequency and 0 indicates high frequency.
    df : pandas dataframe
        Original city data dataframe.
    low : Logival, optional
        Tells us if the filtered dataframe will be low or high frequency.
        The default is True.

    Returns
    -------
    df_done : pandas dataframe
        Filtered pandas dataframe.

    """

    filtered_positions = np.argwhere(filterarray == 1)

    l = len(filtered_positions)
    filtered_positions = filtered_positions.reshape(l)

    if low:
        df_filtered = df[df['start_stat_id'].map(
            id_index).isin(filtered_positions)]
        df_done = df_filtered[df_filtered['end_stat_id'].map(
            id_index).isin(filtered_positions)]

    else:
        df_filtered = df[~df['start_stat_id'].map(
            id_index).isin(filtered_positions)]
        df_done = df_filtered[~df_filtered['end_stat_id'].map(
            id_index).isin(filtered_positions)]

    return df_done


def adjacency_filtered(df, day_index, days, n_tot, id_index, threshold=1, remove_self_loops=True):
    """
    Calculate weighted adjacency matrix (undirected)

    Parameters
    ----------
    days : tuple
        Tuple of days in consideration.
    threshold : int, optional
        Threshold for weights. If an edge has a weight below the threshold
        then the weight is set to zero. The default threshold is 1.
    remove_self_loops : bool, optional
        Does not count trips which start and end at the same station if
        True. The default is True.

    Returns
    -------
    adj : ndarray
        Adjacency matrix of the network.

    """

    adj = np.zeros((n_tot, n_tot))

    si = df['start_stat_id'].map(id_index)
    ei = df['end_stat_id'].map(id_index)
    for day in days:
        if day is max(days):
            for i, j in zip(si[day_index[day]:], ei[day_index[day]:]):
                adj[i, j] += 1
                adj[j, i] += 1

        else:
            for i, j in zip(si[day_index[day]:day_index[day+1]], ei[day_index[day]:day_index[day+1]]):
                adj[i, j] += 1
                adj[j, i] += 1

    adj[adj <= threshold] = 0

    if remove_self_loops == True:
        for i in range(n_tot):
            adj[i, i] = 0

    return adj


def coverage(g, p):
    """
    Calculates the covergae of the partition.

    Parameters
    ----------
    g : networkx graph class
        graph of the data.
    p : dictionary
        tells which verticies belongs to which communities.

    Returns
    -------
    float
        the coverage of our partition.

    """

    d_i = dict(zip(np.arange(g.number_of_nodes()), list(g.nodes())))
    n = g.number_of_nodes()
    ad = nx.adjacency_matrix(g)
    p_sum = np.sum(ad) / 2

    num = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[d_i[i]] == p[d_i[j]]:
                num += ad[i, j]

    return num / p_sum


def distance(lat1, lon1, lat2, lon2):
    """
    Calculates the distance between two coordiantes.

    Parameters
    ----------
    lat1 : float
        Latitude of first coordinate.
    lon1 : float
        Longitude of first coordinate.
    lat2 : float
        Latitude of second coordinate.
    lon2 : float
        Longitude of second coordinate.

    Returns
    -------
    res : float
        Distance between the two coordinates.

    """
    r = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * \
        np.cos(phi2) * np.sin(delta_lambda / 2)**2
    res = 2 * r * np.arcsin(np.sqrt(a))

    return res


def purge_pickles(city, year, month):

    lookfor = f'{city}{year:d}{month:02d}'

    print("Purging in 'python_variables'...")

    for file in os.listdir('python_variables'):
        if lookfor in file:
            os.remove('python_variables/' + file)

    print("Purging in 'python_variables/big_data'...")
    for file in os.listdir('python_variables/big_data'):
        if lookfor in file:
            os.remove('python_variables/big_data/' + file)

    print('Purging done')

def nuke_pickles(city):
    lookfor = f'{city}'

    print("Nuking in 'python_variables'...")

    for file in os.listdir('python_variables'):
        if lookfor in file:
            os.remove('python_variables/' + file)

    print("Nuking in 'python_variables/big_data'...")
    for file in os.listdir('python_variables/big_data'):
        if lookfor in file:
            os.remove('python_variables/big_data/' + file)

    print('Nuke succesful. What have we done...')


def df_subset(df, weekdays=None, days='all', hours='all', minutes='all', activity_type='departures'):
    """
    Get subset of dataframe df.

    Parameters
    ----------
    df : pandas dataframe
        bikeshare dataframe with columns 'start_dt' and 'end_dt'.
    weekdays : list, optional
        list of the weekdays per day in the given month. . 0=monday, 1=tuesday etc. The default is None.
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


def dist_norm(vec1, vec2):
    return np.linalg.norm(vec1-vec2)


def dist_dtw(vec1, vec2):
    return dtw.dtw(vec1, vec2)[1]

def Davies_Bouldin_index(data_mat, labels, centroids, dist_func='norm', verbose=False):
    """
    Calculates the Davies-Bouldin index of clustered data.

    Parameters
    ----------
    data_mat : ndarray
        Array containing the feature vectors.
    labels : itr, optional
        Iterable containg the labels of the feature vectors. If no labels
        are given, they are calculated using the mass_predict method.

    Returns
    -------
    DB_index : float
        Davies-Bouldin index.

    """

    k = len(centroids)

    if dist_func == 'norm':
        dist = dist_norm

    elif dist_func == 'dtw':
        dist = dist_dtw

    if verbose:
        print('Calculating Davies-Bouldin index...')

    pre = time.time()

    S_scores = np.empty(k)

    for i in range(k):
        data_mat_cluster = data_mat[np.where(labels == i)]
        distances = [dist(row, centroids[i]) for row in data_mat_cluster]
        S_scores[i] = np.mean(distances)

    R = np.empty(shape=(k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                R[i, j] = 0
            else:
                R[i, j] = (S_scores[i] + S_scores[j]) / \
                    dist(centroids[i], centroids[j])

    D = [max(row) for row in R]

    DB_index = np.mean(D)

    if verbose:
        print(f'Done. Time taken: {(time.time()-pre):.1f} s')

    return DB_index


def Dunn_index(data_mat, labels, centroids, dist_func='norm', verbose=False):
    """
    Calculates the Dunn index of clustered data. WARNING: VERY SLOW.

    Parameters
    ----------
    data_mat : ndarray
        Array containing the feature vectors.
    labels : itr, optional
        Iterable containg the labels of the feature vectors. If no labels
        are given, they are calculated using the mass_predict method.

    Returns
    -------
    D_index : float
        Dunn index.

    """
    k = len(centroids)

    if dist_func == 'norm':
        dist = dist_norm

    elif dist_func == 'dtw':
        dist = dist_dtw

    if verbose:
        print('Calculating Dunn Index...')

    pre = time.time()

    intra_cluster_distances = np.empty(k)
    inter_cluster_distances = np.full(shape=(k, k), fill_value=np.inf)

    for i in range(k):
        data_mat_cluster = data_mat[np.where(labels == i)]
        cluster_size = len(data_mat_cluster)
        distances = np.empty(shape=(cluster_size, cluster_size))

        for h in range(cluster_size):
            for j in range(cluster_size):
                distances[h, j] = dist(data_mat[h], data_mat[j])

        intra_cluster_distances[i] = np.max(distances)

        for j in range(k):
            if j != i:
                data_mat_cluster_j = data_mat[np.where(labels == j)]
                cluster_size_j = len(data_mat_cluster_j)
                between_cluster_distances = np.empty(
                    shape=(cluster_size, cluster_size_j))
                for m in range(cluster_size):
                    for n in range(cluster_size_j):
                        between_cluster_distances[m, n] = dist(
                            data_mat_cluster[m], data_mat_cluster_j[n])
                inter_cluster_distances[i, j] = np.min(
                    between_cluster_distances)

    D_index = np.min(inter_cluster_distances)/np.max(intra_cluster_distances)

    if verbose:
        print(f'Done. Time taken: {(time.time()-pre):.1f} s')

    return D_index


def silhouette_index(data_mat, labels, centroids, dist_func='norm', verbose=False):
    """
    Calculates the silhouette index of clustered data.

    Parameters
    ----------
    data_mat : ndarray
        Array containing the feature vectors.
    labels : itr, optional
        Iterable containg the labels of the feature vectors. If no labels
        are given, they are calculated using the mass_predict method.

    Returns
    -------
    S_index : float
        Silhouette index.

    """

    k = len(centroids)

    if dist_func == 'norm':
        dist = dist_norm

    elif dist_func == 'dtw':
        dist = dist_dtw

    if verbose:
        print('Calculating Silhouette index...')

    pre = time.time()

    s_coefs = np.empty(len(data_mat))

    for i, vec1 in enumerate(data_mat):
        in_cluster = np.delete(data_mat, i, axis=0)
        in_cluster = in_cluster[np.where(np.delete(labels, i) == labels[i])]

        in_cluster_size = len(in_cluster)

        if in_cluster_size != 0:

            in_cluster_distances = np.empty(in_cluster_size)
            for j, vec2 in enumerate(in_cluster):
                in_cluster_distances[j] = dist(vec1, vec2)

            mean_out_cluster_distances = np.full(k, fill_value=np.inf)

            for j in range(k):
                if j != labels[i]:
                    out_cluster = data_mat[np.where(labels == j)]
                    out_cluster_distances = np.empty(len(out_cluster))

                    for l, vec2 in enumerate(out_cluster):
                        out_cluster_distances[l] = dist(vec1, vec2)

                    mean_out_cluster_distances[j] = np.mean(out_cluster_distances)

            ai = np.mean(in_cluster_distances)
            bi = np.min(mean_out_cluster_distances)

            s_coefs[i] = (bi-ai)/max(ai, bi)

        else:
            s_coefs[i] = 0

    S_index = np.mean(s_coefs)

    if verbose:
        print(f'Done. Time taken: {(time.time()-pre):.1f} s')

    return S_index

def k_test(data_mat, cluster_func, k_max = 10, random_state = 42,
           tests = 'full', plot = False):

    tests = ['SSE', 'DB', 'D', 'S']

    results = np.zeros(shape=(len(tests),k_max-1))

    # print(f'{f"Test result for {cluster_func}":^{spacing}}')
    # print('-'*spacing)

    print(f'{"k":5}{"SSE":15}{"DB_index":15}{"D_index":15}{"S_index":15}')
    print('-'*60)

    for i, k in enumerate(range(2, k_max+1)):
        clusters = cluster_func(k, random_state = random_state).fit(data_mat)
        labels = clusters.predict(data_mat)
        centroids = clusters.cluster_centers_

        results[0, i] = clusters.inertia_
        results[1, i] = Davies_Bouldin_index(data_mat, labels, centroids)
        results[2, i] = Dunn_index(data_mat, labels, centroids)
        results[3, i] = silhouette_index(data_mat, labels, centroids)

        print(
           f'{k:<5,d}{results[0,i]:<15.8f}{results[1,i]:<15.8f}{results[2,i]:<15.8f}{results[3,i]:<15,.8f}')


    res_df = pd.DataFrame(index = range(2,k_max+1),
                          columns = ['SSE', 'DB', 'D', 'S'])
    res_df.index.rename('k', inplace=True)

    for test_i, test in enumerate(res_df.columns):
        res_df[test] = results[test_i]

    if plot:

        plt.subplot(221)
        plt.plot(range(2,k_max+1), res_df['SSE'])
        plt.xticks(range(2,k_max+1))
        # plt.xlabel('$k$')
        plt.legend(['SSE'])

        plt.subplot(222)
        plt.plot(range(2,k_max+1), res_df['DB'], c='tab:orange')
        plt.xticks(range(2,k_max+1))
        # plt.xlabel('$k$')
        plt.legend(['DB_index'])

        plt.subplot(223)
        plt.plot(range(2,k_max+1), res_df['D'], c='tab:green')
        plt.xticks(range(2,k_max+1))
        plt.xlabel('$k$')
        plt.legend(['D_index'])

        plt.subplot(224)
        plt.plot(range(2,k_max+1), res_df['S'], c='tab:red')
        plt.xticks(range(2,k_max+1))
        plt.xlabel('$k$')
        plt.legend(['S_index'])

    return res_df

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
        self.loc = np.array(list(self.locations.values()))
        trans = Transformer.from_crs("EPSG:4326", "EPSG:3857")
        self.loc_merc = np.vstack(trans.transform(
            self.loc[:, 1], self.loc[:, 0])).T
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
    Washington DC: washDC
        Relevant data can be found at https://www.capitalbikeshare.com/system-data
    Chicago: chic
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
    adjacency(self, days, threshold=1, remove_self_loops=True):
        Calculate adjacency matrix. Automatically saves in adj

    adjacency_hour(self, day, hour, threshold=1, remove_self_loops = True):
        Calculates the adjacency matrix for a given hour

    unweightedadjacency_hour(self, day, hour, remove_self_loops=True):
        Calculates the unweighted adjacency matrix for a given hour

    get_degree_matrix(self, days):
        Calculate degree matrix. Automatically saves in deg

    get_laplacian(self, days):
        Calculate laplacian matrix. Automatically saves in lap

    get_busy_stations(self, days, deg_threshold, normalise=True):
        Finds the corresponding degree to each docking station and returns a
        sorted list of the docking stations and their degree.

    get_busy_trips(self, days, directed = False, normalise = True):
        Finds the trips with the largest weights.

    compare_degrees(self, days_ref, days_change, savefig = False):
        Normalises the degree of each station with respect to a reference
        degree and plots the network using the normalised degrees.

    pickle_dump(self):
        dumps Data object as pickle

    """

    def __init__(self, city, year, month=None, blacklist=None, overwrite=False):
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

        if self.month is not None:
            first_weekday, self.num_days = calendar.monthrange(year, month)

            self.weekdays = [(i+(first_weekday)) %
                             7 for i in range(self.num_days)]

            self.df, self.d_index = get_data_month(
                city, year, month, blacklist, overwrite=overwrite)

        else:
            self.df, self.d_index = get_data_year(city, year, blacklist, overwrite=overwrite)

        self.stat = Stations(self.df)

        #self.df['start_stat_index'] = self.df['start_stat_id'].map(
        #    self.stat.id_index)
        #self.df['end_stat_index'] = self.df['end_stat_id'].map(
        #    self.stat.id_index)

    def adjacency(self, days, threshold=1, remove_self_loops=True):
        """
        Calculate weighted adjacency matrix (undirected)

        Parameters
        ----------
        days : tuple
            Tuple of days in consideration.
        threshold : int, optional
            Threshold for weights. If an edge has a weight below the threshold
            then the weight is set to zero. The default threshold is 1.
        remove_self_loops : bool, optional
            Does not count trips which start and end at the same station if
            True. The default is True.

        Returns
        -------
        adj : ndarray
            Adjacency matrix of the network.
        """

        try:
            # If Pickle exists, load it
            with open(f'./python_variables/adjacency_{self.city}{self.year:d}{self.month:02d}{tuple(days)}thr_{threshold:d}_rsl_{remove_self_loops}.pickle', 'rb') as file:
                adj = pickle.load(file)
            print("Pickle loaded")

        except FileNotFoundError:
            # If not, calculate adjacency matrix and create Pickle
            print(
                f"Pickle does not exist. Pickling adjacency matrix: adjacency_{self.city}{self.year:d}{self.month:02d}{tuple(days)}thr_{threshold:d}_rsl_{remove_self_loops}.pickle...")
            adj = np.zeros((self.stat.n_tot, self.stat.n_tot))

            si = self.df['start_stat_id'].map(self.stat.id_index)
            ei = self.df['end_stat_id'].map(self.stat.id_index)

            for day in days:

                if day is max(days):
                    for i, j in zip(si[self.d_index[day]:], ei[self.d_index[day]:]):
                        adj[i, j] += 1
                        adj[j, i] += 1
                    print('Adjacency Day {} loaded...'.format(day))

                else:
                    for i, j in zip(si[self.d_index[day]:self.d_index[day+1]],
                                    ei[self.d_index[day]:self.d_index[day+1]]):
                        adj[i, j] += 1
                        adj[j, i] += 1
                    print('Adjacency Day {} loaded...'.format(day))

            adj[adj <= threshold] = 0

            if remove_self_loops:
                for i in range(self.stat.n_tot):
                    adj[i, i] = 0

            with open(f'./python_variables/adjacency_{self.city}{self.year:d}{self.month:02d}{tuple(days)}thr_{threshold:d}_rsl_{remove_self_loops}.pickle', 'wb') as file:
                pickle.dump(adj, file)
            print("Pickling done.")

        return adj

    def adjacency_hour(self, day, hour, threshold=1, remove_self_loops=True):
        """
        Calculates the adjacency matrix for the given hour

        Parameters
        ----------
        day : int
            day.
        hour : int
            hour.
        threshold : int, optional
            threshold. The default is 1.
        remove_self_loops : bool, optional
            whether or not to remove self loops. The default is True.

        Returns
        -------
        np array
            adjacency matrix for the given hour.

        """

        adj = np.zeros((self.stat.n_tot, self.stat.n_tot))

        if day == self.num_days:
            df_slice = self.df.iloc[self.d_index[day]:]

        else:
            df_slice = self.df.iloc[self.d_index[day]:self.d_index[day+1]]

        df_slice = df_slice.loc[self.df['start_dt'].dt.hour == hour]

        #start_stat_index = id_index(df['start_stat_id'])

        for i, j in zip(df_slice['start_stat_index'], df_slice['end_stat_index']):
            adj[i, j] += 1
        adj = adj + adj.T

        adj[adj <= threshold] = 0

        if remove_self_loops:
            for i in range(self.stat.n_tot):
                adj[i, i] = 0

        return adj

    def unweightedadjacency_hour(self, day, hour, remove_self_loops=True):
        """
        Calculates the adjacency matrix for the given hour

        Parameters
        ----------
        day : int
            day.
        hour : int
            hour.
        threshold : int, optional
            threshold. The default is 1.
        remove_self_loops : bool, optional
            whether or not to remove self loops. The default is True.

        Returns
        -------
        np array
            adjacency matrix for the given hour.

        """

        adj = np.zeros((self.stat.n_tot, self.stat.n_tot))

        if day == self.num_days:
            df_slice = self.df.iloc[self.d_index[day]:]

        else:
            df_slice = self.df.iloc[self.d_index[day]:self.d_index[day+1]]

        df_slice = df_slice.loc[self.df['start_dt'].dt.hour == hour]

        #start_stat_index = id_index(df['start_stat_id'])

        for i, j in zip(df_slice['start_stat_index'], df_slice['end_stat_index']):
            adj[i, j] = 1

        adj = adj + adj.T

        if remove_self_loops:
            for i in range(self.stat.n_tot):
                adj[i, i] = 0

        return adj

    def pickle_dump(self):
        """
        Dumps pickle of entire Data object to the big_data directory

        """

        with open(f'./python_variables/big_data/data_{self.city}{self.year:d}{self.month:02d}.pickle', 'wb') as file:
            pickle.dump(self, file)

    def get_degree_matrix(self, days, threshold=1, remove_self_loops=True):
        """
        Computes the degree matrix of the network.

        Parameters
        ----------
        days : tuple
            Days in consideration.

        Returns
        -------
        ndarray
            Array containing the degree matrix.

        """

        adj_matrix = self.adjacency(days, threshold, remove_self_loops)

        degrees = np.sum(adj_matrix, axis=0)

        deg_matrix = np.diag(degrees)

        return deg_matrix

    def get_laplacian(self, days, threshold=1, remove_self_loops=True):
        """
        Computes the Laplacian matrix of the network.

        Parameters
        ----------
        days : tuple
            Days in consideration.

        Returns
        -------
        ndarray
            Array containing the laplacian matrix.

        """

        adj_matrix = self.adjacency(days, threshold, remove_self_loops)
        deg_matrix = self.get_degree_matrix(days, threshold, remove_self_loops)

        return deg_matrix - adj_matrix

    def get_busy_stations(self, days, normalise=True, sort=True):
        """
        Finds the corresponding degree to each docking station and returns a
        sorted list of the docking stations and their degree.

        Parameters
        ----------
        days : tuple
            Days in consideration.
        normalise : bool, optional
            Normalises the degrees with respect to the number of days if set
            to True, does not if set to False. The default is True.

        Returns
        -------
        list
            List of n tuples with with n being the number of stations. Each
            tuple contains the station ID, the station name and the degree of
            the station. The list is sorted with respect to the degrees in
            descending order.
        """

        deg_matrix = self.get_degree_matrix(days)

        degrees = np.sum(deg_matrix, axis=0)

        if normalise:
            degrees = degrees/len(days)

        busy_stations = []
        for i in range(len(degrees)):
            busy_station = self.stat.names[i]
            busy_stations.append(busy_station)

        temp = list(zip(busy_stations, degrees))

        if sort:
            temp_sorted = sorted(temp, key=lambda x: x[1], reverse=True)
            return temp_sorted
        else:
            return temp

    def get_busy_trips(self, days, directed=False, normalise=True):
        """
        Finds the trips with the largest weights, ignoring trips with zero
        weight. The function assumes a weighted graph.

        Parameters
        ----------
        days : tuple
            Days in consideration.
        directed : bool, optional
            Assumes a directed graph. Computations may be slower. The
            default is False.
        normalise : bool, optional
            Normalises the weights with respect to the number of days. The
            default is True.

        Raises
        ------
        ValueError
            Raised if length of days is zero.

        Returns
        -------
        busy_trips : list
            list of tuples with each tuple containing the index of the start
            station and end station (as a tuple) and the corresponding weigth.
            The list is sorted with respect to the weights in descending order.

        """
        if len(days) == 0:
            raise ValueError('Number of days can not be zero.')

        if directed:
            adj = diradjacency(self.df, self.city, self.year, self.month,
                               self.d_index, days, self.stat)

        else:
            adj = np.tril(self.adjacency(days))

        if normalise:
            adj = adj/len(days)

        mask = adj != 0

        max_indices = []
        weights = []

        while np.sum(mask) != 0:
            w_max = np.max(adj[mask])
            where = np.where(adj == w_max)

            max_indices = max_indices + \
                [(where[0][i], where[1][i]) for i in range(len(where[0]))]

            weights = weights + [w_max for _ in range(len(where[0]))]

            mask[where] = False

        busy_trips = list(zip(max_indices, weights))

        return busy_trips

    def compare_degrees(self, days_ref, days_change, savefig=False):
        """
        Normalises the degree of each station with respect to a reference
        degree and plots the network using the normalised degrees.

        Parameters
        ----------
        days_ref : tuple
            Days used to compute the reference degree of each station.
        days_change : tuple
            Days used to compute the degree of each station. These degrees will
            then be normalised with respect to their reference degrees.
        savefig : bool, optional
            Saves the figure if True. The default is False.

        Returns
        -------
        deg_compare : ndarray
            Array of the normalised degrees used for plotting.

        """

        figsize_dict = {'nyc': (5, 8),
                        'chic': (5, 8),
                        'london': (8, 5),
                        'oslo': (6.6, 5),
                        'sfran': (6, 5),
                        'washDC': (8, 7.7),
                        'madrid': (6, 8),
                        'mexico': (7.2, 8),
                        'taipei': (7.3, 8)}

        adj_ref = self.adjacency(
            days_ref, threshold=0, remove_self_loops=False)

        # +1 to work around division by zero
        ref_degrees = np.sum(adj_ref, axis=0) + 1

        adj_change = self.adjacency(
            days_change, threshold=0, remove_self_loops=False)

        deg_change = np.sum(adj_change, axis=0)

        deg_compare = deg_change / ref_degrees

        # Plot graph
        graph = nx.from_numpy_matrix(adj_ref)

        try:
            fig = plt.subplots(figsize=figsize_dict[self.city])
        except KeyError:
            fig = plt.subplots(figsize=(8, 8))

        plt.grid(False)
        plt.axis(False)

        nx.draw_networkx_nodes(
            graph, self.stat.loc_merc, node_size=20, node_color=deg_compare,
            cmap='jet', vmin=0, vmax=1.5)

        nx.draw_networkx_edges(
            graph, self.stat.loc_merc,
            alpha=0.8, width=0.2, edge_color='black')

        fig.set_figwidth((1+0.24)*fig.get_size_inches()[0])

        cmap = mpl.cm.jet
        norm = mpl.colors.Normalize(vmin=0, vmax=1.5)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                     orientation='vertical', label='degree')

        # plt.subplots_adjust(bottom = 0.1, top = 0.9, right = 0.8)

        # cax = plt.axes([0.85, 0.1, 0.03, 0.8])
        # norm = mpl.colors.Normalize(vmin = 0, vmax = 1.5)
        # plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'),
        #                      orientation='vertical', label='degree', cax=cax)

        plt.tight_layout()

        if savefig:
            print('Saving figure...')
            plt.savefig('figures/graph_compare.png')

        return deg_compare

    def daily_traffic(self, stat_index=None, stat_id=None, month = None, day='all', normalise=True, plot=False, period = None, holidays=True):
        """
        Computes the number of arrivals and departures to and from the station
        in every hour of the specified day.

        Parameters
        ----------
        stat_index : int
            Station index.
        day : int
            Day in the month to compute traffic from.
        plot : bool, optional
            Plots the daily traffic if set to True. The default is False.

        Returns
        -------
        trips_departures : ndarray
            24-dimensional array with number of departures for every hour, eg.
            index 0 yields the number of departures from 00:00:00 to 01:00:00.
        trips_arrivals : ndarray
            24-dimensional array with number of arrivals for every hour, eg.
            index 0 yields the number of arrivals from 00:00:00 to 01:00:00.

        """
        
        if not month:
            month = self.month
        
        if stat_index is None and stat_id is None:
            raise ValueError('Plese provide either a station index or station id')
        
        if stat_index:
            stat_id = self.stat.inverse[stat_index]
        
        if day=='all':
            
            if period == 'b':
                
                if not month:
                    days = [datetime.date(self.year, 1, day) for day in range(
                        1,calendar.monthrange(self.year,1)[1]+1) if calendar.weekday(self.year,1,day) <=4]
                    
                    for m in range(2,13):
                        for day in range(1,calendar.monthrange(self.year,m)[1]+1):
                            if calendar.weekday(self.year, m, day) <= 4:
                                days.append(datetime.date(self.year, m, day))
                
                else:
                    days = [datetime.date(self.year, month, day) for day in range(
                        1,calendar.monthrange(self.year,month)[1]+1) if calendar.weekday(self.year,month,day) <=4]
                    
                
                if not holidays:
                      holiday_year = pd.DataFrame(
                          get_cal(self.city).get_calendar_holidays(self.year), columns=['day', 'name'])
                      holiday_list = holiday_year['day'].tolist()
                      days =  [day for day in days if day not in holiday_list]
                    
            elif period == 'w':
                
                if not month:
                    days = [datetime.date(self.year, 1, day) for day in range(
                        1,calendar.monthrange(self.year,1)[1]+1) if calendar.weekday(self.year,1,day) >4]
                    
                    for m in range(2,13):
                        for day in range(1,calendar.monthrange(self.year,m)[1]+1):
                            if calendar.weekday(self.year, m, day) > 4:
                                days.append(datetime.date(self.year, m, day))
                
                    else:
                        days = [datetime.date(self.year, month, day) for day in range(
                            1,calendar.monthrange(self.year,month)[1]+1) if calendar.weekday(self.year,month,day) > 4]

            else:
                
                if not month:
                    days = [datetime.date(self.year, 1, day) for day in range(1, calendar.monthrange(self.year, 1)[1]+1)]
                
                    for m in range(2,13):
                        for i in range(1, calendar.monthrange(self.year, m)[1]+1):
                            days.append(datetime.date(self.year, m, i))
                
                else:
                    days = [datetime.date(self.year, month, day) for day in range(1, calendar.monthrange(self.year, month)[1]+1)]
                
                if not holidays:
                    holiday_year = pd.DataFrame(
                              get_cal(self.city).get_calendar_holidays(self.year), columns=['day', 'name'])
                    holiday_list = holiday_year['day'].tolist()    
                    
                    days = [day for day in days if day not in holiday_list]
                    
            traffic_mat_dep = np.zeros(shape=(len(days), 24))
            traffic_mat_arr = np.zeros(shape=(len(days), 24))
            
            for i, day in enumerate(days):
                traffic_mat_dep[i,:], traffic_mat_arr[i,:] = self.daily_traffic(
                    stat_index, month=day.month, day=day.day, normalise=normalise,
                    plot=False)

            return traffic_mat_dep, traffic_mat_arr
            
        
        else: 
            
            if not month:
                raise ValueError('Please provide a month.')
            
            df_stat_start = self.df.iloc[np.where(
                self.df['start_stat_id'] == stat_id)]
            df_stat_end = self.df.iloc[np.where(
                self.df['end_stat_id'] == stat_id)]
    
            trips_arrivals = np.zeros(24)
            trips_departures = np.zeros(24)
    
            for hour in range(24):
    
                mask = (df_stat_start['start_dt'].dt.day == day) & (
                    df_stat_start['start_dt'].dt.hour == hour)
                df_hour_start = df_stat_start.loc[mask]
    
                trips_departures[hour] = len(df_hour_start)
    
                mask = (df_stat_end['end_dt'].dt.day == day) & (
                    df_stat_end['end_dt'].dt.hour == hour)
                df_hour_end = df_stat_end.loc[mask]
    
                trips_arrivals[hour] = len(df_hour_end)
    
            if normalise:
                trips_total = sum(trips_departures) + sum(trips_arrivals)
                trips_departures = trips_departures/trips_total
                trips_arrivals = trips_arrivals/trips_total
    
            if plot:
    
                if normalise:
                    plt.plot(np.arange(24), trips_arrivals*100)
                    plt.plot(np.arange(24), trips_departures*100)
                    plt.ylabel('% of total trips')
                else:
                    plt.plot(np.arange(24), trips_arrivals)
                    plt.plot(np.arange(24), trips_departures)
                    plt.ylabel('# trips')
    
                plt.xticks(np.arange(24))
                plt.legend(['Arrivals', 'Departures'])
                plt.xlabel('Hour')
    
                plt.title(
                    f'Hourly traffic for {self.stat.names[self.stat.id_index[stat_id]]} \n on {self.year:d}-{self.month:02d}-{day:02d}')
    
            return trips_departures, trips_arrivals

    def daily_traffic_average(self, stat_index, period='b', normalise=True, plot=False, return_all=False, return_fig=False, return_std=False):
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
        weekdays = [calendar.weekday(self.year, self.month, i) for i in range(
            1, calendar.monthrange(self.year, self.month)[1]+1)]

        if period == 'b':
            days = [date+1 for date, day in enumerate(weekdays) if day <= 4]
        elif period == 'w':
            days = [date+1 for date, day in enumerate(weekdays) if day > 4]
        else:
            raise ValueError(
                "Please provide the period as either 'b' = business days or 'w' = weekends")

        df_start = self.df[self.df['start_stat_id'] ==
                           self.stat.inverse[stat_index]]['start_dt']
        df_end = self.df[self.df['end_stat_id'] ==
                         self.stat.inverse[stat_index]]['end_dt']

        trips_arrivals = np.zeros(shape=(len(days), 24))
        trips_departures = np.zeros(shape=(len(days), 24))

        start_day = df_start.dt.day
        start_hour = df_start.dt.hour

        end_day = df_end.dt.day
        end_hour = df_end.dt.hour

        for i, day in enumerate(days):
            for hour in range(24):
                trips_departures[i, hour] = np.sum(
                    (start_day == day) & (start_hour == hour))
                trips_arrivals[i, hour] = np.sum(
                    (end_day == day) & (end_hour == hour))

        if normalise:
            daily_totals = trips_arrivals.sum(
                axis=1) + trips_departures.sum(axis=1)

            trips_arrivals = np.divide(trips_arrivals.T, daily_totals, out=np.zeros_like(
                trips_arrivals.T), where=daily_totals != 0).T
            trips_departures = np.divide(trips_departures.T, daily_totals, out=np.zeros_like(
                trips_arrivals.T), where=daily_totals != 0).T

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

            month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

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

    def daily_traffic_average_all(self, period='b', normalise=True, plot=False, return_all=False, holidays=True):
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

        # Departures

        if period == 'b':
            df = self.df[['start_stat_id', 'start_dt']][self.df['start_dt'].dt.weekday <= 4]
            if not holidays:
                holiday_year = pd.DataFrame(
                    get_cal(self.city).get_calendar_holidays(self.year), columns=['day', 'name'])
                holiday_list = holiday_year['day'].tolist()
                df = df[~df['start_dt'].dt.date.isin(holiday_list)] # Rows which are not in holiday list
            else:
                holiday_list = []

            weekmask = '1111100'
        elif period == 'w':
            df = self.df[['start_stat_id', 'start_dt']][self.df['start_dt'].dt.weekday > 4]
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


            shap = day_hour_count.shape
            start_mean.append(day_hour_count.sum(axis=0).rename(station) / n_days)
            start_std.append(
                pd.concat(
                    (day_hour_count, pd.DataFrame(np.zeros((n_days - shap[0], shap[1])),columns = day_hour_count.columns))
                          ).std(axis=0).rename(station))

        departures_mean = pd.concat(start_mean, axis=1).fillna(0)
        departures_std = pd.concat(start_std, axis=1).fillna(0)


        # Arrivals

        if period == 'b':
            df = self.df[['end_stat_id', 'end_dt']][self.df['end_dt'].dt.weekday <= 4]
            df = df[~df['end_dt'].dt.date.isin(holiday_list)]
        elif period == 'w':
            df = self.df[['end_stat_id', 'end_dt']][self.df['end_dt'].dt.weekday > 4]

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
                day_hour_count = day_hour_count.loc[day_hour_count.index.month == self.month]

            shap = day_hour_count.shape
            end_mean.append(day_hour_count.sum(axis=0).rename(station) / n_days)
            end_std.append(
                pd.concat(
                    (day_hour_count, pd.DataFrame(np.zeros((n_days - shap[0], shap[1])),columns = day_hour_count.columns))
                          ).std(axis=0).rename(station))

        arrivals_mean = pd.concat(end_mean, axis=1).fillna(0)
        arrivals_std = pd.concat(end_std, axis=1).fillna(0)

# =============================================================================
#         n_dep_mean = departures_mean.sum(axis=0)
#         n_arr_mean = arrivals_mean.sum(axis=0)
#
#         departures_mean = departures_mean.T.loc[n_dep_mean > 1].T # Remove with less than 1 trip per day on average
#         arrivals_mean = arrivals_mean.T.loc[n_arr_mean > 1].T
#         departures_std = departures_std.T.loc[n_dep_mean > 1].T # Remove with less than 1 trip per day on average
#         arrivals_std = arrivals_std.T.loc[n_arr_mean > 1].T
# =============================================================================


        if normalise:
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

                month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

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


    def pickle_daily_traffic(self, normalise=True, plot=False, overwrite=False, holidays=True):
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
        if not overwrite:
            try:
                with open(f'./python_variables/daily_traffic_{self.city}{self.year:d}{monstr}.pickle', 'rb') as file:
                    matrix_b, matrix_w = pickle.load(file)
                return matrix_b, matrix_w
            except FileNotFoundError:
                print("File not found")
        print('Pickling average daily traffic for all stations... \nSit back and relax, this might take a while...')
        pre = time.time()
        departures_b, arrivals_b = self.daily_traffic_average_all(
            'b', normalise=normalise, plot=plot, holidays=holidays)
        print("Hang in there, we're halfway...")
        departures_w, arrivals_w = self.daily_traffic_average_all(
            'w', normalise=normalise, plot=plot, holidays=holidays)

        zeroseries = pd.Series(np.zeros((24,)))
        departures_b = departures_b.add(zeroseries).fillna(0)
        arrivals_b = arrivals_b.add(zeroseries).fillna(0)

        departures_w = departures_w.add(zeroseries).fillna(0)
        arrivals_w = arrivals_w.add(zeroseries).fillna(0)

        id_index = self.stat.id_index
        matrix_b = np.zeros((len(id_index.keys()), 48))
        matrix_w = np.zeros((len(id_index.keys()), 48))
        for id_, index in zip(id_index.keys(), id_index.values()):
            try:
                matrix_b[index, :24] = departures_b.loc[id_]
            except KeyError:
                print(f"Key {id_} not found in departures weekdays.")
            try:
                matrix_b[index, 24:] = arrivals_b.loc[id_]
            except KeyError:
                print(f"Key {id_} not found in arrivals weekdays.")
            try:
                matrix_w[index, :24] = departures_w.loc[id_]
            except KeyError:
                print(f"Key {id_} not found in departures weekend.")
            try:
                matrix_w[index, 24:] = arrivals_w.loc[id_]
            except KeyError:
                print(f"Key {id_} not found in arrivals weekdays.")

        matrix_b = np.nan_to_num(matrix_b, copy=False, nan=0.0)
        matrix_w = np.nan_to_num(matrix_w, copy=False, nan=0.0)

        with open(f'./python_variables/daily_traffic_{self.city}{self.year:d}{monstr}.pickle', 'wb') as file:
            pickle.dump((matrix_b, matrix_w), file)

        print(f'Pickling done. Time taken: {(time.time()-pre):.1f} s')

        return matrix_b, matrix_w

    def subset(self, days='all', hours='all', minutes='all', activity_type='all'):
        """
        Get subset of the dataframe df.

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
        return df_subset(self.df, self.weekdays, days, hours, minutes, activity_type)


class Classifier:
    def __init__(self, dist_func):

        if dist_func == 'norm':
            self.dist = self.dist_norm
        elif dist_func == 'dtw':
            self.dist = self.dist_dtw

        self.centroids = None
        self.Davies_Bouldin_index = None
        self.Dunn_index = None

    def dist_norm(self, vec1, vec2):
        return np.linalg.norm(vec1-vec2)

    def dist_dtw(self, vec1, vec2):
        return dtw.dtw(vec1, vec2)[1]

    def k_means(self, data_matrix, k, init_centroids=None, max_iter=15, seed=None, verbose=False):

        n_stations = len(data_matrix)

        stat_indices = np.arange(n_stations)

        if type(init_centroids) != type(None):
            centroids = init_centroids

        else:
            if seed:
                np.random.seed(seed)

            np.random.shuffle(stat_indices)
            centroid_indices = stat_indices[:k]
            centroids = data_matrix[centroid_indices, :]

        labels_old = np.ones(n_stations)
        labels_new = np.zeros(n_stations)

        if verbose:
            print('Starting clustering...')

        pre = time.time()

        i = 0
        while sum(labels_old-labels_new != 0) > np.floor(n_stations/100) and i < max_iter:
            labels_old = labels_new.copy()

            for stat_index in range(n_stations):
                distances = np.empty(k)
                for center_index in range(k):
                    distances[center_index] = self.dist(
                        data_matrix[stat_index, :], centroids[center_index, :])

                labels_new[stat_index] = np.argmin(distances)

            for label in range(k):
                label_mat = data_matrix[np.where(labels_new == label), :]
                centroids[label, :] = np.mean(label_mat, axis=1)

            i += 1

            if verbose:
                print(
                    f'Iteration: {i} - # Changed labels: {sum(labels_old-labels_new != 0)} - Runtime: {time.time()-pre}s')

        self.centroids = centroids
        if verbose:
            print('Clustering done')

    def k_medoids(self, data_mat, k, verbose=False):

        if verbose:
            print('Starting clustering...')

        n = len(data_mat)

        # BUILD

        if verbose:
            print('Finding distance matrix...')

        d_mat = np.zeros(shape=(n, n))

        for i in range(n-1):
            for j in range(i+1, n):
                d_mat[i, j] = np.linalg.norm(data_mat[i]-data_mat[j])
        d_mat = np.where(d_mat, d_mat, d_mat.T)

        d_sums = np.sum(d_mat, axis=1)

        if verbose:
            print('Building medoids...')

        m_indices = [int(np.argmin(d_sums))]
        medoids = np.zeros(shape=(k, data_mat.shape[1]))
        medoids[0] = data_mat[m_indices[0], :]

        C_mat = np.zeros(shape=(n, n))

        for label in range(1, k):

            for i in range(n):
                for j in range(n):
                    if i and j not in m_indices:

                        m_distances = [
                            d_mat[m_index, j] for m_index in m_indices]

                        closest_m = m_indices[np.argmin(m_distances)]
                        C_mat[j, i] = max(d_mat[closest_m, j] - d_mat[i, j], 0)

            total_gains = np.sum(C_mat, axis=0)

            m_indices.append(np.argmax(total_gains))

            medoids[label, :] = data_mat[m_indices[label], :]

        if verbose:
            print('Assigning initial labels...')

        labels = np.empty(n)
        for stat_index in range(n):
            distances = np.zeros(k)
            for label in range(k):
                distances[label] = d_mat[stat_index, m_indices[label]]
            labels[stat_index] = np.argmin(distances)

        # SWAP

        if verbose:
            print('Swapping medoids...')

        current_cost = 0
        for label, m in enumerate(m_indices):
            current_cost += np.sum(d_mat[m, np.where(labels == label)])

        old_cost = current_cost

        while True:

            best_cost = old_cost

            swap_label = 0
            swap_to = m_indices[0]

            for label, m in enumerate(m_indices):

                h_indices = np.where(labels == label)[0]
                h_indices = h_indices[np.where(h_indices != m)]

                for h in h_indices:
                    m_indices_prop = m_indices.copy()
                    m_indices_prop[label] = h

                    labels_prop = np.empty(n)
                    for stat_index in range(n):
                        distances = np.zeros(k)
                        for l in range(k):
                            distances[l] = d_mat[stat_index, m_indices_prop[l]]
                        labels_prop[stat_index] = np.argmin(distances)

                    cost = 0
                    for l, m_prop in enumerate(m_indices_prop):
                        cost += np.sum(d_mat[m_prop,
                                       np.where(labels_prop == l)])

                    if cost < best_cost:
                        best_cost = cost
                        swap_label = label
                        swap_to = h

            m_indices[swap_label] = swap_to

            medoids[swap_label] = data_mat[swap_to]

            labels = np.empty(n)
            for stat_index in range(n):
                distances = np.zeros(k)
                for label in range(k):
                    distances[label] = d_mat[stat_index, m_indices[label]]
                labels[stat_index] = np.argmin(distances)

            if best_cost >= old_cost:
                break

            old_cost = best_cost

        self.centroids = medoids

        if verbose:
            print('clustering done')

    def h_clustering_find_clusters(self, data_mat, init_distance_filename):

        n = len(data_mat)

        try:
            with open(init_distance_filename, 'rb') as file:
                distance_matrix = pickle.load(file)
            print('pickle loaded.')

        except FileNotFoundError:
            print('No pickle found. Calculating initial distance matrix...')

            distance_matrix = np.full(shape=(n, n), fill_value=np.inf)
            for i in range(n-1):
                for j in range(i+1, n):
                    distance_matrix[i, j] = 1 / \
                        np.sqrt(2)*self.dist(data_mat[i], data_mat[j])

        cluster_list = np.array([set([i]) for i in range(n)])

        print('Starting clustering...')

        clustering_history = [0 for _ in range(n)]
        clustering_history[0] = cluster_list

        temp_mat = data_mat.copy()
        pre = time.time()
        count = 0
        while len(cluster_list) > 1:
            min_indices = np.where(distance_matrix == np.min(distance_matrix))
            stat_1 = np.min(min_indices)
            stat_2 = np.max(min_indices)

            cluster_list[stat_1] = cluster_list[stat_1] | cluster_list[stat_2]
            cluster_list = np.delete(cluster_list, stat_2)

            clustering_history[count+1] = cluster_list

            distance_matrix = np.delete(distance_matrix, stat_2, axis=0)
            distance_matrix = np.delete(distance_matrix, stat_2, axis=1)

            temp_mat = np.delete(temp_mat, stat_2, axis=0)

            cluster_mat = data_mat[list(cluster_list[stat_1])]
            centroid = np.mean(cluster_mat, axis=0)

            temp_mat[stat_1] = centroid

            for i, stat in enumerate(temp_mat):
                stat_cluster_size = len(cluster_list[i])
                centroid_cluster_size = len(cluster_list[stat_1])
                w = np.sqrt(stat_cluster_size * centroid_cluster_size /
                            (stat_cluster_size + centroid_cluster_size))

                if i < stat_1:
                    distance_matrix[i, stat_1] = w*self.dist(stat, centroid)
                elif i > stat_1:
                    distance_matrix[stat_1, i] = w*self.dist(stat, centroid)

            count += 1
            if count % 100 == 0:
                print(
                    f'{count} iterations done. Current runtime: {time.time()-pre}s')

        print(f'Clustering done. Time taken: {(time.time()-pre):.1f} s')

        return clustering_history

    def h_clustering(self, data_mat, k, results_filename, init_distance_filename):

        try:
            with open(results_filename, 'rb') as file:
                clustering_history = pickle.load(file)
                print('Pickle loaded.')

        except FileNotFoundError:
            print('No previous clustering has been found. Performing clustering...')
            clustering_history = self.h_clustering_find_clusters(
                data_mat, init_distance_filename)
            print('Pickling clustering history...')
            # with open(results_filename, 'wb') as file:
            # pickle.dump(clustering_history, file)
            # print('Pickling done.')

        cluster_list = clustering_history[-k]
        centroids = np.empty(shape=(k, data_mat.shape[1]))

        for i in range(k):
            cluster_mat = data_mat[list(cluster_list[i])]
            centroids[i, :] = np.mean(cluster_mat, axis=0)

        self.centroids = centroids

        return centroids, cluster_list

    def predict(self, vec):
        if self.centroids is None:
            raise ValueError(
                'No centroids have been computed. Please run a clustering algorithm first.')

        if len(vec) != len(self.centroids[0]):
            raise ValueError(
                'Vector must be the same dimension as the centroids.')

        distances = np.empty(len(self.centroids))
        for center_index in range(len(self.centroids)):
            distances[center_index] = self.dist(
                vec, self.centroids[center_index])

        return np.argmin(distances)

    def mass_predict(self, data_mat):

        labels = np.empty(len(data_mat))
        for stat_index in range(len(data_mat)):
            labels[stat_index] = self.predict(data_mat[stat_index])

        return labels

    def get_Davies_Bouldin_index(self, data_mat, labels=None, verbose=False):
        """
        Calculates the Davies-Bouldin index of clustered data.

        Parameters
        ----------
        data_mat : ndarray
            Array containing the feature vectors.
        labels : itr, optional
            Iterable containg the labels of the feature vectors. If no labels
            are given, they are calculated using the mass_predict method.

        Returns
        -------
        DB_index : float
            Davies-Bouldin index.

        """
        if labels is None:
            if verbose:
                print('Getting labels...')
            labels = self.mass_predict(data_mat)

        k = len(self.centroids)

        if verbose:
            print('Calculating Davies-Bouldin index...')

        pre = time.time()

        S_scores = np.empty(k)

        for i in range(k):
            data_mat_cluster = data_mat[np.where(labels == i)]
            distances = [self.dist(row, self.centroids[i])
                         for row in data_mat_cluster]
            S_scores[i] = np.mean(distances)

        R = np.empty(shape=(k, k))
        for i in range(k):
            for j in range(k):
                if i == j:
                    R[i, j] = 0
                else:
                    R[i, j] = (S_scores[i] + S_scores[j]) / \
                        self.dist(self.centroids[i], self.centroids[j])

        D = [max(row) for row in R]

        DB_index = np.mean(D)

        if verbose:
            print(f'Done. Time taken: {(time.time()-pre):.1f} s')

        self.Davies_Bouldin_index = DB_index

        return DB_index

    def get_Dunn_index(self, data_mat, labels=None, verbose=False):
        """
        Calculates the Dunn index of clustered data. WARNING: VERY SLOW.

        Parameters
        ----------
        data_mat : ndarray
            Array containing the feature vectors.
        labels : itr, optional
            Iterable containg the labels of the feature vectors. If no labels
            are given, they are calculated using the mass_predict method.

        Returns
        -------
        D_index : float
            Dunn index.

        """
        if labels is None:
            if verbose:
                print('Getting labels...')
            labels = self.mass_predict(data_mat)

        k = len(self.centroids)

        if verbose:
            print('Calculating Dunn Index...')

        pre = time.time()

        intra_cluster_distances = np.empty(k)
        inter_cluster_distances = np.full(shape=(k, k), fill_value=np.inf)

        for i in range(k):
            data_mat_cluster = data_mat[np.where(labels == i)]
            cluster_size = len(data_mat_cluster)
            distances = np.empty(shape=(cluster_size, cluster_size))

            for h in range(cluster_size):
                for j in range(cluster_size):
                    distances[h, j] = self.dist(data_mat[h], data_mat[j])

            intra_cluster_distances[i] = np.max(distances)

            for j in range(k):
                if j != i:
                    data_mat_cluster_j = data_mat[np.where(labels == j)]
                    cluster_size_j = len(data_mat_cluster_j)
                    between_cluster_distances = np.empty(
                        shape=(cluster_size, cluster_size_j))
                    for m in range(cluster_size):
                        for n in range(cluster_size_j):
                            between_cluster_distances[m, n] = self.dist(
                                data_mat_cluster[m], data_mat_cluster_j[n])
                    inter_cluster_distances[i, j] = np.min(
                        between_cluster_distances)

        D_index = np.min(inter_cluster_distances) / \
            np.max(intra_cluster_distances)

        if verbose:
            print(f'Done. Time taken: {(time.time()-pre):.1f} s')

        self.Dunn_index = D_index

        return D_index

    def get_silhouette_index(self, data_mat, labels=None, verbose=False):
        """
        Calculates the silhouette index of clustered data.

        Parameters
        ----------
        data_mat : ndarray
            Array containing the feature vectors.
        labels : itr, optional
            Iterable containg the labels of the feature vectors. If no labels
            are given, they are calculated using the mass_predict method.

        Returns
        -------
        S_index : float
            Silhouette index.

        """
        if labels is None:
            if verbose:
                print('Getting labels...')
            labels = self.mass_predict(data_mat)

        k = len(self.centroids)

        if verbose:
            print('Calculating Silhouette index...')

        pre = time.time()

        s_coefs = np.empty(len(data_mat))

        for i, vec1 in enumerate(data_mat):
            in_cluster = np.delete(data_mat, i, axis=0)
            in_cluster = in_cluster[np.where(
                np.delete(labels, i) == labels[i])]

            in_cluster_size = len(in_cluster)

            in_cluster_distances = np.empty(in_cluster_size)
            for j, vec2 in enumerate(in_cluster):
                in_cluster_distances[j] = self.dist(vec1, vec2)

            mean_out_cluster_distances = np.full(k, fill_value=np.inf)

            for j in range(k):
                if j != labels[i]:
                    out_cluster = data_mat[np.where(labels == j)]
                    out_cluster_distances = np.empty(len(out_cluster))

                    for l, vec2 in enumerate(out_cluster):
                        out_cluster_distances[l] = self.dist(vec1, vec2)

                    mean_out_cluster_distances[j] = np.mean(
                        out_cluster_distances)

            ai = np.mean(in_cluster_distances)
            bi = np.min(mean_out_cluster_distances)

            s_coefs[i] = (bi-ai)/max(ai, bi)

        S_index = np.mean(s_coefs)

        if verbose:
            print(f'Done. Time taken: {(time.time()-pre):.1f} s')

        self.Silhouette_index = S_index

        return S_index

    def k_means_test(self, data_mat, k_min=2, k_max=10, seed=None):

        k_range = range(k_min, k_max+1)

        results = [0 for _ in k_range]

        for i, k in enumerate(k_range):
            self.k_means(data_mat, k, seed=seed, verbose=False)
            labels = self.mass_predict(data_mat)
            DB_index = self.get_Davies_Bouldin_index(
                data_mat, labels, verbose=False)
            D_index = self.get_Dunn_index(data_mat, labels, verbose=False)
            S_index = self.get_silhouette_index(data_mat, labels, verbose=False)

            results[i] = (k, DB_index, D_index, S_index)

        print(f'{"Test result for k-means":^50}')
        print('='*50)
        print(f'{"k":5}{"DB_index":15}{"D_index":15}{"S_index":15}')
        for result in results:
            print(
                f'{result[0]:<5,d}{result[1]:<15.8f}{result[2]:<15.8f}{result[3]:<15,.8f}')

def get_cal(city):
    if city == 'bergen':
        cal = Norway()
    elif city == 'boston':
        cal = Massachusetts()
    elif city == 'buenos_aires':
        cal = Argentina()
    elif city == 'chic':
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
    elif city == 'minn':
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
    elif city == 'washDC':
        cal = DistrictOfColumbia()
    else:
        raise KeyError('Calendar key not found')
    return cal


def get_valid_months(city, year):
    if city in ['trondheim', 'minn']:
        return range(4, 11+1)
    elif city == 'oslo':
        return range(4, 12+1)
    elif city in ['helsinki', 'montreal']:
        return range(4, 10+1)
    else:
        return range(1, 12+1)


name_dict = {
    'bergen': 'Bergen',
    'boston': 'Boston',
    'buenos_aires': 'Buenos Aires',
    'chic': 'Chicago',
    'edinburgh': 'Edinburgh',
    'guadalajara': 'Guadalajara',
    'helsinki': 'Helsinki',
    'london': 'London',
    'la': 'Los Angeles',
    'madrid': 'Madrid',
    'mexico': 'Mexico City',
    'minn': 'Minneapolis',
    'montreal': 'Montreal',
    'nyc': 'New York City',
    'oslo': 'Oslo',
    'sfran': 'San Francisco',
    'taipei': 'Taipei',
    'trondheim': 'Trondheim',
    'washDC': 'Washington DC'}





if __name__ == "__main__":
    pre = time.time()
    data = Data('london', 2019, 1, overwrite=True)
    print(time.time() - pre)
    #traffic_arr, traffic_dep = data.daily_traffic_average_all(plot=False)
    # print(time.time() - pre)
    traffic_arr, traffic_dep = data.pickle_daily_traffic(overwrite=True, holidays=True)
