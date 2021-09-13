"""
Created on Mon Feb 22 15:52:51 2021

@author: Mattek Group 3
"""


import pandas as pd
import numpy as np
import os
import pickle
import calendar
import datetime
import time
import warnings
from pyproj import Transformer
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from requests import get
import dataframe_key


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

        stat_files = [file for file in os.listdir('data') if 'Divvy_Stations' in file]

        col_list = ['id', 'name', 'latitude', 'longitude']
        key = {'ID':'id', 'Station Name':'name', 'Latitude':'latitude','Longitude':'longitude'}

        try:
            stat_df = pd.read_csv(
                'data/Divvy_Bicycle_Stations_-_All_-_Map.csv').rename(columns = key)
            stat_df = stat_df[col_list]
        except FileNotFoundError:
            stat_df = pd.DataFrame(columns = col_list)

        for file in stat_files:
            df = pd.read_csv(f'./data/{file}')[col_list]
            stat_df = pd.concat([stat_df, df], sort = False)

        if stat_df.size == 0:
            raise FileNotFoundError(
                'No data files containing station data found. Please read the docstring for more information.') from exc

        stat_df.drop_duplicates(subset = 'name', inplace = True)

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
        df = df.rename(columns = dataframe_key.get_key('nyc'))

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


def get_data_month(city, year, month, blacklist=None):
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
                        'buenos_aires', 'madrid',
                        'mexico', 'taipei'] # Remember to update this list

    if city not in supported_cities:
        raise ValueError("This city is not currently supported. Supported cities are {}".format(supported_cities))

    # Make folder for dataframes if not found
    if not os.path.exists('python_variables/big_data'):
        os.makedirs('python_variables/big_data')

    try:
        with open(f'./python_variables/big_data/{city}{year:d}{month:02d}_dataframe_blcklst={blacklist}.pickle', 'rb') as file:
            df = pickle.load(file)
        print('Pickle loaded')

    except FileNotFoundError:

        print('No dataframe pickle found. Pickling dataframe...')


        if city == "nyc":

            try:
                df = pd.read_csv(f'./data/{year:d}{month:02d}-citibike-tripdata.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant files can be found at https://www.citibikenyc.com/system-data') from exc

            df = df.rename(columns = dataframe_key.get_key(city))

            try:
                with open('./python_variables/JC_blacklist', 'rb') as file:
                    JC_blacklist = pickle.load(file)

                df = df[~df['start_stat_id'].isin(JC_blacklist)]
                df = df[~df['end_stat_id'].isin(JC_blacklist)]

            except FileNotFoundError:
                print('No JC blacklist found. Continuing...')

            df.dropna(inplace=True)
            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])


        elif city == "washDC":

            try:
                df = pd.read_csv(f'./data/{year:d}{month:02d}-capitalbikeshare-tripdata.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant files can be found at https://www.capitalbikeshare.com/system-data') from exc

            df = df.rename(columns = dataframe_key.get_key(city))

            df['start_stat_lat'] = ''
            df['start_stat_long'] = ''
            df['end_stat_lat'] = ''
            df['end_stat_long'] = ''

            stat_df = pd.read_csv('data/Capital_Bike_Share_Locations.csv')

            for _ , stat in stat_df.iterrows():
                start_matches = np.where(df['start_stat_id'] == stat['TERMINAL_NUMBER'])
                end_matches = np.where(df['end_stat_id'] == stat['TERMINAL_NUMBER'])

                df.at[start_matches[0], 'start_stat_lat'] = stat['LATITUDE']
                df.at[start_matches[0], 'start_stat_long'] = stat['LONGITUDE']
                df.at[end_matches[0], 'end_stat_lat'] = stat['LATITUDE']
                df.at[end_matches[0], 'end_stat_long'] = stat['LONGITUDE']

            df.replace('', np.nan, inplace = True)
            df.dropna(inplace=True)

            max_lat = 38.961029
            min_lat = 38.792686
            max_long= -76.909415
            min_long= -77.139396

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

            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])


        elif city == "chic":

            q = int(np.ceil(month/3))

            try:
                df = pd.read_csv(f'./data/Divvy_Trips_{year:d}_Q{q}.csv')

            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant files can be found at https://www.divvybikes.com/system-data') from exc
            
            if q == 2:
                col_dict = {'01 - Rental Details Rental ID' : 'trip_id',
                            '01 - Rental Details Local Start Time' : 'start_time',
                            '01 - Rental Details Local End Time' : 'end_time',
                            '01 - Rental Details Bike ID' : 'bikeid',
                            '01 - Rental Details Duration In Seconds Uncapped' : 'tripduration',
                            '03 - Rental Start Station ID' : 'from_station_id',
                            '03 - Rental Start Station Name' : 'from_station_name',
                            '02 - Rental End Station ID' : 'to_station_id',
                            '02 - Rental End Station Name' : 'to_station_name',
                            'User Type' : 'usertype',
                            'Member Gender' : 'gender',
                            '05 - Member Details Member Birthday Year' : 'birthyear'}
                df = df.rename(columns = col_dict)
            
            df = df.rename(columns = dataframe_key.get_key(city))

            n_days = calendar.monthrange(year, month)[1]

            df = df.iloc[np.where(df['start_t'] > f'{year:d}-{month:02d}-01 00:00:00')]
            df = df.iloc[np.where(df['start_t'] < f'{year:d}-{month:02d}-{n_days} 23:59:59')]

            df.reset_index(inplace = True, drop = True)

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

            df.replace('', np.nan, inplace = True)
            df.dropna(subset = ['start_stat_lat',
                                'start_stat_long',
                                'end_stat_lat',
                                'end_stat_long'], inplace = True)

            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])
            df['duration'] = df['duration'].str.replace(',', '').astype(float)


        elif city == "sfran":

            try:
                df = pd.read_csv(f'./data/{year:d}{month:02d}-baywheels-tripdata.csv')
            except FileNotFoundError:
                try:
                    df = pd.read_csv(f'./data/{year:d}{month:02d}-fordgobike-tripdata.csv')
                except FileNotFoundError as exc:
                    raise FileNotFoundError('No trip data found. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data') from exc

            df = df.rename(columns = dataframe_key.get_key(city))
            df.dropna(inplace=True)

            df = df.iloc[np.where(df['start_stat_lat'] > 37.593220)]
            df = df.iloc[np.where(df['end_stat_lat'] > 37.593220)]
            df = df.iloc[np.where(df['start_stat_long'] < -80)]
            df = df.iloc[np.where(df['end_stat_long'] < -80)]
            
            
            df.sort_values(by = 'start_t', inplace = True)
            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])


        elif city == "sjose":

            try:
                df = pd.read_csv(f'./data/{year:d}{month:02d}-baywheels-tripdata.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data') from exc

            df = df.rename(columns = dataframe_key.get_key(city))
            df.dropna(inplace=True)

            df = df.iloc[np.where(df['start_stat_lat'] < 37.593220)]
            df = df.iloc[np.where(df['end_stat_lat'] < 37.593220)]

            df.sort_values(by = 'start_t', inplace = True)
            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])


        elif city == "london":

            month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May',
                          6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct',
                          11:'Nov', 12:'Dec'}

            data_files = [file for file in os.listdir('data') if 'JourneyDataExtract' in file]
            data_files = [file for file in data_files if '{}'.format(year)
                          and '{}'.format(month_dict[month]) in file]

            if len(data_files) == 0:
                raise FileNotFoundError('No London data for {}. {} found. All relevant files can be found at https://cycling.data.tfl.gov.uk/.'.format(month_dict[month], year))

            if isinstance(data_files, str):
                warnings.warn('Only one data file found. Please check that you have all available data.')

            df = pd.read_csv('./data/' + data_files[0])

            for file in data_files[1:]:
                df_temp = pd.read_csv('./data/' + file)
                df = pd.concat([df, df_temp], sort = False)

            df.rename(columns = dataframe_key.get_key(city), inplace = True)

            n_days = calendar.monthrange(year, month)[1]

            df = df.iloc[np.where(df['start_t'] >= f'01/{month:02d}/{year} 00:00')]
            df = df.iloc[np.where(df['start_t'] <= f'{n_days}/{month:02d}/{year} 23:59')]

            df.sort_values(by = 'start_t', inplace = True)
            df.reset_index(inplace = True)

            df['start_t'] = pd.to_datetime(df['start_t'], format = '%d/%m/%Y %H:%M').astype(str)
            df['end_t'] = pd.to_datetime(df['end_t'], format = '%d/%m/%Y %H:%M').astype(str)

            stat_df = pd.read_csv('./data/london_stations.csv')
            stat_df.at[np.where(stat_df['station_id'] == 502)[0][0], 'latitude'] = 51.53341

            df['start_stat_lat'] = ''
            df['start_stat_long'] = ''
            df['end_stat_lat'] = ''
            df['end_stat_long'] = ''

            for _ , stat in stat_df.iterrows():
                start_matches = np.where(df['start_stat_name'] == stat['station_name'])
                end_matches = np.where(df['end_stat_name'] == stat['station_name'])

                df.at[start_matches[0], 'start_stat_lat'] = stat['latitude']
                df.at[start_matches[0], 'start_stat_long'] = stat['longitude']
                df.at[end_matches[0], 'end_stat_lat'] = stat['latitude']
                df.at[end_matches[0], 'end_stat_long'] = stat['longitude']

            df.replace('', np.nan, inplace = True)
            df.dropna(inplace = True)

            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])

            df = df[df.start_dt.dt.month == month]

            df.reset_index(inplace = True, drop = True)


        elif city == "oslo":

            try:
                df = pd.read_csv(f'./data/{year:d}{month:02d}-oslo.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant files can be found at https://oslobysykkel.no/en/open-data/historical') from exc

            df = df.rename(columns = dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])


        elif city == "edinburgh":

            try:
                df = pd.read_csv(f'./data/{year:d}{month:02d}-edinburgh.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant files can be found at https://edinburghcyclehire.com/open-data/historical') from exc

            df = df.rename(columns = dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])


        elif city == "bergen":

            try:
                df = pd.read_csv(f'./data/{year:d}{month:02d}-bergen.csv')
            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant files can be found at https://bergenbysykkel.no/en/open-data/historical') from exc

            df = df.rename(columns = dataframe_key.get_key(city))
            df.dropna(inplace=True)
            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])


        elif city == "buenos_aires":

            try:
                df_year = pd.read_csv(f"./data/recorridos-realizados-{year:d}.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant files can be found at https://data.buenosaires.gob.ar/dataset/bicicletas-publicas') from exc

            df_year = df_year.rename(columns = dataframe_key.get_key(city))
            #df_year['month'] = pd.to_datetime(df_year['fecha_origen_recorrido']).dt.month
            df_year['month'] = pd.to_datetime(df_year['start_t']).dt.month
            df = df_year.loc[df_year.month == month]
            df.sort_values(by=['start_t'], inplace=True)
            df.reset_index(inplace = True, drop = True)

            df['start_dt'] = pd.to_datetime(df['start_t'])
            df['end_dt'] = pd.to_datetime(df['end_t'])


        elif city == "madrid":
            #df
            if year == 2019 and month > 7:
                try:
                    df = pd.read_json(f"./data/{year:d}{month:02d}_movements.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError('No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc
                try:
                    df_pre = pd.read_json(f"./data/{year:d}{(month-1):02d}_movements.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError('No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc
                df = df.rename(columns = dataframe_key.get_key(city))
                df_pre = df_pre.rename(columns = dataframe_key.get_key(city))
                
                df['start_dt'] = pd.to_datetime(df['start_t'], format = '%Y-%m-%dT%H:%M:%SZ') + pd.DateOffset(hours=2)
                df_pre['start_dt'] = pd.to_datetime(df_pre['start_t'], format = '%Y-%m-%dT%H:%M:%SZ') + pd.DateOffset(hours=2)
                
                df = df[df['start_dt'].dt.month == month]
                df_pre = df_pre[df_pre['start_dt'].dt.month == month]
                
                df = pd.concat((df_pre, df))
                
            elif year == 2019 and month == 7:
                try:
                    df = pd.read_json(f"./data/{year:d}{month:02d}_movements.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError('No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc
                
                df = df.rename(columns = dataframe_key.get_key(city))
                df['start_dt'] = pd.to_datetime(df['start_t'], format = '%Y-%m-%dT%H:%M:%SZ') + pd.DateOffset(hours=2)
            else:
                try:
                    df = pd.read_json(f"./data/{year:d}{month:02d}_Usage_Bicimad.json", lines=True)
                except FileNotFoundError as exc:
                    raise FileNotFoundError('No trip data found. All relevant files can be found at https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)') from exc
                
                df['unplug_hourTime'] = pd.json_normalize(df['unplug_hourTime'])
                df = df.rename(columns = dataframe_key.get_key(city))
                df['start_t'] = df['start_t'].str[:-6]
                df['start_dt'] = pd.to_datetime(df['start_t'], format = '%Y-%m-%dT%H:%M:%S') #Timezone is correct in older data.
            
            
            
            
            df['start_t'] = df['start_dt'].astype(str)

            df['end_dt'] = df['start_dt'] + pd.to_timedelta(df['duration'], unit='s')
            #df['end_t'] = pd.to_datetime(df['end_dt']).astype(str)
            if year == 2019 and month >=7:
                _ , stations = pd.read_json(
                    f"./data/{year:d}{month:02d}_stations_madrid.json",
                    lines=True).iloc[-1]
            else:
                _ , stations = pd.read_json(
                    f"./data/Bicimad_Stations_{year:d}{month:02d}.json",
                    lines=True).iloc[-1]
            
            stations = pd.DataFrame(stations)

            name_dict = dict(zip(stations['id'], stations['name']))
            long_dict = dict(zip(stations['id'], stations['longitude'].astype(float)))
            lat_dict = dict(zip(stations['id'], stations['latitude'].astype(float)))
            addr_dict = dict(zip(stations['id'], stations['address']))


            df['start_stat_name'] = df['start_stat_id'].map(name_dict)
            df['start_stat_lat'] = df['start_stat_id'].map(lat_dict)
            df['start_stat_long'] = df['start_stat_id'].map(long_dict)
            df['start_stat_desc'] = df['start_stat_id'].map(addr_dict)

            df['end_stat_name'] = df['end_stat_id'].map(name_dict)
            df['end_stat_lat'] = df['end_stat_id'].map(lat_dict)
            df['end_stat_long'] = df['end_stat_id'].map(long_dict)
            df['end_stat_desc'] = df['end_stat_id'].map(addr_dict)

            df.reset_index(inplace = True, drop = True)

        elif city == "mexico":

            try:
                df = pd.read_csv(f"./data/{year:d}-{month:02d}-mexico.csv")
            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant files can be found at https://www.ecobici.cdmx.gob.mx/en/informacion-del-servicio/open-data') from exc

            df.rename(columns = dataframe_key.get_key(city), inplace=True)

            df['start_dt'] = pd.to_datetime(df['start_date'] + df['start_time'],
                                            format='%d/%m/%Y%H:%M:%S')
            df['end_dt'] = pd.to_datetime(df['end_date'] + df['end_time'],
                                          format='%d/%m/%Y%H:%M:%S')
            df.drop(['start_date','start_time','end_date','end_time'], axis=1, inplace=True)
            df['duration'] = (df['end_dt'] - df['start_dt']).dt.total_seconds()

            df['start_t'] = df['start_dt'].astype(str)

            stations = pd.DataFrame(pd.read_json("./data/stations_mexico.json",
                                                 lines=True)['stations'][0])

            name_dict = dict(zip(stations['id'], stations['address']))
            locations = stations['location'].apply(pd.Series)
            long_dict = dict(zip(stations['id'], locations['lon'].astype(float)))
            lat_dict = dict(zip(stations['id'], locations['lat'].astype(float)))
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
            df.reset_index(inplace = True, drop = True)

        elif city == "taipei":
            colnames = ['start_t', 'start_stat_name_zh',
                        'end_t', 'end_stat_name_zh', 'duration', 'rent_date']

            try:
                df = pd.read_csv(f"./data/{year:d}{month:02d}-taipei.csv",
                                 usecols=range(5), names=colnames)
            except FileNotFoundError as exc:
                raise FileNotFoundError('No trip data found. All relevant data can be found at https://data.taipei/#/ and https://drive.google.com/drive/folders/1QsROgp8AcER6qkTJDxpuV8Mt1Dy6lGQO') from exc

            # Update names of stations
            df.replace(to_replace='信義杭州路口(中華電信總公司', value='信義杭州路口(中華電信總公司)', inplace=True)
            df.replace(to_replace='捷運科技大樓站', value='捷運科技大樓站(台北教育大學)', inplace=True)
            df.replace(to_replace='?公公園', value='瑠公公園', inplace=True)
            df.replace(to_replace='饒河夜市', value='饒河夜市(八德路側)', inplace=True)
            df.replace(to_replace='捷運大坪林站(3號出口)', value='捷運大坪林站(1號出口)', inplace=True)
            df.replace(to_replace='新明路321巷口', value='新明路262巷口', inplace=True)

            df['start_dt'] = pd.to_datetime(df['start_t'], format='%Y-%m-%d %H:%M:%S')
            df['end_dt'] = pd.to_datetime(df['end_t'], format='%Y-%m-%d %H:%M:%S')
            df['duration'] = pd.to_timedelta(df.duration).dt.total_seconds()

            try:
                stations = pd.DataFrame.from_dict(
                    list(pd.read_json("./data/YouBikeTP.json")['retVal']))
            except FileNotFoundError as exc:
                raise FileNotFoundError('No station data found. The data can be found at https://tcgbusfs.blob.core.windows.net/blobyoubike/YouBikeTP.json') from exc

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

            df.dropna(inplace=True)
            df.sort_values(by=['start_dt'], inplace=True)
            df.reset_index(inplace=True, drop = True)

        if blacklist:
            df = df[~df['start_stat_id'].isin(blacklist)]
            df = df[~df['end_stat_id'].isin(blacklist)]

        with open(f'./python_variables/big_data/{city}{year:d}{month:02d}_dataframe_blcklst={blacklist}.pickle', 'wb') as file:
            pickle.dump(df, file)

        print('Pickling done.')

    try:
        with open(f'./python_variables/day_index_{city}{year:d}{month:02d}.pickle', 'rb') as file:
            days = pickle.load(file)
    except FileNotFoundError:
        print("Pickle does not exist. Pickling day indices...")
        days = pickle_data(df, city, year, month)
        print("Pickling done.")

    print(f"Data loaded: {city}{year:d}{month:02d}")

    return df, days

def get_data_year(city, year, blacklist=None):
    
    supported_cities = ['nyc', 'sfran', 'washDC', 'chic', 'london'
                        ] # Remember to update this list

    if city not in supported_cities:
        raise ValueError("This city is not currently supported. Supported cities are {}".format(supported_cities))

    # Make folder for dataframes if not found
    if not os.path.exists('python_variables/big_data'):
        os.makedirs('python_variables/big_data')
    
    if city == "nyc":
        
        files = [file for file in os.listdir('data') if f'{year:d}' in file[:4] and 'citibike' in file]
        files.sort()
        
        if len(files) < 12:
            raise FileNotFoundError("Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")
        
        df = pd.read_csv('data/' + files[0])
        
        for file in files[1:]:
            df_temp = pd.read_csv('data/' + file)
            df = pd.concat([df, df_temp], sort = False)
    
        df = df.rename(columns = dataframe_key.get_key(city))

        try:
            with open('./python_variables/JC_blacklist', 'rb') as file:
                JC_blacklist = pickle.load(file)

            df = df[~df['start_stat_id'].isin(JC_blacklist)]
            df = df[~df['end_stat_id'].isin(JC_blacklist)]

        except FileNotFoundError:
            print('No JC blacklist found. Continuing...')

        df.dropna(inplace=True)
        df.reset_index(inplace = True, drop = True)

        df['start_dt'] = pd.to_datetime(df['start_t'])
        df['end_dt'] = pd.to_datetime(df['end_t'])
    
    
    
    elif city == "washDC":
        
        files = [file for file in os.listdir('data') if f'{year:d}' in file[:4] and 'capitalbikeshare' in file]
        files.sort()
        
        if len(files) < 12:
            raise FileNotFoundError("Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")
        
        df = pd.read_csv('data/' + files[0])
        
        for file in files[1:]:
            df_temp = pd.read_csv('data/' + file)
            df = pd.concat([df, df_temp], sort = False)
        
        df.reset_index(inplace = True, drop = True)
        
        df = df.rename(columns = dataframe_key.get_key(city))

        df['start_stat_lat'] = ''
        df['start_stat_long'] = ''
        df['end_stat_lat'] = ''
        df['end_stat_long'] = ''

        stat_df = pd.read_csv('data/Capital_Bike_Share_Locations.csv')

        for _ , stat in stat_df.iterrows():
            start_matches = np.where(df['start_stat_id'] == stat['TERMINAL_NUMBER'])
            end_matches = np.where(df['end_stat_id'] == stat['TERMINAL_NUMBER'])

            df.at[start_matches[0], 'start_stat_lat'] = stat['LATITUDE']
            df.at[start_matches[0], 'start_stat_long'] = stat['LONGITUDE']
            df.at[end_matches[0], 'end_stat_lat'] = stat['LATITUDE']
            df.at[end_matches[0], 'end_stat_long'] = stat['LONGITUDE']

        df.replace('', np.nan, inplace = True)
        df.dropna(inplace=True)

        max_lat = 38.961029
        min_lat = 38.792686
        max_long= -76.909415
        min_long= -77.139396

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

        df.reset_index(inplace = True, drop = True)

        df['start_dt'] = pd.to_datetime(df['start_t'])
        df['end_dt'] = pd.to_datetime(df['end_t'])
    
    elif city == "chic":
        
        files = [file for file in os.listdir('data') if f'Divvy_Trips_{year:d}' in file]
        files.sort()
        
        if len(files) < 4:
            raise FileNotFoundError("Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")
        
        df = pd.read_csv('data/' + files[0])
        
        for file in files[1:]:
            if file == 'Divvy_Trips_2019_Q2.csv':
                col_dict = {'01 - Rental Details Rental ID' : 'trip_id',
                            '01 - Rental Details Local Start Time' : 'start_time',
                            '01 - Rental Details Local End Time' : 'end_time',
                            '01 - Rental Details Bike ID' : 'bikeid',
                            '01 - Rental Details Duration In Seconds Uncapped' : 'tripduration',
                            '03 - Rental Start Station ID' : 'from_station_id',
                            '03 - Rental Start Station Name' : 'from_station_name',
                            '02 - Rental End Station ID' : 'to_station_id',
                            '02 - Rental End Station Name' : 'to_station_name',
                            'User Type' : 'usertype',
                            'Member Gender' : 'gender',
                            '05 - Member Details Member Birthday Year' : 'birthyear'}
                df_temp = pd.read_csv('data/' + file).rename(columns = col_dict)
                # df_temp = df_temp.rename(columns = col_dict)
            
            else: 
                df_temp = pd.read_csv('data/' + file)
            
            df = pd.concat([df, df_temp], sort = False)
        
        df = df.rename(columns = dataframe_key.get_key(city))

        df.reset_index(inplace = True, drop = True)

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

        df.replace('', np.nan, inplace = True)
        df.dropna(subset = ['start_stat_lat',
                            'start_stat_long',
                            'end_stat_lat',
                            'end_stat_long'], inplace = True)

        df.reset_index(inplace = True, drop = True)

        df['start_dt'] = pd.to_datetime(df['start_t'])
        df['end_dt'] = pd.to_datetime(df['end_t'])
        df['duration'] = df['duration'].str.replace(',', '').astype(float)
    
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
            raise FileNotFoundError("Data not found the whole year. Please check that all monthly data is present. All relevant files can be found at https://www.lyft.com/bikes/bay-wheels/system-data")
        
        df = pd.DataFrame(columns = col_list)
        
        for file in files:
            df_temp = pd.read_csv('data/' + file)[col_list]
            df = pd.concat([df, df_temp], sort = False)
        
        df = df.rename(columns = dataframe_key.get_key(city))
        df.dropna(inplace=True)
        
        df = df.iloc[np.where(df['start_stat_lat'] > 37.593220)]
        df = df.iloc[np.where(df['end_stat_lat'] > 37.593220)]
        
        df.sort_values(by = 'start_t', inplace = True)
        df.reset_index(inplace = True, drop = True)
        
        df['start_dt'] = pd.to_datetime(df['start_t'])
        df['end_dt'] = pd.to_datetime(df['end_t'])

    elif city == "london":

        data_files = [file for file in os.listdir('data') if 'JourneyDataExtract' in file]
        data_files = [file for file in data_files if f'{year:d}' in file]
        data_files.sort()

        if len(data_files) == 0:
            raise FileNotFoundError(f'No London data for {year:d} found. All relevant files can be found at https://cycling.data.tfl.gov.uk/.')

        if isinstance(data_files, str):
            warnings.warn('Only one data file found. Please check that you have all available data.')

        df = pd.read_csv('./data/' + data_files[0])

        for file in data_files[1:]:
            df_temp = pd.read_csv('./data/' + file)
            df = pd.concat([df, df_temp], sort = False)

        df.rename(columns = dataframe_key.get_key(city), inplace = True)

        df['start_t'] = pd.to_datetime(df['start_t'], format = '%d/%m/%Y %H:%M').astype(str)
        df['end_t'] = pd.to_datetime(df['end_t'], format = '%d/%m/%Y %H:%M').astype(str)
        
        df = df.iloc[np.where(df['start_t'] >= f'{year:d}-01-01 00:00:00')]
        df = df.iloc[np.where(df['start_t'] <= f'{year:d}-31-12-23:59:59')]

        df.sort_values(by = 'start_t', inplace = True)
        df.reset_index(inplace = True, drop = True)

        stat_df = pd.read_csv('./data/london_stations.csv')
        stat_df.at[np.where(stat_df['station_id'] == 502)[0][0], 'latitude'] = 51.53341

        df['start_stat_lat'] = ''
        df['start_stat_long'] = ''
        df['end_stat_lat'] = ''
        df['end_stat_long'] = ''

        for _ , stat in stat_df.iterrows():
            start_matches = np.where(df['start_stat_name'] == stat['station_name'])
            end_matches = np.where(df['end_stat_name'] == stat['station_name'])

            df.at[start_matches[0], 'start_stat_lat'] = stat['latitude']
            df.at[start_matches[0], 'start_stat_long'] = stat['longitude']
            df.at[end_matches[0], 'end_stat_lat'] = stat['latitude']
            df.at[end_matches[0], 'end_stat_long'] = stat['longitude']

        df.replace('', np.nan, inplace = True)
        df.dropna(inplace = True)

        df.reset_index(inplace = True, drop = True)

        df['start_dt'] = pd.to_datetime(df['start_t'])
        df['end_dt'] = pd.to_datetime(df['end_t'])

    if blacklist:
            df = df[~df['start_stat_id'].isin(blacklist)]
            df = df[~df['end_stat_id'].isin(blacklist)]

    days = days_index(df)
    
    return df, days
    
    
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

    # Create Dictionary Station : Position
    locations = dict()

    for e in id_index.keys():

        if df[df['start_stat_id'] == e]['start_stat_lat'].shape[0]:
            locations[id_index[e]] = (df[df['start_stat_id'] == e]['start_stat_long'].iloc[0],
                                      df[df['start_stat_id'] == e]['start_stat_lat'].iloc[0])
        else:
            locations[id_index[e]] = (df[df['end_stat_id'] == e]['end_stat_long'].iloc[0],
                                      df[df['end_stat_id'] == e]['end_stat_lat'].iloc[0])

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

    # Create Dictionary Station : Names
    names = dict()

    for e in id_index.keys():

        if df[df['start_stat_id'] == e]['start_stat_name'].shape[0]:
            names[id_index[e]] = df[df['start_stat_id'] == e]['start_stat_name'].iloc[0]

        else:
            names[id_index[e]] = df[df['end_stat_id'] == e]['end_stat_name'].iloc[0]

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
        print(f"Pickle does not exist. Pickling directed adjacency matrix: directedadjacency_{city}{year:d}{month:02d}{tuple(days)}thr_{threshold:d}.pickle...")
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
        deg_matrix[i,i] = np.sum(adj[[i],:])

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
            weight += adj[i,n]

        weightlist.append(weight)

    if initialisation == "rdm":
        v = np.random.rand(N, 1)
        v = v / np.linalg.norm(v, ord = 1)

    else: # Uniform initialisation
        v = np.linspace(1/N, 1/N, N)

    for i in range(iterations):

        for n in range(N):
            if weightlist[n] != 0:
                v[n] = v[n]/weightlist[n]

        v =  (1 - d)/(N+1) + d * adj @ v

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
        r = get(query, timeout = 60)
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

            r = get(query, timeout = 60)

            # Only get the json response in case of 200 or 201
            if r.status_code == 200 or r.status_code == 201:
                elevation = np.append(elevation,
                  np.array(pd.json_normalize(r.json(), 'results')['elevation'])
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
    
    cities = ['chic', 'london', 'madrid', 'mexico', 'nyc', 'sfran', 'taipei', 'washDC']

    if city in cities:
        name_dict = {'chic':'Chicago', 'london':'London', 'madrid':'Madrid',
                     'mexico':'Mexico City', 'nyc':'New York City',
                     'sfran':'San Francisco', 'taipei':'Taipei',
                     'washDC':'Washington DC'}
        city = name_dict[city]

    n_days = calendar.monthrange(year, month)[1]
    tp = 1
    query = (f"http://api.worldweatheronline.com/premium/v1/past-weather.ashx?"
            f"key=7886f8387f8c4c0484f83623210305&q={city}&format=json&date={year}-{month}-01&enddate={year}-{month}-{n_days}&tp={tp}")# Request with a timeout for slow responses
    r = get(query, timeout = 60)
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
        str(year) + str(month) + hweather['day'] + hweather['hour'],
        format='%Y%m%d%H')

    hweather['day'] = hweather['day'].astype(int)
    hweather['hour'] = hweather['hour'].astype(int)
    hweather['desc'] = pd.DataFrame(hweather['weatherDesc'].explode().tolist())

    rain = hweather[['day', 'hour', 'time_dt', 'precipMM', 'tempC', 'windspeedKmph', 'desc']]

    return request, rain


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
        T[m] = np.linalg.norm(u[:,m] - (W_tilde @ u[:,m]), ord = 1)
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
        df_filtered = df[df['start_stat_id'].map(id_index).isin(filtered_positions)]
        df_done = df_filtered[df_filtered['end_stat_id'].map(id_index).isin(filtered_positions)]

    else:
        df_filtered = df[~df['start_stat_id'].map(id_index).isin(filtered_positions)]
        df_done = df_filtered[~df_filtered['end_stat_id'].map(id_index).isin(filtered_positions)]

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
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
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

        self.n_start = len(df['start_stat_id'].unique())
        self.n_end = len(df['end_stat_id'].unique())
        total_station_id = set(df['start_stat_id']).union(set(df['end_stat_id']))
        self.n_tot = len(total_station_id)

        self.id_index = dict(zip(sorted(total_station_id), np.arange(self.n_tot)))

        self.inverse = dict(zip(np.arange(self.n_tot),sorted(total_station_id)))

        self.locations = station_locations(df, self.id_index)
        self.loc = np.array(list(self.locations.values()))
        trans = Transformer.from_crs("EPSG:4326", "EPSG:3857")
        self.loc_merc = np.vstack(trans.transform(self.loc[:,1], self.loc[:,0])).T

        self.names = station_names(df, self.id_index)

        print("Stations loaded")


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

    def __init__(self, city, year, month = None, blacklist = None):
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
    
            self.weekdays = [(i+(first_weekday))%7 for i in range(self.num_days)]
    
            self.df, self.d_index = get_data_month(city, year, month, blacklist)
        
        else:
            self.df, self.d_index = get_data_year(city, year, blacklist)
        
        self.stat = Stations(self.df)

        self.df['start_stat_index'] = self.df['start_stat_id'].map(self.stat.id_index)
        self.df['end_stat_index'] = self.df['end_stat_id'].map(self.stat.id_index)


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
            print(f"Pickle does not exist. Pickling adjacency matrix: adjacency_{self.city}{self.year:d}{self.month:02d}{tuple(days)}thr_{threshold:d}_rsl_{remove_self_loops}.pickle...")
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


    def adjacency_hour(self, day, hour, threshold=1, remove_self_loops = True):
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


    def get_degree_matrix(self, days, threshold = 1, remove_self_loops = True):
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

        degrees = np.sum(adj_matrix, axis = 0)

        deg_matrix = np.diag(degrees)

        return deg_matrix

    def get_laplacian(self, days, threshold = 1, remove_self_loops = True):
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

        degrees = np.sum(deg_matrix, axis = 0)

        if normalise:
            degrees = degrees/len(days)

        busy_stations = []
        for i in range(len(degrees)):
            busy_station = self.stat.names[i]
            busy_stations.append(busy_station)

        temp = list(zip(busy_stations, degrees))
        
        if sort:
            temp_sorted = sorted(temp, key = lambda x: x[1], reverse = True)
            return temp_sorted
        else:
            return temp


    def get_busy_trips(self, days, directed = False, normalise = True):
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

            max_indices = max_indices + [(where[0][i], where[1][i]) for i in range(len(where[0]))]

            weights = weights + [w_max for _ in range(len(where[0]))]

            mask[where] = False

        busy_trips = list(zip(max_indices, weights))

        return busy_trips


    def compare_degrees(self, days_ref, days_change, savefig = False):
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

        figsize_dict = {'nyc': (5,8),
                        'chic': (5,8),
                        'london': (8,5),
                        'oslo': (6.6,5),
                        'sfran': (6,5),
                        'washDC': (8,7.7),
                        'madrid': (6,8),
                        'mexico': (7.2,8),
                        'taipei': (7.3,8)}

        adj_ref = self.adjacency(
            days_ref, threshold=0, remove_self_loops=False)

        ref_degrees = np.sum(adj_ref, axis = 0) + 1 # +1 to work around division by zero

        adj_change = self.adjacency(
            days_change, threshold=0, remove_self_loops=False)

        deg_change = np.sum(adj_change, axis = 0)

        deg_compare = deg_change / ref_degrees

        # Plot graph
        graph = nx.from_numpy_matrix(adj_ref)

        try:
            fig = plt.subplots(figsize=figsize_dict[self.city])
        except KeyError:
            fig = plt.subplots(figsize=(8,8))

        plt.grid(False)
        plt.axis(False)

        nx.draw_networkx_nodes(
            graph, self.stat.loc_merc, node_size=20, node_color=deg_compare,
            cmap='jet', vmin = 0, vmax = 1.5)

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
    
    def daily_traffic(self, stat_index, day, plot = False):
    
        df_stat_start = self.df.iloc[np.where(self.df['start_stat_index'] == stat_index)]
        df_stat_end = self.df.iloc[np.where(self.df['end_stat_index'] == stat_index)]
        
        trips_arrivals = np.zeros(24)
        trips_departures = np.zeros(24)
        
        for hour in range(24):
                df_hour_start = df_stat_start.iloc[np.where(df_stat_start['start_t'] > f'{self.year:d}-{self.month:02d}-{day:02d} {hour:02d}:00:00')]
                df_hour_start = df_hour_start.iloc[np.where(df_hour_start['start_t'] < f'{self.year:d}-{self.month:02d}-{day:02d} {hour:02d}:59:59')]
                
                trips_departures[hour] = len(df_hour_start)
                
                df_hour_end = df_stat_end.iloc[np.where(df_stat_end['end_t'] > f'{self.year:d}-{self.month:02d}-{day:02d} {hour:02d}:00:00')]
                df_hour_end = df_hour_end.iloc[np.where(df_hour_end['end_t'] < f'{self.year:d}-{self.month:02d}-{day:02d} {hour:02d}:59:59')]
                
                trips_arrivals[hour] = len(df_hour_end)
        
        if plot:
            plt.plot(np.arange(24), trips_arrivals)
            plt.plot(np.arange(24), trips_departures)
            plt.xticks(np.arange(24))
            plt.legend(['Arrivals','Departures'])
            plt.xlabel('Hour')
            plt.ylabel('# trips')
            plt.title(f'Hourly traffic for {self.stat.names[stat_index]} \n on {self.year:d}-{self.month:02d}-{day:02d}')
        
        return trips_departures, trips_arrivals
    
    def daily_traffic_average(self, stat_index, period = 'b', plot = False):
        
        weekdays = [calendar.weekday(self.year,self.month,i) for i in range(1,calendar.monthrange(self.year,self.month)[1]+1)]
        
        if period == 'b': 
            days = [date+1 for date, day in enumerate(weekdays) if day <= 4]
        elif period == 'w':
            days = [date+1 for date, day in enumerate(weekdays) if day > 4]
        else:
            raise ValueError("Please provide the period as either 'b' = business days or 'w' = weekends")
        
        df_stat_start = self.df.iloc[np.where(self.df['start_stat_index'] == stat_index)]
        df_stat_end = self.df.iloc[np.where(self.df['end_stat_index'] == stat_index)]
        
        trips_arrivals = np.zeros(shape=(len(days), 24))
        trips_departures = np.zeros(shape=(len(days), 24))
        
        for i, day in enumerate(days):
            
            df_stat_start_day = df_stat_start.iloc[np.where(df_stat_start.start_dt.dt.day == day)]
            df_stat_end_day = df_stat_end.iloc[np.where(df_stat_end.start_dt.dt.day == day)]
            
            trips_arrivals_daily = np.zeros(24)
            trips_departures_daily = np.zeros(24)
            
            for hour in range(24):
                df_hour_start = df_stat_start_day.iloc[np.where(df_stat_start_day['start_t'] > f'{self.year:d}-{self.month:02d}-{day:02d} {hour:02d}:00:00')]
                df_hour_start = df_hour_start.iloc[np.where(df_hour_start['start_t'] < f'{self.year:d}-{self.month:02d}-{day:02d} {hour:02d}:59:59')]
                
                trips_departures_daily[hour] = len(df_hour_start)
                
                df_hour_end = df_stat_end_day.iloc[np.where(df_stat_end_day['end_t'] > f'{self.year:d}-{self.month:02d}-{day:02d} {hour:02d}:00:00')]
                df_hour_end = df_hour_end.iloc[np.where(df_hour_end['end_t'] < f'{self.year:d}-{self.month:02d}-{day:02d} {hour:02d}:59:59')]
                
                trips_arrivals_daily[hour] = len(df_hour_end)
        
            trips_departures[i,:] = trips_departures_daily
            trips_arrivals[i,:] = trips_arrivals_daily
        
        trips_departures_average = np.mean(trips_departures, axis=0)
        trips_arrivals_average = np.mean(trips_arrivals, axis=0)
        
        if plot:
            
            trips_departures_std = np.std(trips_departures, axis=0)
            trips_arrivals_std = np.std(trips_arrivals, axis=0)
            
            plt.plot(np.arange(24), trips_arrivals_average)
            plt.plot(np.arange(24), trips_departures_average)
            plt.xticks(np.arange(24))
            plt.fill_between(np.arange(24), trips_arrivals_average-trips_arrivals_std, 
                             trips_arrivals_average+trips_arrivals_std, 
                             facecolor='b',alpha=0.2)
            plt.fill_between(np.arange(24), trips_departures_average-trips_departures_std, 
                             trips_departures_average+trips_departures_std, 
                             facecolor='orange',alpha=0.2)
            
            
            plt.legend(['Arrivals','Departures','$\pm$std - arrivals','$\pm$std - departures'])
            plt.xlabel('Hour')
            plt.ylabel('# trips')
            
            month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
                  7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
            
            if period == 'b':
                plt.title(f'Average hourly traffic for {self.stat.names[stat_index]} \n in {month_dict[self.month]} {self.year} on business days')
            
            elif period == 'w':
                plt.title(f'Average hourly traffic for {self.stat.names[stat_index]} \n in {month_dict[self.month]} {self.year} on weekends')
        
            plt.show()
            
        return trips_departures_average, trips_arrivals_average
        
if __name__ == "__main__":
    pre = time.time()
    data = Data('nyc', 2019, 9)
    print(time.time() - pre)
    
    pre = time.time()
    dep, arr = data.daily_traffic_average(407, 'b', plot=True)
    print(time.time() - pre)
    
    
    
    
    
    
    
    
    
    
    
    
    
    