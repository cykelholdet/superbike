# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:22:19 2021

@author: nweinr
"""
import pickle

#import fiona
import numpy as np
import pandas as pd
import geopandas as gpd
import holoviews as hv

import shapely.ops

from shapely.geometry import Point
# from shapely.ops import nearest_points
from geopy.distance import great_circle
import matplotlib.colors as mpl_colors

import bikeshare as bs
import dataframe_key

def df_key(city):
    
    if city == 'nyc':
        
        key = {'index' : 'stat_id',
               'ZONEDIST' : 'zone_dist',
               'BoroCT2020' : 'census_tract',
               'Shape__Area' : 'CT_area',
               'Pop_20' : 'population'}
    elif city == 'madrid':
        key = {}
    
    elif city == 'chic':
        key = {'index' : 'stat_id',
               'zone_class' : 'zone_dist',
               'geoid10' : 'census_block',
               'TOTAL POPULATION' : 'population'}
    
    elif city == 'washDC':
        key = {'ZONING LABEL' : 'zone_dist',
               'GEOID' : 'census_tract',
               'P0010001' : 'population'}
    
    elif city == 'minn':
        key = {'ZONE_CODE' : 'zone_dist',
               'GEOID20' : 'census_tract',
               'ALAND20' : 'CT_area'}
    
    else:
        key = {}
    
    return key
    
    
def zone_dist_transform(city, zone_dist):
    
    if pd.notnull(zone_dist):
        
        if city == 'nyc':
            
            if 'PARK' in zone_dist or 'PLAYGROUND' in zone_dist:
                zone_type = 'recreational'
            elif 'R' in zone_dist and '/' not in zone_dist:
                zone_type = 'residential'
            elif 'C' in zone_dist and '/' not in zone_dist:
                zone_type = 'commercial'
            elif 'M' in zone_dist and '/' not in zone_dist:
                zone_type = 'manufacturing'
            elif '/' in zone_dist:
                zone_type = 'mixed'
            else:
                zone_type = 'UNKNOWN'
        
        elif city in ['madrid', 'helsinki']:
            
            if zone_dist in ['11100', '11210', '11220', '11230']: # Continuous urban fabric (S.L. : > 80%), Discontinuous dense urban fabric (S.L. : 50% -  80%), Discontinuous medium density urban fabric (S.L. : 30% - 50%), Discontinuous low density urban fabric (S.L. : 10% - 30%)
                zone_type = 'residential' 
            elif zone_dist in ['12220', '12210']: # Other roads and associated land, Fast transit roads and associated land
                zone_type = 'road'
            elif zone_dist in ['12100']: # Industrial, commercial, public, military and private units
                zone_type = 'commercial'
            elif zone_dist in ['14100', '14200', '31000', '32000']: # Green urban areas, Sports and leisure facilities, Forests, Herbaceous vegetation associations (natural grassland, moors...)
                zone_type = 'recreational'
            elif zone_dist in ['12230']: # Railways and associated land
                zone_type = 'transportation'
            elif zone_dist in ['12300']: # Port areas
                zone_type = 'port'
            elif zone_dist in ['13100', 'Construction sites']: # Mineral extraction and dump sites
                zone_type = 'manufacturing'
            elif zone_dist in ['50000']:
                zone_type = 'water'
            else:
                zone_type = 'UNKNOWN'
        
        elif city == 'chic':
            
            com_zones = ['B1-1', 'B1-2', 'B1-3', 'B1-5', 'B2-1', 'B2-2', 'B3-1',
                         'B3-2', 'B3-3', 'B3-5', 'C1-1', 'C1-2', 'C1-3', 'C1-5',
                         'C2-2','C2-3', 'C3-2', 'C3-3', 'C3-5', 'DC-16', 'DS-3',
                         'DS-5']
            
            res_zones = ['DR-10', 'DR-3', 'RM-5', 'RM-6', 'RM-6.5', 'RS-1', 
                         'RS-2', 'RS-3', 'RT-3.5', 'RT-4']
            
            man_zones = ['M1-1', 'M1-2', 'M1-3', 'M2-2', 'M2-3', 'PMD 11', 'PMD 2',
                         'PMD 3', 'PMD 4', 'PMD 7', 'PMD 8', 'PMD 9']
            
            rec_zones = ['POS-1', 'POS-2']
        
            if zone_dist in com_zones: # Business and commercial zones and downtown core and services
                zone_type = 'commercial'
            elif zone_dist in res_zones: # Residential and downtown residential
                zone_type = 'residential'
            elif zone_dist in man_zones: # Manufacturing and planned manufactoring development
                zone_type = 'manufacturing'
            elif zone_dist == 'T': # Transportation
                zone_type = 'transportation'
            elif zone_dist in rec_zones: # Green areas
                zone_type = 'recreational'
            elif 'DX' in zone_dist or 'PD' in zone_dist: # Mixed downtown and planned development
                zone_type = 'mixed'
            else:
                zone_type = 'UNKNOWN'
        
        elif city == 'washDC':
            
            res_zones = ['RF-4', 'RF-3', 'RF-2', 'RF-1', 'RC-3', 'RC-2', 'RC-1',
                         'RA-1', 'RA-2', 'RA-3', 'RA-4', 'RA-5', 'RA-6', 'RA-7',
                         'RA-8', 'RA-9', 'RA-10', 'R-9', 'R-8', 'R-6', 'R-3',
                         'R-21', 'R-20', 'R-2', 'R-19', 'R-17', 'R-16', 'R-15', 
                         'R-14', 'R-13', 'R-12', 'R-11', 'R-10', 'R-1-B', 
                         'R-1-A', 'ARTS-2', 'CG-1', 'CG-2', 'D-1-R', 'D-4-R',
                         'MU-15', 'MU-16', 'MU-18', 'MU-19', 'MU-23', 'MU-5A',
                         'MU-5B', 'MU-6', 'NC-10', 'NC-11', 'NC-13', 'NC-5',
                         'NC-9']
            
            com_zones = ['ARTS-3', 'CG-3', 'D-3', 'D-4', 'D-5', 'D-6-R', 'D-7',
                         'MU-20', 'MU-21', 'MU-28', 'MU-8', 'M-9', 'NC-16', 
                         'NC-17', 'NC-8']
            
            mix_zones = ['ARTS-1', 'ARTS-4', 'CG-5', 'D-5-R', 'D-6', 'MU-1',
                         'MU-10', 'MU-12', 'MU-13', 'MU-14', 'MU-17', 'MU-2',
                         'MU-22', 'MU-24', 'MU-25', 'MU-26', 'MU-27', 'MU-29',
                         'MU-3A', 'MU-3B', 'MU-4', 'MU-7', 'NC-1', 'NC-2',
                         'NC-3', 'NC-4', 'NC-6', 'NC-7', 'NHR', 'SEFC-1A',
                         'SEFC-1B', 'SEFC-2', 'SEFC-3', 'SEFC-4']
            
            rec_zones = ['MU-11', 'NC-14', 'NC-15']
            
            if zone_dist in res_zones:
                zone_type = 'residential'
            elif zone_dist in com_zones:
                zone_type = 'commercial'
            elif zone_dist in rec_zones:
                zone_type = 'recreational'
            elif 'PDR' in zone_dist:
                zone_type = 'manufacturing'
            elif zone_dist in mix_zones or 'WR' in zone_dist:
                zone_type = 'mixed'
            elif 'StE' in zone_dist:
                zone_type = 'educational'
            
            else:
                zone_type = 'UNKNOWN'
        
        elif city == 'minn':
            
            if 'OR' in zone_dist:
                zone_type = 'mixed'
            elif 'R' in zone_dist:
                zone_type = 'residential'
            elif 'C' in zone_dist or 'B' in zone_dist:
                zone_type = 'commercial'
            elif 'I' in zone_dist:
                zone_type = 'industrial'
            
            else:
                zone_type = 'UNKNOWN'
            
            
        else:
            raise KeyError('city transform not found')
        
    else:
        zone_type = 'UNKNOWN'
    
    return zone_type
    

def make_station_df(data, holidays=True, return_land_use=False, overwrite=False):
    postfix = "" if data.month == None else f"{data.month:02d}"
    postfix = postfix + "" if holidays else postfix + "_no_holidays"
    
    if not overwrite:
        try:
            with open(f'./python_variables/station_df_{data.city}{data.year:d}{postfix}.pickle', 'rb') as file:
                df, land_use = pickle.load(file)
            if return_land_use:
                return df, land_use
            else:
                return df
        except FileNotFoundError:
            print("Pickle does not exist. Pickling station_df...")
    
    df = pd.DataFrame(data.stat.locations).T.rename(columns={0: 'long', 1: 'lat'})
    
    df['stat_id'] = df.index.map(data.stat.inverse)
    
    df['easting'], df['northing'] = hv.util.transform.lon_lat_to_easting_northing(df['long'], df['lat'])
    
    df['name'] = data.stat.names.values()
    df['n_arrivals'] = df['stat_id'].map(data.df['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    df['n_departures'] = df['stat_id'].map(data.df['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)

    df['n_trips'] = df['n_arrivals'] + df['n_departures']
    
    
    
    df_s = data.df[['start_stat_id', 'start_dt']][data.df['start_dt'].dt.weekday <= 4]
    if not holidays:
        holiday_year = pd.DataFrame(
            bs.get_cal(data.city).get_calendar_holidays(data.year), columns=['day', 'name'])
        holiday_list = holiday_year['day'].tolist()
        df_s = df_s[~df_s['start_dt'].dt.date.isin(holiday_list)] # Rows which are not in holiday list
    df['b_departures'] = df['stat_id'].map(df_s['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    
    df_s = data.df[['start_stat_id', 'start_dt']][data.df['start_dt'].dt.weekday > 4]
    df['w_departures'] = df['stat_id'].map(df_s['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)

    df_s = data.df[['end_stat_id', 'end_dt']][data.df['end_dt'].dt.weekday <= 4]
    if not holidays:
        df_s = df_s[~df_s['end_dt'].dt.date.isin(holiday_list)] # Rows which are not in holiday list
    df['b_arrivals'] = df['stat_id'].map(df_s['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    
    df_s = data.df[['end_stat_id', 'end_dt']][data.df['end_dt'].dt.weekday > 4]
    df['w_arrivals'] = df['stat_id'].map(df_s['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    
    df['b_trips'] = df['b_arrivals'] + df['b_departures']
    df['w_trips'] = df['w_arrivals'] + df['w_departures']

    
    df['coords'] = list(zip(df['long'], df['lat']))
    df['coords'] = df['coords'].apply(Point)
    
    df['label'] = np.nan
    df['color'] = "gray"
    
    extent = {'lat': [df['lat'].min(), df['lat'].max()], 
              'long': [df['long'].min(), df['long'].max()]}
    
    if data.city == 'nyc':
    
        zoning_df = gpd.read_file('./data/other_data/nyc_zoning_data.json')
        zoning_df = zoning_df[['ZONEDIST', 'geometry']]
        
        land_use = zoning_df[['ZONEDIST', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        df['zone_type'] = df['ZONEDIST'].apply(lambda x: zone_dist_transform(data.city, x))
        
        
        CTracts_df = gpd.read_file('./data/other_data/nyc_CT_data.json')
        
        
        CTracts_df = CTracts_df[['BoroCT2020', 'geometry', 'Shape__Area']]
        # CTracts_df.rename({'Shape__Area':'CT_area'}, axis=1, inplace=True)
        
        df = gpd.tools.sjoin(df, CTracts_df, op='within', how='left')
        df['BoroCT2020'] = df['BoroCT2020'].apply(int)
        df['Shape__Area'] = df['Shape__Area']/10.764 # convert to m^2
        df.drop('index_right', axis=1, inplace=True)
        
        census_df = pd.read_excel('./data/other_data/nyc_census_data.xlsx', sheet_name=1, skiprows=[0,1,2])
        census_df = census_df[['BCT2020', 'Pop_20']].dropna()
        
        pop_map = dict(zip(census_df['BCT2020'], census_df['Pop_20']))
        
        df['population'] = df['BoroCT2020'].map(pop_map)
        df['pop_density'] = df['population'] / df['Shape__Area']
        
        subways_df = gpd.read_file('./data/other_data/nyc_subways_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    elif data.city in ['madrid', 'helsinki']:
        
        land_use_df = gpd.read_file(f'data/other_data/{data.city}_UA2018_v013.gpkg')
        land_use_df = land_use_df[['code_2018', 'class_2018', 'area', 'Pop2018', 'geometry']].to_crs('EPSG:4326')
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs='EPSG:4326')
        df = gpd.tools.sjoin(df, land_use_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        df['zone_type'] = df['code_2018'].apply(lambda x: zone_dist_transform(data.city, x))

        land_use = land_use_df[['code_2018', 'geometry']]
        land_use = land_use.cx[extent['long'][0]:extent['long'][1], extent['lat'][0]:extent['lat'][1]]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
        land_use = land_use[land_use['zone_type'] != 'road']
        land_use = land_use[land_use['zone_type'] != 'water']
    
    elif data.city == 'chic':
        
        zoning_df = gpd.read_file('./data/other_data/chic_zoning_data.geojson')
        zoning_df = zoning_df[['zone_class', 'geometry']]
        
        land_use = zoning_df[['zone_class', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
    
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['zone_type'] = df['zone_class'].apply(lambda x: zone_dist_transform(data.city, x))
        
        CBlocks_df = gpd.read_file('./data/other_data/chic_CB_data.geojson')
        CBlocks_df = CBlocks_df[['geoid10', 'geometry']]
        
        CBlocks_df_cart = CBlocks_df.copy()
        CBlocks_df_cart = CBlocks_df_cart.to_crs({'proj': 'cea'})
        CBlocks_df['CB_area'] = CBlocks_df_cart['geometry'].area
        
        df = gpd.tools.sjoin(df, CBlocks_df, op='within', how='left')
        df['geoid10'] = df['geoid10'].apply(lambda x: int(x) if pd.notnull(x) else x)
        df.drop('index_right', axis=1, inplace=True)
        
        census_df = pd.read_csv('./data/other_data/chic_census_data.csv')
        census_df = census_df[['CENSUS BLOCK FULL', 'TOTAL POPULATION']]
        
        pop_map = dict(zip(census_df['CENSUS BLOCK FULL'], census_df['TOTAL POPULATION']))
        
        df['population'] = df['geoid10'].map(pop_map)
        df['pop_density'] = df['population'] / df['CB_area']
        
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
        subways_df = gpd.read_file('./data/other_data/chic_subways_data.kml', driver='KML')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    elif data.city == 'washDC':
        
        zoning_df = gpd.read_file('./data/other_data/washDC_zoning_data.geojson')
        zoning_df = zoning_df[['ZONING_LABEL', 'geometry']]
        
        land_use = zoning_df[['ZONING_LABEL', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
            
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['zone_type'] = df['ZONING_LABEL'].apply(lambda x: zone_dist_transform(data.city, x))
        
        census_df = gpd.read_file('./data/other_data/washDC_census_data.geojson')
        census_df = census_df[['GEOID', 'P0010001', 'geometry']]
        
        census_df_cart = census_df.copy()
        census_df_cart = census_df_cart.to_crs({'proj': 'cea'})
        census_df['CT_area'] = census_df_cart['geometry'].area
        
        census_df = census_df[['GEOID', 'CT_area', 'P0010001', 'geometry']]
        
        df = gpd.tools.sjoin(df, census_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['pop_density'] = df['P0010001'] / df['CT_area']
    
        subways_df = gpd.read_file('./data/other_data/washDC_subways_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    
    elif data.city == 'minn':
        
        zoning_df = gpd.read_file('./data/other_data/minn_zoning_data.geojson')
        zoning_df = zoning_df[['ZONE_CODE', 'geometry']]
        
        land_use = zoning_df[['ZONE_CODE', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['zone_type'] = df['ZONE_CODE'].apply(lambda x: zone_dist_transform(data.city, x))
    
        CTracts_df = gpd.read_file('./data/other_data/minn_CT_data.shp')
        CTracts_df.to_crs(crs = zoning_df.crs, inplace=True)
        CTracts_df = CTracts_df[['GEOID20', 'ALAND20', 'geometry']]
        
        df = gpd.tools.sjoin(df, CTracts_df, op='within', how='left')
        df['GEOID20'] = df['GEOID20'].apply(lambda x: int(x) if pd.notnull(x) else x)
        df.drop('index_right', axis=1, inplace=True)
        
        census_df = pd.read_excel('./data/other_data/minn_census_data.xlsx')
        census_df = census_df[['GEOID2', 'POPTOTAL']]
        
        pop_map = dict(zip(census_df['GEOID2'], census_df['POPTOTAL']))
        
        df['population'] = df['GEOID20'].map(pop_map)
        df['pop_density'] = df['population'] / df['ALAND20']
        
        subways_df = gpd.read_file('./data/other_data/minn_subways_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    
    elif data.city == 'boston':
        
        zoning_df = gpd.read_file('./data/other_data/boston_zoning_data.geojson')
        zoning_df = zoning_df[['ZONE_', 'SUBDISTRIC', 'geometry']]
        
        land_use = zoning_df[['ZONE_',  'SUBDISTRIC', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        #land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        # df['zone_type'] = df['ZONE_CODE'].apply(lambda x: zone_dist_transform(data.city, x))
    
        CTracts_df = gpd.read_file('./data/other_data/boston_CT_data.shp')
        CTracts_df = CTracts_df.to_crs(epsg=4326)
        CTracts_df = CTracts_df[['GEOID20', 'ALAND20', 'geometry']]
        
        df = gpd.tools.sjoin(df, CTracts_df, op='within', how='left')
        df['GEOID20'] = df['GEOID20'].apply(lambda x: int(x) if pd.notnull(x) else x)
        df.drop('index_right', axis=1, inplace=True)
        
        census_df = pd.read_csv('./data/other_data/boston_census_data.csv')
        census_df = census_df[['GEOCODE', 'P0020001']].iloc[1:]
        census_df['GEOCODE'] = census_df['GEOCODE'].apply(lambda x: int(x) if pd.notnull(x) else x)
        
        pop_map = dict(zip(census_df['GEOCODE'], census_df['P0020001']))
        
        df['population'] = df['GEOID20'].map(pop_map).apply(lambda x: int(x) if pd.notnull(x) else x)
        df['pop_density'] = df['population'] / df['ALAND20']
        
        
        subways_df = gpd.read_file('./data/other_data/boston_subways_data.shp').to_crs(epsg = 4326)
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    else:
        
        df['population'] = 0
        df['pop_density'] = 0
        df['zone_type'] = 0
        land_use = pd.DataFrame([])
        land_use['zone_type'] = 'UNKNOWN'
        
    df.rename(mapper=df_key(data.city), axis=1, inplace=True)  
    
    land_use['color'] = land_use['zone_type'].map(color_dict).fillna("pink")
    
    with open(f'./python_variables/station_df_{data.city}{data.year:d}{postfix}.pickle', 'wb') as file:
        pickle.dump([df, land_use], file)
    
    if return_land_use:
        return df, land_use
    else:
        return df


color_dict = {
    'residential': mpl_colors.to_hex('tab:purple'), # 4
    'commercial': mpl_colors.to_hex('tab:orange'),  # 1
    'recreational': mpl_colors.to_hex('tab:green'),  # 2
    'manufacturing': mpl_colors.to_hex('tab:red'), # 3
    'mixed': mpl_colors.to_hex('tab:blue'), # 0
    'educational': mpl_colors.to_hex('tab:brown'), # 5
    'UNKNOWN': mpl_colors.to_hex('gray'), # 7
    'road': mpl_colors.to_hex('tab:pink'),
    'port': mpl_colors.to_hex('tab:olive'),
    'transportation': mpl_colors.to_hex('tab:olive'),
    }

color_num_dict = {
    'residential': 4, # 4
    'commercial': 1,  # 1
    'recreational': 2,  # 2
    'manufacturing': 3, # 3
    'mixed': 0, # 0
    'educational': 5, # 5
    'UNKNOWN': 7 # 7
    }

    
if __name__ == "__main__":
    import bikeshare as bs
    import time
    
    data = bs.Data('helsinki', 2019, 9)
    pre = time.time()
    station_df, land_use = make_station_df(data, return_land_use=True)
    print(f'station_df took {time.time() - pre:.2f} seconds')
