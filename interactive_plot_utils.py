# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:22:19 2021

@author: nweinr
"""
import pickle
import time
import os
import contextlib
from functools import partial

#import fiona
import numpy as np
import pandas as pd
import geopandas as gpd
import holoviews as hv
import pyproj

import shapely.ops

from shapely.geometry import Point, Polygon
# from shapely.ops import nearest_points
from geopy.distance import great_circle
import matplotlib.colors as mpl_colors
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import skimage.color as skcolor

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
               'GEO_ID' : 'census_tract',
               'B01001_001E' : 'population'}
    
    elif city == 'minn':
        key = {'ZONE_CODE' : 'zone_dist',
               'GEOID20' : 'census_tract',
               'ALAND20' : 'CT_area'}
    
    elif city == 'boston':
        key = {'ZONE_' : 'zone_dist',
               'GEO_ID' : 'census_tract',
               'B01001_001E' : 'population'}
    
    else:
        key = {}
    
    return key
    
    
def zone_dist_transform(city, zone_dist):
    
    # TODO: Change manufacturing to industrial?
    
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
        
        elif city in ['madrid', 'helsinki', 'london', 'oslo']:
            
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
                         'NC-9', 'R-10', 'R-10T', 'R-20', 'R-5', 'R-6', 'R-8',
                         'R15-30T', 'R2-7', 'RA-H', 'RA-H-3.2', 'RA14-26', 
                         'RA4.8', 'RA6-15', 'RA7-15', 'RA7-16', 'RA8-18',
                         'R12', 'R2-5', 'R20', 'R5', 'R8', 'RA', 'RB', 'RC',
                         'RCX', 'RD', 'RM', 'RT']
            
            com_zones = ['ARTS-3', 'CG-3', 'D-3', 'D-4', 'D-5', 'D-6-R', 'D-7',
                         'MU-20', 'MU-21', 'MU-28', 'MU-8', 'M-9', 'NC-16', 
                         'NC-17', 'NC-8', 'C-1', 'C-1-0', 'C-1-R', 'C-2', 'C-3',
                         'C-R', 'C-TH', 'CC', 'CD', 'CDX', 'CG', 'CL', 'CSL',
                         'OC', 'OCH', 'OCM(100)', 'OCM(50)']
            
            mix_zones = ['ARTS-1', 'ARTS-4', 'CG-5', 'D-5-R', 'D-6', 'MU-1',
                         'MU-10', 'MU-12', 'MU-13', 'MU-14', 'MU-17', 'MU-2',
                         'MU-22', 'MU-24', 'MU-25', 'MU-26', 'MU-27', 'MU-29',
                         'MU-3A', 'MU-3B', 'MU-4', 'MU-7', 'NC-1', 'NC-2',
                         'NC-3', 'NC-4', 'NC-6', 'NC-7', 'NHR', 'SEFC-1A',
                         'SEFC-1B', 'SEFC-2', 'SEFC-3', 'SEFC-4', 'C-0', 
                         'C-0-1.0', 'C-0-1.5', 'C-0-2.5', 'C-0-A', 'C-0-CC',
                         'C-0-ROSS', 'CP-FBC', 'MU-VS', 'RC', 'CRMU/H', 
                         'CRMU/L', 'CRMU/M', 'CRMU/X', 'KR', 'NR', 'W-1']
            
            rec_zones = ['MU-11', 'NC-14', 'NC-15', 'POS', 'WPR', 'UNZONED']
            
            man_zones = ['CM', 'M-1', 'M-2', 'I']
            
            
            if zone_dist in res_zones:
                zone_type = 'residential'
            elif zone_dist in com_zones:
                zone_type = 'commercial'
            elif zone_dist in rec_zones:
                zone_type = 'recreational'
            elif zone_dist in man_zones or 'PDR' in zone_dist:
                zone_type = 'manufacturing'
            elif zone_dist in mix_zones or 'WR' in zone_dist or 'CDD' in zone_dist:
                zone_type = 'mixed'
            elif 'StE' in zone_dist:
                zone_type = 'educational'
            elif zone_dist == 'UT':
                zone_type = 'transportation'
            
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
                zone_type = 'manufacturing'
            
            else:
                zone_type = 'UNKNOWN'
        
        elif city == 'boston':
            
            res_zones = ['A-1', 'A-2', 'B', 'C', 'C-1', 'C-1A', 'C-2', 'C-2A',
                         'C-2B', 'C-3', 'C-3A', 'C-3B', 'SD-10(H)', 'SD-12',
                         'SD-13', 'SD-14', 'SD-2', 'SD-4A', 'SD-6', 'SD-9']
        
            com_zones = ['BA', 'BA-1', 'BA-2', 'BA-3', 'BA-4', 'BB', 'BB-1',
                         'BB-2', 'O-1', 'O-2', 'O-2A', 'O-3', 'O-3A', 'SD-1',
                         'SD-10(F)', 'SD-11', 'SD-4', 'SD-5', 'SD-7']
            
            mix_zones = ['ASD', 'CRDD', 'MXD', 'NP', 'SD-3', 'SD-8', 'SD-8A']
        
            rec_zones = ['OS']
            
            man_zones = ['IA', 'IA-1', 'IA-2', 'IB', 'IB-1', 'IB-2', 'SD-15']
            
            if zone_dist in res_zones:
                zone_type = 'residential'
            
            elif zone_dist in com_zones:
                zone_type = 'commercial'
            
            elif zone_dist in mix_zones:
                zone_type = 'mixed'
            
            elif zone_dist in rec_zones:
                zone_type = 'recreational'
            
            elif zone_dist in man_zones:
                zone_type = 'manufacturing'
                
            
            # TODO: Make boston zone_types mre precise using Zone_Desc
            
            elif zone_dist == 'Residential':
                zone_type = 'residential'
            elif zone_dist == 'Open Space':
                zone_type = 'recreational'
            elif zone_dist == 'Business':
                zone_type = 'commercial'
            elif zone_dist == 'Mixed use':
                zone_type = 'mixed'
            elif zone_dist == 'Industrial':
                zone_type = 'manufacturing'
            elif zone_dist == 'Comm/Instit':
                zone_type = 'educational'
            
            else:
                zone_type = 'UNKNOWN'
            
            
        else:
            raise KeyError('city transform not found')
        
    else:
        zone_type = 'UNKNOWN'
    
    return zone_type
    

def make_neighborhoods(city, year, station_df, land_use):
    
    pre = time.time()
    print(f"Loading data: {city}{year}")
    data_year = bs.Data(city, year)
    neighborhoods = gpd.GeoDataFrame(index = list(data_year.stat.inverse.keys()))
    neighborhoods['stat_id'] = list(data_year.stat.id_index.keys())
    neighborhoods['coords'] = [Point(data_year.stat.locations[i][0],
                                     data_year.stat.locations[i][1]) 
                               for i in neighborhoods.index]
    neighborhoods.set_geometry('coords', inplace=True)
    # neighborhoods.set_crs(epsg=4326, inplace=True)
    # neighborhoods.to_crs(epsg=3857, inplace=True)        
    print("Getting the hood ready to go", end="")
    if len(land_use) > 0:
        # land_use.to_crs(epsg=3857, inplace=True)
        
        lu_merge = {}
        for zone_type in land_use['zone_type'].unique():
            lu_merge[zone_type] = land_use[land_use['zone_type'] == zone_type].unary_union
            print(".", end="")
        print(" ", end="")
        
        buffers = neighborhoods['coords'].apply(lambda coord: geodesic_point_buffer(coord.y, coord.x, 1000))
        
        for zone_type in lu_merge.keys():
            neighborhoods[f"neighborhood_{zone_type}"] = buffers.intersection(lu_merge[zone_type])
            neighborhoods[f"neighborhood_{zone_type}"].set_crs(epsg=4326, inplace=True)
            print(".", end="")
        print(" ")
        
        neighborhoods.drop(columns=['coords'], inplace=True)
        
        with open(f'./python_variables/neighborhoods_{city}{year}.pickle', 'wb') as file:
            pickle.dump(neighborhoods, file)
        
        print(f'Pickling done. Time taken: {time.time()-pre:.2f} seconds.')
        
        land_use.to_crs(epsg=4326, inplace=True)
    
    return neighborhoods
    

def make_station_df(data, holidays=True, return_land_use=False, overwrite=False):
    postfix = "" if data.month == None else f"{data.month:02d}"
    postfix = postfix + "" if holidays else postfix + "_no_holidays"
    
    if not overwrite:
        try:
            with open(f'./python_variables/station_df_{data.city}{data.year:d}{postfix}.pickle', 'rb') as file:
                df, land_use = pickle.load(file)
            
            try:
                with open(f'./python_variables/neighborhoods_{data.city}{data.year}.pickle', 'rb') as file:
                    neighborhoods = pickle.load(file)
            
            except FileNotFoundError:
                print(f'No neighborhoods found. Pickling neighborhoods using {data.city}{data.year} data...')
                neighborhoods = make_neighborhoods(data.city, data.year, df, land_use)
            
            df = df.merge(neighborhoods, on='stat_id')
            
            if return_land_use:
                return df, land_use
            else:
                return df
        except FileNotFoundError:
            print("Pickle does not exist. Pickling station_df...")
    
    print("Making Station DataFrame", end="")
    
    df = pd.DataFrame(data.stat.locations).T.rename(columns={0: 'long', 1: 'lat'})
    
    df['stat_id'] = df.index.map(data.stat.inverse)
    
    df['easting'], df['northing'] = hv.util.transform.lon_lat_to_easting_northing(df['long'], df['lat'])
    
    df['name'] = data.stat.names.values()
    df['n_arrivals'] = df['stat_id'].map(data.df['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    df['n_departures'] = df['stat_id'].map(data.df['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)

    df['n_trips'] = df['n_arrivals'] + df['n_departures']
    
    
    
    df_s = data.df[['start_stat_id', 'start_dt']][data.df['start_dt'].dt.weekday <= 4]
    
    print(".", end="")
    
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
    
    land_use_extent = Polygon(
        [(df['easting'].min()-1000, df['northing'].min()-1000),
         (df['easting'].min()-1000, df['northing'].max()+1000),
         (df['easting'].max()+1000, df['northing'].max()+1000),
         (df['easting'].max()+1000, df['northing'].min()-1000)])
    print(". ", end="")
    
    # poly_gdf = gpd.GeoDataFrame(index=[0], columns=['poly'], geometry='poly')
    # poly_gdf.loc[0,'poly'] = land_use_extent
    # poly_gdf.set_crs(epsg=3857, inplace=True)
    # poly_gdf.to_crs(epsg=4326, inplace=True)
    
    # return poly_gdf
    # print(poly_gdf.iloc.poly)
    # extent = {'lat': [df['lat'].min(), df['lat'].max()], 
              # 'long': [df['long'].min(), df['long'].max()]}
    
    if data.city == 'nyc':
    
        zoning_df = gpd.read_file('./data/other_data/nyc_zoning_data.json')
        zoning_df = zoning_df[['ZONEDIST', 'geometry']]
        
        land_use = zoning_df[['ZONEDIST', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
        
        land_use.to_crs(epsg=3857, inplace=True)
        land_use = land_use.cx[df['easting'].min()-1000:df['easting'].max()+1000,
                               df['northing'].min()-1000:df['northing'].max()+1000]
        
        land_use['geometry'] = land_use['geometry'].apply(lambda area: area.buffer(0).intersection(land_use_extent))
        land_use.to_crs(epsg=4326, inplace=True)
        
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
    
    elif data.city in ['madrid', 'helsinki', 'london', 'oslo']:
        
        bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        land_use = gpd.read_file(f'data/other_data/{data.city}_UA2018_v013.gpkg', bbox=bbox)
        land_use = land_use[['code_2018', 'class_2018', 'area', 'Pop2018', 'geometry']].to_crs('EPSG:4326')
        print(".", end="")
        df = gpd.GeoDataFrame(df, geometry='coords', crs='EPSG:4326')
        df = gpd.tools.sjoin(df, land_use, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        df['Pop2018'].fillna(0, inplace=True)
        df['area'].fillna(0.1, inplace=True)
        print(".", end="")
        df['zone_type'] = df['code_2018'].apply(lambda x: zone_dist_transform(data.city, x))
        df.loc[df['zone_type'] == 'water', 'zone_type'] = "UNKNOWN"
        print(".", end="")
        land_use = land_use[['code_2018', 'geometry']]
        print(".", end="")
        land_use.to_crs(epsg=3857, inplace=True)
        print(".", end="")
        land_use = land_use.cx[df['easting'].min()-1000:df['easting'].max()+1000,
                               df['northing'].min()-1000:df['northing'].max()+1000]
        print(".", end="")
        land_use['geometry'] = land_use['geometry'].apply(lambda area: area.intersection(land_use_extent))
        land_use.to_crs(epsg=4326, inplace=True)
        print(".", end="")
        
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
        # land_use = land_use[land_use['zone_type'] != 'road']
        land_use = land_use[land_use['zone_type'] != 'water']
        print(".", end="")
        subways_df = gpd.read_file(f'./data/other_data/{data.city}_transit_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
        
        df['pop_density'] = df['Pop2018'] / df['area']
    
    elif data.city == 'chic':
        
        zoning_df = gpd.read_file('./data/other_data/chic_zoning_data.geojson')
        zoning_df = zoning_df[['zone_class', 'geometry']]
        
        land_use = zoning_df[['zone_class', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
    
        land_use.to_crs(epsg=3857, inplace=True)
        land_use = land_use.cx[df['easting'].min()-1000:df['easting'].max()+1000,
                               df['northing'].min()-1000:df['northing'].max()+1000]
        
        land_use['geometry'] = land_use['geometry'].apply(lambda area: area.buffer(0).intersection(land_use_extent))
        land_use.to_crs(epsg=4326, inplace=True)
    
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
        
        zoning_DC = gpd.read_file('./data/other_data/washDC_zoning_data.geojson')
        zoning_DC = zoning_DC[['ZONING_LABEL', 'geometry']]
        
        zoning_arlington = gpd.read_file('./data/other_data/arlington_zoning_data.geojson')
        zoning_arlington = zoning_arlington[['ZN_DESIG', 'geometry']]
        zoning_arlington = zoning_arlington.rename({'ZN_DESIG' : 'ZONING_LABEL'}, axis=1)
        
        zoning_alexandria = gpd.read_file('./data/other_data/alexandria_zoning_data.geojson')
        zoning_alexandria = zoning_alexandria[['ZONING', 'geometry']]
        zoning_alexandria = zoning_alexandria.rename({'ZONING' : 'ZONING_LABEL'}, axis=1)
        
        zoning_df = gpd.GeoDataFrame(pd.concat([zoning_DC, 
                                                zoning_arlington, 
                                                zoning_alexandria], ignore_index=True),
                                     geometry='geometry', crs='EPSG:4326')
        
        land_use = zoning_df[['ZONING_LABEL', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
        
        land_use.to_crs(epsg=3857, inplace=True)
        land_use = land_use.cx[df['easting'].min()-1000:df['easting'].max()+1000,
                               df['northing'].min()-1000:df['northing'].max()+1000]
        
        land_use['geometry'] = land_use['geometry'].apply(lambda area: area.buffer(0).intersection(land_use_extent))
        land_use.to_crs(epsg=4326, inplace=True)
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['zone_type'] = df['ZONING_LABEL'].apply(lambda x: zone_dist_transform(data.city, x))
        
        
        census_df = pd.read_csv('./data/other_data/washDC_census_data.csv', 
                                usecols=['B01001_001E', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        DC_CTracts_df = gpd.read_file('./data/other_data/washDC_CT_data.shp')
        DC_CTracts_df = DC_CTracts_df[['GEOID', 'geometry']]
        
        VA_CTracts_df = gpd.read_file('./data/other_data/VA_CT_data.shp')
        VA_CTracts_df = VA_CTracts_df[['GEOID', 'geometry']]
        
        CTracts_df = DC_CTracts_df.append(VA_CTracts_df)
        CTracts_df = CTracts_df.rename({'GEOID' : 'GEO_ID'}, axis=1)
        
        census_df = gpd.GeoDataFrame(census_df.merge(CTracts_df, on='GEO_ID'),
                                     geometry='geometry', crs='EPSG:4326')
        
        census_df.to_crs(epsg=3857, inplace=True)
        census_df['CT_area'] = census_df['geometry'].area/1000000
        census_df.to_crs(epsg=4326, inplace=True)
        
        # census_df = gpd.read_file('./data/other_data/washDC_census_data.geojson')
        # census_df = census_df[['GEOID', 'P0010001', 'geometry']]
        
        # census_df_cart = census_df.copy()
        # census_df_cart = census_df_cart.to_crs({'proj': 'cea'})
        # census_df['CT_area'] = census_df_cart['geometry'].area
        
        # census_df = census_df[['GEOID', 'CT_area', 'P0010001', 'geometry']]
        
        df = gpd.tools.sjoin(df, census_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['pop_density'] = df['B01001_001E'] / df['CT_area']
    
        subways_df = gpd.read_file('./data/other_data/washDC_subways_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    
    elif data.city == 'minn':
        
        zoning_df = gpd.read_file('./data/other_data/minn_zoning_data.geojson')
        zoning_df = zoning_df[['ZONE_CODE', 'geometry']]
        
        land_use = zoning_df[['ZONE_CODE', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
        
        land_use.to_crs(epsg=3857, inplace=True)
        land_use = land_use.cx[df['easting'].min()-1000:df['easting'].max()+1000,
                               df['northing'].min()-1000:df['northing'].max()+1000]
        
        land_use['geometry'] = land_use['geometry'].apply(lambda area: area.buffer(0).intersection(land_use_extent))
        land_use.to_crs(epsg=4326, inplace=True)
        
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
        
        zoning_boston = gpd.read_file('./data/other_data/boston_zoning_data.geojson')
        zoning_boston = zoning_boston[['ZONE_', 'SUBDISTRIC', 'Zone_Desc','geometry']]
        zoning_boston['zone_type'] = zoning_boston['SUBDISTRIC'].apply(lambda x: zone_dist_transform(data.city, x))
        zoning_boston = zoning_boston[['ZONE_', 'geometry']]
        
        
        zoning_cambridge = gpd.read_file('./data/other_data/Cambridge_zoning_data.shp')
        zoning_cambridge = zoning_cambridge[['ZONE_TYPE', 'geometry']]
        zoning_cambridge = zoning_cambridge.rename({'ZONE_TYPE' : 'ZONE_'}, axis=1)

        zoning_cambridge['zone_type'] = zoning_cambridge['ZONE_'].apply(lambda x: zone_dist_transform(data.city, x))
        zoning_cambridge.to_crs(epsg=4326, inplace=True)
        
        zoning_df = gpd.GeoDataFrame(pd.concat([zoning_boston, 
                                                zoning_cambridge], ignore_index=True),
                                     geometry='geometry', crs='EPSG:4326')
        
        
        land_use = zoning_df[['zone_type', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        #land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_dist_transform(data.city, x))
        
        land_use.to_crs(epsg=3857, inplace=True)
        land_use = land_use.cx[df['easting'].min()-1000:df['easting'].max()+1000,
                               df['northing'].min()-1000:df['northing'].max()+1000]
        
        land_use['geometry'] = land_use['geometry'].apply(lambda area: area.buffer(0).intersection(land_use_extent))
        land_use.to_crs(epsg=4326, inplace=True)
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left') # TODO: adds more stations for some reason
        df.drop('index_right', axis=1, inplace=True)
    
        # df['zone_type'] = df['ZONE_CODE'].apply(lambda x: zone_dist_transform(data.city, x))
        
        census_df = pd.read_csv('./data/other_data/boston_census_data.csv', 
                                usecols=['B01001_001E', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        
        CTracts_df = gpd.read_file('./data/other_data/boston_CT_data.shp')
        CTracts_df = CTracts_df[['GEOID', 'geometry']]
        
        CTracts_df = CTracts_df.rename({'GEOID' : 'GEO_ID'}, axis=1)
        
        census_df = gpd.GeoDataFrame(census_df.merge(CTracts_df, on='GEO_ID'),
                                     geometry='geometry', crs='EPSG:4326')
        
        census_df.to_crs(epsg=3857, inplace=True)
        census_df['CT_area'] = census_df['geometry'].area/1000000
        census_df.to_crs(epsg=4326, inplace=True)
        
        # census_df = pd.read_file('./data/other_data/boston_census_data.csv')
        
        
        
        # CTracts_df.set_crs(epsg=4326)
        
        
        
        # CTracts_df = CTracts_df[['GEOID20', 'ALAND20', 'geometry']]
        
        df = gpd.tools.sjoin(df, census_df, op='within', how='left')
        # df['GEOID20'] = df['GEOID20'].apply(lambda x: int(x) if pd.notnull(x) else x)
        df.drop('index_right', axis=1, inplace=True)
        
        # census_df = pd.read_csv('./data/other_data/boston_census_data.csv')
        # census_df = census_df[['GEOCODE', 'P0020001']].iloc[1:]
        # census_df['GEOCODE'] = census_df['GEOCODE'].apply(lambda x: int(x) if pd.notnull(x) else x)
        
        # pop_map = dict(zip(census_df['GEOCODE'], census_df['P0020001']))
        
        # df['population'] = df['GEOID20'].map(pop_map).apply(lambda x: int(x) if pd.notnull(x) else x)
        df['pop_density'] = df['B01001_001E'] / df['CT_area']
        
        
        subways_df = gpd.read_file('./data/other_data/boston_subways_data.shp').to_crs(epsg = 4326)
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    else:
        
        df['population'] = 0
        df['pop_density'] = 0
        df['zone_type'] = 0
        land_use = pd.DataFrame([])
        land_use['zone_type'] = 'UNKNOWN'
    
    print(".")
        
    df.rename(mapper=df_key(data.city), axis=1, inplace=True)  
    
    land_use['color'] = land_use['zone_type'].map(color_dict).fillna("pink")
    
    with open(f'./python_variables/station_df_{data.city}{data.year:d}{postfix}.pickle', 'wb') as file:
        pickle.dump([df, land_use], file)
    
    
    try:
        with open(f'./python_variables/neighborhoods_{data.city}{data.year}.pickle', 'rb') as file:
            neighborhoods = pickle.load(file)
    
    except FileNotFoundError:
        print(f'No neighborhoods found. Pickling neighborhoods using {data.city}{data.year} data...')
        neighborhoods = make_neighborhoods(data.city, data.year, df, land_use)
    
    df = df.merge(neighborhoods, on='stat_id')
    
    print("Done")
    
    if return_land_use:
        return df, land_use
    else:
        return df


def mask_traffic_matrix(traffic_matrices, station_df, day_type, min_trips, holidays=False, return_mask=False):
    """
    Applies a mask to the daily traffic matrix based on the minimum number of 
    trips to include.

    Parameters
    ----------
    day_type : str
        'business_days' or 'weekend'.
    min_trips : int
        the minimum number of trips for a station. If the station has fewer
        trips than this, exclude it.
    holidays : bool, optional
        Whether to include holidays in business days (True) or remove them from
        the business days (False). The default is False.

    Returns
    -------
    np array
        masked traffic matrix, that is, the number of 48-dimensional vectors 
        which constitute the rows of the traffic matrix is reduced.

    """
    if day_type == 'business_days':
        traffic_matrix = traffic_matrices[0]
        x_trips = 'b_trips'
    elif day_type == "weekend":
        traffic_matrix = traffic_matrices[1]
        x_trips = 'w_trips'
    else:
        raise ValueError("Please enter 'business_days' or 'weekend'.")
    mask = station_df[x_trips] >= min_trips
    if return_mask:
        return traffic_matrix[mask], mask, x_trips
    else:
        return traffic_matrix[mask]


def get_clusters(traffic_matrices, station_df, day_type, min_trips, clustering, k, random_state=None):
    """
    From a station dataframe and associated variables, return the updated 
    station df and clustering and labels

    Parameters
    ----------
    station_df : pandas dataframe
        has each station as a row.
    day_type : str
        'weekend' or 'business_days'.
    min_trips : int
        minimum number of trips.
    clustering : str
        clustering type.
    k : int
        number of clusters.
    random_state : int
        the seed for the random generator.

    Returns
    -------
    station_df : pandas dataframe
        has each station as a row and color and label columns populated.
    clusters : sklearn.clustering cluster
        can be used for stuff later.
    labels : np array
        the labels of the masked traffic matrix.

    """
    traffic_matrix, mask, x_trips = mask_traffic_matrix(
        traffic_matrices, station_df, day_type, min_trips, holidays=False, return_mask=True)
    
    if clustering == 'k_means':
        clusters = KMeans(k, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        station_df['color'] = station_df['label'].map(cluster_color_dict)

    elif clustering == 'k_medoids':
        clusters = KMedoids(k, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        station_df['color'] = station_df['label'].map(cluster_color_dict)
        
    elif clustering == 'h_clustering':
        clusters = None
        labels = AgglomerativeClustering(k).fit_predict(traffic_matrix)
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        station_df['color'] = station_df['label'].map(cluster_color_dict)
    
    elif clustering == 'gaussian_mixture':
        clusters = GaussianMixture(k, n_init=10, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict_proba(traffic_matrix)
        lab_mat = np.array(lab_color_list[:k]).T
        lab_cols = [np.sum(labels[i] * lab_mat, axis=1) for i in range(len(traffic_matrix))]
        labels_rgb = skcolor.lab2rgb(lab_cols)
        station_df.loc[mask, 'label'] = pd.Series(list(labels), index=mask[mask].index)
        station_df.loc[~mask, 'label'] = np.nan
        station_df.loc[mask, 'color'] = ['#%02x%02x%02x' % tuple(label.astype(int)) for label in labels_rgb*255]
        station_df.loc[~mask, 'color'] = 'gray'
        
    elif clustering == 'none':
        clusters = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = station_df[x_trips].tolist()
    
    elif clustering == 'zoning':
        clusters = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = [cluster_color_dict[zone] for zone in pd.factorize(station_df['zone_type'])[0]]
        
    else:
        clusters = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = None
    
    return station_df, clusters, labels


proj_wgs84 = pyproj.Proj('+proj=longlat +datum=WGS84')

def geodesic_point_buffer(lat, lon, m):
    """
    Stolen from https://gis.stackexchange.com/questions/289044/creating-buffer-circle-x-kilometers-from-point-using-python

    Parameters
    ----------
    lat : float
        latitude in WGS 84.
    lon : float
        longitude in WGS84.
    m : float
        buffer radius in meter.

    Returns
    -------
    shapely
        DESCRIPTION.

    """
    # Azimuthal equidistant projection
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        proj_wgs84)
    buf = Point(0, 0).buffer(m)  # distance in metres
    return shapely.ops.transform(project, buf)


def create_all_pickles(city, year, holidays=False, overwrite=False):
    if isinstance(city, str): # If city is a str (therefore not a list)
        data = bs.Data(city, year, overwrite=overwrite)
        make_station_df(data, holidays=holidays, overwrite=overwrite)
        data.pickle_daily_traffic(holidays=holidays, overwrite=overwrite)
        
        for month in bs.get_valid_months(city, year):
            print(f"Pickling month = {month}")
            if city in ['nyc', 'washDC', 'chic', 'la', 'sfran', 'london', 'mexico', 'buenos_aires', ]:
                data = bs.Data(city, year, month, overwrite=overwrite)
            else: #For other cities, the dataframe has been made already during the yearly dataframe.
                data = bs.Data(city, year, month, overwrite=False)
            make_station_df(data, holidays=holidays, overwrite=overwrite)
            data.pickle_daily_traffic(holidays=holidays, overwrite=overwrite)
    else:
        try:
            for cities_i in city:
                pre = time.time()
                with contextlib.suppress(FileNotFoundError):
                    os.remove(f"python_variables/neighborhoods_{cities_i}{year}.pickle")
                create_all_pickles(cities_i, year, holidays=holidays, overwrite=overwrite)
                print(f'{bs.name_dict[city]} took {time.time() - pre:.2f} seconds')
        except TypeError:
            print(city, "is not iterable, no pickle was made")


def create_all_pickles_all_cities(year, holidays=False, overwrite=False):
    for city in bs.name_dict.keys():
        pre = time.time()
        create_all_pickles(city, 2019, overwrite=True)
        print(f'{bs.name_dict[city]} took {time.time() - pre:.2f} seconds')


cluster_color_dict = {0 : 'blue', 1 : 'red', 2 : 'yellow', 3 : 'green', #tab:
              4 : 'purple', 5 : 'cyan', 6: 'pink',
              7 : 'brown', 8 : 'olive', 9 : 'magenta', np.nan: 'gray'}

mpl_color_dict = {i: mpl_colors.to_rgb(cluster_color_dict[i]) for i in range(10)}
lab_color_dict = {i: skcolor.rgb2lab(mpl_color_dict[i]) for i in range(10)}
lab_color_list = [lab_color_dict[i] for i in range(10)]


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

    
    # create_all_pickles('washDC', 2019, overwrite=True)

    data = bs.Data('madrid', 2019, 9)

    pre = time.time()
    station_df, land_use = make_station_df(data, return_land_use=True, overwrite=True)
    print(f'station_df took {time.time() - pre:.2f} seconds')
    
    for i, station in station_df.iterrows():
        a = geodesic_point_buffer(station['lat'], station['long'], 1000)
    
