# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:22:19 2021

@author: nweinr
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import holoviews as hv

import shapely.ops

from shapely.geometry import Point
# from shapely.ops import nearest_points
from geopy.distance import great_circle

def df_key(city):
    
    if city == 'nyc':
        
        key = {'Index' : 'stat_id',
               'ZONEDIST' : 'zone_dist',
               'BoroCT2020' : 'census_tract',
               'Shape__Area' : 'CT_area',
               'Pop_20' : 'population'}
    elif city == 'madrid':
        key = {}
    else:
        key = {}
    
    return key
    
    
def zone_dist_transform(city, zone_dist):
    
    if city == 'nyc':
        
        if 'PARK' in zone_dist or 'PLAYGROUND' in zone_dist:
            zone_type = 'recreation'
        
        elif 'R' in zone_dist and '/' not in zone_dist:
            zone_type = 'residential'
        
        elif 'C' in zone_dist and '/' not in zone_dist:
            zone_type = 'commercial'
            
        elif 'M' in zone_dist and '/' not in zone_dist:
            zone_type = 'manufacturing'
            
        else:
            zone_type = 'mixed'
    
    elif city == 'madrid':
        if zone_dist in ['11100', '11210', '11220']: # Continuous urban fabric (S.L. : > 80%), Discontinuous dense urban fabric (S.L. : 50% -  80%), Discontinuous medium density urban fabric (S.L. : 30% - 50%)
            zone_type = 'residential' 
        elif zone_dist in ['12220']: # Other roads and associated land
            zone_type = 'road'
        elif zone_dist in ['12100']: # Industrial, commercial, public, military and private units
            zone_type = 'commercial'
        elif zone_dist in ['14100']: # Green urban areas
            zone_type = 'recreational'
        else:
            zone_type = 'UNKNOWN'
    
    else:
        raise KeyError('city transform not found')
        
    return zone_type


def make_station_df(data):
        
    df = pd.DataFrame(data.stat.locations).T.rename(columns={0: 'long', 1: 'lat'})
    
    df['stat_id'] = df.index.map(data.stat.inverse)
    
    df['easting'], df['northing'] = hv.util.transform.lon_lat_to_easting_northing(df['long'], df['lat'])

    df['name'] = data.stat.names.values()
    df['n_arrivals'] = data.df['start_stat_id'].value_counts()
    df['n_departures'] = data.df['end_stat_id'].value_counts()
    df['n_arrivals'].fillna(0, inplace=True)
    df['n_departures'].fillna(0, inplace=True)

    df['n_trips'] = data.df['start_stat_id'].value_counts().add(data.df['end_stat_id'].value_counts(), fill_value=0)

    df['coords'] = list(zip(df['long'], df['lat']))
    df['coords'] = df['coords'].apply(Point)
    
    if data.city == 'nyc':
    
        zoning_df = gpd.read_file('./data/other_data/nyc_zoning_data.json')
        zoning_df = zoning_df[['ZONEDIST', 'geometry']]
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        df['zone_type'] = df['ZONEDIST'].apply(lambda x: zone_dist_transform(data.city, x))
        
        
        CTracts_df = gpd.read_file('./data/other_data/nyc_CT_data.json')
        CTracts_df = CTracts_df[['BoroCT2020', 'geometry', 'Shape__Area']]
        # CTracts_df.rename({'Shape__Area':'CT_area'}, axis=1, inplace=True)
        
        df = gpd.tools.sjoin(df, CTracts_df, op='within', how='left')
        df['BoroCT2020'] = df['BoroCT2020'].apply(int)
        df.drop('index_right', axis=1, inplace=True)
        
        census_df = pd.read_excel('./data/other_data/nyc_census_data.xlsx', sheet_name=1, skiprows=[0,1,2])
        census_df = census_df[['BCT2020', 'Pop_20']]
        census_df.rename(columns={'BCT2020': 'BoroCT2020'}, inplace=True)
        
        df = df.reset_index().merge(census_df, on='BoroCT2020').set_index('index').sort_index()
        
        df['pop_density'] = df['2020 Data'] / df['Shape__Area']
        
        subways_df = gpd.read_file('./data/other_data/nyc_subways_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    elif data.city == 'madrid':
        
        land_use_df = gpd.read_file('data/other_data/madrid_UA2018_v013.gpkg')
        land_use_df = land_use_df[['code_2018', 'class_2018', 'area', 'Pop2018', 'geometry']].to_crs('EPSG:4326')
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs='EPSG:4326')
        df = gpd.tools.sjoin(df, land_use_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        df['zone_type'] = df['code_2018'].apply(lambda x: zone_dist_transform(data.city, x))

    
    
    
    
    
    
    
    
    elif data.city == 'chic':
        
        zoning_df = gpd.read_file('./data/other_data/chic_zoning_data.geojson')
        zoning_df = zoning_df[['zone_class', 'geometry']]
    
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        # df['zone_type'] = df['zone_class'].apply(lambda x: zone_dist_transform(data.city, x))
        
        CBlocks_df = gpd.read_file('./data/other_data/chic_CB_data.geojson')
        CBlocks_df = CBlocks_df[['BoroCT2020', 'geometry', 'Shape__Area']]
    
    
    
    
    
    df.rename(mapper=df_key(data.city), axis=1, inplace=True)    
    
    
    return df
    
    
if __name__ == "__main__":
    import bikeshare as bs
    import time
    
    data = bs.Data('nyc', 2019, 9)
    pre = time.time()
    station_df = make_station_df(data)
    print(f'station_df took {time.time() - pre:.2f} seconds')
