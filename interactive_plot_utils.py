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

def make_station_df(data):
    
    station_df = pd.DataFrame(data.stat.locations).T.rename(columns={0: 'long', 1: 'lat'}, index=data.stat.inverse)

    station_df['easting'], station_df['northing'] = hv.util.transform.lon_lat_to_easting_northing(station_df['long'], station_df['lat'])
    
    station_df['name'] = data.stat.names.values()
    station_df['n_arrivals'] = data.df['start_stat_id'].value_counts()
    station_df['n_departures'] = data.df['end_stat_id'].value_counts()
    station_df['n_arrivals'].fillna(0, inplace=True)
    station_df['n_departures'].fillna(0, inplace=True)

    station_df['n_trips'] = data.df['start_stat_id'].value_counts().add(data.df['end_stat_id'].value_counts(), fill_value=0)

    station_df['coords'] = list(zip(station_df['long'], station_df['lat']))
    station_df['coords'] = station_df['coords'].apply(Point)
    
    if data.city == 'nyc':
    
        zoning_df = gpd.read_file('./data/nyc_zoning_data.json')
        zoning_df = zoning_df[['ZONEDIST', 'geometry']]
        
        station_df = gpd.GeoDataFrame(station_df, geometry='coords', crs=zoning_df.crs)
        station_df = gpd.tools.sjoin(station_df, zoning_df, op='within', how='left')
        station_df.drop('index_right', axis=1, inplace=True)
        
        CTracts_df = gpd.read_file('./data/nyc_CT_data.json')
        CTracts_df = CTracts_df[['BoroCT2020', 'geometry', 'Shape__Area']]
        CTracts_df.rename({'Shape__Area':'CT_area'}, axis=1, inplace=True)
        
        station_df = gpd.tools.sjoin(station_df, CTracts_df, op='within', how='left')
        station_df['BoroCT2020'] = station_df['BoroCT2020'].apply(int)
        station_df.drop('index_right', axis=1, inplace=True)
        
        census_df = pd.read_excel('./data/nyc_census_data.xlsx', sheet_name=1)
        census_df = census_df[['Unnamed: 4', '2020 Data']]
        census_df.rename({'Unnamed: 4':'BoroCT2020','2020 Data':'population'}, axis=1, inplace=True)
        
        station_df = pd.merge(station_df, census_df, on = 'BoroCT2020') # resetter index, måske ikke godt
        station_df['pop_density'] = station_df['population'] / station_df['CT_area']
        
        subways_df = gpd.read_file('./data/nyc_subways_data.geojson')
        
        station_df['nearest_subway'] = station_df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        station_df['nearest_subway_dist'] = station_df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    return station_df
    
    
    
    
    
    
    
    
    
    
    
    