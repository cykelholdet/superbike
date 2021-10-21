# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 21:22:19 2021

@author: nweinr
"""
import fiona
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
    
    elif city == 'chic':
        key = {'index' : 'stat_id',
               'zone_class' : 'zone_dist',
               'geoid10' : 'census_block',
               'TOTAL POPULATION' : 'population'}
    
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
            elif zone_dist in ['12220']: # Other roads and associated land
                zone_type = 'road'
            elif zone_dist in ['12100']: # Industrial, commercial, public, military and private units
                zone_type = 'commercial'
            elif zone_dist in ['14100', '14200', '31000']: # Green urban areas, Sports and leisure facilities, Forests
                zone_type = 'recreational'
            elif zone_dist in ['12230']: # Railways and associated land
                zone_type = 'transport'
            elif zone_dist in ['12300']: # Port areas
                zone_type = 'port'
            elif zone_dist in ['13100', 'Construction sites']: # Mineral extraction and dump sites
                zone_type = 'industrial'
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
                         'R-1-A']
            
            if zone_dist in res_zones:
                zone_type = 'residential'
            elif 'PDR' in zone_dist:
                zone_type = 'manufacturing'
            elif 'MU' in zone_dist or 'NC' in zone_dist or 'NHR' in zone_dist:
                zone_type = 'mixed'
            
            else:
                zone_type = 'UNKNOWN'
        
        
        else:
            raise KeyError('city transform not found')
        
    else:
        zone_type = 'UNKNOWN'
    
    return zone_type
    

def make_station_df(data):

    df = pd.DataFrame(data.stat.locations).T.rename(columns={0: 'long', 1: 'lat'})
    
    df['stat_id'] = df.index.map(data.stat.inverse)
    
    df['easting'], df['northing'] = hv.util.transform.lon_lat_to_easting_northing(df['long'], df['lat'])
    
    df['name'] = data.stat.names.values()
    df['n_arrivals'] = df['stat_id'].map(data.df['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    df['n_departures'] = df['stat_id'].map(data.df['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)

    df['n_trips'] = df['n_arrivals'] + df['n_departures']
    
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

    
    elif data.city == 'chic':
        
        zoning_df = gpd.read_file('./data/other_data/chic_zoning_data.geojson')
        zoning_df = zoning_df[['zone_class', 'geometry']]
    
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
            
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        # df['zone_type'] = df['ZONING_LABEL'].apply(lambda x: zone_dist_transform(data.city, x))
        
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
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        # df['zone_type'] = df['ZONE_CODE'].apply(lambda x: zone_dist_transform(data.city, x))
    
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
    
    df.rename(mapper=df_key(data.city), axis=1, inplace=True)    
    
    return df
    
    
if __name__ == "__main__":
    import bikeshare as bs
    import time
    
    data = bs.Data('minn', 2019, 9)
    pre = time.time()
    station_df = make_station_df(data)
    print(f'station_df took {time.time() - pre:.2f} seconds')
