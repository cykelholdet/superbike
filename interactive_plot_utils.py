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

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import skimage.color as skcolor
import statsmodels.api as sm
from shapely.geometry import Point, Polygon, LineString
# from shapely.ops import nearest_points
from geopy.distance import great_circle
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statsmodels.discrete.discrete_model import MNLogit
from scipy.spatial import Voronoi

import bikeshare as bs
import dataframe_key

def df_key(city):
    
    if city == 'nyc':
        
        key = {'index' : 'stat_id',
               'ZONEDIST' : 'zone_code',
               'BoroCT2020' : 'census_tract',
               'Shape__Area' : 'census_area',
               'Pop_20' : 'population'}
    elif city == 'madrid':
        key = {}
    
    elif city == 'chic':
        key = {'index' : 'stat_id',
               'zone_class' : 'zone_code',
               'geoid10' : 'census_block',
               'TOTAL POPULATION' : 'population'}
    
    elif city == 'washDC':
        key = {'ZONING LABEL' : 'zone_code',
               'GEO_ID' : 'census_tract',
               'B01001_001E' : 'population'}
    
    elif city == 'minn':
        key = {'ZONE_CODE' : 'zone_code',
               'GEOID20' : 'census_tract',
               'ALAND20' : 'census_area'}
    
    elif city == 'boston':
        key = {'ZONE_' : 'zone_code',
               'GEO_ID' : 'census_tract',
               'P1_001N' : 'population'}
    
    else:
        key = {}
    
    return key
    
    
def zone_code_transform(city, zone_code):
    
    if pd.notnull(zone_code):
        
        if city == 'nyc':
            
            if 'PARK' in zone_code or 'PLAYGROUND' in zone_code:
                zone_type = 'recreational'
            elif 'R' in zone_code and '/' not in zone_code:
                zone_type = 'residential'
            elif 'C' in zone_code and '/' not in zone_code:
                zone_type = 'commercial'
            elif 'M' in zone_code and '/' not in zone_code:
                zone_type = 'industrial'
            elif '/' in zone_code:
                zone_type = 'mixed'
            else:
                zone_type = 'UNKNOWN'
        
        elif city in ['madrid', 'helsinki', 'london', 'oslo', 'bergen', 'trondheim', 'edinburgh']:
            
            if zone_code in ['11100', '11210', '11220', '11230']: # Continuous urban fabric (S.L. : > 80%), Discontinuous dense urban fabric (S.L. : 50% -  80%), Discontinuous medium density urban fabric (S.L. : 30% - 50%), Discontinuous low density urban fabric (S.L. : 10% - 30%)
                zone_type = 'residential' 
            elif zone_code in ['12220', '12210']: # Other roads and associated land, Fast transit roads and associated land
                zone_type = 'road'
            elif zone_code in ['12100']: # Industrial, commercial, public, military and private units
                zone_type = 'commercial'
            elif zone_code in ['14100', '14200', '31000', '32000']: # Green urban areas, Sports and leisure facilities, Forests, Herbaceous vegetation associations (natural grassland, moors...)
                zone_type = 'recreational'
            elif zone_code in ['12230']: # Railways and associated land
                zone_type = 'transportation'
            elif zone_code in ['13100', 'Construction sites', '12300']: # Mineral extraction and dump sites, Construction, Port areas
                zone_type = 'industrial'
            elif zone_code in ['50000']:
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
        
            if zone_code in com_zones: # Business and commercial zones and downtown core and services
                zone_type = 'commercial'
            elif zone_code in res_zones: # Residential and downtown residential
                zone_type = 'residential'
            elif zone_code in man_zones: # Manufacturing and planned manufactoring development
                zone_type = 'industrial'
            elif zone_code == 'T': # Transportation
                zone_type = 'transportation'
            elif zone_code in rec_zones: # Green areas
                zone_type = 'recreational'
            elif 'DX' in zone_code or 'PD' in zone_code: # Mixed downtown and planned development
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
            
            
            if zone_code in res_zones:
                zone_type = 'residential'
            elif zone_code in com_zones:
                zone_type = 'commercial'
            elif zone_code in rec_zones:
                zone_type = 'recreational'
            elif zone_code in man_zones or 'PDR' in zone_code:
                zone_type = 'industrial'
            elif zone_code in mix_zones or 'WR' in zone_code or 'CDD' in zone_code:
                zone_type = 'mixed'
            elif 'StE' in zone_code:
                zone_type = 'educational'
            elif zone_code == 'UT':
                zone_type = 'transportation'
            
            else:
                zone_type = 'UNKNOWN'
        
        elif city == 'minn':
            
            if 'OR' in zone_code:
                zone_type = 'mixed'
            elif 'R' in zone_code:
                zone_type = 'residential'
            elif 'C' in zone_code or 'B' in zone_code:
                zone_type = 'commercial'
            elif 'I' in zone_code:
                zone_type = 'industrial'
            
            else:
                zone_type = 'UNKNOWN'
        
        elif city == 'boston':
            
            res_zones = ['A-1', 'A-2', 'B', 'C', 'C-1', 'C-1A', 'C-2', 'C-2A',
                         'C-2B', 'C-3', 'C-3A', 'C-3B', 'SD-10(H)', 'SD-12',
                         'SD-13', 'SD-14', 'SD-2', 'SD-4A', 'SD-6', 'SD-9',
                         'NR', 'UR', 'Apartment Residential', 
                         'Multifamily Residential',
                         'Multifamiy Residential/Local Services',
                         'Neighborhood Development Area', 
                         'One-Family Residential', 'Residential', 
                         'Row House Residential', 'Three-Family Residential',
                         'Two-Family Residential', 'Waterfront Residential']
        
            com_zones = ['BA', 'BA-1', 'BA-2', 'BA-3', 'BA-4', 'BB', 'BB-1',
                         'BB-2', 'O-1', 'O-2', 'O-2A', 'O-3', 'O-3A', 'SD-1',
                         'SD-10(F)', 'SD-11', 'SD-4', 'SD-5', 'SD-7', 'FAB',
                         'CI', 'CB', 'CC4', 'CC5', 'C3', 'Chalestown Gateway',
                         'Commercial', 'Community Commercial', 
                         'Economic Development Area', 'General Business', 
                         'Local Business', 'Local Convenience',
                         'Neighborhood Shopping', 
                         'New Boston Garden Development Area',
                         'Parcel-to-Parcel Linkage Development Area', 
                         'Pilot House Extension', 'Protection Area',
                         'Transition Zone', 'Waterfront Commercial', 
                         'Waterfront Service', 'Waterfront Transition Zone']
            
            mix_zones = ['ASD', 'CRDD', 'MXD', 'NP', 'SD-3', 'SD-8', 'SD-8A',
                         'MR3', 'MR4', 'MR5', 'MR6', 'ASMD', 'HR', 'PS', 
                         'Central Artery Area', 'General Area', 
                         'Huntington Avenue Boulevard Area', 
                         'Institutional', 'Leather District', 
                         'Medium Density Area', 'Mixed Use Area 1',
                         'Mixed use Area 2', 'Mixed use Area 3', 
                         'Mixed use Area 4', 'Restricted Growth Area',
                         'Turnpike Air-Rights Special Study Area']
        
            rec_zones = ['OS', 'CIV', 'Air-Right Open Space', 
                         'Botanical/Zoological Garden Open Space',
                         'Cemetery Open Space', 'Charlestown Navy Yard',
                         'Chestnut Hill Waterworks Protection', 
                         'Community Garden Open Space', 'Conservation Protection',
                         'Corridor Enhancement', 'Cultural Facilities',
                         'Enterprise Protection', 'Open Space', 
                         'Parkland Open Space', 'Recreation Open Space',
                         'Shoreland Open Space', 'Urban Plaza Open Space',
                         'Urban Wild Open Space', 'Waterfront', 
                         'Waterfront Access Open Space', 
                         'Waterfront Community Facilities']
            
            man_zones = ['IA', 'IA-1', 'IA-2', 'IB', 'IB-1', 'IB-2', 'SD-15',
                         'General Industrial', 'Industrial Commercial',
                         'Industrial Development Area', 'Local Industrial', 
                         'Maritime Economy Reserve', 'Restricted Manufacturing',
                         'Waterfront Manufacturing']
            
            edu_zones = ['TU', 'Community Facilities', 
                         'Neighborhood Institutional', 
                         'Special Study Area']
            
            if zone_code in res_zones:
                zone_type = 'residential'
            
            elif zone_code in com_zones:
                zone_type = 'commercial'
            
            elif zone_code in mix_zones:
                zone_type = 'mixed'
            
            elif zone_code in rec_zones:
                zone_type = 'recreational'
            
            elif zone_code in man_zones:
                zone_type = 'industrial'
            
            elif zone_code in edu_zones:
                zone_type = 'educational'
            
            elif zone_code == 'Residential':
                zone_type = 'residential'
            elif zone_code == 'Open Space':
                zone_type = 'recreational'
            elif zone_code == 'Business':
                zone_type = 'commercial'
            elif zone_code == 'Mixed use':
                zone_type = 'mixed'
            elif zone_code == 'Industrial':
                zone_type = 'industrial'
            elif zone_code == 'Comm/Instit':
                zone_type = 'educational'
    
            
            elif zone_code in ['APARTMENT HOUSE', 'SINGLE-FAMILY', 
                               'TWO-FAMILY & ATTACHED SINGLE-FAMILY',
                               'SINGLE-FAMILY & CONVERTED FOR TWO-FAMILY',
                               'THREE-FAMILY', 'SPECIAL DISTRICT']:
                zone_type = 'residential'
            elif zone_code in ['GENERAL BUSINESS', 'LOCAL BUSINESS',
                               'BUSINESS AND PROFFESSIONAL OFFICE',
                               'LIMITED SERVICE HOTEL',
                               'GENERAL BUSINESS AND MEDICAL RESEARCH']:
                zone_type = 'commercial'
            elif zone_code == 'INDUSTRIAL SERVICES':
                zone_type = 'industrial'
            
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
    print("Getting the hood ready to go", end="")
    if len(land_use) > 0:
        
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
    

def make_station_df(data, holidays=True, return_land_use=False, 
                    return_census=False, overwrite=False):
    postfix = "" if data.month == None else f"{data.month:02d}"
    postfix = postfix + "" if holidays else postfix + "_no_holidays"
    
    if not overwrite:
        try:
            with open(f'./python_variables/station_df_{data.city}{data.year:d}{postfix}.pickle', 'rb') as file:
                df, land_use, census_df = pickle.load(file)
            
            try:
                with open(f'./python_variables/neighborhoods_{data.city}{data.year}.pickle', 'rb') as file:
                    neighborhoods = pickle.load(file)
            
            except FileNotFoundError:
                print(f'No neighborhoods found. Pickling neighborhoods using {data.city}{data.year} data...')
                neighborhoods = make_neighborhoods(data.city, data.year, df, land_use)
            
            df = df.merge(neighborhoods, on='stat_id')
            
            if return_land_use and not return_census:
                return df, land_use
            elif return_census and not return_land_use:
                return df, census_df
            elif return_land_use and return_census:
                return df, land_use, census_df
            else:
                return df
        except FileNotFoundError:
            print("Pickle does not exist. ", end="")
    
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
    
    census_df = pd.DataFrame([])
    
    if data.city == 'nyc':
        
        bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        
        zoning_df = gpd.read_file('./data/other_data/nyc_zoning_data.json', bbox=bbox)
        zoning_df = zoning_df[['ZONEDIST', 'geometry']]
        
        land_use = zoning_df[['ZONEDIST', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_code_transform(data.city, x))
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        df['zone_type'] = df['ZONEDIST'].apply(lambda x: zone_code_transform(data.city, x))
        
        census_df = pd.read_csv('./data/other_data/nyc_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        
        bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        
        CTracts_df = gpd.read_file('./data/other_data/nyc_CT_data.shp', bbox=bbox)
        CTracts_df.set_crs(epsg=4326, allow_override=True, inplace=True)
        CTracts_df = CTracts_df[['GEOID', 'geometry']]
        CTracts_df = CTracts_df.rename({'GEOID' : 'GEO_ID'}, axis=1)
        
        census_df = gpd.GeoDataFrame(census_df.merge(CTracts_df, on='GEO_ID'),
                                     geometry='geometry', crs='EPSG:4326')
        
        census_df.to_crs(epsg=3857, inplace=True)
        census_df['census_area'] = census_df['geometry'].area/1000000
        census_df.to_crs(epsg=4326, inplace=True)
        census_df['pop_density'] = census_df['P1_001N'] / census_df['census_area']
        
        census_df.rename(columns=dataframe_key.get_census_key(data.city), inplace=True)
        
        # census_df = pd.read_excel('./data/other_data/nyc_census_data.xlsx', sheet_name=1, skiprows=[0,1,2])
        # census_df = census_df[['BCT2020', 'Pop_20']].dropna()
        
        # CTracts_df = gpd.read_file('./data/other_data/nyc_CT_data.json', bbox=bbox)
        # CTracts_df = CTracts_df[['BoroCT2020', 'geometry']]
        # CTracts_df.rename(columns={'BoroCT2020' : 'BCT2020'}, inplace=True)
        # CTracts_df['BCT2020'] = CTracts_df['BCT2020'].apply(int)
        
        # census_df = gpd.GeoDataFrame(census_df.merge(CTracts_df, on='BCT2020'),
        #                      geometry='geometry', crs='EPSG:4326')
        
        # census_df.to_crs(epsg=3857, inplace=True)
        # census_df['census_area'] = census_df['geometry'].area/1000000
        # census_df.to_crs(epsg=4326, inplace=True)
        # census_df['pop_density'] = census_df['Pop_20'] / census_df['census_area']
        
        # census_df.rename(columns=dataframe_key.get_census_key(data.city), inplace=True)
        
        df = gpd.tools.sjoin(df, census_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        subways_df = gpd.read_file('./data/other_data/nyc_subways_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    elif data.city in ['madrid', 'helsinki', 'london', 'oslo', 'bergen', 'trondheim', 'edinburgh']:
        
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
        df['zone_type'] = df['code_2018'].apply(lambda x: zone_code_transform(data.city, x))
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
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_code_transform(data.city, x))
        # land_use = land_use[land_use['zone_type'] != 'road']
        land_use = land_use[land_use['zone_type'] != 'water']
        print(".", end="")
        subways_df = gpd.read_file(f'./data/other_data/{data.city}_transit_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
        
        df['pop_density'] = (df['Pop2018'] / df['area'])*1000000
        
        census_df = pd.DataFrame([])
        
    elif data.city == 'chic':
        
        bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        
        zoning_df = gpd.read_file('./data/other_data/chic_zoning_data.geojson', bbox=bbox)
        zoning_df = zoning_df[['zone_class', 'geometry']]
        
        land_use = zoning_df[['zone_class', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_code_transform(data.city, x))
    
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['zone_type'] = df['zone_class'].apply(lambda x: zone_code_transform(data.city, x))
        
        # CBlocks_df = gpd.read_file('./data/other_data/chic_CB_data.geojson')
        # CBlocks_df = CBlocks_df[['geoid10', 'geometry']]
        # CBlocks_df.rename(columns={'geoid10' : 'CENSUS BLOCK FULL'}, inplace=True)
        # CBlocks_df['CENSUS BLOCK FULL'] = CBlocks_df['CENSUS BLOCK FULL'].apply(int)
        
        census_df = pd.read_csv('./data/other_data/chic_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        
        bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        
        CTracts_df = gpd.read_file('./data/other_data/chic_CT_data.shp', bbox=bbox)
        CTracts_df.set_crs(epsg=4326, allow_override=True, inplace=True)
        CTracts_df = CTracts_df[['GEOID', 'geometry']]
        CTracts_df = CTracts_df.rename({'GEOID' : 'GEO_ID'}, axis=1)
        
        census_df = gpd.GeoDataFrame(census_df.merge(CTracts_df, on='GEO_ID'),
                                     geometry='geometry', crs='EPSG:4326')
        
        census_df.to_crs(epsg=3857, inplace=True)
        census_df['census_area'] = census_df['geometry'].area/1000000
        census_df.to_crs(epsg=4326, inplace=True)
        census_df['pop_density'] = census_df['P1_001N'] / census_df['census_area']
        
        census_df.rename(columns=dataframe_key.get_census_key(data.city), inplace=True)
        
        df = gpd.tools.sjoin(df, census_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
        subways_df = gpd.read_file('./data/other_data/chic_subways_data.kml', driver='KML')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    elif data.city == 'washDC':
        
        bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        
        zoning_DC = gpd.read_file('./data/other_data/washDC_zoning_data.geojson', bbox=bbox)
        zoning_DC = zoning_DC[['ZONING_LABEL', 'geometry']]
        
        zoning_arlington = gpd.read_file('./data/other_data/arlington_zoning_data.geojson', bbox=bbox)
        zoning_arlington = zoning_arlington[['ZN_DESIG', 'geometry']]
        zoning_arlington = zoning_arlington.rename({'ZN_DESIG' : 'ZONING_LABEL'}, axis=1)
        
        zoning_alexandria = gpd.read_file('./data/other_data/alexandria_zoning_data.geojson', bbox=bbox)
        zoning_alexandria = zoning_alexandria[['ZONING', 'geometry']]
        zoning_alexandria = zoning_alexandria.rename({'ZONING' : 'ZONING_LABEL'}, axis=1)
        
        zoning_df = gpd.GeoDataFrame(pd.concat([zoning_DC, 
                                                zoning_arlington, 
                                                zoning_alexandria], ignore_index=True),
                                     geometry='geometry', crs='EPSG:4326')
        
        zoning_df.rename(columns={'ZONING_LABEL' : 'zone_code'}, inplace=True)
        
        land_use = zoning_df[['zone_code', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_code'].apply(lambda x: zone_code_transform(data.city, x))
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['zone_type'] = df['zone_code'].apply(lambda x: zone_code_transform(data.city, x))
        
        
        DC_census_df = pd.read_csv('./data/other_data/washDC_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        VA_census_df = pd.read_csv('./data/other_data/VA_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df = pd.concat([DC_census_df, VA_census_df])
        
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        DC_CTracts_df = gpd.read_file('./data/other_data/washDC_CT_data.shp', bbox=bbox)
        DC_CTracts_df = DC_CTracts_df[['GEOID', 'geometry']]
        
        VA_CTracts_df = gpd.read_file('./data/other_data/VA_CT_data.shp', bbox=bbox)
        VA_CTracts_df = VA_CTracts_df[['GEOID', 'geometry']]
        
        CTracts_df = DC_CTracts_df.append(VA_CTracts_df)
        CTracts_df = CTracts_df.rename({'GEOID' : 'GEO_ID'}, axis=1)
        
        census_df = gpd.GeoDataFrame(census_df.merge(CTracts_df, on='GEO_ID'),
                                     geometry='geometry', crs='EPSG:4326')
        
        census_df.to_crs(epsg=3857, inplace=True)
        census_df['census_area'] = census_df['geometry'].area/1000000
        census_df.to_crs(epsg=4326, inplace=True)
        census_df['pop_density'] = census_df['P1_001N'] / census_df['census_area']
        census_df.rename(columns=dataframe_key.get_census_key(data.city), inplace=True)
        
        
        df = gpd.tools.sjoin(df, census_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        subways_df = gpd.read_file('./data/other_data/washDC_subways_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    
    elif data.city == 'minn':
        
        bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        
        zoning_df = gpd.read_file('./data/other_data/minn_zoning_data.geojson', bbox=bbox)
        zoning_df = zoning_df[['ZONE_CODE', 'geometry']]
        
        land_use = zoning_df[['ZONE_CODE', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_code_transform(data.city, x))
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['zone_type'] = df['ZONE_CODE'].apply(lambda x: zone_code_transform(data.city, x))
        
        census_df = pd.read_csv('./data/other_data/minn_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        
        bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        
        CTracts_df = gpd.read_file('./data/other_data/minn_CT_data.shp', bbox=bbox)
        CTracts_df.set_crs(epsg=4326, allow_override=True, inplace=True)
        CTracts_df = CTracts_df[['GEOID', 'geometry']]
        CTracts_df = CTracts_df.rename({'GEOID' : 'GEO_ID'}, axis=1)
        
        census_df = gpd.GeoDataFrame(census_df.merge(CTracts_df, on='GEO_ID'),
                                     geometry='geometry', crs='EPSG:4326')
        
        census_df.to_crs(epsg=3857, inplace=True)
        census_df['census_area'] = census_df['geometry'].area/1000000
        census_df.to_crs(epsg=4326, inplace=True)
        census_df['pop_density'] = census_df['P1_001N'] / census_df['census_area']
        
        census_df.rename(columns=dataframe_key.get_census_key(data.city), inplace=True)
        
        df = gpd.tools.sjoin(df, census_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        subways_df = gpd.read_file('./data/other_data/minn_subways_data.geojson')
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    
    elif data.city == 'boston':
        
        # bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        
        zoning_boston = gpd.read_file('./data/other_data/boston_zoning_data.geojson')
        zoning_boston = zoning_boston[['ZONE_', 'Zone_Desc','geometry']]
        zoning_boston['zone_type'] = zoning_boston['Zone_Desc'].apply(lambda x: zone_code_transform(data.city, x))
        zoning_boston = zoning_boston[['ZONE_', 'zone_type', 'geometry']]
        
        
        zoning_cambridge = gpd.read_file('./data/other_data/Cambridge_zoning_data.shp')
        zoning_cambridge = zoning_cambridge[['ZONE_TYPE', 'geometry']]
        zoning_cambridge = zoning_cambridge.rename({'ZONE_TYPE' : 'ZONE_'}, axis=1)

        zoning_cambridge['zone_type'] = zoning_cambridge['ZONE_'].apply(lambda x: zone_code_transform(data.city, x))
        crs = zoning_cambridge.crs
        zoning_cambridge.to_crs(epsg=4326, inplace=True)
        
        zoning_brookline = gpd.read_file('./data/other_data/brookline_zoning_data.geojson')
        zoning_brookline = zoning_brookline[['ZONECLASS', 'ZONEDESC' ,'geometry']]
        zoning_brookline['zone_type'] = zoning_brookline['ZONEDESC'].apply(lambda x: zone_code_transform(data.city, x))
        zoning_brookline = zoning_brookline.rename({'ZONECLASS' : 'ZONE_'}, axis=1)
        zoning_brookline = zoning_brookline[['ZONE_', 'zone_type', 'geometry']]
        
        # bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857).to_crs(crs)
        
        zoning_somerville = gpd.read_file('./data/other_data/somerville_zoning_data.shp')
        zoning_somerville.set_crs(crs, inplace=True)
        zoning_somerville.to_crs(epsg=4326, inplace=True)
        zoning_somerville = zoning_somerville[['ZoneCode', 'geometry']]
        zoning_somerville['zone_type'] = zoning_somerville['ZoneCode'].apply(lambda x: zone_code_transform(data.city, x))
        zoning_somerville = zoning_somerville.rename({'ZoneCode' : 'ZONE_'}, axis=1)
        
        zoning_df = gpd.GeoDataFrame(pd.concat([zoning_boston, 
                                                zoning_cambridge,
                                                zoning_brookline,
                                                zoning_somerville], ignore_index=True),
                                     geometry='geometry', crs='EPSG:4326')
        
        # Transform splitted zones
        
        zoning_df.loc[566, 'geometry'] = zoning_df.loc[566, 'geometry'].difference(
            shapely.ops.unary_union([zoning_df.loc[777, 'geometry'],
                                     zoning_df.loc[757, 'geometry'],
                                     zoning_df.loc[762, 'geometry'],
                                     zoning_df.loc[775, 'geometry']])) 
                                     
        zoning_df.loc[569, 'geometry'] = zoning_df.loc[569, 'geometry'].difference(
            shapely.ops.unary_union([zoning_df.loc[766, 'geometry'],
                                     zoning_df.loc[772, 'geometry'],
                                     zoning_df.loc[769, 'geometry'],
                                     zoning_df.loc[773, 'geometry'],
                                     zoning_df.loc[774, 'geometry'],
                                     zoning_df.loc[768, 'geometry']])) 
            
        
        zoning_df.drop([770, 761, 570, 568], inplace=True) # drop redundant zones
        
        # zoning_df['zone_type'] = zoning_df['zone_type'].apply(lambda x: x if pd.notnull(x) else 'UNKNOWN')
        
        land_use = zoning_df[['ZONE_', 'zone_type', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        
        land_use.to_crs(epsg=3857, inplace=True)
        land_use = land_use.cx[df['easting'].min()-1000:df['easting'].max()+1000,
                                df['northing'].min()-1000:df['northing'].max()+1000]
        
        land_use['geometry'] = land_use['geometry'].apply(lambda area: area.buffer(0).intersection(land_use_extent))
        land_use.to_crs(epsg=4326, inplace=True)
        
        df = gpd.GeoDataFrame(df, geometry='coords', crs=zoning_df.crs)
        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left') 
        df.drop('index_right', axis=1, inplace=True)
    
        # df['zone_type'] = df['ZONE_CODE'].apply(lambda x: zone_code_transform(data.city, x))
        
        census_df = pd.read_csv('./data/other_data/boston_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        
        bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
        
        CTracts_df = gpd.read_file('./data/other_data/boston_CT_data.shp', bbox=bbox)
        CTracts_df.set_crs(epsg=4326, allow_override=True, inplace=True)
        CTracts_df = CTracts_df[['GEOID', 'geometry']]
        CTracts_df = CTracts_df.rename({'GEOID' : 'GEO_ID'}, axis=1)
        
        # CTracts_df.to_crs(epsg=3857, inplace=True)
        # CTracts_df = CTracts_df.cx[df['easting'].min()-1000:df['easting'].max()+1000,
        #                            df['northing'].min()-1000:df['northing'].max()+1000]
        
        # CTracts_df['geometry'] = CTracts_df['geometry'].apply(lambda area: area.buffer(0).intersection(land_use_extent))
        # CTracts_df.to_crs(epsg=4326, inplace=True)
        
        
        census_df = gpd.GeoDataFrame(census_df.merge(CTracts_df, on='GEO_ID'),
                                     geometry='geometry', crs='EPSG:4326')
        
        census_df.to_crs(epsg=3857, inplace=True)
        census_df['census_area'] = census_df['geometry'].area/1000000
        census_df.to_crs(epsg=4326, inplace=True)
        census_df['pop_density'] = census_df['P1_001N'] / census_df['census_area']
        
        census_df.rename(columns=dataframe_key.get_census_key(data.city), inplace=True)
        
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
        # df['pop_density'] = df['P1_001N'] / df['census_area']
        
        
        subways_df = gpd.read_file('./data/other_data/boston_subways_data.shp').to_crs(epsg = 4326)
        
        df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
    
    else:
        
        df['population'] = 0
        df['pop_density'] = 0
        df['zone_type'] = 0
        land_use = gpd.GeoDataFrame([])
        land_use.set_geometry([], inplace=True)
        land_use.set_crs(epsg=4326, inplace=True)
        land_use['zone_type'] = 'UNKNOWN'
    
    print(".")
        
    df.rename(mapper=df_key(data.city), axis=1, inplace=True)  
    df['zone_type'] = df['zone_type'].apply(lambda x: x if pd.notnull(x) else 'UNKNOWN')
    
    land_use['color'] = land_use['zone_type'].map(color_dict).fillna("pink")
    
    with open(f'./python_variables/station_df_{data.city}{data.year:d}{postfix}.pickle', 'wb') as file:
        pickle.dump([df, land_use, census_df], file)
    
    
    try:
        with open(f'./python_variables/neighborhoods_{data.city}{data.year}.pickle', 'rb') as file:
            neighborhoods = pickle.load(file)
    
    except FileNotFoundError:
        print(f'No neighborhoods found. Pickling neighborhoods using {data.city}{data.year} data...')
        neighborhoods = make_neighborhoods(data.city, data.year, df, land_use)
    
    df = df.merge(neighborhoods, on='stat_id')
    
    print("Done")
    
    if return_land_use and not return_census:
        return df, land_use
    elif return_census and not return_land_use:
        return df, census_df
    elif return_land_use and return_census:
        return df, land_use, census_df
    
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
        means = clusters.cluster_centers_
        station_df, means, labels = sort_clusters(station_df, means, labels, traffic_matrices, day_type, k)
        clusters.cluster_centers_ = means
        station_df['color'] = station_df['label'].map(cluster_color_dict)

    elif clustering == 'k_medoids':
        clusters = KMedoids(k, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        means = clusters.cluster_centers_
        station_df, means, labels = sort_clusters(station_df, means, labels, traffic_matrices, day_type, k)
        clusters.cluster_centers_ = means
        station_df['color'] = station_df['label'].map(cluster_color_dict)
        
    elif clustering == 'h_clustering':
        clusters = None
        labels = AgglomerativeClustering(k).fit_predict(traffic_matrix)
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        means = cluster_mean(traffic_matrix, station_df, labels, k)
        station_df, means, labels = sort_clusters(station_df, means, labels, traffic_matrices, day_type, k)
        clusters = means
        station_df['color'] = station_df['label'].map(cluster_color_dict)
    
    elif clustering == 'gaussian_mixture':
        clusters = GaussianMixture(k, n_init=10, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict_proba(traffic_matrix)

        station_df.loc[mask, 'label'] = pd.Series(list(labels), index=mask[mask].index)
        station_df.loc[~mask, 'label'] = np.nan
        means = clusters.means_
        station_df, means, labels = sort_clusters(station_df, means, labels, traffic_matrices, day_type, k, cluster_type='gaussian_mixture', mask=mask)
        clusters.means_ = means
        lab_mat = np.array(lab_color_list[:k]).T
        lab_cols = [np.sum(labels[i] * lab_mat, axis=1) for i in range(len(traffic_matrix))]
        labels_rgb = skcolor.lab2rgb(lab_cols)
        station_df.loc[mask, 'color'] = ['#%02x%02x%02x' % tuple(label.astype(int)) for label in labels_rgb*255]
        station_df.loc[~mask, 'color'] = 'gray'
        station_df['label'].loc[mask] = [np.argmax(x) for x in station_df['label'].loc[mask]]
        
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


def sort_clusters(station_df, cluster_means, labels, traffic_matrices, day_type, k, cluster_type=None, mask=None):
    # Order the clusters by setting cluster 0 to be closest to the mean traffic.
    
    if day_type == 'business_days':
        mean = np.mean(traffic_matrices[0][station_df.index], axis=0)
    elif day_type == 'weekend':
        mean = np.mean(traffic_matrices[1][station_df.index], axis=0)
    
    morning_hours = np.array([6,7,8,9,10])
    afternoon_hours = np.array([15,16,17,18,19])
    
    #mean = mean/np.max(mean)
    
    peakiness = [] # Distance to mean
    rush_houriness = [] # difference between arrivals and departures
    for center in cluster_means:
        dist_from_mean = np.linalg.norm(center-mean)
        peakiness.append(dist_from_mean)
        rhn = np.sum((center[morning_hours] - center[morning_hours+24]) - (center[afternoon_hours] - center[afternoon_hours+24]))
        rush_houriness.append(rhn)
    
    rush_houriness = np.array(rush_houriness)
    first = np.argmin(np.array(peakiness)*0.5 + np.abs(rush_houriness))
    order = np.argsort(rush_houriness)
    
    order_new = order.copy()
    for i, item in enumerate(order):
        if item == first:
            order_new[i] = order[0]
            order_new[0] = first
    order = order_new
    
    # print(f"first = {first}")
    # print(order)
    # print(f"rush-houriness = {rush_houriness}")
    # print(f"peakiness = {peakiness}")
    
    # for i in range(k):
    #     if abs(rush_houriness[i]) < 0.05:
    #         print(f"small_rushouriness {i}")
    #     if peakiness[i] > 0.1:
    #         print(f'large peakiness {i}')
    
    # for i in range(k):
    #     if abs(rush_houriness[i]) < 0.05 and peakiness[i] > 0.1:
    #         temp = order[i]
    #         order[i] = order[-1]
    #         order[-1] = temp
    #         print(f'swapped {order[-1]} for {order[i]}')
    # print(labels[0:2])
    
        
    
    labels_dict = dict(zip(order, range(len(order))))
    print(labels_dict)
    
    if cluster_type == 'gaussian_mixture':
        values = np.zeros_like(labels)
        values[:,order] = labels[:,range(k)]
        
        station_df['label'].loc[mask] = pd.Series(list(values), index=mask[mask].index)
    else:
        station_df = station_df.replace({'label' : labels_dict})
    labels = station_df['label']
    
    centers = np.zeros_like(cluster_means)
    for i in range(k):
        centers[labels_dict[i]] = cluster_means[i]
    cluster_means = centers
    return station_df, cluster_means, labels


def cluster_mean(traffic_matrix, station_df, labels, k):
    mean_vector = np.zeros((k, traffic_matrix.shape[1]))
    for j in range(k):
        mean_vector[j,:] = np.mean(traffic_matrix[np.where(labels == j)], axis=0)
    return mean_vector


def service_areas(city, station_df, land_use, service_radius=500, use_road=False):
    t_start = time.time()
    if 'service_area' in station_df.columns:
        station_df.drop(columns='service_area', inplace=True)
        station_df.set_geometry('coords', inplace=True)
    
    points = station_df[['easting', 'northing']].to_numpy()
    points_gdf = gpd.GeoDataFrame(geometry = [Point(station_df.iloc[i]['easting'], 
                                                  station_df.iloc[i]['northing'])
                           for i in range(len(station_df))], crs='EPSG:3857')

    points_gdf['point'] = points_gdf['geometry']
    
    mean_point= np.mean(points, axis=0)
    edge_dist = 1000000
    edge_points = np.array([[mean_point[0]-edge_dist, mean_point[1]-edge_dist],
                            [mean_point[0]-edge_dist, mean_point[1]+edge_dist],
                            [mean_point[0]+edge_dist, mean_point[1]+edge_dist],
                            [mean_point[0]+edge_dist, mean_point[1]-edge_dist]])

    vor = Voronoi(np.concatenate([points, edge_points], axis=0))
    
    lines = [LineString(vor.vertices[line])
        for line in vor.ridge_vertices
        if -1 not in line]
    
    poly_gdf = gpd.GeoDataFrame()
    poly_gdf['vor_poly'] = [poly for poly in shapely.ops.polygonize(lines)]
    poly_gdf['geometry'] = poly_gdf['vor_poly']
    poly_gdf.set_crs(epsg=3857, inplace=True)

    poly_gdf = gpd.tools.sjoin(points_gdf, poly_gdf, op='within', how='left')
    poly_gdf.drop('index_right', axis=1, inplace=True)
    poly_gdf['geometry'] = poly_gdf['vor_poly']
    
    buffers = poly_gdf['point'].buffer(service_radius)
    
    poly_gdf['service_area'] = poly_gdf.intersection(buffers)

    poly_gdf['geometry'] = poly_gdf['service_area']
    poly_gdf.set_crs(epsg=3857, inplace=True)
    poly_gdf.to_crs(epsg=4326, inplace=True)
    poly_gdf['service_area'] = poly_gdf['geometry']
    
    station_df = gpd.tools.sjoin(station_df, poly_gdf, op='within', how='left')
    station_df.drop(columns=['index_right', 'vor_poly', 'point'], inplace=True)
    
    
    union = land_use_union(city, land_use)
    
    
    station_df['service_area'] = station_df['service_area'].apply(lambda area: area.intersection(union) if area else shapely.geometry.Polygon())
    
    service_area_trim = []
    for i, row in station_df.iterrows():
        # if isinstance(row['service_area'], shapely.geometry.multipolygon.MultiPolygon):
        #     count=1
        #     for poly in row['service_area'].geoms:

        #         if poly.contains(row['coords']):
        #             service_area_trim.append(poly)
                # else:
                #     service_area_trim.append(row['service_area']) # hotfix, find better solution
                    
                #     if count != len(row['service_area']):
                #         service_area_trim = service_area_trim[:-1]
                # count+=1

        if isinstance(row['service_area'], shapely.geometry.collection.GeometryCollection):
            service_area_trim.append(shapely.ops.unary_union(row['service_area']))
        
        else:
            service_area_trim.append(row['service_area'])
    
    station_df['service_area'] = service_area_trim
    station_df.set_geometry('service_area', inplace=True)
    

    # station_df['service_area'] = station_df['service_area'].to_crs(epsg=3857)
    # land_use['geometry'] = land_use['geometry'].to_crs(epsg=3857)
    
    if 'road' in station_df['zone_type'].unique() and use_road == 'False': # Ignore road

        station_df['service_area_no_road'] = [
            stat['service_area'].difference(stat['neighborhood_road'])
            for i, stat in station_df.iterrows()] # ShapelyDeprecationWarning for 2.0 here. Probably needs geopandas to fix it

        zone_types = station_df['zone_type'].unique()[
            station_df['zone_type'].unique() != 'road']

        
        geo_sdf = station_df.set_geometry('service_area_no_road')
        geo_sdf.to_crs(epsg=3857, inplace=True)
        
        geo_sdf.geometry = geo_sdf.buffer(0)

        for zone_type in zone_types:
        
            zone_percents = np.zeros(len(station_df))
            
            mask = ~station_df[f'neighborhood_{zone_type}'].is_empty
            
            sdf_zone = geo_sdf[f'neighborhood_{zone_type}'].to_crs(epsg=3857)[mask]
            
            print(zone_type)
            zone_percents[mask] = geo_sdf[mask].intersection(sdf_zone).area/geo_sdf[mask].area
            
            station_df[f'percent_{zone_type}'] = zone_percents
        
    else:
            
        for zone_type in station_df['zone_type'].unique(): #This is where all the time goes
            
            zone_percents = np.zeros(len(station_df))
            
            for i, stat in station_df.iterrows():
                
                if stat['service_area']:
                    
                    if stat[f'neighborhood_{zone_type}']:
                        
                        zone_percents[i] = stat['service_area'].buffer(0).intersection(stat[f'neighborhood_{zone_type}']).area/stat['service_area'].area
                    
                else:
                    zone_percents[i] = np.nan
                    
            station_df[f'percent_{zone_type}'] = zone_percents

    station_df['service_area_size'] = station_df['service_area'].apply(lambda area: area.area/1000000)
    
    station_df['service_area'] = station_df['service_area'].to_crs(epsg=4326)
    land_use['geometry'] = land_use['geometry'].to_crs(epsg=4326)
    
    station_df['service_area'] = station_df['service_area'].apply(
        lambda poly: Point(0,0).buffer(0.0001) if poly.area==0 else poly)
    
    # serv = gpd.GeoSeries(station_df.service_area)
    # mask = serv.area == 0
    # station_df = station_df[~mask]
    
    station_df['geometry'] = station_df['service_area']
    print(f"total time on service areas spent = {time.time()-t_start:.2f}")
    return station_df


def land_use_union(city, land_use):
    try:
        with open(f'./python_variables/union_{city}.pickle', 'rb') as file:
            union = pickle.load(file)
    except FileNotFoundError:
        print(f'No union for {city} found. Pickling union...')
        land_use.to_crs(epsg=3857, inplace=True)
        union = shapely.ops.unary_union(land_use.geometry)
        union_gpd = gpd.GeoSeries(union)
        union_gpd.set_crs(epsg=3857, inplace=True)
        union_gpd = union_gpd.to_crs(epsg=4326)
        union = union_gpd.loc[0].buffer(0)
        land_use.to_crs(epsg=4326, inplace=True)
        with open(f'./python_variables/union_{city}.pickle', 'wb') as file:
            pickle.dump(union, file)
        print('Pickling done')
    
    return union


def stations_logistic_regression(station_df, zone_columns, other_columns, 
                                 use_points_or_percents='points', 
                                 make_points_by='station location', 
                                 const=False, test_model=False, test_ratio=0.2, 
                                 test_seed=None, plot_cm=False, 
                                 normalise_cm=None):
    df = station_df[~station_df['label'].isna()]
    
    if len(df) == 0:
        raise ValueError("station_df['label'] is empty")
    
    X = df[zone_columns]
    
    p_columns = [column for column in X.columns 
                  if 'percent_' in column]
    
    nop_columns = [col[8:] for col in p_columns]
    
    if use_points_or_percents == 'points':
        
        if make_points_by == 'station location':
            X = pd.get_dummies(df['zone_type'])
            
            
        elif make_points_by == 'station land use':
            
            p_df = df[p_columns]
            
            # get the zonetype with the largest percentage
            zone_types = [p_df.iloc[i].index[p_df.iloc[i].argmax()][8:] 
                          for i in range(len(p_df))]
    
            df['zone_type_by_percent'] = zone_types
        
            X = pd.get_dummies(df['zone_type_by_percent'])
        
        else:
            print('"station location" or "station land use"')
        
        X = X[nop_columns]
    
    elif use_points_or_percents == 'percents':
        X = df[p_columns]
    
    for column in other_columns:
        X[column] = df[column]
    
    if const:
        X = sm.add_constant(X)
    
    y = df['label'][~X.isna().any(axis=1)]

    X = X[~X.isna().any(axis=1)]
    
    X_scaled = X.copy()
    if 'n_trips' in X_scaled.columns:
        X_scaled['n_trips'] = X_scaled['n_trips']/X_scaled['n_trips'].sum()
    if 'nearest_subway_dist' in X_scaled.columns:
        X_scaled['nearest_subway_dist'] = X_scaled['nearest_subway_dist']/1000
    
    param_names = {'percent_industrial' : '% industrial',
                   'percent_commercial' : '% commercial',
                   'percent_residential' : '% residential',
                   'percent_recreational' : '% recreational',
                   'percent_mixed' : '% mixed',
                   'pop_density' : 'pop density',
                   'nearest_subway_dist' : 'nearest subway dist'}
    
    X_scaled = X_scaled.rename(param_names)
    
    if test_model:
        
        if test_ratio < 0 or test_ratio > 1:
            raise ValueError("test_ratio must be between 0 and 1")
        
        
        if test_seed:
            if isinstance(test_seed, int):
                np.random.seed(test_seed)
            else:
                raise ValueError("test_seed must be an integer")
        
        mask = np.random.rand(len(X_scaled)) < test_ratio
        
        X_test = X_scaled[mask]
        y_test = y[mask]
        
        X_train = X_scaled[~mask]
        y_train = y[~mask]
        
        success_rate, cm, predictions = logistic_regression_test(X_train, y_train, 
                                                    X_test, y_test,
                                                    plot_cm=plot_cm,
                                                    normalise_cm=normalise_cm)
    
    LR_model = MNLogit(y, X_scaled.rename(param_names))
    
    try:
        LR_results = LR_model.fit_regularized(maxiter=10000)
        #print(LR_model.loglikeobs(LR_results.params.to_numpy()))
    except np.linalg.LinAlgError:
        print("Singular matrix")
        LR_results = None
    
    if test_model:
        return LR_results, X, y, predictions
    else:
        return LR_results, X, y

def logistic_regression_test(X_train, y_train, X_test, y_test, plot_cm=True, normalise_cm=None):
    
    if (X_train_set := set(X_train.columns.to_list())) != (X_test_set := set(X_test.columns.to_list())):
        print(f'WARNING: One or more columns are not shared between X_train and X_test and will be omitted:\n{X_train_set.symmetric_difference(X_test_set)}\n')
        
        columns_shared = list(X_train_set.intersection(X_test_set))
        
        X_train = X_train[columns_shared]
        X_test = X_test[columns_shared]
    
    try:
        LR_train_res = MNLogit(y_train, X_train).fit_regularized(maxiter=10000, disp=0)
    except np.linalg.LinAlgError:
          print("Singular matrix. Test aborted.")
          return None, None
    
    predictions = LR_train_res.predict(X_test)
    predictions['label'] = predictions.idxmax(axis=1)
    
    success_rate = (y_test == predictions['label']).sum(0)/len(y_test)
    cm = confusion_matrix(y_test, predictions['label'], normalize=normalise_cm)
    
    print(f"\nTest completed. Success rate: {success_rate*100:.2f}%\n")
    
    if plot_cm:
        
        if normalise_cm == 'true':
            title='Normalised wrt. true label'
        
        if normalise_cm == 'pred':
            title='Normalised wrt. predicted label'
        
        if normalise_cm == 'all':
            title='Normalised wrt. number of predictions'
        
        
        
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(title)

    return success_rate, cm, predictions



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


def big_station_df(cities, year=2019, month=None, service_radius=500,
                   use_road=False, day_type='business_days',
                   min_trips=100, clustering='k_means', k=3,
                   random_state=42):
    station_df_list = []
    traffic_matrix_b_list = []
    traffic_matrix_w_list = []
    for city in cities:
        data = bs.Data(city, year, month)
        station_df, land_use = make_station_df(data, holidays=False, return_land_use=True)
        traffic_matrices = data.pickle_daily_traffic(holidays=False)

        station_df = service_areas(data.city, station_df, land_use, service_radius=service_radius, use_road=use_road)
        
        station_df_list.append(pd.concat({city: station_df}))
        traffic_matrix_b_list.append(traffic_matrices[0])
        traffic_matrix_w_list.append(traffic_matrices[1])


    big_station_df = pd.concat(station_df_list)
    big_station_df = big_station_df.reset_index()

    big_tm_b = np.concatenate(traffic_matrix_b_list)
    big_tm_w = np.concatenate(traffic_matrix_w_list)
    
    traffic_matrices = [big_tm_b, big_tm_w]

    big_station_df, clusters, labels = get_clusters(
        traffic_matrices, big_station_df, day_type, min_trips, 
        clustering, 
        k, 
        random_state=random_state)
    
    return big_station_df, traffic_matrices, labels

def make_station_df_year(city, year=2019, months=None, service_radius=500,
                   use_road=False, day_type='business_days',
                   min_trips=100, clustering='k_means', k=3,
                   random_state=42):
    
    month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
          7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    station_df_list = []
    traffic_matrix_b_list = []
    traffic_matrix_w_list = []
    
    if not months:
        months = bs.get_valid_months(city, year)
    elif isinstance(months, list):
        for month in months:
            if month not in bs.get_valid_months(city, year):
                raise ValueError(f'month {month} is not a valid month for {city}')
    else:
        raise ValueError('months must be an iterable')
    
    for month in months:
        data = bs.Data(city, year, month)
        station_df, land_use = make_station_df(data, holidays=False, return_land_use=True)
        traffic_matrices = data.pickle_daily_traffic(holidays=False)

        station_df = service_areas(data.city, station_df, land_use, service_radius=service_radius, use_road=use_road)
        station_df[month_dict[month]] = 1
        
        station_df_list.append(pd.concat({city: station_df}))
        traffic_matrix_b_list.append(traffic_matrices[0])
        traffic_matrix_w_list.append(traffic_matrices[1])


    big_station_df = pd.concat(station_df_list)
    big_station_df = big_station_df.reset_index()
    
    for month in month_dict.values():
        big_station_df[month] = big_station_df[month].fillna(0)
    
    big_tm_b = np.concatenate(traffic_matrix_b_list)
    big_tm_w = np.concatenate(traffic_matrix_w_list)
    
    traffic_matrices = [big_tm_b, big_tm_w]

    big_station_df, clusters, labels = get_clusters(
        traffic_matrices, big_station_df, day_type, min_trips, 
        clustering, 
        k, 
        random_state=random_state)
    
    return big_station_df, traffic_matrices, labels


def create_all_pickles(city, year, holidays=False, overwrite=False):
    if isinstance(city, str): # If city is a str (therefore not a list)
        data = bs.Data(city, year, overwrite=overwrite)
        land_use = make_station_df(data, return_land_use=True, holidays=holidays, overwrite=overwrite)[1]
        
        if overwrite:
            
            print(f'Pickling union for {city}...', end='')
            land_use.to_crs(epsg=3857, inplace=True)
            union = shapely.ops.unary_union(land_use.geometry)
            union_gpd = gpd.GeoSeries(union)
            union_gpd.set_crs(epsg=3857, inplace=True)
            union_gpd = union_gpd.to_crs(epsg=4326)
            union = union_gpd.loc[0].buffer(0)
            land_use.to_crs(epsg=4326, inplace=True)
            with open(f'./python_variables/union_{city}.pickle', 'wb') as file:
                pickle.dump(union, file)
            print('Done')
            
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
    'industrial': mpl_colors.to_hex('tab:red'), # 3
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
    'industrial': 3, # 3
    'mixed': 0, # 0
    'educational': 5, # 5
    'UNKNOWN': 7 # 7
    }

    
if __name__ == "__main__":

    

    # create_all_pickles('boston', 2019, overwrite=True)

    data = bs.Data('nyc', 2019, 9)

    pre = time.time()
    traffic_matrices = data.pickle_daily_traffic(holidays=False, normalise=False, overwrite=True)
    station_df, land_use, census_df = make_station_df(data, return_land_use=True, return_census=True, overwrite=True)
    print(f'station_df took {time.time() - pre:.2f} seconds')

    
    # for i, station in station_df.iterrows():
    #     a = geodesic_point_buffer(station['lat'], station['long'], 1000)
    
    
    # overlaps = []
    
    # for i in range(len(land_use)):
    #     for j in range(i+1, len(land_use)):
    #         if land_use.iloc[i].geometry.intersection(land_use.iloc[j].geometry).area > 0:
    #             overlap_perc = land_use.iloc[i].geometry.intersection(land_use.iloc[j].geometry).area/land_use.iloc[i].geometry.union(land_use.iloc[j].geometry).area*100
    #             print(f'Zone {land_use.iloc[i].name} overlaps with zone {land_use.iloc[j].name}. Pecentage overlap: {overlap_perc:.2f}%')
    #             overlaps.append([overlap_perc,(land_use.iloc[i].name, land_use.iloc[j].name)])

    # overlaps2 = []
    
    # for i in range(len(poly_gdf)):
    #     for j in range(i+1, len(poly_gdf)):
    #         if poly_gdf.iloc[i].geometry.intersection(poly_gdf.iloc[j].geometry).area > 0:
    #             overlap_perc = poly_gdf.iloc[i].geometry.intersection(poly_gdf.iloc[j].geometry).area/poly_gdf.iloc[i].geometry.union(poly_gdf.iloc[j].geometry).area*100
    #             print(f'Zone {poly_gdf.iloc[i].name} overlaps with zone {poly_gdf.iloc[j].name}. Pecentage overlap: {overlap_perc:.2f}%')
    #             overlaps2.append([overlap_perc,(poly_gdf.iloc[i].name, poly_gdf.iloc[j].name)])

    

