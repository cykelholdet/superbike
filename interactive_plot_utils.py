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

import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
import shapely.ops
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import skimage.color as skcolor
import statsmodels.api as sm

from holoviews.util.transform import lon_lat_to_easting_northing
from shapely.geometry import Point, Polygon, LineString
from geopy.distance import great_circle
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import BallTree
from sklearn_extra.cluster import KMedoids
from statsmodels.discrete.discrete_model import MNLogit
from scipy.spatial import Voronoi

import bikeshare as bs
import dataframe_key

def census_key(city):
    """
    Dictionary for renaming census data columns

    Parameters
    ----------
    city : str
        the city of interest.

    Returns
    -------
    key : dict
        maps census data columns to common format.

    """
    
    if city == 'nyc':
        key = {'index' : 'stat_id',
               'ZONEDIST' : 'zone_code',
               'BoroCT2020' : 'census_tract',
               'Shape__Area' : 'census_area',
               'Pop_20' : 'population'}
    
    elif city == 'chicago':
        key = {'index' : 'stat_id',
               'zone_class' : 'zone_code',
               'geoid10' : 'census_block',
               'TOTAL POPULATION' : 'population'}
    
    elif city == 'washdc':
        key = {'ZONING LABEL' : 'zone_code',
               'GEO_ID' : 'census_tract',
               'B01001_001E' : 'population'}
    
    elif city == 'minneapolis':
        key = {'ZONE_CODE' : 'zone_code',
               'GEOID20' : 'census_tract',
               'ALAND20' : 'census_area'}
    
    elif city == 'boston':
        key = {'ZONE_' : 'zone_code',
               'GEO_ID' : 'census_tract',
               'P1_001N' : 'population'}

    else:
        key = {}  # No renaming of the columns.
    
    return key
    
    
def zone_code_transform(city, zone_code):
    """
    map a land use code to a common format.

    Parameters
    ----------
    city : str
        the city of interest.
    zone_code : str
        name of the zone type.

    Returns
    -------
    zone_type : str
        the corresponding name in the common format.

    """
    
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
        
        elif city in ['madrid', 'helsinki', 'london', 'oslo', 'bergen',
                      'trondheim', 'edinburgh']:
            
            if zone_code in ['11100', '11210', '11220', '11230', '11240']: # Continuous urban fabric (S.L. : > 80%), Discontinuous dense urban fabric (S.L. : 50% -  80%), Discontinuous medium density urban fabric (S.L. : 30% - 50%), Discontinuous low density urban fabric (S.L. : 10% - 30%), Discontinuous very low density urban fabric (S.L. < 10%)
                zone_type = 'residential' 
            elif zone_code in ['12220', '12210']: # Other roads and associated land, Fast transit roads and associated land
                zone_type = 'road'
            elif zone_code in ['12100']: # Industrial, commercial, public, military and private units
                zone_type = 'commercial'
            elif zone_code in ['14100', '14200', '31000', '32000']: # Green urban areas, Sports and leisure facilities, Forests, Herbaceous vegetation associations (natural grassland, moors...)
                zone_type = 'recreational'
            elif zone_code in ['12230']: # Railways and associated land
                zone_type = 'transportation'
            elif zone_code in ['13100', '12300']: # Mineral extraction and dump sites, Port areas
                zone_type = 'industrial'
            elif zone_code in ['50000']:
                zone_type = 'water'
            else:
                zone_type = 'UNKNOWN'
        
        elif city == 'chicago':
            
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
        
        elif city == 'washdc':
            
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
        
        elif city == 'minneapolis':
            
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
    """
    Determine which land use/zoning polygons are near each station in station_df.
    The neighborhoods are determined at a 1000 m radius.

    Parameters
    ----------
    city : str
        the city of interest.
    year : int
        the year.
    station_df : pandas DataFrame
        contains information about the stations.
    land_use : geopandas GeoDataFrame
        polygons for land use.

    Returns
    -------
    neighborhoods : GeoDataFrame
        contains polygons in the neighborhood of station.

    """
    pre = time.time()
    print(f"Determining neighborhoods for {bs.name_dict[city]} {year}")
    data_year = bs.Data(city, year)
    neighborhoods = gpd.GeoDataFrame(index = list(data_year.stat.inverse.keys()))
    neighborhoods['stat_id'] = list(data_year.stat.id_index.keys())
    neighborhoods['coords'] = [Point(data_year.stat.locations.loc[i, 'long'],
                                     data_year.stat.locations.loc[i, 'lat']) 
                               for i in neighborhoods.index]
    neighborhoods.set_geometry('coords', inplace=True)
    print("Getting the hood ready to go", end="")
    if len(land_use) > 0:
        
        # Merge the polygons of each different type of land use.
        lu_merge = {}
        for zone_type in land_use['zone_type'].unique():
            lu_merge[zone_type] = land_use[land_use['zone_type'] == zone_type].unary_union
            print(".", end="")
        print(" ", end="")
        
        # Add buffer
        buffers = neighborhoods['coords'].apply(
            lambda coord: geodesic_point_buffer(coord.y, coord.x, 1000))
        
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
    
    else:
        neighborhoods.drop(columns=['coords'], inplace=True)
        neighborhoods['neighborhood_UNKNOWN'] = None
    
    return neighborhoods
    

def make_station_df(data, holidays=True, return_land_use=False, 
                    return_census=False, overwrite=False):
    """
    Get a pandas DataFrame with the stations of the bikeshare network as rows.
    The columns contain varoius information about the station.

    Parameters
    ----------
    data : bikeshare Data object.
        Containing data for a year, month or day.
    holidays : bool, optional
        Whether to include holidays (True) or remove holidays (False). The 
        default is True.
    return_land_use : bool, optional
        If True: returns land use as GeoDataFrame. The default is False.
    return_census : bool, optional
        If True: returns census DataFrame. The default is False.
    overwrite : bool, optional
        If True: does not load data from pickle regardless if it exists
        already. Overwrites pickle if it does exist. The default is False.

    Returns
    -------
    Always returns
    
    pandas DataFrame
        the station DataFrame.
    
    Depending on return_land_use and return census
    
    pandas DataFrame
        Land use DataFrame.
    pandas DataFrame
        Census DataFrame

    """
    # Postfix to the pickle filename.
    postfix = "" if data.month == None else f"{data.month:02d}"
    postfix = postfix + "" if holidays else postfix + "_no_holidays"
    
    # If the data only contains a day we don't save a pickle. If overwrite is
    # True, we create a new pickle regardless of whether it exists.
    if (not overwrite) and (data.day == None): 
        try:
            with open(f'./python_variables/station_df_{data.city}{data.year:d}{postfix}.pickle', 'rb') as file:
                df, land_use, census_df = pickle.load(file)
            
            try:
                with open(f'./python_variables/neighborhoods_{data.city}{data.year}.pickle', 'rb') as file:
                    neighborhoods = pickle.load(file)
            
            except FileNotFoundError:
                print(f'No neighborhoods found. Pickling neighborhoods using {data.city}{data.year} data...')
                neighborhoods = make_neighborhoods(data.city, data.year, df, land_use)
            
            # Neighborhoods are pickled per year, while station_df is pickled
            # per month. Here they are merged.
            df = df.merge(neighborhoods, on='stat_id')
            
            # Neighborhood percentage columns are calculated and added to the
            # station DataFrame.
            df = df.join(
                neighborhood_percentages(
                    data.city, df, land_use, service_radius=500,
                    use_road=False
                    ))
            
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
    
    # return empty DataFrames if data.df is empty
    if len(data.df) == 0:
        df = gpd.GeoDataFrame(
            columns = ['long', 'lat', 'stat_id', 'easting', 'northing', 'name', 
                       'n_arrivals', 'n_departures', 'n_trips', 'b_departures', 
                       'sub_b_departures', 'cus_b_departures', 'w_departures', 
                       'sub_w_departures', 'cus_w_departures', 'b_arrivals', 
                       'sub_b_arrivals', 'cus_b_arrivals', 'w_arrivals', 
                       'sub_w_arrivals', 'cus_w_arrivals', 'b_trips', 'w_trips', 
                       'coords', 'label', 'color', 'zone_type', 'census_geo_id', 
                       'population', 'census_area', 'pop_density', 
                       'nearest_subway', 'nearest_subway_dist', 
                       'nearest_railway', 'nearest_railway_dist', 
                       'service_area', 'service_area_size', 
                       'neighborhood_recreational', 'neighborhood_industrial', 
                       'neighborhood_residential', 'neighborhood_commercial', 
                       'neighborhood_mixed', 'percent_industrial', 
                       'percent_commercial', 'percent_residential', 
                       'percent_recreational','percent_mixed'], 
            geometry = 'service_area')
        
        land_use = gpd.GeoDataFrame(columns = ['zone_type', 'geometry', 'color'])
        
        census_df = gpd.GeoDataFrame(columns = ['census_geo_id', 'population', 
                                                'geometry', 'census_area',
                                                'pop_density'])
        
        if return_land_use and not return_census:
            return df, land_use
        elif return_census and not return_land_use:
            return df, census_df
        elif return_land_use and return_census:
            return df, land_use, census_df
        
        else:
            return df
            
    
    # Basic information about stations from data object
    print("Making Station DataFrame", end="")
    df = gpd.GeoDataFrame(data.stat.locations)
    df['stat_id'] = df.index.map(data.stat.inverse)
    df['easting'], df['northing'] = lon_lat_to_easting_northing(df['long'], df['lat'])
    
    df['name'] = data.stat.names.values()
    df['n_arrivals'] = df['stat_id'].map(data.df['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    df['n_departures'] = df['stat_id'].map(data.df['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)

    df['n_trips'] = df['n_arrivals'] + df['n_departures']

    # First look at the start station (origin / departure) information
    # Include user type information if it exists
    column_list = ['start_stat_id', 'start_dt']
    if 'user_type' in data.df.columns:
        column_list.append('user_type')
    
    # First, we look at business days
    df_s = data.df[column_list][data.df['start_dt'].dt.weekday <= 4]
    
    print(".", end="")
    
    if not holidays:
        holiday_year = pd.DataFrame(
            bs.get_cal(data.city).get_calendar_holidays(data.year), columns=['day', 'name'])
        holiday_list = holiday_year['day'].tolist()
        df_s = df_s[~df_s['start_dt'].dt.date.isin(holiday_list)] # Rows which are not in holiday list
    df['b_departures'] = df['stat_id'].map(df_s['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    
    if 'user_type' in data.df.columns:
        df_sub = df_s[df_s['user_type'] == 'Subscriber']
        df_cus = df_s[df_s['user_type'] == 'Customer']
        df['sub_b_departures'] = df['stat_id'].map(df_sub['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
        df['cus_b_departures'] = df['stat_id'].map(df_cus['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)


    # Then we look at the weekend
    df_s = data.df[column_list][data.df['start_dt'].dt.weekday > 4]
    df['w_departures'] = df['stat_id'].map(df_s['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)

    if 'user_type' in data.df.columns:
        df_sub = df_s[df_s['user_type'] == 'Subscriber']
        df_cus = df_s[df_s['user_type'] == 'Customer']
        df['sub_w_departures'] = df['stat_id'].map(df_sub['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
        df['cus_w_departures'] = df['stat_id'].map(df_cus['start_stat_id'].value_counts().sort_index().to_dict()).fillna(0)

    # Then look at the end station (destination / arrival) information
    column_list = ['end_stat_id', 'end_dt']
    if 'user_type' in data.df.columns:
        column_list.append('user_type')
        
    # Business days
    df_s = data.df[column_list][data.df['end_dt'].dt.weekday <= 4]
    if not holidays:
        df_s = df_s[~df_s['end_dt'].dt.date.isin(holiday_list)] # Rows which are not in holiday list
    df['b_arrivals'] = df['stat_id'].map(df_s['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    
    if 'user_type' in data.df.columns:
        df_sub = df_s[df_s['user_type'] == 'Subscriber']
        df_cus = df_s[df_s['user_type'] == 'Customer']
        df['sub_b_arrivals'] = df['stat_id'].map(df_sub['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
        df['cus_b_arrivals'] = df['stat_id'].map(df_cus['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    
    # Weekend
    df_s = data.df[column_list][data.df['end_dt'].dt.weekday > 4] # DataFrame subset consisting of the end station id and time of trips which end in the weekend
    df['w_arrivals'] = df['stat_id'].map(df_s['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    
    if 'user_type' in data.df.columns:
        df_sub = df_s[df_s['user_type'] == 'Subscriber']
        df_cus = df_s[df_s['user_type'] == 'Customer']
        df['sub_w_arrivals'] = df['stat_id'].map(df_sub['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
        df['cus_w_arrivals'] = df['stat_id'].map(df_cus['end_stat_id'].value_counts().sort_index().to_dict()).fillna(0)
    
    # Add departures and arrivals
    df['b_trips'] = df['b_arrivals'] + df['b_departures']
    df['w_trips'] = df['w_arrivals'] + df['w_departures']
    
    # add Shapely Point column
    df['coords'] = list(zip(df['long'], df['lat']))
    df['coords'] = df['coords'].apply(Point)
    
    # Placeholder information which will be changed by the clustering
    df['label'] = np.nan
    df['color'] = "gray"
    
    # Extend the min and max by 1000m (in projected WGS84 coordinates)
    land_use_extent = Polygon(
        [(df['easting'].min()-1000, df['northing'].min()-1000),
         (df['easting'].min()-1000, df['northing'].max()+1000),
         (df['easting'].max()+1000, df['northing'].max()+1000),
         (df['easting'].max()+1000, df['northing'].min()-1000)])
    
    bbox = gpd.GeoDataFrame([land_use_extent], geometry=0).set_crs(epsg=3857)
    print(". ", end="")
    
    df = gpd.GeoDataFrame(df, geometry='coords', crs='EPSG:4326')
    
    if data.city == 'nyc':
        # Zoning data
        land_use = gpd.read_file('./data/nyc/nyc_zoning_data.json', bbox=bbox)
        
        land_use = land_use[['ZONEDIST', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(
            lambda x: zone_code_transform(data.city, x))
        
        df = gpd.tools.sjoin(df, land_use, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        # Census data
        census_df = pd.read_csv('./data/nyc/nyc_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        CTracts_df = gpd.read_file('./data/nyc/nyc_CT_data.shp', bbox=bbox)
        CTracts_df.set_crs(epsg=4326, allow_override=True, inplace=True)
        CTracts_df = CTracts_df[['GEOID', 'geometry']]
        CTracts_df = CTracts_df.rename({'GEOID' : 'GEO_ID'}, axis=1)
        
        census_df = gpd.GeoDataFrame(census_df.merge(CTracts_df, on='GEO_ID'),
                                     geometry='geometry', crs='EPSG:4326')
        
        # Calculate area of census polygons in projected crs.
        census_df['census_area'] = census_df['geometry'].to_crs(epsg=3857).area/1000000
        
        census_df['pop_density'] = census_df['P1_001N'] / census_df['census_area']
        
        census_df.rename(columns=dataframe_key.get_census_key(data.city), inplace=True)
        
        df = gpd.tools.sjoin(df, census_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)

    elif data.city in ['madrid', 'helsinki', 'london', 'oslo', 'bergen', 'trondheim', 'edinburgh']:
        # Land use data
        land_use = gpd.read_file(f'data/{data.city}/{data.city}_UA2018_v013.gpkg', bbox=bbox)
        land_use = land_use[['code_2018', 'class_2018', 'area', 'Pop2018', 'geometry']].to_crs('EPSG:4326')
        print(".", end="")
        df = gpd.tools.sjoin(df, land_use, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        df['Pop2018'].fillna(0, inplace=True)
        df['area'].fillna(0.1, inplace=True)
        print(".", end="")
        df['zone_type'] = df['code_2018'].apply(lambda x: zone_code_transform(data.city, x))
        df.loc[df['zone_type'] == 'water', 'zone_type'] = "UNKNOWN"
        print(".", end="")
        census_df = land_use[['Pop2018', 'area', 'geometry']]
        census_df['pop_density'] = (census_df['Pop2018'] / census_df['area'])*1000000
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
        
        df['pop_density'] = (df['Pop2018'] / df['area'])*1000000
        
        census_df = land_use[['Pop2018', 'area', 'geometry']]
        census_df['pop_density'] = (census_df['Pop2018'] / census_df['area'])*1000000
        
        land_use = land_use[['zone_type', 'geometry']]
        
    elif data.city == 'chicago':
        
        land_use = gpd.read_file('./data/chicago/chicago_zoning_data.geojson', bbox=bbox)
        
        land_use = land_use[['zone_class', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_code_transform(data.city, x))

        df = gpd.tools.sjoin(df, land_use, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        census_df = pd.read_csv('./data/chicago/chicago_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        CTracts_df = gpd.read_file('./data/chicago/chicago_CT_data.shp', bbox=bbox)
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
        
    elif data.city == 'washdc':
        
        zoning_DC = gpd.read_file('./data/washdc/washdc_zoning_data.geojson', bbox=bbox)
        zoning_DC = zoning_DC[['ZONING_LABEL', 'geometry']]
        
        zoning_arlington = gpd.read_file('./data/washdc/arlington_zoning_data.geojson', bbox=bbox)
        zoning_arlington = zoning_arlington[['ZN_DESIG', 'geometry']]
        zoning_arlington = zoning_arlington.rename({'ZN_DESIG' : 'ZONING_LABEL'}, axis=1)
        
        zoning_alexandria = gpd.read_file('./data/washdc/alexandria_zoning_data.geojson', bbox=bbox)
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

        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
    
        df['zone_type'] = df['zone_code'].apply(lambda x: zone_code_transform(data.city, x))
        
        
        DC_census_df = pd.read_csv('./data/washdc/washdc_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        VA_census_df = pd.read_csv('./data/washdc/VA_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df = pd.concat([DC_census_df, VA_census_df])
        
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        DC_CTracts_df = gpd.read_file('./data/washdc/washdc_CT_data.shp', bbox=bbox)
        DC_CTracts_df = DC_CTracts_df[['GEOID', 'geometry']]
        
        VA_CTracts_df = gpd.read_file('./data/washdc/VA_CT_data.shp', bbox=bbox)
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
       
    elif data.city == 'minneapolis':
        
        land_use = gpd.read_file('./data/minneapolis/minneapolis_zoning_data.geojson', bbox=bbox)
        
        land_use = land_use[['ZONE_CODE', 'geometry']]
        land_use.rename(columns=dataframe_key.get_land_use_key(data.city), inplace=True)
        land_use['zone_type'] = land_use['zone_type'].apply(lambda x: zone_code_transform(data.city, x))

        df = gpd.tools.sjoin(df, land_use, op='within', how='left')
        df.drop('index_right', axis=1, inplace=True)
        
        census_df = pd.read_csv('./data/minneapolis/minneapolis_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        
        
        
        CTracts_df = gpd.read_file('./data/minneapolis/minneapolis_CT_data.shp', bbox=bbox)
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
    
    elif data.city == 'boston':
        
        zoning_boston = gpd.read_file('./data/boston/boston_zoning_data.geojson')
        zoning_boston = zoning_boston[['ZONE_', 'Zone_Desc','geometry']]
        zoning_boston['zone_type'] = zoning_boston['Zone_Desc'].apply(lambda x: zone_code_transform(data.city, x))
        zoning_boston = zoning_boston[['ZONE_', 'zone_type', 'geometry']]
        
        
        zoning_cambridge = gpd.read_file('./data/boston/Cambridge_zoning_data.shp')
        zoning_cambridge = zoning_cambridge[['ZONE_TYPE', 'geometry']]
        zoning_cambridge = zoning_cambridge.rename({'ZONE_TYPE' : 'ZONE_'}, axis=1)

        zoning_cambridge['zone_type'] = zoning_cambridge['ZONE_'].apply(lambda x: zone_code_transform(data.city, x))
        crs = zoning_cambridge.crs
        zoning_cambridge.to_crs(epsg=4326, inplace=True)
        
        zoning_brookline = gpd.read_file('./data/boston/brookline_zoning_data.geojson')
        zoning_brookline = zoning_brookline[['ZONECLASS', 'ZONEDESC' ,'geometry']]
        zoning_brookline['zone_type'] = zoning_brookline['ZONEDESC'].apply(lambda x: zone_code_transform(data.city, x))
        zoning_brookline = zoning_brookline.rename({'ZONECLASS' : 'ZONE_'}, axis=1)
        zoning_brookline = zoning_brookline[['ZONE_', 'zone_type', 'geometry']]
        
        zoning_somerville = gpd.read_file('./data/boston/somerville_zoning_data.shp')
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

        df = gpd.tools.sjoin(df, zoning_df, op='within', how='left') 
        df.drop('index_right', axis=1, inplace=True)
        
        census_df = pd.read_csv('./data/boston/boston_census_data.csv', 
                                usecols=['P1_001N', 'GEO_ID'],
                                skiprows=[1])
        
        census_df['GEO_ID'] = census_df['GEO_ID'].apply(lambda x: x[9:]) # remove state code from GEO_ID
        
        CTracts_df = gpd.read_file('./data/boston/boston_CT_data.shp', bbox=bbox)
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
        
    else:
        
        df['population'] = 0
        df['pop_density'] = 0
        df['zone_type'] = 'UNKNOWN'
        subways_df = gpd.GeoDataFrame([])
        census_df = gpd.GeoDataFrame([])
        census_df.set_geometry([], inplace=True)
        census_df.set_crs(epsg=4326, inplace=True)
        land_use = gpd.GeoDataFrame([])
        land_use.set_geometry([], inplace=True)
        land_use.set_crs(epsg=4326, inplace=True)
        land_use['zone_type'] = 'UNKNOWN'
    
    print(".")
    try:
        subways_df = gpd.read_file(f'./data/{data.city}/{data.city}-transit-data.geojson')
    except FileNotFoundError:
        print(f'File ./data/{data.city}/{data.city}-transit-data.geojson not '
              'found. Please get transit data for {bs.name_dict[data.city]} '
              'from OpenStreetMap. Continuing without transit data, i.e. no '
              '"nearest_subway" or "nearest_subway_dist"')
    # If transit data is available and there is a subway_df.    
    if len(subways_df) > 0:
        
        subways = subways_df[
            (subways_df['station']=='subway') | (subways_df['subway']=='yes')]
        
        railways = subways_df[
            ~((subways_df['station']=='subway') | (subways_df['subway']=='yes'))]
        
        # df['nearest_subway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways.geometry.unary_union)[1], axis=1)
        # df['nearest_subway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]).meters, axis=1)
        
        nearest_subway = nearest_neighbor(df, subways[['geometry']])
        df['nearest_subway'] = nearest_subway['geometry']
        df['nearest_subway_dist'] = nearest_subway['distance']
        
        nearest_railway = nearest_neighbor(df, railways[['geometry']])
        df['nearest_railway'] = nearest_railway['geometry']
        df['nearest_railway_dist'] = nearest_railway['distance']
        
        nearest_transit = nearest_neighbor(df, subways_df[['geometry']])
        df['nearest_transit'] = nearest_transit['geometry']
        df['nearest_transit_dist'] = nearest_transit['distance']
        
        # df['nearest_railway'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], railways.geometry.unary_union)[1], axis=1)
        # df['nearest_railway_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_railway'].coords[0][::-1]).meters, axis=1)
        
        # df['nearest_transit'] = df.apply(lambda stat: shapely.ops.nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
        # df['nearest_transit_dist'] = df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_transit'].coords[0][::-1]).meters, axis=1)
        
    # Fix some issues with Shapely polygons.
    census_df.geometry = census_df.geometry.buffer(0)
    
    df['service_area'], df['service_area_size'] = get_service_area(data.city, df, land_use, service_radius=500)
    if len(census_df) > 0:
        df['pop_density'] = pop_density_in_service_area(df, census_df)

    df.rename(mapper=census_key(data.city), axis=1, inplace=True)  
    df['zone_type'] = df['zone_type'].apply(lambda x: x if pd.notnull(x) else 'UNKNOWN')
    
    land_use['color'] = land_use['zone_type'].map(color_dict).fillna("pink")
    
    if data.day is None:
        with open(f'./python_variables/station_df_{data.city}{data.year:d}{postfix}.pickle', 'wb') as file:
            pickle.dump([df, land_use, census_df], file)
    
    
    try:
        with open(f'./python_variables/neighborhoods_{data.city}{data.year}.pickle', 'rb') as file:
            neighborhoods = pickle.load(file)
    
    except FileNotFoundError:
        print(f'No neighborhoods found. Pickling neighborhoods using {data.city}{data.year} data...')
        neighborhoods = make_neighborhoods(data.city, data.year, df, land_use)
    
    df = df.merge(neighborhoods, on='stat_id')

    df = df.join(
        neighborhood_percentages(
            data.city, df, land_use, service_radius=500,
            use_road=False
            ))
    
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
    
    traffic_matrix = traffic_matrix[:,:24] - traffic_matrix[:,24:]
    
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
        
        traffic_matrices = traffic_matrices[0][station_df.index][:,:24] - traffic_matrices[0][station_df.index][:,24:]
        mean = np.mean(traffic_matrices)
        
        # mean = np.mean(traffic_matrices[0][station_df.index], axis=0)
    elif day_type == 'weekend':
        mean = np.mean(traffic_matrices[1][station_df.index], axis=0)
    
    morning_hours = np.array([6,7,8,9,10])
    afternoon_hours = np.array([15,16,17,18,19])
    
    #mean = mean/np.max(mean)
    
    peakiness = [] # Distance to mean
    rush_houriness = [] # difference between arrivals and departures
    for center in cluster_means:
        # dist_from_mean = np.linalg.norm(center-mean)
        dist_from_mean = np.linalg.norm(center)
        peakiness.append(dist_from_mean)
        # rhn = np.sum((center[morning_hours] - center[morning_hours+24]) - (center[afternoon_hours] - center[afternoon_hours+24]))
        
        rhn = np.sum(center[morning_hours] - center[afternoon_hours])
        
        
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


def neighborhood_percentages(city, station_df, land_use, service_radius=500, use_road='False'):
    
    if 'service_area' not in station_df.columns or service_radius != 500:
        print('Making service area...')
        station_df['service_area'], station_df['service_area_size'] = get_service_area(city, station_df, land_use, service_radius=service_radius)
    
    percentages = pd.DataFrame()
    
    if 'road' in station_df['zone_type'].unique() and use_road in ['False', False]: # Ignore road
        
    
        sa_no_road = station_df['service_area'].difference(station_df['neighborhood_road']).rename('sa_no_road').to_crs(epsg=3857)  
        
        zone_types = station_df['zone_type'].unique()[
            station_df['zone_type'].unique() != 'road']

        sa_no_road = sa_no_road.buffer(0)
        
        

        for zone_type in zone_types:
        
            zone_percents = np.zeros(len(station_df))
            
            mask = ~station_df[f'neighborhood_{zone_type}'].is_empty
            
            sdf_zone = station_df[f'neighborhood_{zone_type}'].to_crs(epsg=3857)[mask]
            
            print(zone_type)
            zone_percents[mask] = sa_no_road[mask].intersection(sdf_zone).area/sa_no_road[mask].area
            
            percentages[f'percent_{zone_type}'] = zone_percents
        
    else:
        
        for zone_type in station_df['zone_type'].unique(): #This is where all the time goes
            
            zone_percents = np.zeros(len(station_df))
            
            for i, stat in station_df.iterrows():
                
                if stat['service_area']:
                    
                    if stat[f'neighborhood_{zone_type}']:
                        
                        zone_percents[i] = stat['service_area'].buffer(0).intersection(stat[f'neighborhood_{zone_type}']).area/stat['service_area'].area
                    
                else:
                    zone_percents[i] = np.nan
                    
            percentages[f'percent_{zone_type}'] = zone_percents

    return percentages


def get_service_area(city, station_df, land_use, service_radius=500):
    """
    

    Parameters
    ----------
    city : TYPE
        DESCRIPTION.
    station_df : TYPE
        DESCRIPTION.
    land_use : TYPE
        DESCRIPTION.
    service_radius : TYPE, optional
        DESCRIPTION. The default is 500.

    Returns
    -------
    service_areas : TYPE
        DESCRIPTION.

    """
    
    points = station_df[['easting', 'northing']].to_numpy()
    points_gdf = gpd.GeoDataFrame(
        geometry=[Point(station_df.iloc[i]['easting'], 
                        station_df.iloc[i]['northing'])
                  for i in range(len(station_df))],
        crs='EPSG:3857')

    points_gdf['point'] = points_gdf['geometry']
    
    mean_point = np.mean(points, axis=0)
    edge_dist = 1000000
    edge_points = np.array([[mean_point[0]-edge_dist, mean_point[1]-edge_dist],
                            [mean_point[0]-edge_dist, mean_point[1]+edge_dist],
                            [mean_point[0]+edge_dist, mean_point[1]+edge_dist],
                            [mean_point[0]+edge_dist, mean_point[1]-edge_dist]])

    vor = Voronoi(np.concatenate([points, edge_points], axis=0))
    
    lines = [LineString(vor.vertices[line])
        for line in vor.ridge_vertices if -1 not in line]
    
    poly_gdf = gpd.GeoDataFrame()
    poly_gdf['vor_poly'] = [poly for poly in shapely.ops.polygonize(lines)]
    poly_gdf['geometry'] = poly_gdf['vor_poly']
    poly_gdf.set_crs(epsg=3857, inplace=True)

    poly_gdf = gpd.tools.sjoin(points_gdf, poly_gdf, op='within', how='left')
    poly_gdf.drop('index_right', axis=1, inplace=True)
    poly_gdf['geometry'] = poly_gdf['vor_poly']
    
    buffers = poly_gdf['point'].buffer(service_radius)
    
    poly_gdf['service_area'] = poly_gdf.intersection(buffers)
    
    poly_gdf['service_area_size'] = poly_gdf['service_area'].apply(lambda area: area.area/1000000)
    
    poly_gdf['geometry'] = poly_gdf['service_area']
    poly_gdf.set_crs(epsg=3857, inplace=True)
    poly_gdf.to_crs(epsg=4326, inplace=True)
    poly_gdf['service_area'] = poly_gdf['geometry']
    
    sa_df = gpd.tools.sjoin(station_df[['coords']], poly_gdf, op='within', how='left')
    sa_df.drop(columns=['index_right', 'vor_poly', 'point'], inplace=True)
    
    service_areas = sa_df['service_area']
    service_area_size = sa_df['service_area_size']
    
    if len(land_use) > 0:
        union = land_use_union(city, land_use)
    
        service_areas = service_areas.apply(lambda area, union=union: area.intersection(union) if area else shapely.geometry.Polygon())
    else:
        print("No land use available, continuing service area without intersection.")
    
    mask = service_areas.apply(isinstance, args=[shapely.geometry.collection.GeometryCollection])
    
    if mask.sum() > 0:
        print(f"GeometryCollection found!\nNumber of 'GeometryCollection's: {mask.sum()}")
    
    service_areas[mask].apply(shapely.ops.unary_union)
    
    service_areas = service_areas.apply(
        lambda poly: Point(0,0).buffer(0.0001) if poly.area==0 else poly)
    
    service_areas = service_areas.buffer(0)
    
    
    return gpd.GeoSeries(service_areas), service_area_size


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


def pop_density_in_service_area(station_df, census_df):
    """
    Compute the population density of each station as a weighted average of the
    population densities in the service area.

    Parameters
    ----------
    station_df : GeoDataFrame
        Dataframe containing the column 'service_area' for each station.
    census_df : GeoDataFrame
        Contains census polygons and corresponding population density.

    Returns
    -------
    pop_ds : list
        weighted average population density for each station.

    """
    pop_ds = []
    stat_sa = station_df['service_area'].to_crs(epsg=3857)
    cens_df = census_df.to_crs(epsg=3857)
    for station in stat_sa:
        intersection = cens_df.intersection(station)
        mask = (intersection != Polygon())
        sec_masked = intersection.loc[mask]
        pop_ds.append((cens_df.loc[mask, 'pop_density'] * sec_masked.area).sum() / station.area)
    return pop_ds


def stations_logistic_regression(station_df, zone_columns, other_columns, 
                                 use_points_or_percents='points', 
                                 make_points_by='station location', 
                                 const=False, test_model=False, test_ratio=0.2, 
                                 test_seed=None, plot_cm=False, 
                                 normalise_cm=None, return_scaled=False):
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
        X_scaled['n_trips'] = X_scaled['n_trips']/X_scaled['n_trips'].sum() # percentage of total trips
    if 'b_trips' in X_scaled.columns:
        X_scaled['b_trips'] = X_scaled['b_trips']/X_scaled['b_trips'].sum() # percentage of business trips
    if 'w_trips' in X_scaled.columns:
        X_scaled['w_trips'] = X_scaled['w_trips']/X_scaled['w_trips'].sum() # percentage of weekend trips
    if 'nearest_subway_dist' in X_scaled.columns:
        X_scaled['nearest_subway_dist'] = X_scaled['nearest_subway_dist']/1000 # Convert to km
    if 'nearest_railway_dist' in X_scaled.columns:
        X_scaled['nearest_railway_dist'] = X_scaled['nearest_railway_dist']/1000 # Convert to km
    
    if 'pop_density' in X_scaled.columns:
        X_scaled['pop_density'] = X_scaled['pop_density']/10000 # population per 100 m²
    
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
    
    if return_scaled:
        X = X_scaled
        
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


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, return_dist=True):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    
    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """
    
    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name
    
    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)
    
    # Parse coordinates from points and insert them into a numpy array as RADIANS
    # Notice: should be in Lat/Lon format 
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.y * np.pi / 180, geom.x * np.pi / 180)).to_list())
    
    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)
    
    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]
    
    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)
    
    # Add distance if requested 
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius
        
    return closest_points


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
        
        #station_df['service_area'], station_df['service_area_size'] = get_service_area(data.city, station_df, land_use, service_radius=service_radius)
        station_df = station_df.merge(
            neighborhood_percentages(
                data.city, station_df, land_use, 
                service_radius=service_radius, use_road=use_road
                ),
            how='outer', left_index=True, right_index=True)
        
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
                   random_state=42, return_land_use=False):
    
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

        #station_df['service_area'], station_df['service_area_size'] = get_service_area(data.city, station_df, land_use, service_radius=service_radius)
        station_df = station_df.merge(
            neighborhood_percentages(
                data.city, station_df, land_use, 
                service_radius=service_radius, use_road=use_road
                ),
            how='outer', left_index=True, right_index=True)
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
    
    if return_land_use:
        return big_station_df, traffic_matrices, labels, land_use
    else:
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
            if city in ['nyc', 'washdc', 'chicago', 'la', 'sfran', 'london', 'mexico', 'buenos_aires', ]:
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

    data = bs.Data('oslo', 2019)

    pre = time.time()
    traffic_matrices = data.pickle_daily_traffic(holidays=False, normalise=True, overwrite=False)
    station_df, land_use, census_df = make_station_df(data, return_land_use=True, return_census=True, overwrite=True)
    #station_df['service_area'], station_df['service_area_size'] = get_service_area(data.city, station_df, land_use, service_radius=500)
    
    # percent_cols = [column for column in station_df.columns if "percent_" in column]

    # station_df = station_df.drop(columns=percent_cols).merge(
    #     neighborhood_percentages(
    #         data.city, station_df, land_use, 
    #         service_radius=500, use_road=False
    #         ),
    #     how='outer', left_index=True, right_index=True)
    
    station_df, clusters, labels = get_clusters(
        traffic_matrices, station_df, day_type='business_days', min_trips=100,
        clustering='k_means', k=3, random_state=42)
    
    # print(f'station_df took {time.time() - pre:.2f} seconds')
    
    # lr = stations_logistic_regression(
    #     station_df, ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational'],
    #     ['pop_density', 'nearest_subway_dist'], use_points_or_percents='percents', 
    #     make_points_by='station location', const=True, test_model=False,
    #     test_ratio=0.2, test_seed=None, plot_cm=False, normalise_cm=None)
    # print(lr[0].summary())


    
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

    

