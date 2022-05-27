#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:48:02 2022

@author: ubuntu
"""

import numpy as np
import geopandas as gpd
import pandas as pd

import bikeshare as bs
import interactive_plot_utils as ipu
from optimization import *
from clustering import get_clusters


data = bs.Data('nyc', 2019, 9)

station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
sub_polygons = gpd.read_file('data/nyc/nyc_expansion_subdivision_2.geojson')

months = [1,2,3,4,5,6,7,8,9]
asdf = asdf_months(data, months)

pops = []
for polygon in sub_polygons['geometry']:
    intersections = census_df.intersection(polygon)
    selection = ~intersections.is_empty
    census_intersect = census_df.loc[selection, 'pop_density']
    # Area in km²
    areas = intersections[selection].to_crs(data.laea_crs).area/1000000
    population = np.sum(areas * census_intersect)
    pops.append(population)

sub_polygons['population'] = pops

# Number of stations per person
station_density = len(station_df) / station_df['population'].sum()

proportional_n_stations = sub_polygons['population'] * station_density

# Scale up and round to add up to 60
n_stations = np.floor(proportional_n_stations*3.15)

sub_polygons['n_stations'] = n_stations

traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')
cols = ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational',
        'pop_density', 'nearest_subway_dist', 'nearest_railway_dist']
day_type = 'business_days'
min_trips = 8
clustering = 'k_means'
k = 5
seed = 42
triptype = 'b_trips'
asdf, clusters, labels = get_clusters(
    traffic_matrices, asdf, day_type, min_trips, clustering, k, seed)

if data.city in ['helsinki', 'oslo', 'madrid', 'london']:
    df_cols = [col for col in cols if col != 'percent_industrial']
else:
    df_cols = cols

model_results = ipu.linear_regression(asdf, df_cols, triptype)

minima = []
n_per = 50000

rng = np.random.default_rng(42)

for i, polygon in sub_polygons.iterrows():
    int_exp = get_intersections(polygon['geometry'], data=data)
   
    existing_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')
   
    existing_stations['lon'] = existing_stations['long']
    existing_stations['geometry'] = existing_stations['coords']
   
    n_existing = len(existing_stations)
   
    print(f"{n_existing} existing, {polygon['n_stations']} total")
   
    int_exp = pd.concat((existing_stations[['lat', 'lon', 'coords', 'geometry']], int_exp))
    int_exp = int_exp.reset_index()
    
    print(f"There are {len(int_exp)} Intersections")
    
    point_info = get_point_info(data, int_exp, land_use, census_df)
    
    months = [1,2,3,4,5,6,7,8,9]
    # asdf = asdf_months(data, months)
    
    int_proj = int_exp.to_crs(data.laea_crs)
    
    n_stations = polygon['n_stations']
    
    n_combinations = binom(len(int_exp), n_stations)
    fig, ax = plt.subplots()
    gpd.GeoSeries(polygon['geometry']).plot(ax=ax)
    int_exp.plot(ax=ax, color='red')
    ax.set_title(f"{n_stations} stations : {n_combinations} combinations")
    print(n_combinations)
    
    n = len(point_info)
    
    n_select = int(n_stations)
    
    n_total = n_select + n_existing
    
    distances = np.zeros((n, n))
    for i in range(n):
        distances[i] = int_proj.distance(int_proj.geometry.loc[i])
        
    
    pred = model_results.predict(point_info[['const', *df_cols]])
    
    def obj_fun(x):
        return -np.sum(x*pred)
    
    def condition(x):
        xb = x.astype(bool)
        return np.min(distances[xb][:,xb][distances[xb][:,xb] != 0])
    
    x0 = np.zeros(n-n_existing)
    x0[:n_select] = 1
    np.random.seed(42)
    x0 = np.random.permutation(x0)
    
    n_permutations = np.floor(np.min((n_per, n_combinations*100))).astype(int)
    
    population = rng.permuted(np.tile(x0, n_permutations).reshape(n_permutations, x0.size), axis=1)
    
    existing_population = np.ones((n_permutations, n_existing))
    
    population = np.hstack((existing_population, population))
    
    score = parallel_apply_along_axis(obj_fun, 1, population)
    if n_select > 1:
        cond = parallel_apply_along_axis(condition, 1, population)
    else:
        cond = np.sum(population, axis=1)*400
    mask = np.where(cond < 250)
    if len(score[mask]) == len(score):
        print('mask condition not fulfilled, changing to 200')
        mask = np.where(cond < 200)
        if len(score[mask]) == len(score):
            print('mask condition not fulfilled, changing to 100')
            mask = np.where(cond < 100)
    
    score[mask] = 0
    
    print(f"min: {population[np.argmin(score)]}, score: {np.min(score)}, condition = {cond}")
    
    
    # minimum = so.minimize(obj_fun, x0=x0, constraints=(sum_constraint), bounds=bounds, method='SLSQP', options={'maxiter': 10})
    # print(minimum.message)
    # minima.append(minimum)
    # selection_idx = np.argpartition(minimum.x, -n_select)[-n_select:]
    # minima.append([np.min(score[mask]), population[np.argmin(score[mask])]])
    score[mask] = 0
    minima.append(population[np.argmin(score)]) 
#%% Results

results = [
    [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ]

results = [np.array(res) for res in results]
results = minima 
selected_intersections = []

for i, polygon in sub_polygons.iterrows():
    int_exp = get_intersections(polygon['geometry'], data=data)
    
    existing_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')

    existing_stations['lon'] = existing_stations['long']
    existing_stations['geometry'] = existing_stations['coords']

    n_existing = len(existing_stations)

    print(f"{n_existing} existing, {polygon['n_stations']} total")
    
    int_exp['existing'] = False
    
    existing_stations['existing'] = True

    int_exp = pd.concat((existing_stations[['lat', 'lon', 'coords', 'geometry', 'existing']], int_exp))
    int_exp = int_exp.reset_index()

    selected_intersections.append(int_exp[results[i] == 1])
    
selected_intersections = pd.concat(selected_intersections)


import full_model

selected_intersections = selected_intersections.reset_index()

data = bs.Data('nyc', 2019, 9)

station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
selected_point_info = get_point_info(data, selected_intersections[['geometry', 'coords']], land_use, census_df)


variables_list = ['percent_residential', 'percent_commercial',
                  'percent_recreational', 
                  'pop_density', 'nearest_subway_dist',
                  'nearest_railway_dist', 'center_dist']

months = [1,2,3,4,5,6,7,8,9]
asdf = asdf_months(data, months)

traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')

asdf, b, c = get_clusters(traffic_matrices, asdf, day_type='business_days', min_trips=8, clustering='k_means', k=5)



# data, asdf, traf_mat = full_model.load_city('nyc')

model = full_model.FullModel(variables_list)
asdf2 = model.fit(asdf, traffic_matrices)

