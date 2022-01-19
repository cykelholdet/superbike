# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 09:30:43 2022

@author: Nicolai
"""
import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import bikeshare as bs
import interactive_plot_utils as ipu
import matplotlib.pyplot as plt

CITY = 'chic'
YEAR = 2019
MONTH = 1

day_type = 'business_days' # 'business_days' or 'weekend'
min_trips = 100
clustering = 'k_means'
k = 3
seed = 42

service_radius=500
use_points_or_percents='percents'
make_points_by='station_location'
add_const=True
use_road=False
use_whole_year = False

#%% Make station_df

if use_whole_year:
    for month in bs.get_valid_months(city, YEAR):
        
        data = bs.Data(CITY, YEAR, month)
        
        station_df, land_use = ipu.make_station_df(data, return_land_use=True)
        
        traffic_matrices = data.pickle_daily_traffic(holidays=False)
         
        station_df, clusters, labels = ipu.get_clusters(traffic_matrices, station_df, 
                                                        day_type, min_trips, 
                                                        clustering, k, seed)
            
        station_df = ipu.service_areas(CITY, station_df, land_use, 
                                       service_radius=service_radius, 
                                       use_road=use_road)
        
        station_df['month'] = month
        
        if month == 1:
            station_df_year = station_df
            
        else:
            station_df_year = pd.concat([station_df_year, station_df])
    
else:
    data = bs.Data(CITY, YEAR, MONTH)
        
    station_df, land_use = ipu.make_station_df(data, return_land_use=True)
    
    traffic_matrices = data.pickle_daily_traffic(holidays=False)
     
    station_df, clusters, labels = ipu.get_clusters(traffic_matrices, station_df, 
                                                    day_type, min_trips, 
                                                    clustering, k, seed)
        
    station_df = ipu.service_areas(CITY, station_df, land_use, 
                                   service_radius=service_radius, 
                                   use_road=use_road)
    
    
#%% Make logistic regression

zone_columns = [column for column in station_df.columns if 'percent_' in column]
# zone_columns.remove('percent_mixed')
# zone_columns.remove('percent_educational')


other_columns = ['n_trips', 'pop_density', 'nearest_subway_dist']

LR_results, X, y, predictions = ipu.stations_logistic_regression(station_df, zone_columns, 
                                                    other_columns, 
                                                    use_points_or_percents=use_points_or_percents, 
                                                    make_points_by=make_points_by, 
                                                    const=add_const,
                                                    test_model=True)

print(LR_results.summary())

#%% Compare cities

city_train = 'nyc'
city_test = ['nyc', 'chic', 'washDC', 'boston', 'minn', 'london', 'helsinki', 'madrid', 'oslo']

success_rates_all = dict()

for city in city_test:
    success_rates = []
    for month in bs.get_valid_months(city, YEAR):
    
        data_train = bs.Data(city_train, YEAR, month)
        stat_df_train, land_use_train = ipu.make_station_df(data_train, return_land_use=True)
        traffic_matrices_train = data_train.pickle_daily_traffic(holidays=False)
        stat_df_train = ipu.get_clusters(traffic_matrices_train, stat_df_train, 
                                                        day_type, min_trips,
                                                        clustering, k, seed)[0]
        stat_df_train = ipu.service_areas(city_train, stat_df_train, land_use_train, 
                                       service_radius=service_radius, 
                                       use_road=use_road)
        

        data_test = bs.Data(city, YEAR, month)
        stat_df_test, land_use_test = ipu.make_station_df(data_test, return_land_use=True)
        traffic_matrices_test = data_test.pickle_daily_traffic(holidays=False)
        stat_df_test = ipu.get_clusters(traffic_matrices_test, stat_df_test, 
                                                        day_type, min_trips,
                                                        clustering, k, seed)[0]
        stat_df_test = ipu.service_areas(city, stat_df_test, land_use_test, 
                                       service_radius=service_radius, 
                                       use_road=use_road)
        
        zone_columns_train = [column for column in stat_df_train.columns if 'percent_' in column]
        zone_columns_test  = [column for column in stat_df_test.columns  if 'percent_' in column]
        
        
        
        other_columns = ['n_trips', 'pop_density', 'nearest_subway_dist']
        
        
        LR_train, X_train, y_train = ipu.stations_logistic_regression(stat_df_train, zone_columns_train, 
                                                            other_columns, 
                                                            use_points_or_percents=use_points_or_percents, 
                                                            make_points_by=make_points_by, 
                                                            const=add_const)
        
        LR_test, X_test, y_test = ipu.stations_logistic_regression(stat_df_test, zone_columns_test, 
                                                            other_columns, 
                                                            use_points_or_percents=use_points_or_percents, 
                                                            make_points_by=make_points_by, 
                                                            const=add_const)
        
        success_rate, cm = ipu.logistic_regression_test(X_train, y_train, X_test, y_test, plot_cm=False)
        
        success_rates.append(success_rate)
    
    success_rates_all[city] = success_rates

#%%

# fig = plt.Figure()

for city in city_test:
    plt.plot(bs.get_valid_months(city, year), success_rates_all[city], label=city)
    
plt.plot(range(1,13), np.full(shape=(12,1), fill_value=1/k), color='grey', ls='--', label='random guessing')

plt.style.use('seaborn-darkgrid')
plt.ylim([0,1])
plt.legend()
plt.xticks(ticks = range(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.yticks(np.linspace(0,1,11))
plt.ylabel('Success rate')
plt.title(f'Success rates from model trained on {city_train} and tested on all cities')


