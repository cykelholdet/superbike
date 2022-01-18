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

CITY = 'nyc'
YEAR = 2019
MONTH = 9

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

city_test = 'helsinki'
month_test = 9

data = bs.Data(CITY, YEAR, MONTH)

station_df, land_use = ipu.make_station_df(data, return_land_use=True)

traffic_matrices = data.pickle_daily_traffic(holidays=False)
 
station_df, clusters, labels = ipu.get_clusters(traffic_matrices, station_df, 
                                                day_type, min_trips, 
                                                clustering, k, seed)
    
station_df = ipu.service_areas(CITY, station_df, land_use, 
                               service_radius=service_radius, 
                               use_road=use_road)

ohe_zone_test = pd.get_dummies(stat_df_test['zone_type'])
X_test = pd.concat([ohe_zone_test, stat_df_test[['nearest_subway_dist', 'pop_density', ]]], axis=1)

zone_columns = [column for column in station_df.columns if 'percent_' in column]

other_columns = ['n_trips', 'pop_density', 'nearest_subway_dist']

LR_results, X, y, predictions = ipu.stations_logistic_regression(station_df, zone_columns, 
                                                    other_columns, 
                                                    use_points_or_percents=use_points_or_percents, 
                                                    make_points_by=make_points_by, 
                                                    const=add_const,
                                                    test_model=True)

print(LR_results.summary())
