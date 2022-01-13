#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 08:52:14 2022

@author: dbvd
"""

import numpy as np
import pandas as pd

import bikeshare as bs
import interactive_plot_utils as ipu

CITY = 'nyc'
YEAR = 2019
MONTH = None

k = 3
day_type = 'business_days'
min_trips = 100

data = bs.Data(CITY, YEAR, MONTH)
#%%
import interactive_plot_utils as ipu

station_df, land_use = ipu.make_station_df(data, holidays=False, return_land_use=True)
traffic_matrices = data.pickle_daily_traffic(holidays=False)

station_df, clusters, labels = ipu.get_clusters(traffic_matrices, 
                                                station_df, 
                                                day_type, 
                                                100, 
                                                'k_means', 
                                                k, 
                                                random_state=42)
station_df = ipu.service_areas(CITY, station_df, land_use, service_radius=500, use_road=False)

zone_columns = [column for column in station_df.columns if 'percent_' in column]

other_columns = ['n_trips', 'pop_density', 'nearest_subway_dist']

lr_results, X, y = ipu.stations_logistic_regression(station_df, zone_columns, other_columns, use_points_or_percents='points', make_points_by='station location', const=False)

print(lr_results.summary())

#%% Plot the centers
traffic_matrix, mask, _ = ipu.mask_traffic_matrix(traffic_matrices, station_df, day_type, min_trips, holidays=False, return_mask=True)

for j in range(k):
    mean_vector = np.mean(traffic_matrix[np.where(labels[mask] == j)], axis=0)
    
    cc_df = pd.DataFrame([mean_vector[:24], mean_vector[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
    
    cc_df.plot(title=f"Cluster {j}")
