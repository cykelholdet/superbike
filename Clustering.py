# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:35:06 2021

@author: Nicolai
"""

import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import bikeshare as bs
import interactive_plot_utils as ipu

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

#%% Load data

city = 'nyc'
year = 2019
month = 2
period = 'b' # 'b' = business days or 'w' = weekends

# if city == 'nyc':
#     gov_stations = [3254, 3182, 3479]
#     data = bs.Data(city, year, month, blacklist=gov_stations)

data = bs.Data(city,year,month)

try:
    
    if data.month:
        with open(f'./python_variables/daily_traffic_{data.city}{data.year:d}{data.month:02d}.pickle', 'rb') as file:
            if period == 'b':
                traffic_matrix=pickle.load(file)[0]
            elif period == 'w':
                traffic_matrix=pickle.load(file)[1]
    else:
        with open(f'./python_variables/daily_traffic_{data.city}{data.year:d}.pickle', 'rb') as file:
            if period == 'b':
                traffic_matrix=pickle.load(file)[0]
            elif period == 'w':
                traffic_matrix=pickle.load(file)[1]
except FileNotFoundError:
    if period == 'b':
        traffic_matrix = data.pickle_daily_traffic()[0]
    elif period == 'w':
        traffic_matrix = data.pickle_daily_traffic()[1]

#%% k-test

cluster_func = GaussianMixture

k_test = bs.k_test(traffic_matrix, KMeans, plot=True)

if cluster_func == KMeans:
    clustering = 'KMeans'
elif cluster_func == KMedoids:
    clustering = 'KMedoids'
elif cluster_func == AgglomerativeClustering:
    clustering = 'AgglomerativeClustering'
elif cluster_func == GaussianMixture:
    clustering = 'GaussianMixture'

plt.savefig(f'./figures/k_tests/{data.city}{data.year}{data.month:02d}{period}_{clustering}_k-test.pdf')

#%% Correlation

k = 5
cluster_func = KMedoids
seed = 42

clusters = cluster_func(k, random_state=seed).fit(traffic_matrix)

station_df = ipu.make_station_df(data)
station_df['label'] = clusters.predict(traffic_matrix)

percs=dict()
mean_dist_to_subway = np.zeros(k)
mean_pop_density = np.zeros(k)

for c in range(k):
    cluster = station_df[station_df['label']==c] 
    zone_counts = cluster['zone_type'].value_counts()
    percs[c] = zone_counts/np.sum(zone_counts)*100
    mean_dist_to_subway[c] = np.mean(cluster['nearest_subway_dist'])
    mean_pop_density[c] = np.mean(cluster['pop_density'])


zone_counts_df = pd.DataFrame()
for zone in station_df['zone_type'].unique():
    zone_stats = station_df[station_df['zone_type'] == zone]
    label_counts = zone_stats['label'].value_counts()
    label_counts.sort_index(0, inplace=True)
    zone_counts_df[zone] = label_counts/np.sum(label_counts)*100
zone_counts_df = zone_counts_df.fillna(0)

