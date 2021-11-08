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

from matplotlib.offsetbox import AnchoredText
#%% Load data

city = 'nyc'
year = 2019
month = 12
period = 'b' # 'b' = business days or 'w' = weekends
holidays = False
min_trips = 100

# if city == 'nyc':
#     gov_stations = [3254, 3182, 3479]
#     data = bs.Data(city, year, month, blacklist=gov_stations)

data = bs.Data(city,year,month)

if period == 'b':
    traffic_matrix = data.pickle_daily_traffic(holidays=holidays)[0]
    x_trips = 'b_trips'
elif period == 'w':
    traffic_matrix = data.pickle_daily_traffic(holidays=holidays)[1]
    x_trips = 'w_trips'

station_df = ipu.make_station_df(data, holidays = holidays)
mask = station_df[x_trips] > min_trips
station_df = station_df[mask]
traffic_matrix = traffic_matrix[mask]

#%% k-test

cluster_func = KMeans

k_test = bs.k_test(traffic_matrix, KMeans, plot=True, k_max=6)

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
cluster_func = KMeans
seed = 42

clusters = cluster_func(k, random_state=seed).fit(traffic_matrix)

station_df = station_df[mask]
station_df['label'] = clusters.predict(traffic_matrix)

percs=dict()
mean_dist_to_subway = np.zeros(k)
std_dist_to_subway = np.zeros(k)
mean_pop_density = np.zeros(k)
std_pop_density = np.zeros(k)


for c in range(k):
    cluster = station_df[station_df['label']==c] 
    zone_counts = cluster['zone_type'].value_counts()
    percs[c] = zone_counts/np.sum(zone_counts)*100
    mean_dist_to_subway[c] = np.mean(cluster['nearest_subway_dist'])
    std_dist_to_subway[c] = np.std(cluster['nearest_subway_dist'])
    mean_pop_density[c] = np.mean(cluster['pop_density'])
    std_pop_density[c] = np.std(cluster['pop_density'])

zone_counts_df = pd.DataFrame()
for zone in station_df['zone_type'].unique():
    zone_stats = station_df[station_df['zone_type'] == zone]
    label_counts = zone_stats['label'].value_counts()
    label_counts.sort_index(0, inplace=True)
    zone_counts_df[zone] = label_counts/np.sum(label_counts)*100
zone_counts_df = zone_counts_df.fillna(0)

#%%

print(mean_dist_to_subway)
print(std_dist_to_subway)
