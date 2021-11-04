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
month = 5
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

min_trips = 100

station_df = ipu.make_station_df(data)
mask = station_df.n_trips > min_trips
station_df = station_df[mask]
traffic_matrix = traffic_matrix[mask]

#%% k-test

cluster_func = KMeans

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
cluster_func = KMeans
seed = 42

clusters = cluster_func(k, random_state=seed).fit(traffic_matrix)

station_df = station_df[mask]
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

#%% Correlation fig

city = 'nyc'
period = 'b'

month_abbr = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

# 0 = mild morning source, 1 = extreme morning source, 2 = mild morning sink
# 3 = extreme morning sink, 4 = average, 5 = leisure behaviour

# (month, ID of cluster type) : corresponding label
# 'k_list' : list of k's used in order
# 'zone_names' : list of zone types in alphabetical order
nyc_clust_dict={
    (1,0) : 3, (1,1) : 0, (1,2) : 1, (1,3) : 2, (1,4) : 4, (1,5) : None,
    (2,0) : 4, (2,1) : 3, (2,2) : 0, (2,3) : 2, (2,4) : 1, (2,5) : None,
    (3,0) : 0, (3,1) : 3, (3,2) : 1, (3,3) : 4, (3,4) : 2, (3,5) : None,
    (4,0) : 2, (4,1) : 1, (4,2) : 3, (4,3) : 0, (4,4) : 4, (4,5) : None,
    (5,0) : None, (5,1) : 3, (5,2) : 0, (5,3) : 2, (5,4) : 1, (5,5) : None,
    (6,0) : 0, (6,1) : 4, (6,2) : 3, (6,3) : 2, (6,4) : 1, (6,5) : None,
    (7,0) : 5, (7,1) : 0, (7,2) : 3, (7,3) : 2, (7,4) : 1, (7,5) : 4,
    (8,0) : 4, (8,1) : 0, (8,2) : 1, (8,3) : 5, (8,4) : 3, (8,5) : 2,
    (9,0) : 1, (9,1) : 3, (9,2) : 5, (9,3) : 2, (9,4) : 0, (9,5) : 4,
    (10,0) : 3, (10,1) : 0, (10,2) : 1, (10,3) : 2, (10,4) : None, (10,5) : None,
    (11,0) : 0, (11,1) : 2, (11,2) : 1, (11,3) : 3, (11,4) : None, (11,5) : None,
    (12,0) : 2, (12,1) : 0, (12,2) : 1, (12,3) : 4, (12,4) : None, (12,5) : 3,
    'cluster_types' : ['Mild morning source', 'High morning source',
                       'Mild morning sink', 'High morning sink',
                       'Average', 'Leisure'],
    'k_list' : [5,5,5,5,4,5,6,6,6,4,4,5],
    'zone_names' : ['Commercial', 'Manufacturing', 'Recreational', 
                    'Residential', 'Mixed'],
    'zone_colors' : ['tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:blue']}



main_clust_dict = {'nyc' : nyc_clust_dict}

clust_dict = main_clust_dict[city]
zone_names_lower = list(map(lambda x: x.lower(), clust_dict['zone_names']))

plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(12, 6, sharex=True, sharey=True, figsize=(10, 14))

for row in range(12):
    
    data = bs.Data(city, 2019, row+1)
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
    
    min_trips = 100
    
    station_df = ipu.make_station_df(data)
    mask = station_df.n_trips > min_trips
    station_df = station_df[mask]
    traffic_matrix = traffic_matrix[mask]

    
    clusters = KMeans(clust_dict['k_list'][row], random_state=42).fit(traffic_matrix)
    station_df['label'] = clusters.predict(traffic_matrix)
    
    for col in range(6):
        if clust_dict[row+1,col] != None:
            cluster = station_df[station_df['label']==clust_dict[row+1,col]]
            
            zone_counts = pd.DataFrame(np.zeros(len(zone_names_lower)),
                                       index=zone_names_lower,
                                       columns=['zone_type'])
            counts = cluster['zone_type'].value_counts()
            for i in range(len(zone_counts)):
                if zone_counts.iloc[i].name in counts.index.to_list():
                    zone_counts.iloc[i]['zone_type'] = counts[zone_counts.iloc[i].name]
            zone_counts = zone_counts/np.sum(zone_counts)*100
            
            ax[row,col].bar(clust_dict['zone_names'], zone_counts['zone_type'], 
                            color=clust_dict['zone_colors'])
            
            # zone_counts.plot(kind='bar', ax = ax[row,col], color=clust_dict['zone_colors'])
            
            
            # bar_list=ax[row,col].bar(zone_counts.index.to_list(), zone_counts.values)
            # for i, bar in enumerate(bar_list):
            #     bar.set_color(clust_dict['zone_colors'][i])
        
        if row == 0:
            ax[row,col].set_title(clust_dict['cluster_types'][col])
        
        if row == 11:        
            ax[row,col].set_xticklabels(clust_dict['zone_names'], rotation = 90)            
        
        if col == 0:
            ax[row,col].set_ylabel('%')
            text_box = AnchoredText(f'{month_abbr[row+1]}', frameon=False, loc='upper left', pad=0.3)        
            ax[row,col].add_artist(text_box)
plt.tight_layout()
plt.savefig(f'./figures/zone_distributions/{city}{year}_zone_distributions.pdf')
plt.close()




