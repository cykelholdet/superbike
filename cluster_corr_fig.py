# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:50:49 2021

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


city = 'nyc'
year = 2019
period = 'b'
holidays = False

month_abbr = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

# 0 = mild morning source, 1 = extreme morning source, 2 = mild morning sink
# 3 = extreme morning sink, 4 = average, 5 = leisure behaviour

# (month, ID of cluster type) : corresponding label
# 'k_list' : list of k's used in order
# 'zone_names' : list of zone types in alphabetical order
nyc_clust_dict={
    (1,0) : (2,173), (1,1) : (3,129), (1,2) : (0,152), (1,3) : (4,49), (1,4) : (1,249), (1,5) : None,
    (2,0) : (1,206), (2,1) : (0,153), (2,2) : (4,97), (2,3) : (2,68), (2,4) : (3,228), (2,5) : None,
    (3,0) : (0,206), (3,1) : (3,106), (3,2) : (1,167), (3,3) : (4,59), (3,4) : (2,228), (3,5) : None,
    (4,0) : (3,212), (4,1) : (2,140), (4,2) : (4,156), (4,3) : (1,49), (4,4) : (0,220), (4,5) : None,
    (5,0) : (1,152), (5,1) : (3,164), (5,2) : (2,155), (5,3) : (4,37), (5,4) : (0,232), (5,5) : (5,47),
    (6,0) : (0,18), (6,1) : (4,194), (6,2) : (3,142), (6,3) : (2,29), (6,4) : (1,247), (6,5) : None,
    (7,0) : (3,214), (7,1) : (0,172), (7,2) : (4,152), (7,3) : (1,26), (7,4) : (2,209), (7,5) : (5,13),
    (8,0) : (4,195), (8,1) : (0,147), (8,2) : (1,146), (8,3) : (5,29), (8,4) : (3,243), (8,5) : (2,33),
    (9,0) : (2,204), (9,1) : (1,154), (9,2) : (4,151), (9,3) : (3,40), (9,4) : (0,245), (9,5) : None,
    (10,0) : (1,151), (10,1) : (2,192), (10,2) : (0,168), (10,3) : (4,49), (10,4) : (3,258), (10,5) : None,
    (11,0) : (0,293), (11,1) : (1,222), (11,2) : (3,231), (11,3) : (2,79), (11,4) : None, (11,5) : None,
    (12,0) : (2,249), (12,1) : (0,203), (12,2) : (3,191), (12,3) : (1,75), (12,4) : None, (12,5) : (4,97),
    'cluster_types' : ['Mild morning source', 'High morning source',
                       'Mild morning sink', 'High morning sink',
                       'Average', 'Leisure'],
    'k_list' : [5,5,5,5,6,5,6,6,5,5,4,5],
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
    
    if period == 'b':
        x_trips = 'b_trips'
        traffic_matrix = data.pickle_daily_traffic(holidays=holidays)[0]
    elif period == 'w':
        x_trips = 'w_trips'
        traffic_matrix = data.pickle_daily_traffic(holidays=holidays)[1]
    
    min_trips = 100
    
    station_df = ipu.make_station_df(data, holidays=holidays)
    mask = station_df[x_trips] > min_trips
    station_df = station_df[mask]
    traffic_matrix = traffic_matrix[mask]

    
    clusters = KMeans(clust_dict['k_list'][row], random_state=42).fit(traffic_matrix)
    station_df['label'] = clusters.predict(traffic_matrix)
    
    for col in range(6):
        if clust_dict[row+1,col] != None:
            cluster = station_df[station_df['label']==clust_dict[row+1,col][0]]
            
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
            
            count_box = AnchoredText(f'(n={clust_dict[row+1,col][1]})', frameon=False, loc='upper right', pad=0.3)
            ax[row,col].add_artist(count_box)
            
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
            month_box = AnchoredText(f'{month_abbr[row+1]}', frameon=False, loc='upper left', pad=0.3)        
            ax[row,col].add_artist(month_box)
plt.tight_layout()
plt.savefig(f'./figures/zone_distributions/{city}{year}_zone_distributions.pdf')
plt.close()




