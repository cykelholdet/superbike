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
nyc_clust_dict_b={
    (1,0) : (2,173), (1,1) : (3,129), (1,2) : (0,152), (1,3) : (4,49), (1,4) : (1,249), (1,5) : None,
    (2,0) : (1,206), (2,1) : (0,153), (2,2) : (4,97), (2,3) : (2,68), (2,4) : (3,228), (2,5) : None,
    (3,0) : (0,206), (3,1) : (3,106), (3,2) : (1,167), (3,3) : (4,59), (3,4) : (2,228), (3,5) : None,
    (4,0) : (3,212), (4,1) : (2,140), (4,2) : (4,156), (4,3) : (1,49), (4,4) : (0,220), (4,5) : None,
    (5,0) : (1,152), (5,1) : (3,164), (5,2) : (2,155), (5,3) : (4,37), (5,4) : (0,232), (5,5) : (5,47),
    (6,0) : (0,180), (6,1) : (4,194), (6,2) : (3,142), (6,3) : (2,29), (6,4) : (1,247), (6,5) : None,
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
    'min_trips' : 100,
    'zone_names' : ['Commercial', 'Manufacturing', 'Recreational', 
                    'Residential', 'Mixed'],
    'cluster_type_colors' : ['tab:cyan', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:green']}

nyc_clust_dict_w={
    (1,0) : (1,207), (1,1) : (0,231), (1,2) : (3,110), (1,3) : None, 
    (1,4) : (2,168), (1,5) : None, (1,6) : None, (1,7) : None,
    (2,0) : (3,143), (2,1) : (0,220), (2,2) : (2,142), (2,3) : None, 
    (2,4) : (1,222), (2,5) : None, (2,6) : None, (2,7) : None,
    (3,0) : (0,154), (3,1) : (1,215), (3,2) : (2,141), (3,3) : None, 
    (3,4) : None, (3,5) : (3,244), (3,6) : None, (3,7) : None,
    (4,0) : (2,189), (4,1) : (0,186), (4,2) : (3,159), (4,3) : (1,232), 
    (4,4) : None, (4,5) : None, (4,6) : None, (4,7) : None,
    (5,0) : (2,163), (5,1) : (4,162), (5,2) : None, (5,3) : (3,176), 
    (5,4) : None, (5,5) : (1,179), (5,6) : (0,95), (5,7) : None,
    (6,0) : (3,197), (6,1) : (1,156), (6,2) : None, (6,3) : (0,226), 
    (6,4) : None, (6,5) : (2,211), (6,6) : None, (6,7) : None,
    (7,0) : (2,221), (7,1) : (0,118), (7,2) : None, (7,3) : (1,264), 
    (7,4) : None, (7,5) : (3,177), (7,6) : None, (7,7) : None,
    (8,0) : (2,193), (8,1) : (1,158), (8,2) : None, (8,3) : (0,233), 
    (8,4) : None, (8,5) : (3,190), (8,6) : None, (8,7) : (4,17),
    (9,0) : (0,279), (9,1) : (2,176), (9,2) : None, (9,3) : (1,339), 
    (9,4) : None, (9,5) : None, (9,6) : None, (9,7) : None,
    (10,0) : (2,246), (10,1) : None, (10,2) : (1,132), (10,3) : (0,425), 
    (10,4) : None, (10,5) : None, (10,6) : None, (10,7) : None,
    (11,0) : (2,271), (11,1) : (1,310), (11,2) : (0,223), (11,3) : None, 
    (11,4) : None, (11,5) : None, (11,6) : None, (11,7) : None,
    (12,0) : (0,258), (12,1) : (1,240), (12,2) : None, (12,3) : None, 
    (12,4) : (2,273), (12,5) : None, (12,6) : None, (12,7) : None,
    'cluster_types' : ['NS', 'MAN',
                       'MAN-N', 'MAN-F',
                       'EAN', 'LAN', 'LAN-N', '11-rush'],
    'k_list' : [4,4,4,4,5,4,4,5,3,3,3,3],
    'min_trips' : 70,
    'zone_names' : ['Commercial', 'Manufacturing', 'Recreational', 
                    'Residential', 'Mixed'],
    'cluster_type_colors' : ['tab:cyan', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:green']}

main_clust_dict = {('nyc','b') : nyc_clust_dict_b,
                   ('nyc','w') : nyc_clust_dict_w}

clust_dict = main_clust_dict[city, period]
zone_names_lower = list(map(lambda x: x.lower(), clust_dict['zone_names']))

plt.style.use('seaborn-darkgrid')
n_cols = len(clust_dict['zone_names'])
fig, ax = plt.subplots(12, n_cols, sharex=True, sharey=True, figsize=(10, 14))

for row in range(12):
    
    data = bs.Data(city, year, row+1)
    station_df = ipu.make_station_df(data, holidays=holidays)
    
    if period == 'b':
        traffic_matrix = data.pickle_daily_traffic(holidays=holidays)[0]
        x_trips = 'b_trips'
    elif period == 'w':
        traffic_matrix = data.pickle_daily_traffic(holidays=holidays)[1]
        x_trips = 'w_trips'
    
    min_trips = clust_dict['min_trips']
    
    mask = station_df[x_trips] > min_trips
    # station_df = station_df[mask]
    traffic_matrix = traffic_matrix[mask]

    k = clust_dict['k_list'][row]
    clusters = KMeans(k, random_state=42).fit(traffic_matrix)
    station_df['label'].iloc[mask] = clusters.predict(traffic_matrix)
    station_df['label'].loc[~mask] = np.nan
    
    zone_counts_df = pd.DataFrame(index = list(range(len(clust_dict['cluster_types']))))
    for zone in zone_names_lower:
        station_df = station_df[mask]
        zone_stats = station_df[station_df['zone_type'] == zone]
        label_counts = zone_stats['label'].value_counts()
        label_counts.sort_index(0, inplace=True)
        zone_counts_df[zone] = label_counts/np.sum(label_counts)*100
    zone_counts_df = zone_counts_df.fillna(0)
    
    for col in range(n_cols):
        
        bar_vals = np.zeros(len(clust_dict['cluster_types']))
        for clust_type_id in range(len(clust_dict['cluster_types'])):
            if clust_dict[row+1,clust_type_id] != None:
                bar_vals[clust_type_id] = zone_counts_df[zone_names_lower[col]].iloc[clust_dict[row+1,clust_type_id][0]]
        
        ax[row,col].bar(clust_dict['cluster_types'], bar_vals, 
                        color = clust_dict['cluster_type_colors'])
        
        
        
        
        
        
        
        # if clust_dict[row+1,col] != None:
        #     cluster = station_df[station_df['label']==clust_dict[row+1,col][0]]
            
        #     zone_counts = pd.DataFrame(np.zeros(len(zone_names_lower)),
        #                                index=zone_names_lower,
        #                                columns=['zone_type'])
        #     counts = cluster['zone_type'].value_counts()
        #     for i in range(len(zone_counts)):
        #         if zone_counts.iloc[i].name in counts.index.to_list():
        #             zone_counts.iloc[i]['zone_type'] = counts[zone_counts.iloc[i].name]
        #     zone_counts = zone_counts/np.sum(zone_counts)*100
            
        #     ax[row,col].bar(clust_dict['zone_names'], zone_counts['zone_type'], 
        #                     color=clust_dict['zone_colors'])
            
        #     count_box = AnchoredText(f'(n={clust_dict[row+1,col][1]})', frameon=False, loc='upper right', pad=0.3)
        #     ax[row,col].add_artist(count_box)
            
            # zone_counts.plot(kind='bar', ax = ax[row,col], color=clust_dict['zone_colors'])
            
            
            # bar_list=ax[row,col].bar(zone_counts.index.to_list(), zone_counts.values)
            # for i, bar in enumerate(bar_list):
            #     bar.set_color(clust_dict['zone_colors'][i])
        
        if row == 0:
            ax[row,col].set_title(clust_dict['zone_names'][col])
        
        if row == 11:        
            ax[row,col].set_xticklabels(clust_dict['cluster_types'], rotation = 90)            
        
        if col == 0:
            ax[row,col].set_ylabel('%')
            ax[row,col].set_yticks([10,20,30,40,50])
            month_box = AnchoredText(f'{month_abbr[row+1]}', frameon=False, loc='upper left', pad=0.3)        
            ax[row,col].add_artist(month_box)
plt.tight_layout()
plt.savefig(f'./figures/zone_distributions/{city}{year}_{period}_zone_distributions_2.pdf')

# plt.savefig(f'./figures/zone_distributions/test.pdf')
plt.close()




