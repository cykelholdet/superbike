# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:41:18 2021

@author: Cykelholdet
"""

import os
import calendar
import pickle
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextily as ctx
import bikeshare as bs
import plotting
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

#%% Load data

city = 'nyc'
year = 2019
month = 9
period = 'b' # 'b' = business days or 'w' = weekends

# if city == 'nyc':
#     gov_stations = [3254, 3182, 3479]
#     data = bs.Data(city, year, month, blacklist=gov_stations)

data=bs.Data(city,year,month)

with open(f'./python_variables/daily_traffic_{data.city}{data.year:d}{data.month:02d}_{period}.pickle', 'rb') as file:
        traffic_matrix=pickle.load(file)
    
#%% Clustering

k = 5

clf = bs.Classifier()

clf.k_means(traffic_matrix, k)

labels = clf.mass_predict(traffic_matrix)

#%% Plotting

month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
figsize_dict = {'nyc': (5,8),
                'chic': (5,8),
                'london': (8,5),
                'oslo': (6.6,5),
                'sfran': (6,5),
                'washDC': (8,7.7),
                'madrid': (6,8),
                'mexico': (7.2,8), 
                'taipei': (7.3,8)}

scalebars = {'chic': 5000,
             'london': 5000,
             'madrid': 2000,
             'mexico': 2000,
             'nyc':5000,
             'sfran':5000,
             'taipei':5000,
             'washDC':5000}

lat = [data.stat.loc_merc[i][0] for i in range(data.stat.n_tot)]
long = [data.stat.loc_merc[i][1] for i in range(data.stat.n_tot)]
    
extent = np.array([np.min(lat)-1000, np.max(lat)+1000, np.min(long)-1000, np.max(long)+1000])

try:
    fig, ax = plt.subplots(figsize=figsize_dict[data.city])
except KeyError:
    fig, ax = plt.subplots(figsize=(8,8))
    
ax.axis(extent)
ax.axis('off')

print('Drawing network...')

color_dict = {0 : 'tab:blue', 1 : 'tab:orange', 2 : 'tab:green', 3 : 'tab:red',
              4 : 'tab:purple', 5 : 'tab:brown', 6: 'tab:pink',
              7 : 'tab:gray', 8 : 'tab:olive', 9 : 'tab:cyan'}

color_map = [color_dict[label] for label in labels]

ax.scatter(lat, long, c = color_map)

print('Adding basemap... ')
ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain,
                    attribution='(C) Stamen Design, (C) OpenStreetMap contributors')

print('Adding scalebar...')
scalebar = AnchoredSizeBar(ax.transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                            pad=0.2, color='black', frameon=False, size_vertical=50)

ax.add_artist(scalebar)

markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in list(color_dict.values())[:k]]
plt.legend(markers, color_dict.keys(), numpoints=1)

if period == 'b':
    plt.title(f'Clustering for {data.city} in {month_dict[data.month]} {data.year:d} on business days')
elif period == 'w':
    plt.title(f'Clustering for {data.city} in {month_dict[data.month]} {data.year:d} on weekends')

fig.tight_layout()
fig.show()

for i in range(k):
    
    traffic_cluster = traffic_matrix[np.where(labels == i)]
    
    arrivals_std = np.std(traffic_cluster[:,:24]*100, axis=0)
    departures_std = np.std(traffic_cluster[:,24:]*100, axis=0)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(24), clf.centroids[i][:24]*100)
    ax.plot(np.arange(24), clf.centroids[i][24:]*100)
    plt.xticks(np.arange(24))
    
    plt.fill_between(np.arange(24), clf.centroids[i][:24]*100-arrivals_std, 
                             clf.centroids[i][:24]*100+arrivals_std, 
                             facecolor='b',alpha=0.2)
    plt.fill_between(np.arange(24), clf.centroids[i][24:]*100-departures_std, 
                     clf.centroids[i][24:]*100+departures_std, 
                     facecolor='orange',alpha=0.2)
    
    
    plt.legend(['Arrivals','Departures'])
    plt.xlabel('Hour')
    plt.ylabel('% of total trips')
    plt.title(f'Hourly traffic for centroid of cluster {i} (n = {len(traffic_cluster)})')
    fig.show()









