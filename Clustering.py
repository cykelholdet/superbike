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

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

#%% Load data

city = 'mexico'
year = 2019
month = 3
period = 'b' # 'b' = business days or 'w' = weekends

# if city == 'nyc':
#     gov_stations = [3254, 3182, 3479]
#     data = bs.Data(city, year, month, blacklist=gov_stations)

data = bs.Data(city,year,month)

try:
    with open(f'./python_variables/daily_traffic_{data.city}{data.year:d}{data.month:02d}.pickle', 'rb') as file:
            
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

cluster_func = AgglomerativeClustering

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