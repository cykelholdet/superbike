# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:35:06 2021

@author: Nicolai
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
import simpledtw as dtw
import plotting
import time

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

#%% Load data

city = 'nyc'
year = 2019
month = 4
period = 'w' # 'b' = business days or 'w' = weekends

# if city == 'nyc':
#     gov_stations = [3254, 3182, 3479]
#     data = bs.Data(city, year, month, blacklist=gov_stations)

data = bs.Data(city,year,month)

with open(f'./python_variables/daily_traffic_{data.city}{data.year:d}{data.month:02d}.pickle', 'rb') as file:
        
    if period == 'b':
        traffic_matrix=pickle.load(file)[0]
    else:
        traffic_matrix=pickle.load(file)[1]

#%% k_tests

k_max = 10

index_mat = np.zeros(16, k_max-1)

DB_indices = np.zeros(k_max-1)
D_indices = np.zeros(k_max-1)
S_indices = np.zeros(k_max-1)

for i, k in enumerate(range(2, k_max+1)):
    clf = KMeans(k, )
    
    
    DB_indices[i] = 








