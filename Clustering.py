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

if city == 'nyc':
    gov_stations = [3254, 3182, 3479]
    data = bs.Data(city, year, month, blacklist=gov_stations)

with open(f'./python_variables/daily_traffic_{data.city}{data.year:d}{data.month:02d}_{period}.pickle', 'rb') as file:
        traffic_matrix=pickle.load(file)
    
#%% Clustering

clf = bs.Classifier()

clf.k_means(traffic_matrix, k = 5)

labels = clf.mass_predict(traffic_matrix)









