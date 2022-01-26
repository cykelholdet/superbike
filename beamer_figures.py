# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:21:31 2022

@author: Nicolai
"""

import numpy as np
import matplotlib.pyplot as plt

import bikeshare as bs
import interactive_plot_utils as ipu

#%% 

station_id = 494

data = bs.Data('nyc', 2019, 9)

traffic_matrices = data.daily_traffic(stat_id=station_id, day='all', normalise=False, holidays=True)
station_df, land_use, census_df = make_station_df(data, return_land_use=True, return_census=True, overwrite=True)

#%%

day = 10

fig = plt.figure()

plt.style.use('seaborn-darkgrid')

plt.plot(range(24), traffic_matrices[0][day], 
         color = 'tab:blue', label = 'departures')
plt.plot(range(24), traffic_matrices[1][day], 
         color = 'tab:red', label = 'arrivals')
plt.xticks(range(24))
# plt.yticks([0,5,10,15,20])
plt.xlim(0,23)
plt.ylim(bottom=0)
plt.ylabel('# rides')
plt.xlabel('Hour')
plt.legend()
plt.savefig(f'./figures/beamer_figures/stat_{station_id}_{day}.pdf')