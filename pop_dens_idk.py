# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:08:44 2021

@author: Nicolai
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#%% Load data

nyc_census = pd.read_excel('nyc_decennialcensusdata_2020_2010.xlsx', sheet_name = 1)
nyc_NTA = gpd.read_file('nyc_2020_NTA.json')

#%% Do stuff

pop = np.empty(len(nyc_NTA))

for i in range(len(nyc_NTA)):
    neighbor_NTA2020 = nyc_NTA['NTA2020'].iloc[i]
    neighbor_CT2020 = int(nyc_NTA['BoroCT2020'].iloc[i])
    
    where = np.where(nyc_census['Unnamed: 4'] == neighbor_CT2020)
    pop[i] = nyc_census['2020 Data'].iloc[where]/nyc_NTA['Shape__Area'].iloc[i]
    
    # except FileNotFoundError:
    #     where = np.where(nyc_census['Unnamed: 3'] == neighbor_GEOID)
    #     pop[i] = nyc_census['2020 Data'].iloc[where]


nyc_NTA['population'] = pop
