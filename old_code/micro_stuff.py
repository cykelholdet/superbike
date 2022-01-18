# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:00:03 2021

@author: nweinr
"""
import calendar

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import bikeshare as bs
import interactive_plot_utils as ipu

city = 'nyc'
year = 2019
month = 9
period = 'b' # 'b' = business days or 'w' = weekends
station = 247

data = bs.Data('nyc', 2019, 9)
station_df = ipu.make_station_df(data, holidays=False)

days = [day for day in range(1, calendar.monthrange(year,month)[1]+1) if calendar.weekday(year, month, day) < 5]

stat_traffic = np.zeros(shape=(len(days), 48))

for i, day in enumerate(days):
    dep, arr = data.daily_traffic(data.stat.inverse[station], day, normalise=False)
    stat_traffic[i,:] = np.concatenate([dep,arr])
    
std = np.std(stat_traffic, axis=0)
mean = np.mean(stat_traffic, axis=0)
