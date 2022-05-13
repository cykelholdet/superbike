# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:08:19 2022

@author: Nicolai
"""

import numpy as np
import pandas as pd
import pickle

import bikeshare as bs
import interactive_plot_utils as ipu
from clustering import get_clusters


class FullModel:
    def __init__(self, variables, clustering='k_means', k=5, min_trips=8, cluster_seed=42, use_dtw=False,
                 linkage = 'average', user_type='Subscriber', day_type='business_days'):
        
        self.variables = variables
        self.clustering = clustering
        self.k = k
        self.min_trips = min_trips
        self.cluster_seed = cluster_seed
        self.use_dtw = use_dtw
        self.linkage = linkage
        self.user_type = user_type
        if day_type in ['business_days', 'weekend']:
            self.day_type = day_type
        else:
            raise ValueError("Please provide day_type as either 'business_days' or 'weekend")
        self.labels = None
        self.centers = None
        self.logit_model = None
        self.linear_model = None
        
    def fit(self, stat_df, traf_mat, const=True):
        
        for var in self.variables:
            if var not in stat_df.columns:
                raise KeyError(f"{var} not found in stat_df columns")
        
        stat_df, self.centers, self.labels = get_clusters(traf_mat, stat_df, 
                                                          day_type=self.day_type, 
                                                          min_trips=self.min_trips, 
                                                          clustering=self.clustering, k=self.k, 
                                                          random_state=self.cluster_seed,
                                                          use_dtw=self.use_dtw, linkage=self.linkage)
        
        zone_columns  = [var for var in self.variables if 'percent' in var]
        other_columns = [var for var in self.variables if 'percent' not in var]
        
        self.logit_model, X, y = ipu.stations_logistic_regression(
                stat_df, zone_columns, other_columns, 
                use_points_or_percents='percents', 
                make_points_by='station land use', 
                const=const, test_model=False)
        
        print(self.logit_model.summary())
            
        if (x_trips := 'b_trips' if self.day_type == 'business_days' else 'w_trips') in stat_df.columns:
            
            self.linear_model = ipu.linear_regression(stat_df, 
                                                      [*zone_columns, *other_columns], 
                                                      x_trips)
        
    
    def predict(self, stat_row, plot_pattern=True, verbose=True):
        # give row from stat_df and predict traffic pattern and number of trips
        # and maybe plot pedicted traffic pattern
        
        if isinstance(stat_row, pd.core.series.Series):
            
            
            var_values = np.zeros(len(self.variables)+1)
            var_values[0] = 1
            for i, var in enumerate(self.variables):
                if var in stat_row.index:
                    var_values[i+1] = stat_row[var]
            
            label_predict = self.logit_model.predict(exog=var_values)
            trips_predict = self.linear_model.predict(exog=var_values)
            trips_predict = np.exp(trips_predict[0])
            
            if verbose:
                print("\nProbabilities of labels:")
                string=''
                for i, p in enumerate(label_predict[0]):
                    string += f'label {i}: {p*100} %\n'
                print(string)
                print(f"Predicted label:   {label_predict.argmax()}")
                print(f"Predicted # trips: {trips_predict}")
                
                print(f"\nActual label:   {stat_row.label}")
                print(f"Actual # trips: {stat_row.b_trips}")
        
        pass

#%% Do data

CITY = 'nyc'
YEAR = 2019
MONTH = None

variables_list = ['percent_residential', 'percent_commercial',
                  'percent_recreational', 
                  'pop_density', 'nearest_subway_dist',
                  'nearest_railway_dist', 'center_dist']

# Read asdf

if MONTH is None:
    filestr = f'./python_variables/{CITY}{YEAR}_avg_stat_df.pickle'
else:
    filestr = f'./python_variables/{CITY}{YEAR}{MONTH:02d}_avg_stat_df.pickle'
    
with open(filestr, 'rb') as file:
    asdf = pickle.load(file)

# Make Data object and traffic matrix

data = bs.Data(CITY, YEAR, MONTH)
traf_mat = data.pickle_daily_traffic(holidays=False, 
                                      user_type='Subscriber',
                                      day_type='business_days',
                                      overwrite=False)
        
mask = ~asdf['n_trips'].isna()

asdf = asdf[mask]
asdf = asdf.reset_index(drop=True)

try:
    traf_mat = traf_mat[mask]
except IndexError:
    pass

#%%

model = FullModel(variables_list)

model.fit(asdf, traf_mat)
model.predict(asdf.iloc[308])

