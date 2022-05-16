# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:08:19 2022

@author: Nicolai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            
            mask = stat_df[x_trips]>=self.min_trips
            stat_slice = stat_df[mask]
            self.linear_model = ipu.linear_regression(stat_slice, 
                                                      [*zone_columns, *other_columns], 
                                                      x_trips)
        
    
    def predict(self, stat, verbose=True):
        # give row from stat_df and predict traffic pattern and number of trips
        # and maybe plot pedicted traffic pattern
        
        if isinstance(stat, pd.core.series.Series):
            
            
            var_values = np.zeros(len(self.variables)+1)
            var_values[0] = 1
            for i, var in enumerate(self.variables):
                if var in stat.index:
                    var_values[i+1] = stat[var]
            
            label_predict = self.logit_model.predict(exog=var_values)
            trips_predict = self.linear_model.predict(exog=var_values)
            # trips_predict = np.exp(trips_predict[0])
            
            if verbose:
                print("\nProbabilities of labels:")
                string=''
                for i, p in enumerate(label_predict[0]):
                    string += f'label {i}: {p*100} %\n'
                print(string)
                print(f"Predicted label:   {label_predict.argmax()}")
                print(f"Predicted # trips: {trips_predict}")
            
            return label_predict.argmax(), trips_predict
            
        
        elif isinstance(stat, pd.core.frame.DataFrame):
            
            label_predicts = np.zeros(len(stat))
            trips_predicts = np.zeros(len(stat))
            
            for i in range(len(stat)):
                label_predicts[i], trips_predicts[i] = self.predict(
                    stat.iloc[i], verbose=False)
            
            return label_predicts, trips_predicts
            
def trips_predict_test(stat_df, traf_mat, variables, by_cluster=False, 
                       error='residual', show_id=False, plotfig=True, savefig=False):
    
    model = FullModel(variables)
    model.fit(stat_df, traf_mat)
    trip_mask = stat_df['b_trips']>=model.min_trips
    trips_predicted = model.predict(stat_df[trip_mask])[1]
    
    error_list = ['residual', 'absolute', 'relative']
    
    if error == 'residual':
        errors = stat_df[trip_mask]['b_trips'].to_numpy() - trips_predicted
    elif error == 'absolute':
        errors = np.abs(stat_df[trip_mask]['b_trips'].to_numpy() - trips_predicted)
    elif error == 'relative':
        errors = (np.abs(stat_df[trip_mask]['b_trips'].to_numpy() - trips_predicted))/stat_df[trip_mask]['b_trips'].to_numpy()
    
    else:
        raise ValueError(f"Please choose an error from {error_list}")
    
    if by_cluster:
        
        errors_by_cluster = []
        
        for l in range(model.k):
            
            mask = stat_df[trip_mask]['label'] == l
            
            stat_slice = stat_df[trip_mask][mask]
            traf_slice = traf_mat[trip_mask][mask]
                   
            error_df = pd.DataFrame(index=stat_slice['stat_id'])
            error_df['true'] = stat_slice['b_trips']
            error_df['predicted'] = trips_predicted[mask]
            error_df['error'] = errors[mask]
            
            errors_by_cluster.append(error_df)
            
            plt.style.use('seaborn-darkgrid')
            fig, ax = plt.subplots(figsize=(10,8))
            ax.scatter(trips_predicted[mask], errors[mask],
                       c=stat_slice['b_trips'], cmap='viridis')
            
            if show_id:
                for i in range(len(stat_slice)):
                    ax.annotate(str(stat_slice.iloc[i]['stat_id']), 
                                (trips_predicted[mask][i], errors[mask][i]))
            
            ax.set_xlabel('Predicted # trips')
            ax.set_ylabel(f'Error ({error})')
            ax.set_title(f'Cluster {l}')
            
        return errors_by_cluster
            
    else:
        
        error_df = pd.DataFrame(index=stat_df[trip_mask]['stat_id'])
        error_df['true'] = stat_df[trip_mask]['b_trips'].to_numpy()
        error_df['predicted'] = trips_predicted
        error_df['error'] = errors
        
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(figsize=(10,8))
        ax.scatter(error_df['predicted'], error_df['error'], c=np.log(error_df['true']), cmap='viridis')
        # ax.scatter(error_df['true'], error_df['error'])
        
        if show_id:
            for i in range(len(stat_df[trip_mask])):
                ax.annotate(str(stat_df[trip_mask].iloc[i]['stat_id']), (trips_predicted[i], errors[i]))
        
        ax.set_xlabel('Predicted # trips')
        ax.set_ylabel(f'Error ({error})')
    
        return error_df





#%% Do data

CITY = 'chicago'
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



# errors = trips_predict_test(asdf, traf_mat, variables_list, error='residual', 
#                             by_cluster=False, show_id=False)

model = FullModel(variables_list)

model.fit(asdf, traf_mat)
# model.predict(asdf.iloc[907])

