#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 13:36:10 2022

@author: dvd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

import bikeshare as bs

import full_model
import capacity_opt
import optimization

if __name__ == '__main__':
    
    CITY = 'nyc'
    YEAR = 2019
    MONTH = None
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
              'london', 'helsinki', 'oslo', 'madrid']
    
    variables_list = ['percent_residential', 'percent_commercial',
                      'percent_recreational', 
                      'pop_density', 'nearest_subway_dist',
                      'nearest_railway_dist', 'center_dist']
    
    # data, asdf, traf_mat = full_model.load_city(CITY)
    
    # model = full_model.FullModel(variables_list)
    # asdf=model.fit(asdf, traf_mat)
    
    # low_err, mid_err, high_err = test_model_stratisfied(asdf, traf_mat, variables_list, test_seed=42)
    
    #%% residual plots
    
    # plt.style.use('seaborn-darkgrid')
    # big_fig, big_ax = plt.subplots(figsize=(8,12), nrows=4, ncols=2)
    
    # count=0
    # for row in range(4):
    #     for col in range(2):
    #         city = cities[count]
    #         data, asdf, traf_mat = full_model.load_city(city)
    #         # errors = full_model.trips_predict_error_plot(asdf, traf_mat, variables_list, error='residual', 
    #         #                                   by_cluster=False, show_id=True)
    #         # big_ax[row,col].scatter(errors['predicted'], errors['error'])
    #         # # big_ax[row,col].scatter(errors['predicted'], errors['error'], 
    #         # #                         c=np.log(errors.true), cmap='viridis')
            
    #         # line_stop = max(big_ax[row,col].get_xlim()[1], big_ax[row,col].get_ylim()[1])
            
            
    #         # if row == 3:
    #         #     big_ax[row,col].set_xlabel('Predicted # trips')
            
    #         # if col == 0:
    #         #     big_ax[row,col].set_ylabel('Residual')
            
    #         # big_ax[row,col].set_title(f'{bs.name_dict[city]}')
            
    #         count+=1
    
    # big_fig.tight_layout()
    
    
    #%% Predict daily traffic
    
    data, asdf, traf_mat = full_model.load_city(CITY)
    
    model = full_model.FullModel(variables_list)
    asdf=model.fit(asdf, traf_mat)
    
    traf_mat_true = full_model.load_city(CITY, normalise=False)[2]
    
    stat_id = 72
    
    traffic_est = model.predict_daily_traffic(asdf[asdf.stat_id==stat_id].squeeze(),
                                              predict_cluster=True,
                                              plotfig=False,
                                              verbose=True)
    
    # Compare prediction to actual traffic
    
    traffic_true = traf_mat_true[data.stat.id_index[stat_id]]
    
    plt.style.use('seaborn-darkgrid')
    fig_dep, ax_dep = plt.subplots(figsize=(10,3))
    
    ax_dep.plot(np.arange(24)+0.5, traffic_true[:24], label='True traffic')
    ax_dep.plot(np.arange(24)+0.5, traffic_est[:24], label='Estimated traffic')
    
    ax_dep.set_xlim(0,24)
    ax_dep.set_xticks(range(24))
    
    ax_dep.set_xlabel('Hour')
    ax_dep.set_ylabel('# Trips')
    # ax_dep.set_title(f'Predicted number of departures each hour for {data.stat.names[data.stat.id_index[stat_id]]} (ID: {stat_id})')
    
    ax_dep.legend()
    
    fig_arr, ax_arr = plt.subplots(figsize=(10,3))
    
    ax_arr.plot(np.arange(24)+0.5, traffic_true[24:], label='True traffic')
    ax_arr.plot(np.arange(24)+0.5, traffic_est[24:], label='Estimated traffic')
    
    ax_arr.set_xlim(0,24)
    ax_arr.set_xticks(range(24))
    
    ax_arr.set_xlabel('Hour')
    ax_arr.set_ylabel('# Trips')
    # ax_arr.set_title(f'Predicted number of arrivals each hour for {data.stat.names[data.stat.id_index[stat_id]]} (ID: {stat_id})')
    
    
    ax_arr.legend()
    
    #%% Predict daily traffic all stations
    
    traf_est = pd.DataFrame()
    traf_true = pd.DataFrame()
    
    for stat_id in asdf['stat_id']:
        traffic_est = model.predict_daily_traffic(asdf[asdf.stat_id==stat_id].squeeze(),
                                                  predict_cluster=True,
                                                  plotfig=False,
                                                  verbose=True)
        traffic_true = traf_mat_true[data.stat.id_index[stat_id]]
        
        traf_est[stat_id] = traffic_est
        traf_true[stat_id] = traffic_true

    traf_est.to_csv(f'python_variables/traf_est_{CITY}{YEAR}{MONTH}.csv')
    traf_true.to_csv(f'python_variables/traf_true_{CITY}{YEAR}{MONTH}.csv')
    
    def capacity_helper(vec):
        tau = 1 / 6
        K = 18
        d = vec[:24]
        a = vec[24:]
        return capacity_opt.min_size(a,d,tau,K,0.9)
        
    min_sizes_est = optimization.parallel_apply_along_axis(capacity_helper, 1, traf_est)
    
    print("")
    print(min_sizes_est)
    print("")
    with open(f'./python_variables/min_sizes_est.pickle', 'wb') as file:
        pickle.dump(min_sizes_est, file)
        
    min_sizes_true = optimization.parallel_apply_along_axis(capacity_helper, 1, traf_true)
    print("")
    print(min_sizes_true)
    print("")
    with open(f'./python_variables/min_sizes_true.pickle', 'wb') as file:
        pickle.dump(min_sizes_true, file)
    
    
    
    
    # print(f"{stat_id}: traf_est: ")
    
    # d = traffic_true[:24]
    # a = traffic_true[24:]
    
    # print(f"{stat_id}: traf_true: {capacity_opt.min_size(a,d,tau,K,0.9)}")