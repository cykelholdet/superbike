#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 08:52:14 2022

@author: dbvd
"""

import numpy as np
import pandas as pd

import bikeshare as bs
import interactive_plot_utils as ipu


def lr_coefficients(data, min_trips=100, clustering='k_means', k=3, random_state=None, day_type='business_days', service_radius=500, use_points_or_percents='points', make_points_by='station_location', add_const=False):
    station_df, land_use = ipu.make_station_df(data, holidays=False, return_land_use=True)
    traffic_matrices = data.pickle_daily_traffic(holidays=False)

    station_df, clusters, labels = ipu.get_clusters(traffic_matrices, 
                                                    station_df, 
                                                    day_type, 
                                                    100, 
                                                    'k_means', 
                                                    k, 
                                                    random_state=42)
    station_df = ipu.service_areas(data.city, station_df, land_use, service_radius=500, use_road=False)

    zone_columns = [column for column in station_df.columns if 'percent_' in column]

    other_columns = ['n_trips', 'pop_density', 'nearest_subway_dist']

    lr_results, X, y = ipu.stations_logistic_regression(station_df, zone_columns, other_columns, use_points_or_percents='points', make_points_by='station location', const=False)

    print(lr_results.summary())

    traffic_matrix, mask, _ = ipu.mask_traffic_matrix(traffic_matrices, station_df, day_type, min_trips, holidays=False, return_mask=True)

    for j in range(k):
        mean_vector = np.mean(traffic_matrix[np.where(labels[mask] == j)], axis=0)
        
        cc_df = pd.DataFrame([mean_vector[:24], mean_vector[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
        
        cc_df.plot(title=f"{bs.name_dict[data.city]} {data.year} cluster {j}")

    single_index = lr_results.params[0].index

    parameters = np.concatenate([lr_results.params[i] for i in range(0, k-1)])
    stdev = np.concatenate([lr_results.bse[i] for i in range(0, k-1)])
    pvalues = np.concatenate([lr_results.pvalues[i] for i in range(0, k-1)])
    
    index = np.concatenate([lr_results.params.index for i in range(0, k-1)])

    multiindex = pd.MultiIndex.from_product([range(1,k), single_index], names=['cluster', 'coef_name'])

    pars = pd.Series(parameters, index=multiindex, name='coef')
    sts = pd.Series(stdev, index=multiindex, name='stdev')
    pvs = pd.Series(pvalues, index=multiindex, name='pvalues')
    
    coefs = pd.DataFrame(pars).join((sts, pvs))
    return pd.concat({bs.name_dict[data.city]: coefs}, names=['city'], axis=1)


#%%
if __name__ == '__main__':
    
    YEAR = 2019
    MONTH = 9
    
    k = 3
    day_type = 'business_days'
    min_trips = 100
    
    table_type = 'city'
    
    if table_type == 'month':
        month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None'}
        
        month_table = pd.DataFrame([])
        for month in range(1,13):
            data = bs.Data('nyc', YEAR, month)
            
            month_table[month_dict[month]] = lr_coefficients(
                data, min_trips, 'k_means', k, random_state=42,
                day_type='business_days', service_radius=500,
                use_points_or_percents='points', make_points_by='station_location')
        
    elif table_type == 'city':
        city_list = ['nyc', 'boston', 'washDC', 'london']
        
        table = pd.DataFrame([])
        for city in city_list:
            data = bs.Data(city, YEAR, MONTH)
            
            table = pd.concat((table, lr_coefficients(
                data, min_trips, 'k_means', k, random_state=42,
                day_type='business_days', service_radius=500,
                use_points_or_percents='points', make_points_by='station_location')
            ), axis=1)
        
        print(table.to_latex(multicolumn_format='c', multirow=True, formatters = {'coef': 'hej', 'stdev': 'dav', 'pvalues': 'yo'}))
            
formatters = {'coef': 'hej', 'stdev': 'dav', 'pvalues': 'yo'}