# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 09:30:43 2022

@author: Nicolai
"""
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon, Point
import geopandas as gpd


import bikeshare as bs
import interactive_plot_utils as ipu
from logistic_table import lr_coefficients

CITY = 'nyc'
YEAR = 2019
MONTH = 1

day_type = 'business_days' # 'business_days' or 'weekend'
min_trips = 100
clustering = 'k_means'
k = 3
seed = 42

service_radius=500
use_points_or_percents='percents'
make_points_by='station_location'
add_const=False
use_road=False
use_whole_year = False

month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
          7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

# #%% Make station_df

# if MONTH is None:
#     for month in bs.get_valid_months(CITY, YEAR):
        
#         data = bs.Data(CITY, YEAR, month)
        
#         station_df, land_use = ipu.make_station_df(data, return_land_use=True, holidays=False)
        
#         traffic_matrices = data.pickle_daily_traffic(holidays=False)
         
#         station_df, clusters, labels = ipu.get_clusters(traffic_matrices, station_df, 
#                                                         day_type, min_trips, 
#                                                         clustering, k, seed)
            
#         station_df = station_df.merge(
#             ipu.neighborhood_percentages(
#                 data.city, station_df, land_use, 
#                 service_radius=service_radius, use_road=use_road
#                 ),
#             how='outer', left_index=True, right_index=True)
        
#         station_df['month'] = month
        
#         if month == 1:
#             station_df_year = station_df
            
#         else:
#             station_df_year = pd.concat([station_df_year, station_df])
    
# else:
#     data = bs.Data(CITY, YEAR, MONTH)
        
#     station_df, land_use = ipu.make_station_df(data, return_land_use=True, holidays=False)
    
#     traffic_matrices = data.pickle_daily_traffic(holidays=False)
     
#     station_df, clusters, labels = ipu.get_clusters(traffic_matrices, station_df, 
#                                                     day_type, min_trips, 
#                                                     clustering, k, seed)
        
#     station_df = station_df.merge(
#         ipu.neighborhood_percentages(
#             data.city, station_df, land_use, 
#             service_radius=service_radius, use_road=use_road
#             ),
#         how='outer', left_index=True, right_index=True)
    
    
# #%% Make logistic regression

# zone_columns = [column for column in station_df.columns if 'percent_' in column]
# # zone_columns.remove('percent_mixed')
# # zone_columns.remove('percent_educational')


# other_columns = ['n_trips', 'pop_density', 'nearest_subway_dist']

# LR_results, X, y, predictions = ipu.stations_logistic_regression(station_df, zone_columns, 
#                                                     other_columns, 
#                                                     use_points_or_percents=use_points_or_percents, 
#                                                     make_points_by=make_points_by, 
#                                                     const=add_const,
#                                                     test_model=True,
#                                                     test_seed=69,
#                                                     plot_cm=True, 
#                                                     normalise_cm='true')

# print(LR_results.summary())

# plt.savefig(f'./figures/norm_tests/test_/{CITY}{YEAR}{MONTH:02d}.png')
# plt.close()
# #%% Compare whole year model with monthly models 1

# city_train = 'nyc'


# omit_columns = {
#                            'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
#                            'chicago': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
#                            'nyc': ['percent_mixed', 'n_trips'],
#                            'washdc': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
#                            'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
#                            'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips'],
#                            'madrid': ['n_trips'],
#                            'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
#                            }


# data_train_year = bs.Data(city_train, YEAR, None)
# LR_train_year, X_train_year, y_train_year = lr_coefficients(data_train_year, data.city, min_trips=min_trips, clustering=clustering, k=k, 
#                                                 random_state=seed, day_type=day_type, 
#                                                 service_radius=service_radius, use_points_or_percents=use_points_or_percents, 
#                                                 make_points_by=make_points_by, add_const=add_const, 
#                                                 use_road=use_road, remove_columns=omit_columns[city_train], title='City', 
#                                                 big_station_df=False, return_model=True)[1:]

# success_rates_year = []
# success_rates_month = []
# for month in bs.get_valid_months(city_train, YEAR):
    
#     data_train = bs.Data(city_train, YEAR, month)
#     stat_df_train, land_use_train = ipu.make_station_df(data_train, return_land_use=True)
#     traffic_matrices_train = data_train.pickle_daily_traffic(holidays=False)
#     stat_df_train = ipu.get_clusters(traffic_matrices_train, stat_df_train, 
#                                                     day_type, min_trips,
#                                                     clustering, k, seed)[0]
#     station_df = station_df.merge(
#         ipu.neighborhood_percentages(
#             data.city, station_df, land_use, 
#             service_radius=service_radius, use_road=use_road
#             ),
#         how='outer', left_index=True, right_index=True)
    
#     data_test = bs.Data(city_train, YEAR, month)
#     stat_df_test, land_use_test = ipu.make_station_df(data_test, return_land_use=True)
#     traffic_matrices_test = data_test.pickle_daily_traffic(holidays=False)
#     stat_df_test = ipu.get_clusters(traffic_matrices_test, stat_df_test, 
#                                                     day_type, min_trips,
#                                                     clustering, k, seed)[0]
#     station_df = station_df.merge(
#         ipu.neighborhood_percentages(
#             data.city, station_df, land_use, 
#             service_radius=service_radius, use_road=use_road
#             ),
#         how='outer', left_index=True, right_index=True)
    
#     zone_columns_train = [column for column in stat_df_train.columns if 'percent_' in column]
#     zone_columns_train = [column for column in zone_columns_train if column not in omit_columns[city_train]]
           
    
#     zone_columns_test  = [column for column in stat_df_test.columns  if 'percent_' in column]
#     zone_columns_test =  [column for column in zone_columns_test if column not in omit_columns[city_train]]
    
#     other_columns = ['pop_density', 'nearest_subway_dist']

#     LR_train, X_train, y_train = ipu.stations_logistic_regression(stat_df_train, zone_columns_train, 
#                                                                 other_columns, 
#                                                                 use_points_or_percents=use_points_or_percents, 
#                                                                 make_points_by=make_points_by, 
#                                                                 const=add_const)

#     LR_test, X_test, y_test = ipu.stations_logistic_regression(stat_df_test, zone_columns_test, 
#                                                         other_columns, 
#                                                         use_points_or_percents=use_points_or_percents, 
#                                                         make_points_by=make_points_by, 
#                                                         const=add_const)

            
#     success_rates_year.append(ipu.logistic_regression_test(X_train_year, y_train_year, X_test, y_test, plot_cm=False)[0])
#     success_rates_month.append(ipu.logistic_regression_test(X_train, y_train, X_test, y_test, plot_cm=False)[0])

# #%% Compare whole year model with monthly models 2

# city_train = 'nyc'
# split_seed = 42

# omit_columns = {
#                            'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
#                            'chicago': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
#                            'nyc': ['percent_mixed', 'n_trips'],
#                            'washdc': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
#                            'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
#                            'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips'],
#                            'madrid': ['n_trips'],
#                            'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
#                            }


# stat_df_train_year_bad, traffic_matrices_train_year_bad, labels, land_use = ipu.make_station_df_year(
#     city_train, year=YEAR, months=None, service_radius=service_radius,
#     use_road=use_road, day_type=day_type,
#     min_trips=min_trips, clustering=clustering, k=k,
#     random_state=seed, return_land_use=True)

# # stat_df_train_year_bad = ipu.get_clusters(traffic_matrices_train_year_bad, stat_df_train_year_bad, 
# #                                                     day_type, min_trips,
# #                                                     clustering, k, seed)[0]
# # stat_df_train_year_bad = ipu.service_areas(city_train, stat_df_train_year_bad, land_use, 
# #                                 service_radius=service_radius, 
# #                                 use_road=use_road)


 
# other_columns_year = ['pop_density', 'nearest_subway_dist', 'Jan', 'Feb', 'Mar',
#                  'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']         


# other_columns_month = ['pop_density', 'nearest_subway_dist']         


# # LR_train_year_bad, X_train_year_bad, y_train_year_bad = ipu.stations_logistic_regression(stat_df_train_year, zone_columns_train_year, 
# #                                                                 other_columns_month, 
# #                                                                 use_points_or_percents=use_points_or_percents, 
# #                                                                 make_points_by=make_points_by, 
# #                                                                 const=add_const)
# stat_df_train_bad_list = []
# stat_df_train_list = []
# stat_df_test_list = []
# for month in bs.get_valid_months(city_train, YEAR):
        
#     data = bs.Data(city_train, YEAR, month)
    
#     station_df, land_use = ipu.make_station_df(data, return_land_use=True)
    
#     traffic_matrices = data.pickle_daily_traffic(holidays=False)
     
#     station_df, clusters, labels = ipu.get_clusters(traffic_matrices, station_df, 
#                                                     day_type, min_trips, 
#                                                     clustering, k, seed)
        
#     station_df = station_df.merge(
#         ipu.neighborhood_percentages(
#             data.city, station_df, land_use, 
#             service_radius=service_radius, use_road=use_road
#             ),
#         how='outer', left_index=True, right_index=True)
    
#     for i in bs.get_valid_months(city_train, YEAR):
#         if i == month:
#             station_df[month_dict[i]] = 1
#         else:
#             station_df[month_dict[i]] = 0
    
#     stat_df_train, stat_df_test = train_test_split(station_df, test_size=0.2, random_state=split_seed)
    
#     stat_df_train_bad = stat_df_train_year_bad[stat_df_train_year_bad[month_dict[month]]==1]
#     stat_df_train_bad = stat_df_train_bad[~stat_df_train_bad['stat_id'].isin(stat_df_test['stat_id'].to_list())]
    
#     stat_df_train_bad_list.append(stat_df_train_bad)
#     stat_df_train_list.append(stat_df_train)
#     stat_df_test_list.append(stat_df_test)
    
# stat_df_train_year = pd.concat(stat_df_train_list)
# stat_df_train_year_bad = pd.concat(stat_df_train_bad_list)

# zone_columns_train_year = [column for column in stat_df_train_year.columns if 'percent_' in column]
# zone_columns_train_year = [column for column in zone_columns_train_year if column not in omit_columns[city_train]]

# LR_train_year, X_train_year, y_train_year = ipu.stations_logistic_regression(stat_df_train_year, zone_columns_train_year, 
#                                                                 other_columns_month, 
#                                                                 use_points_or_percents=use_points_or_percents, 
#                                                                 make_points_by=make_points_by, 
#                                                                 const=add_const)


# LR_train_year_bad, X_train_year_bad, y_train_year_bad = ipu.stations_logistic_regression(stat_df_train_year_bad, zone_columns_train_year, 
#                                                                 other_columns_month, 
#                                                                 use_points_or_percents=use_points_or_percents, 
#                                                                 make_points_by=make_points_by, 
#                                                                 const=add_const)
    

# success_rates_year = []
# success_rates_year_bad = []
# success_rates_month = []
# for i, month in enumerate(bs.get_valid_months(city_train, YEAR)):
    
#     # data_train = bs.Data(city_train, YEAR, month)
#     # stat_df_train, land_use_train = ipu.make_station_df(data_train, return_land_use=True)
#     # traffic_matrices_train = data_train.pickle_daily_traffic(holidays=False)
#     # stat_df_train = ipu.get_clusters(traffic_matrices_train, stat_df_train, 
#     #                                                 day_type, min_trips,
#     #                                                 clustering, k, seed)[0]
#     # stat_df_train = ipu.service_areas(city_train, stat_df_train, land_use_train, 
#     #                                service_radius=service_radius, 
#     #                                use_road=use_road)
    
#     # data_test = bs.Data(city_train, YEAR, month)
#     # stat_df_test, land_use_test = ipu.make_station_df(data_test, return_land_use=True)
#     # traffic_matrices_test = data_test.pickle_daily_traffic(holidays=False)
#     # stat_df_test = ipu.get_clusters(traffic_matrices_test, stat_df_test, 
#     #                                                 day_type, min_trips,
#     #                                                 clustering, k, seed)[0]
#     # stat_df_test = ipu.service_areas(city_train, stat_df_test, land_use_test, 
#     #                                service_radius=service_radius, 
#     #                                use_road=use_road)
    
#     # for i in get_valid_months(city_train, YEAR):
#     #     if i == month:
#     #         stat_df_test[month_dict[i]] = 1
#     #     else:
#     #         stat_df_test[month_dict[i]] = 0
    
    
    
    
    
#     zone_columns_train = [column for column in stat_df_train.columns if 'percent_' in column]
#     zone_columns_train = [column for column in zone_columns_train if column not in omit_columns[city_train]]
           
    
#     zone_columns_test  = [column for column in stat_df_test.columns  if 'percent_' in column]
#     zone_columns_test =  [column for column in zone_columns_test if column not in omit_columns[city_train]]
    

#     LR_train, X_train, y_train = ipu.stations_logistic_regression(stat_df_train_list[i], zone_columns_train, 
#                                                                 other_columns_month, 
#                                                                 use_points_or_percents=use_points_or_percents, 
#                                                                 make_points_by=make_points_by, 
#                                                                 const=add_const)

#     LR_test, X_test, y_test = ipu.stations_logistic_regression(stat_df_test_list[i], zone_columns_test, 
#                                                         other_columns_year, 
#                                                         use_points_or_percents=use_points_or_percents, 
#                                                         make_points_by=make_points_by, 
#                                                         const=add_const)

            
#     success_rates_year.append(ipu.logistic_regression_test(X_train_year, y_train_year, X_test, y_test, plot_cm=False)[0])
#     success_rates_year_bad.append(ipu.logistic_regression_test(X_train_year_bad, y_train_year_bad, X_test, y_test, plot_cm=False)[0])
#     success_rates_month.append(ipu.logistic_regression_test(X_train, y_train, X_test, y_test, plot_cm=False)[0])



# #%%

# fig =  plt.figure(figsize=(8,5))

# plt.plot(bs.get_valid_months(city_train, YEAR), success_rates_month, label='Trained monthly, clustered monthly')
# plt.plot(bs.get_valid_months(city_train, YEAR), success_rates_year, label='Trained whole year, clustered monthly')
# plt.plot(bs.get_valid_months(city_train, YEAR), success_rates_year_bad, label='Trained whole year, clustered whole year')


# plt.plot(range(1,13), np.full(shape=(12,1), fill_value=1/k), color='grey', ls='--', label='random guessing')

# plt.style.use('seaborn-darkgrid')
# plt.ylim([0,1])
# plt.legend()
# plt.xticks(ticks = range(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# plt.yticks(np.linspace(0,1,11))
# plt.xlabel('Month')
# plt.ylabel('Success rate')
# # plt.title(f'Success rates from model trained on {bs.name_dict[city_train]} and tested on different cities')
# # plt.savefig(f'./figures/LR_model_tests/train_on_{city_train}_whole_year_vs_monthly.pdf')
# plt.savefig(f'./figures/LR_model_tests/{city_train}_yealy_vs_monthly_no_month_cols.pdf')












# #%% Compare cities

# city_train = 'nyc'
# city_test = ['nyc', 'chicago', 'washdc', 'boston', 'london', 'madrid', 'helsinki', 'oslo']
# # city_test = ['nyc', 'chicago', 'washdc', 'boston']
# # city_test =  ['london', 'madrid', 'helsinki', 'oslo']

# # city_test= ['minneapolis']

# success_rates_all = dict()

# for city in city_test:
#     success_rates = []
#     for month in bs.get_valid_months(city, YEAR):
#         if month in bs.get_valid_months(city_train, YEAR):
#             data_train = bs.Data(city_train, YEAR, month)
#             stat_df_train, land_use_train = ipu.make_station_df(data_train, return_land_use=True)
#             traffic_matrices_train = data_train.pickle_daily_traffic(holidays=False)
#             stat_df_train = ipu.get_clusters(traffic_matrices_train, stat_df_train, 
#                                                             day_type, min_trips,
#                                                             clustering, k, seed)[0]
#             station_df = station_df.merge(
#                 ipu.neighborhood_percentages(
#                     data.city, station_df, land_use, 
#                     service_radius=service_radius, use_road=use_road
#                     ),
#                 how='outer', left_index=True, right_index=True)
            
    
#             data_test = bs.Data(city, YEAR, month)
#             stat_df_test, land_use_test = ipu.make_station_df(data_test, return_land_use=True)
#             traffic_matrices_test = data_test.pickle_daily_traffic(holidays=False)
#             stat_df_test = ipu.get_clusters(traffic_matrices_test, stat_df_test, 
#                                                             day_type, min_trips,
#                                                             clustering, k, seed)[0]
#             station_df = station_df.merge(
#                 ipu.neighborhood_percentages(
#                     data.city, station_df, land_use, 
#                     service_radius=service_radius, use_road=use_road
#                     ),
#                 how='outer', left_index=True, right_index=True)
            
#             omit_columns = {
#                             'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
#                             'chicago': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
#                             'nyc': ['percent_mixed', 'n_trips'],
#                             'washdc': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
#                             'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
#                             'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips'],
#                             'madrid': ['n_trips'],
#                             'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
#                             }
            
            
#             zone_columns_train = [column for column in stat_df_train.columns if 'percent_' in column]
#             zone_columns_train = [column for column in zone_columns_train if column not in omit_columns[city_train]]
            
#             zone_columns_test  = [column for column in stat_df_test.columns  if 'percent_' in column]
#             zone_columns_test = [column for column in zone_columns_test if column not in omit_columns[city]]
            
#             other_columns = ['pop_density', 'nearest_subway_dist']
            
            
#             LR_train, X_train, y_train = ipu.stations_logistic_regression(stat_df_train, zone_columns_train, 
#                                                                 other_columns, 
#                                                                 use_points_or_percents=use_points_or_percents, 
#                                                                 make_points_by=make_points_by, 
#                                                                 const=add_const)
            
#             LR_test, X_test, y_test = ipu.stations_logistic_regression(stat_df_test, zone_columns_test, 
#                                                                 other_columns, 
#                                                                 use_points_or_percents=use_points_or_percents, 
#                                                                 make_points_by=make_points_by, 
#                                                                 const=add_const)
            
#             success_rate, cm, predictions = ipu.logistic_regression_test(X_train, y_train, X_test, y_test, plot_cm=False)
            
#             success_rates.append(success_rate)
        
#     success_rates_all[city] = success_rates

# #%%

# fig =  plt.figure(figsize=(8,5))

# for city in city_test:
#     plt.plot(bs.get_valid_months(city, YEAR), success_rates_all[city], label=bs.name_dict[city])
    
# plt.plot(range(1,13), np.full(shape=(12,1), fill_value=1/k), color='grey', ls='--', label='Random guessing')


# plt.style.use('seaborn-darkgrid')
# plt.ylim([0,1])

# lines, labels = plt.gca().get_legend_handles_labels()

# lines.insert(4, plt.Line2D([],[], alpha=0))
# labels.insert(4,'')

# plt.legend(lines, labels, ncol=2)
# plt.xticks(ticks = range(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# plt.yticks(np.linspace(0,1,11))
# plt.xlabel('Month')
# plt.ylabel('Success rate')
# plt.tight_layout()
# # plt.title(f'Success rates from model trained on {bs.name_dict[city_train]} and tested on different cities')
# plt.savefig(f'./figures/LR_model_tests/train_on_{city_train}_test_on_{city_test}.pdf')




#%% Linear Regression
from statsmodels.api import OLS, GLS, WLS
from statsmodels.tools import add_constant
import statsmodels.formula.api as smf
import smopy

CITY = 'nyc'
YEAR = 2019
MONTH = 9

data = bs.Data(CITY, YEAR, MONTH)

station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)
traffic_matrices = data.pickle_daily_traffic(holidays=False)
station_df, clusters, labels = ipu.get_clusters(
    traffic_matrices, station_df, day_type, min_trips, clustering, k, seed)
# station_df = station_df.merge(
#     ipu.neighborhood_percentages(
#         data.city, station_df, land_use, 
#         service_radius=service_radius, use_road=use_road
#         ),
#     how='outer', left_index=True, right_index=True)

df = station_df

cols = ['percent_residential', 'percent_commercial', 'percent_industrial',
        'pop_density', 'nearest_subway_dist']

triptype = 'b_departures'

#cols = ['percent_residential']
X = df[cols]

y = df[triptype][~X.isna().any(axis=1)]

y = np.sqrt(y)

X = X[~X.isna().any(axis=1)]

X_scaled = X.copy()
if triptype in X_scaled.columns:
    X_scaled[triptype] = X_scaled[triptype]/X_scaled[triptype].sum()
if 'nearest_subway_dist' in X_scaled.columns:
    X_scaled['nearest_subway_dist'] = X_scaled['nearest_subway_dist']/1000
if 'pop_density' in X_scaled.columns:
    X_scaled['pop_density'] = X_scaled['pop_density']/10000

    

param_names = {'percent_industrial' : '% industrial',
               'percent_commercial' : '% commercial',
               'percent_residential' : '% residential',
               'percent_recreational' : '% recreational',
               'percent_mixed' : '% mixed',
               'pop_density' : 'pop density',
               'nearest_subway_dist' : 'nearest subway dist'}

X_scaled = X_scaled.rename(param_names)

X_scaled = add_constant(X_scaled)

OLS_model = OLS(y, X_scaled)

OLS_results = OLS_model.fit(maxiter=10000)

print(OLS_results.summary())


# OLS_pred = OLS_results.get_prediction()

# iv_l = OLS_pred.summary_frame()["obs_ci_lower"]
# iv_u = OLS_pred.summary_frame()["obs_ci_upper"]


# variable = 'nearest_subway_dist'


# x = X_scaled[variable]

# a = pd.concat([x, y], axis=1).sort_values(variable)

# x = a[a.columns[0]]
# y = a[a.columns[1]]


# fig, ax = plt.subplots(figsize=(8, 6))

# ax.plot(x, y, "o", label="Data")
# ax.plot(x, OLS_results.fittedvalues, "r--.", label="Predicted")
# # ax.plot(x, iv_u, "r--")
# # ax.plot(x, iv_l, "r--")
# ax.set_xlabel(variable)
# ax.set_ylabel('n_trips')
# legend = ax.legend(loc="best")

# cols = ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational',
#         'pop_density', 'nearest_subway_dist', triptype]

# df = station_df[cols]
    
# df['nearest_subway_dist'] = df['nearest_subway_dist']/1000
# df['pop_density'] = df['pop_density']/10000
# df[triptype] = np.sqrt(df[triptype])

# model = smf.ols(formula=f"""{triptype} ~ 
#                 percent_residential + percent_commercial + percent_industrial + percent_recreational + pop_density + nearest_subway_dist
#                 + percent_residential:percent_commercial + percent_residential:percent_industrial + percent_residential:percent_recreational
#                 + percent_commercial:percent_industrial + percent_commercial:percent_recreational
#                 + percent_industrial:percent_recreational
#                 """, data=df)

# model = smf.ols(formula=f"""{triptype} ~ 
#                 percent_residential + percent_commercial + percent_industrial + percent_recreational + pop_density + nearest_subway_dist
#                 + percent_commercial:percent_industrial
#                 """, data=df)

# result = model.fit()

# print(result.summary())

# For point, get percentages and pop density and nearest subway dist

def heatmap_grid(city, land_use, census_df, bounds, resolution):
    latmin, lonmin, latmax, lonmax = bounds
    grid_points = []
    for lat in np.arange(latmin, latmax, resolution):
        for lon in np.arange(lonmin, lonmax, resolution):
            grid_points.append(Point((round(lat,4), round(lon,4))))
    
    
    grid_points = gpd.GeoDataFrame(geometry=grid_points, crs='epsg:3857')
    
    grid_points['coords'] = grid_points['geometry'].to_crs(epsg=4326)
            
    grid_points = grid_points.set_geometry('coords')
    
    neighborhoods = ipu.point_neighborhoods(grid_points['coords'], land_use)

    grid_points = grid_points.join(neighborhoods)

    service_area = ipu.get_service_area(city, grid_points, land_use)
    
    grid_points['service_area'] = service_area[0]
    
    percentages = ipu.neighborhood_percentages(city, grid_points, land_use)
    pop_density = ipu.pop_density_in_service_area(grid_points, census_df)
    nearest_subway = ipu.nearest_transit(city, grid_points)

    point_info = pd.DataFrame(index=percentages.index)
    point_info['const'] = 1.0
    point_info[['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational']] = percentages[['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational']]
    point_info['pop_density'] = np.array(pop_density)/10000
    point_info['nearest_subway_dist'] = nearest_subway['nearest_subway_dist']/1000
    point_info['nearest_railway_dist'] = nearest_subway['nearest_railway_dist']/1000
    
    return grid_points, point_info


def plot_heatmap(function, grid_points, point_info, zlabel="Demand (Average daily business day departures)", cmap='magma', ax=None):
    if isinstance(function, str):
        color = point_info[function]
    else:
        color = function(point_info)
    grid_points['color'] = color

    sas = grid_points.set_geometry('service_area')
    sas.plot('color', legend=True, legend_kwds={'label': zlabel}, cmap=plt.get_cmap(cmap), ax=ax)
    
    return grid_points, point_info


def plot_multi_heatmaps(grid_points, point_info, savefig=True):
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 13))
    plt.setp(ax, xticks=[], yticks=[])
    plot_heatmap(OLS_predict_partial, grid_points, point_info, ax=ax[0,0])
    plot_heatmap('pop_density', grid_points, point_info, zlabel='Population Density (pop/10000m²)', ax=ax[0,1])
    
    plot_heatmap('percent_residential', grid_points, point_info, zlabel='Percentage of residential usage', ax=ax[1,0])
    plot_heatmap('percent_commercial', grid_points, point_info, zlabel='Percentage of commercial usage', ax=ax[1,1])
    plot_heatmap('percent_recreational', grid_points, point_info, zlabel='Percentage of recreational usage', ax=ax[2,0])
    plot_heatmap('percent_industrial', grid_points, point_info, zlabel='Percentage of industrial usage', ax=ax[2,1])
    plot_heatmap('nearest_subway_dist', grid_points, point_info, zlabel='Nearest subway distance (km)', cmap='magma_r', ax=ax[3,0])
    plot_heatmap('nearest_railway_dist', grid_points, point_info, zlabel='Nearest railway distance (km)', cmap='magma_r', ax=ax[3,1])
    plt.tight_layout()
    if savefig:
        plt.savefig(f'figures/heatmaps_{CITY}{YEAR}{MONTH}.pdf')

def OLS_predict(dataframe, OLS_results, cols):
    return OLS_results.predict(dataframe[[*cols, 'const']])

polygon = Polygon(
    [(station_df['easting'].min()-1000, station_df['northing'].min()-1000),
     (station_df['easting'].min()-1000, station_df['northing'].max()+1000),
     (station_df['easting'].max()+1000, station_df['northing'].max()+1000),
     (station_df['easting'].max()+1000, station_df['northing'].min()-1000)])

latmin, lonmin, latmax, lonmax = polygon.bounds

resolution = 250

grid_points, point_info = heatmap_grid(CITY, land_use, census_df, polygon.bounds, resolution)

OLS_predict_partial = partial(OLS_predict, OLS_results=OLS_results, cols=cols)

plot_multi_heatmaps(grid_points, point_info)

extent = (station_df['lat'].min(), station_df['long'].min(), 
      station_df['lat'].max(), station_df['long'].max())

tileserver = 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg' # Stamen Terrain
# tileserver = 'http://a.tile.stamen.com/toner/{z}/{x}/{y}.png' # Stamen Toner
# tileserver = 'http://c.tile.stamen.com/watercolor/{z}/{x}/{y}.png' # Stamen Watercolor
# tileserver = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png' # OSM Default

m = smopy.Map(extent, tileserver=tileserver)

fig, ax = plt.subplots(figsize=(7,10))

m.show_mpl(ax=ax)
