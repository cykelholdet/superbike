# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 09:30:43 2022

@author: Nicolai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import bikeshare as bs
import interactive_plot_utils as ipu
from logistic_table import lr_coefficients

CITY = 'nyc'
YEAR = 2019
MONTH = 9

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

#%% Make station_df

if use_whole_year:
    for month in bs.get_valid_months(city, YEAR):
        
        data = bs.Data(CITY, YEAR, month)
        
        station_df, land_use = ipu.make_station_df(data, return_land_use=True)
        
        traffic_matrices = data.pickle_daily_traffic(holidays=False)
         
        station_df, clusters, labels = ipu.get_clusters(traffic_matrices, station_df, 
                                                        day_type, min_trips, 
                                                        clustering, k, seed)
            
        station_df = ipu.service_areas(CITY, station_df, land_use, 
                                       service_radius=service_radius, 
                                       use_road=use_road)
        
        station_df['month'] = month
        
        if month == 1:
            station_df_year = station_df
            
        else:
            station_df_year = pd.concat([station_df_year, station_df])
    
else:
    data = bs.Data(CITY, YEAR, MONTH)
        
    station_df, land_use = ipu.make_station_df(data, return_land_use=True)
    
    traffic_matrices = data.pickle_daily_traffic(holidays=False)
     
    station_df, clusters, labels = ipu.get_clusters(traffic_matrices, station_df, 
                                                    day_type, min_trips, 
                                                    clustering, k, seed)
        
    station_df = ipu.service_areas(CITY, station_df, land_use, 
                                   service_radius=service_radius, 
                                   use_road=use_road)
    
    
#%% Make logistic regression

zone_columns = [column for column in station_df.columns if 'percent_' in column]
# zone_columns.remove('percent_mixed')
# zone_columns.remove('percent_educational')


other_columns = ['n_trips', 'pop_density', 'nearest_subway_dist']

LR_results, X, y, predictions = ipu.stations_logistic_regression(station_df, zone_columns, 
                                                    other_columns, 
                                                    use_points_or_percents=use_points_or_percents, 
                                                    make_points_by=make_points_by, 
                                                    const=add_const,
                                                    test_model=True,
                                                    plot_cm=True, 
                                                    normalise_cm='true')

print(LR_results.summary())

#%% Compare whole year model with monthly models 1

city_train = 'nyc'


omit_columns = {
                           'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
                           'chic': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
                           'nyc': ['percent_mixed', 'n_trips'],
                           'washDC': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
                           'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
                           'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips'],
                           'madrid': ['n_trips'],
                           'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
                           }


data_train_year = bs.Data(city_train, YEAR, None)
LR_train_year, X_train_year, y_train_year = lr_coefficients(data_train, data.city, min_trips=min_trips, clustering=clustering, k=k, 
                                                random_state=seed, day_type=day_type, 
                                                service_radius=service_radius, use_points_or_percents=use_points_or_percents, 
                                                make_points_by=make_points_by, add_const=add_const, 
                                                use_road=use_road, remove_columns=omit_columns[city_train], title='City', 
                                                big_station_df=False, return_model=True)[1:]

success_rates_year = []
success_rates_month = []
for month in bs.get_valid_months(city_train, YEAR):
    
    data_train = bs.Data(city_train, YEAR, month)
    stat_df_train, land_use_train = ipu.make_station_df(data_train, return_land_use=True)
    traffic_matrices_train = data_train.pickle_daily_traffic(holidays=False)
    stat_df_train = ipu.get_clusters(traffic_matrices_train, stat_df_train, 
                                                    day_type, min_trips,
                                                    clustering, k, seed)[0]
    stat_df_train = ipu.service_areas(city_train, stat_df_train, land_use_train, 
                                   service_radius=service_radius, 
                                   use_road=use_road)
    
    data_test = bs.Data(city_train, YEAR, month)
    stat_df_test, land_use_test = ipu.make_station_df(data_test, return_land_use=True)
    traffic_matrices_test = data_test.pickle_daily_traffic(holidays=False)
    stat_df_test = ipu.get_clusters(traffic_matrices_test, stat_df_test, 
                                                    day_type, min_trips,
                                                    clustering, k, seed)[0]
    stat_df_test = ipu.service_areas(city_train, stat_df_test, land_use_test, 
                                   service_radius=service_radius, 
                                   use_road=use_road)
    
    zone_columns_train = [column for column in stat_df_train.columns if 'percent_' in column]
    zone_columns_train = [column for column in zone_columns_train if column not in omit_columns[city_train]]
           
    
    zone_columns_test  = [column for column in stat_df_test.columns  if 'percent_' in column]
    zone_columns_test =  [column for column in zone_columns_test if column not in omit_columns[city_train]]
    
    other_columns = ['pop_density', 'nearest_subway_dist']

    LR_train, X_train, y_train = ipu.stations_logistic_regression(stat_df_train, zone_columns_train, 
                                                                other_columns, 
                                                                use_points_or_percents=use_points_or_percents, 
                                                                make_points_by=make_points_by, 
                                                                const=add_const)

    LR_test, X_test, y_test = ipu.stations_logistic_regression(stat_df_test, zone_columns_test, 
                                                        other_columns, 
                                                        use_points_or_percents=use_points_or_percents, 
                                                        make_points_by=make_points_by, 
                                                        const=add_const)

            
    success_rates_year.append(ipu.logistic_regression_test(X_train_year, y_train_year, X_test, y_test, plot_cm=False)[0])
    success_rates_month.append(ipu.logistic_regression_test(X_train, y_train, X_test, y_test, plot_cm=False)[0])

#%% Compare whole year model with monthly models 2

city_train = 'nyc'
split_seed = 42

omit_columns = {
                           'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
                           'chic': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
                           'nyc': ['percent_mixed', 'n_trips'],
                           'washDC': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
                           'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
                           'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips'],
                           'madrid': ['n_trips'],
                           'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
                           }


stat_df_train_year_bad, traffic_matrices_train_year_bad, labels, land_use = ipu.make_station_df_year(
    city_train, year=YEAR, months=None, service_radius=service_radius,
    use_road=use_road, day_type=day_type,
    min_trips=min_trips, clustering=clustering, k=k,
    random_state=seed, return_land_use=True)

# stat_df_train_year_bad = ipu.get_clusters(traffic_matrices_train_year_bad, stat_df_train_year_bad, 
#                                                     day_type, min_trips,
#                                                     clustering, k, seed)[0]
# stat_df_train_year_bad = ipu.service_areas(city_train, stat_df_train_year_bad, land_use, 
#                                 service_radius=service_radius, 
#                                 use_road=use_road)


 
other_columns_year = ['pop_density', 'nearest_subway_dist', 'Jan', 'Feb', 'Mar',
                 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']         


other_columns_month = ['pop_density', 'nearest_subway_dist']         


# LR_train_year_bad, X_train_year_bad, y_train_year_bad = ipu.stations_logistic_regression(stat_df_train_year, zone_columns_train_year, 
#                                                                 other_columns_month, 
#                                                                 use_points_or_percents=use_points_or_percents, 
#                                                                 make_points_by=make_points_by, 
#                                                                 const=add_const)
stat_df_train_bad_list = []
stat_df_train_list = []
stat_df_test_list = []
for month in bs.get_valid_months(city_train, YEAR):
        
    data = bs.Data(city_train, YEAR, month)
    
    station_df, land_use = ipu.make_station_df(data, return_land_use=True)
    
    traffic_matrices = data.pickle_daily_traffic(holidays=False)
     
    station_df, clusters, labels = ipu.get_clusters(traffic_matrices, station_df, 
                                                    day_type, min_trips, 
                                                    clustering, k, seed)
        
    station_df = ipu.service_areas(CITY, station_df, land_use, 
                                   service_radius=service_radius, 
                                   use_road=use_road)
    
    for i in bs.get_valid_months(city_train, YEAR):
        if i == month:
            station_df[month_dict[i]] = 1
        else:
            station_df[month_dict[i]] = 0
    
    stat_df_train, stat_df_test = train_test_split(station_df, test_size=0.2, random_state=split_seed)
    
    stat_df_train_bad = stat_df_train_year_bad[stat_df_train_year_bad[month_dict[month]]==1]
    stat_df_train_bad = stat_df_train_bad[~stat_df_train_bad['stat_id'].isin(stat_df_test['stat_id'].to_list())]
    
    stat_df_train_bad_list.append(stat_df_train_bad)
    stat_df_train_list.append(stat_df_train)
    stat_df_test_list.append(stat_df_test)
    
stat_df_train_year = pd.concat(stat_df_train_list)
stat_df_train_year_bad = pd.concat(stat_df_train_bad_list)

zone_columns_train_year = [column for column in stat_df_train_year.columns if 'percent_' in column]
zone_columns_train_year = [column for column in zone_columns_train_year if column not in omit_columns[city_train]]

LR_train_year, X_train_year, y_train_year = ipu.stations_logistic_regression(stat_df_train_year, zone_columns_train_year, 
                                                                other_columns_month, 
                                                                use_points_or_percents=use_points_or_percents, 
                                                                make_points_by=make_points_by, 
                                                                const=add_const)


LR_train_year_bad, X_train_year_bad, y_train_year_bad = ipu.stations_logistic_regression(stat_df_train_year_bad, zone_columns_train_year, 
                                                                other_columns_month, 
                                                                use_points_or_percents=use_points_or_percents, 
                                                                make_points_by=make_points_by, 
                                                                const=add_const)
    

success_rates_year = []
success_rates_year_bad = []
success_rates_month = []
for i, month in enumerate(bs.get_valid_months(city_train, YEAR)):
    
    # data_train = bs.Data(city_train, YEAR, month)
    # stat_df_train, land_use_train = ipu.make_station_df(data_train, return_land_use=True)
    # traffic_matrices_train = data_train.pickle_daily_traffic(holidays=False)
    # stat_df_train = ipu.get_clusters(traffic_matrices_train, stat_df_train, 
    #                                                 day_type, min_trips,
    #                                                 clustering, k, seed)[0]
    # stat_df_train = ipu.service_areas(city_train, stat_df_train, land_use_train, 
    #                                service_radius=service_radius, 
    #                                use_road=use_road)
    
    # data_test = bs.Data(city_train, YEAR, month)
    # stat_df_test, land_use_test = ipu.make_station_df(data_test, return_land_use=True)
    # traffic_matrices_test = data_test.pickle_daily_traffic(holidays=False)
    # stat_df_test = ipu.get_clusters(traffic_matrices_test, stat_df_test, 
    #                                                 day_type, min_trips,
    #                                                 clustering, k, seed)[0]
    # stat_df_test = ipu.service_areas(city_train, stat_df_test, land_use_test, 
    #                                service_radius=service_radius, 
    #                                use_road=use_road)
    
    # for i in get_valid_months(city_train, YEAR):
    #     if i == month:
    #         stat_df_test[month_dict[i]] = 1
    #     else:
    #         stat_df_test[month_dict[i]] = 0
    
    
    
    
    
    zone_columns_train = [column for column in stat_df_train.columns if 'percent_' in column]
    zone_columns_train = [column for column in zone_columns_train if column not in omit_columns[city_train]]
           
    
    zone_columns_test  = [column for column in stat_df_test.columns  if 'percent_' in column]
    zone_columns_test =  [column for column in zone_columns_test if column not in omit_columns[city_train]]
    

    LR_train, X_train, y_train = ipu.stations_logistic_regression(stat_df_train_list[i], zone_columns_train, 
                                                                other_columns_month, 
                                                                use_points_or_percents=use_points_or_percents, 
                                                                make_points_by=make_points_by, 
                                                                const=add_const)

    LR_test, X_test, y_test = ipu.stations_logistic_regression(stat_df_test_list[i], zone_columns_test, 
                                                        other_columns_year, 
                                                        use_points_or_percents=use_points_or_percents, 
                                                        make_points_by=make_points_by, 
                                                        const=add_const)

            
    success_rates_year.append(ipu.logistic_regression_test(X_train_year, y_train_year, X_test, y_test, plot_cm=False)[0])
    success_rates_year_bad.append(ipu.logistic_regression_test(X_train_year_bad, y_train_year_bad, X_test, y_test, plot_cm=False)[0])
    success_rates_month.append(ipu.logistic_regression_test(X_train, y_train, X_test, y_test, plot_cm=False)[0])



#%%

fig =  plt.figure(figsize=(8,5))

plt.plot(bs.get_valid_months(city_train, YEAR), success_rates_month, label='Trained monthly, clustered monthly')
plt.plot(bs.get_valid_months(city_train, YEAR), success_rates_year, label='Trained whole year, clustered monthly')
plt.plot(bs.get_valid_months(city_train, YEAR), success_rates_year_bad, label='Trained whole year, clustered whole year')


plt.plot(range(1,13), np.full(shape=(12,1), fill_value=1/k), color='grey', ls='--', label='random guessing')

plt.style.use('seaborn-darkgrid')
plt.ylim([0,1])
plt.legend()
plt.xticks(ticks = range(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.yticks(np.linspace(0,1,11))
plt.xlabel('Month')
plt.ylabel('Success rate')
# plt.title(f'Success rates from model trained on {bs.name_dict[city_train]} and tested on different cities')
# plt.savefig(f'./figures/LR_model_tests/train_on_{city_train}_whole_year_vs_monthly.pdf')
plt.savefig(f'./figures/LR_model_tests/{city_train}_yealy_vs_monthly_no_month_cols.pdf')












#%% Compare cities

city_train = 'nyc'
city_test = ['nyc', 'chic', 'washDC', 'boston', 'london', 'madrid', 'helsinki', 'oslo']
# city_test = ['nyc', 'chic', 'washDC', 'boston']
# city_test =  ['london', 'madrid', 'helsinki', 'oslo']

# city_test= ['minn']

success_rates_all = dict()

for city in city_test:
    success_rates = []
    for month in bs.get_valid_months(city, YEAR):
        if month in bs.get_valid_months(city_train, YEAR):
            data_train = bs.Data(city_train, YEAR, month)
            stat_df_train, land_use_train = ipu.make_station_df(data_train, return_land_use=True)
            traffic_matrices_train = data_train.pickle_daily_traffic(holidays=False)
            stat_df_train = ipu.get_clusters(traffic_matrices_train, stat_df_train, 
                                                            day_type, min_trips,
                                                            clustering, k, seed)[0]
            stat_df_train = ipu.service_areas(city_train, stat_df_train, land_use_train, 
                                           service_radius=service_radius, 
                                           use_road=use_road)
            
    
            data_test = bs.Data(city, YEAR, month)
            stat_df_test, land_use_test = ipu.make_station_df(data_test, return_land_use=True)
            traffic_matrices_test = data_test.pickle_daily_traffic(holidays=False)
            stat_df_test = ipu.get_clusters(traffic_matrices_test, stat_df_test, 
                                                            day_type, min_trips,
                                                            clustering, k, seed)[0]
            stat_df_test = ipu.service_areas(city, stat_df_test, land_use_test, 
                                           service_radius=service_radius, 
                                           use_road=use_road)
            
            omit_columns = {
                            'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
                            'chic': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
                            'nyc': ['percent_mixed', 'n_trips'],
                            'washDC': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
                            'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
                            'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips'],
                            'madrid': ['n_trips'],
                            'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
                            }
            
            
            zone_columns_train = [column for column in stat_df_train.columns if 'percent_' in column]
            zone_columns_train = [column for column in zone_columns_train if column not in omit_columns[city_train]]
            
            zone_columns_test  = [column for column in stat_df_test.columns  if 'percent_' in column]
            zone_columns_test = [column for column in zone_columns_test if column not in omit_columns[city]]
            
            other_columns = ['pop_density', 'nearest_subway_dist']
            
            
            LR_train, X_train, y_train = ipu.stations_logistic_regression(stat_df_train, zone_columns_train, 
                                                                other_columns, 
                                                                use_points_or_percents=use_points_or_percents, 
                                                                make_points_by=make_points_by, 
                                                                const=add_const)
            
            LR_test, X_test, y_test = ipu.stations_logistic_regression(stat_df_test, zone_columns_test, 
                                                                other_columns, 
                                                                use_points_or_percents=use_points_or_percents, 
                                                                make_points_by=make_points_by, 
                                                                const=add_const)
            
            success_rate, cm, predictions = ipu.logistic_regression_test(X_train, y_train, X_test, y_test, plot_cm=False)
            
            success_rates.append(success_rate)
        
    success_rates_all[city] = success_rates

#%%

fig =  plt.figure(figsize=(8,5))

for city in city_test:
    plt.plot(bs.get_valid_months(city, YEAR), success_rates_all[city], label=bs.name_dict[city])
    
plt.plot(range(1,13), np.full(shape=(12,1), fill_value=1/k), color='grey', ls='--', label='Random guessing')


plt.style.use('seaborn-darkgrid')
plt.ylim([0,1])

lines, labels = plt.gca().get_legend_handles_labels()

lines.insert(4, plt.Line2D([],[], alpha=0))
labels.insert(4,'')

plt.legend(lines, labels, ncol=2)
plt.xticks(ticks = range(1,13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.yticks(np.linspace(0,1,11))
plt.xlabel('Month')
plt.ylabel('Success rate')
plt.tight_layout()
# plt.title(f'Success rates from model trained on {bs.name_dict[city_train]} and tested on different cities')
plt.savefig(f'./figures/LR_model_tests/train_on_{city_train}_test_on_{city_test}.pdf')



















