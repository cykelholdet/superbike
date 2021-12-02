#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:01:55 2021

@author: dbvd
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import bikeshare as bs
import interactive_plot_utils as ipu

k = 3

data = bs.Data('nyc', 2019, 8)
station_df = ipu.make_station_df(data, holidays=False)
traffic_matrices = data.pickle_daily_traffic(holidays=False)

station_df, clusters, labels = ipu.get_clusters(traffic_matrices, 
                                                station_df, 
                                                'business_days', 
                                                100, 
                                                'k_means', 
                                                k, 
                                                random_state=42)
station_df = station_df.dropna()

mean = np.mean(traffic_matrices[0][station_df.index], axis=0)

dist_list=[]
for label, center in enumerate(clusters.cluster_centers_):
    dist = np.linalg.norm(center/np.max(center)-mean/np.max(mean))
    dist_list.append(dist)

avg_label = np.argmin(dist_list)

new_labels = [avg_label]
for i in range(1,k):
    if i == avg_label:
        new_labels.append(0)
    else:
        new_labels.append(i)

labels_dict = dict(zip(range(len(new_labels)), new_labels))

station_df.replace({'label' : labels_dict}, inplace=True)

#%% sklearn


ohe_zone = pd.get_dummies(station_df['zone_type'])

X = pd.concat([ohe_zone, station_df[['nearest_subway_dist', 'pop_density', 'n_trips',]], pd.DataFrame(traffic_matrices[0])], axis=1)

y = station_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression(max_iter=10000).fit(X_train, y_train)

coefs = pd.DataFrame(lr.coef_, columns=X.columns)

y_pred = lr.predict(X_test)

accuracy = np.sum(y_pred == y_test) / y_test.shape[0]

print(f"accuracy is {accuracy*100:.2f}%")

# zone_vec = np.array([(ohe_zone.columns == zone_type).astype(int) for zone_type in ohe_zone.columns], dtype=float)
# vectors = np.concatenate((zone_vec, np.ones((zone_vec.shape[0],1))*station_df['long'].mean(), np.ones((zone_vec.shape[0],1))*station_df['lat'].mean(), np.ones((zone_vec.shape[0],1))*station_df['nearest_subway_dist'].mean()), axis=1)
# probs = lr.predict_proba(vectors)

# for i, zone_type in enumerate(ohe_zone.columns):
#     print(zone_type)
#     print(probs[i])
#     print(probs[i].argmax())

#%% Compare cities

k = 2
min_trips = 100
seed = None

city_train = 'nyc'
month_train = 9

city_test = 'helsinki'
month_test = 9

data_train = bs.Data(city_train, 2019, month_train)
stat_df_train = ipu.make_station_df(data_train, holidays=False)
traffic_matrices_train = data_train.pickle_daily_traffic(holidays=False)
stat_df_train, clusters_train, labels_train = ipu.get_clusters(traffic_matrices_train, stat_df_train, 'business_days', min_trips, 'k_means', k, random_state=seed)
stat_df_train.dropna(inplace=True)

data_test = bs.Data(city_test, 2019, month_test)
stat_df_test = ipu.make_station_df(data_test, holidays=False)
traffic_matrices_test = data_test.pickle_daily_traffic(holidays=False)
stat_df_test, clusters_test, labels_test = ipu.get_clusters(traffic_matrices_test, stat_df_test, 'business_days', min_trips, 'k_means', k, random_state=seed)
stat_df_test.dropna(inplace=True)

ohe_zone_train = pd.get_dummies(stat_df_train['zone_type'])
X_train = pd.concat([ohe_zone_train, stat_df_train[['nearest_subway_dist', 'pop_density']]], axis=1)
y_train = stat_df_train['label']

ohe_zone_test = pd.get_dummies(stat_df_test['zone_type'])
X_test = pd.concat([ohe_zone_test, stat_df_test[['nearest_subway_dist', 'pop_density', ]]], axis=1)

drop_indices = np.where(X_test.transportation == 1)[0]

X_test.drop(drop_indices, inplace=True)
X_test.drop(columns=['transportation'], axis=1, inplace=True)
y_test = stat_df_test['label']
y_test.drop(drop_indices, inplace=True)

lr = LogisticRegression(max_iter=10000).fit(X_train, y_train)
coefs = pd.DataFrame(lr.coef_, columns=X_train.columns)

y_pred = lr.predict(X_test)

accuracy = np.sum(y_pred == y_test) / y_test.shape[0]

print(f"accuracy is {accuracy*100:.2f}%")


#%% Statsmodels

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

# data = sm.datasets.scotland.load_pandas()

ohe_zone = pd.get_dummies(station_df['zone_type'])
X = pd.concat([ohe_zone, station_df[['nearest_subway_dist', 'pop_density', 'n_trips']]], axis=1)
X['n_trips'] = X['n_trips']/X['n_trips'].sum()
X['nearest_subway_dist'] = X['nearest_subway_dist']/X['nearest_subway_dist'].sum()

# X = ohe_zone

X = sm.add_constant(X)
y = station_df['label']


LR_model = MNLogit(y, X)
LR_results = LR_model.fit(maxiter=1000)
print(LR_results.summary())

# sk_model = LogisticRegression(max_iter=10000)
# sk_results = sk_model.fit(X,y)

