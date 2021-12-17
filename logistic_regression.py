#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:01:55 2021

@author: dbvd
"""
import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import bikeshare as bs
import interactive_plot_utils as ipu

k = 5

data = bs.Data('nyc', 2019, 9)
station_df = ipu.make_station_df(data, holidays=False)
traffic_matrices = data.pickle_daily_traffic(holidays=False)

station_df, clusters, labels = ipu.get_clusters(traffic_matrices, 
                                                station_df, 
                                                'business_days', 
                                                100, 
                                                'k_means', 
                                                k, 
                                                random_state=42)
station_df = station_df[~station_df['label'].isna()]

mean = np.mean(traffic_matrices[0][station_df.index], axis=0)

mean = mean/np.max(mean)

dist1_list = []
for center in clusters.cluster_centers_:
    dist_from_mean = np.linalg.norm(center/np.max(center)-mean)
    dist1_list.append(dist_from_mean)

avg_candidates = np.argsort(dist1_list)[:2]

dist2_list=[]
for candidate_label in avg_candidates:
    center = clusters.cluster_centers_[candidate_label]
    dist_from_zero = np.linalg.norm(center[:24]-center[24:])
    dist2_list.append(dist_from_zero)

avg_label = avg_candidates[np.argmin(dist2_list)]

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

X = pd.concat([ohe_zone, station_df[['nearest_subway_dist', 'pop_density', 'n_trips',]]], axis=1)

y = station_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

lr = LogisticRegression(max_iter=10000).fit(X_train, y_train)

coefs = pd.DataFrame(lr.coef_, columns=X.columns)

y_pred = lr.predict(X_test)

accuracy = np.sum(y_pred == y_test) / y_test.shape[0]

print(f"accuracy is {accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)#, normalize="true")

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
# zone_vec = np.array([(ohe_zone.columns == zone_type).astype(int) for zone_type in ohe_zone.columns], dtype=float)
# vectors = np.concatenate((zone_vec, np.ones((zone_vec.shape[0],1))*station_df['long'].mean(), np.ones((zone_vec.shape[0],1))*station_df['lat'].mean(), np.ones((zone_vec.shape[0],1))*station_df['nearest_subway_dist'].mean()), axis=1)
# probs = lr.predict_proba(vectors)

# for i, zone_type in enumerate(ohe_zone.columns):
#     print(zone_type)
#     print(probs[i])
#     print(probs[i].argmax())

#%% Compare cities

k = 5
min_trips = 100
seed = None

city_train = 'nyc'
month_train = 9

city_test = 'helsinki'
month_test = 9

data_train = bs.Data(city_train, 2019, month_train)
stat_df_train = ipu.make_station_df(data_train, holidays=False, overwrite=True)
traffic_matrices_train = data_train.pickle_daily_traffic(holidays=False)
stat_df_train, clusters_train, labels_train = ipu.get_clusters(traffic_matrices_train, stat_df_train, 'business_days', min_trips, 'k_means', k, random_state=seed)
#stat_df_train.dropna(inplace=True)
stat_df_train = stat_df_train[~stat_df_train['label'].isna()]

data_test = bs.Data(city_test, 2019, month_test)
stat_df_test = ipu.make_station_df(data_test, holidays=False, overwrite=True)
traffic_matrices_test = data_test.pickle_daily_traffic(holidays=False)
stat_df_test, clusters_test, labels_test = ipu.get_clusters(traffic_matrices_test, stat_df_test, 'business_days', min_trips, 'k_means', k, random_state=seed)
#stat_df_test.dropna(inplace=True)
stat_df_test = stat_df_test[~stat_df_test['label'].isna()]

ohe_zone_train = pd.get_dummies(stat_df_train['zone_type'])
X_train = pd.concat([ohe_zone_train, stat_df_train[['nearest_subway_dist', 'pop_density']]], axis=1)
y_train = stat_df_train['label']

ohe_zone_test = pd.get_dummies(stat_df_test['zone_type'])
X_test = pd.concat([ohe_zone_test, stat_df_test[['nearest_subway_dist', 'pop_density', ]]], axis=1)

#drop_indices = np.where(X_test.transportation == 1)[0]

#X_test.drop(drop_indices, inplace=True)
#X_test.drop(columns=['transportation'], axis=1, inplace=True)
y_test = stat_df_test['label']
#y_test.drop(drop_indices, inplace=True)

lr = LogisticRegression(max_iter=10000).fit(X_train, y_train)
coefs = pd.DataFrame(lr.coef_, columns=X_train.columns)

y_pred = lr.predict(X_test[['commercial', 'manufacturing', 'UNKNOWN', 'recreational', 'residential', 'nearest_subway_dist', 'pop_density']])

accuracy = np.sum(y_pred == y_test) / y_test.shape[0]

print(f"accuracy is {accuracy*100:.2f}%")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


#%% Statsmodels

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

# data = sm.datasets.scotland.load_pandas()

ohe_zone = pd.get_dummies(station_df['zone_type'])

X = ohe_zone

# X = pd.concat([X, station_df['pop_density']/station_df['pop_density'].max()], axis=1)
# X = pd.concat([X, station_df['nearest_subway_dist']/station_df['nearest_subway_dist'].max()], axis=1)
# X = pd.concat([X, station_df['n_trips']], axis=1)

# X['n_trips'] = X['n_trips']/X['n_trips'].sum()
# X['nearest_subway_dist'] = X['nearest_subway_dist']/X['nearest_subway_dist'].sum()

# X = sm.add_constant(X)

X.reset_index(inplace=True, drop=True)

bad_columns = ['mixed', 'transportation', ]
bad_indices = []
for column in bad_columns:
    if column in X.columns:
        np.concatenate([bad_indices, np.where(X[column] == 1)[0]])
        X.drop(columns = [column], inplace=True)
X.drop(index = bad_indices, inplace=True)

y = station_df['label']
y.reset_index(inplace=True, drop=True)
y.drop(index=bad_indices, inplace=True)

LR_model = MNLogit(y, X)

# LR_results = LR_model.fit(maxiter=10000)
LR_results = LR_model.fit_regularized(maxiter=10000)
# LR_results = LR_model.fit(start_params = LR_results.params, method = 'bfgs', maxiter=1000)
# print(LR_results.summary())

if not os.path.exists('stat_results'):
    os.makedirs('stat_results')


with open(f'./stat_results/{data.city}{data.year}{data.month:02d}_MNLogit_results.txt', 'w') as file:
    print(LR_results.summary())
    print(LR_results.summary(), file = file)


