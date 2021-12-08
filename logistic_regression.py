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

import bikeshare as bs
import interactive_plot_utils as ipu

k = 5

data = bs.Data('london', 2019, 9)
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

# X = pd.concat([ohe_zone, station_df[['nearest_subway_dist', 'pop_density', 'n_trips',]], pd.DataFrame(traffic_matrices[0])], axis=1)
X = pd.concat([ohe_zone, station_df[['nearest_subway_dist', 'pop_density', 'n_trips']]], axis=1)
# X = ohe_zone
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

#%% Statsmodels

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

# data = sm.datasets.scotland.load_pandas()

ohe_zone = pd.get_dummies(station_df['zone_type'])

X = ohe_zone

X = pd.concat([X, station_df['pop_density']/station_df['pop_density'].max()], axis=1)
X = pd.concat([X, station_df['nearest_subway_dist']/station_df['nearest_subway_dist'].max()], axis=1)
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


