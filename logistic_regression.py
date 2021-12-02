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

