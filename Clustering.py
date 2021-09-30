# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:41:18 2021

@author: Cykelholdet
"""

import os
import calendar
import pickle
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextily as ctx
import bikeshare as bs
import simpledtw as dtw
import plotting
import time
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

#%% Load data

city = 'nyc'
year = 2019
month = 9
period = 'b' # 'b' = business days or 'w' = weekends

# if city == 'nyc':
#     gov_stations = [3254, 3182, 3479]
#     data = bs.Data(city, year, month, blacklist=gov_stations)

data = bs.Data(city,year,month)

if period == 'b':
    traffic_matrix = data.pickle_daily_traffic()[0]

elif period == 'w':
    traffic_matrix = data.pickle_daily_traffc()[1]

# with open(f'./python_variables/daily_traffic_{data.city}{data.year:d}{data.month:02d}_{period}.pickle', 'rb') as file:
#         traffic_matrix=pickle.load(file)

#%% Make data

means = np.array([[5,5],
                  [3,8],
                  [8,12]])

cluster_sizes = [150,200,250]

n = np.sum(cluster_sizes)

samples_0 = np.random.multivariate_normal(means[0], np.array([[4,1],[9,2]]), size = cluster_sizes[0])
samples_1 = np.random.multivariate_normal(means[1], np.array([[4,1],[9,2]]), size = cluster_sizes[1])
samples_2 = np.random.multivariate_normal(means[2], np.array([[4,1],[9,2]]), size = cluster_sizes[2])

samples = np.append(samples_0, samples_1, axis = 0)
samples = np.append(samples, samples_2, axis = 0)

plt.scatter(samples_0[:,0],samples_0[:,1], c='b')
plt.scatter(samples_1[:,0],samples_1[:,1], c='r')
plt.scatter(samples_2[:,0],samples_2[:,1], c='g')

#%% k-test

clf = bs.Classifier(dist_func = 'norm')

clf.k_means_test(traffic_matrix, k_max = 10, seed = 69)

#%% Clustering

k = 3

clf = bs.Classifier(dist_func = 'norm')

results_filename = f'./python_variables/h_clustering_{data.city}{data.year}{data.month:02d}_{period}.pickle'
init_distance_filename = f'./python_variables/distance_matrix_{data.city}{data.year}{data.month:02d}_{period}.pickle'
# clf.h_clustering(traffic_matrix, k, results_filename, init_distance_filename)
# clf.k_means(traffic_matrix, k, seed=69)
pre = time.time()
clf.k_medioids(traffic_matrix, k)

print(time.time()-pre)

labels = clf.mass_predict(traffic_matrix)

#%% Plotting

month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
figsize_dict = {'nyc': (5,8),
                'chic': (5,8),
                'london': (8,5),
                'oslo': (6.6,5),
                'sfran': (6,5),
                'washDC': (8,7.7),
                'madrid': (6,8),
                'mexico': (7.2,8), 
                'taipei': (7.3,8)}

scalebars = {'chic': 5000,
             'london': 5000,
             'madrid': 2000,
             'mexico': 2000,
             'nyc':5000,
             'sfran':5000,
             'taipei':5000,
             'washDC':5000}

lat = [data.stat.loc_merc[i][0] for i in range(data.stat.n_tot)]
long = [data.stat.loc_merc[i][1] for i in range(data.stat.n_tot)]
    
extent = np.array([np.min(lat)-1000, np.max(lat)+1000, np.min(long)-1000, np.max(long)+1000])

try:
    fig, ax = plt.subplots(figsize=figsize_dict[data.city])
except KeyError:
    fig, ax = plt.subplots(figsize=(8,8))
    
ax.axis(extent)
ax.axis('off')

print('Drawing network...')

color_dict = {0 : 'tab:blue', 1 : 'tab:orange', 2 : 'tab:green', 3 : 'tab:red',
              4 : 'tab:purple', 5 : 'tab:brown', 6: 'tab:pink',
              7 : 'tab:gray', 8 : 'tab:olive', 9 : 'tab:cyan'}

color_map = [color_dict[label] for label in labels]

ax.scatter(lat, long, c = color_map)

print('Adding basemap... ')
ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain,
                    attribution='(C) Stamen Design, (C) OpenStreetMap contributors')

print('Adding scalebar...')
scalebar = AnchoredSizeBar(ax.transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                            pad=0.2, color='black', frameon=False, size_vertical=50)

ax.add_artist(scalebar)

markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in list(color_dict.values())[:k]]
plt.legend(markers, color_dict.keys(), numpoints=1)

if period == 'b':
    plt.title(f'Clustering for {data.city} in {month_dict[data.month]} {data.year:d} on business days')
elif period == 'w':
    plt.title(f'Clustering for {data.city} in {month_dict[data.month]} {data.year:d} on weekends')

fig.tight_layout()
fig.show()

for i in range(k):
    
    traffic_cluster = traffic_matrix[np.where(labels == i)]
    
    arrivals_std = np.std(traffic_cluster[:,24:]*100, axis=0)
    departures_std = np.std(traffic_cluster[:,:24]*100, axis=0)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(24), clf.centroids[i][24:]*100)
    ax.plot(np.arange(24), clf.centroids[i][:24]*100)
    plt.xticks(np.arange(24))
    
    plt.fill_between(np.arange(24), clf.centroids[i][24:]*100-arrivals_std, 
                             clf.centroids[i][24:]*100+arrivals_std, 
                             facecolor='b',alpha=0.2)
    plt.fill_between(np.arange(24), clf.centroids[i][:24]*100-departures_std, 
                     clf.centroids[i][:24]*100+departures_std, 
                     facecolor='orange',alpha=0.2)
    
    
    plt.legend(['Arrivals','Departures'])
    plt.xlabel('Hour')
    plt.ylabel('% of total trips')
    plt.title(f'Hourly traffic for centroid of cluster {i} (n = {len(traffic_cluster)})')
    fig.show()

#%% Plot all of one cluster

label = 3

label_indices = np.where(labels == label)

count = 0
for stat_index in label_indices[0]:
    data.daily_traffic_average(stat_index, plot=True)
    count += 1
    print(count)

#%% k-medioids

# data_mat = traffic_matrix[:800]
data_mat = samples
k = 3

n = len(data_mat)

# BUILD

d_mat = np.zeros(shape=(n,n))

for i in range(n-1):
    for j in range(i+1,n):
        d_mat[i,j] = np.linalg.norm(data_mat[i]-data_mat[j])
d_mat = np.where(d_mat, d_mat, d_mat.T)

d_sums = np.sum(d_mat, axis = 1)

m_indices = [int(np.argmin(d_sums))]
medioids = np.zeros(shape=(k,data_mat.shape[1]))
medioids[0] = data_mat[m_indices[0],:]

plt.scatter(samples_0[:,0],samples_0[:,1], c='b')
plt.scatter(samples_1[:,0],samples_1[:,1], c='r')
plt.scatter(samples_2[:,0],samples_2[:,1], c='g')
plt.scatter(medioids[:,0], medioids[:,1], c = 'k')    
plt.show()

C_mat = np.zeros(shape=(n,n))

for label in range(1,k):

    for i in range(n):
        for j in range(n):
            if i and j not in m_indices:
                
                m_distances = [
                    d_mat[m_index,j] for m_index in m_indices]
                
                closest_m = m_indices[np.argmin(m_distances)]                    
                C_mat[j,i] = max(d_mat[closest_m,j] - d_mat[i,j], 0)
    
    total_gains = np.sum(C_mat, axis = 0)
    
    m_indices.append(np.argmax(total_gains))
    
    medioids[label,:] = data_mat[m_indices[label],:]

    plt.scatter(samples_0[:,0],samples_0[:,1], c='b')
    plt.scatter(samples_1[:,0],samples_1[:,1], c='r')
    plt.scatter(samples_2[:,0],samples_2[:,1], c='g')
    plt.scatter(medioids[:,0], medioids[:,1], c = 'k')    
    plt.show()

# ASSIGN

labels = np.empty(n)
for stat_index in range(n):
    distances = np.zeros(k)    
    for label in range(k):
        distances[label] = d_mat[stat_index, m_indices[label]]
    labels[stat_index] = np.argmin(distances)

# SWAP

current_cost = 0
for label, m in enumerate(m_indices):
    current_cost += np.sum(d_mat[m,np.where(labels == label)])

old_cost = current_cost

while True:
    
    best_cost = old_cost
    
    for label, m in enumerate(m_indices):
        for h in [index for index in range(n) if index not in m_indices]:
            m_indices_prop = m_indices.copy()
            m_indices_prop[label] = h
            
            labels_prop = np.empty(n)
            for stat_index in range(n):
                distances = np.zeros(k)    
                for l in range(k):
                    distances[l] = d_mat[stat_index, m_indices_prop[l]]
                labels_prop[stat_index] = np.argmin(distances)
            
            cost = 0
            for l, m_prop in enumerate(m_indices_prop):
                cost += np.sum(d_mat[m_prop, np.where(labels_prop == l)])
            
            if cost < best_cost:
                best_cost = cost
                swap_label = label
                swap_to = h
    
    m_indices[swap_label] = swap_to
    
    medioids[swap_label] = data_mat[swap_to]

    plt.scatter(samples_0[:,0],samples_0[:,1], c='b')
    plt.scatter(samples_1[:,0],samples_1[:,1], c='r')
    plt.scatter(samples_2[:,0],samples_2[:,1], c='g')
    plt.scatter(medioids[:,0], medioids[:,1], c = 'k')    
    plt.show()
    
    
    print(m_indices)
    
    if best_cost >= old_cost:
        break

    old_cost = best_cost












# while True:
#     C_mat = np.zeros(shape=(n,n,n))
#     for i in m_indices:
#         for h in [index for index in range(n) if index not in m_indices]:
#             for j in range(n):
#                 if j != i and j != h:
                    
#                     dist_to_m = [d_mat[m,j] for m in m_indices]
                    
#                     closest_m = int(np.argmin(dist_to_m))
                    
#                     other_m = [m for m in m_indices if m != i]
#                     dist_to_other_m = [d_mat[m,j] for m in other_m]
#                     dist_to_second_closest_m = np.min(dist_to_other_m)
                    
#                     dist_to_h = d_mat[j,h]
                    
#                     if closest_m == i:
                        
#                         if dist_to_h < dist_to_second_closest_m:
#                             C_mat[j,i,h] = dist_to_h - d_mat[j,i]
                        
#                         else:
#                             C_mat[j,i,h] = dist_to_second_closest_m - d_mat[i,j]
                        
#                     elif closest_m != i and dist_to_h < dist_to_second_closest_m:
#                         C_mat[j,i,h] = dist_to_h - dist_to_second_closest_m
    
#     T_mat = np.zeros(shape=(n,n))
    
#     for m in m_indices:
#         for h in range(n):
#             T_sum = 0
#             for j in range(n):
#                 T_sum += C_mat[j,i,h]
            
#             T_mat[i,h] = T_sum
    
#     # T_mat = np.sum(C_mat, axis = 0)
#     T_min = np.min(T_mat)
    
#     print(m_indices)
#     print(T_min)
#     if T_min <= 0:
        
#         swap_from = np.where(T_mat == T_min)[0][0]
#         swap_to = np.where(T_mat == T_min)[1][0]
        
#         m_indices[np.where(m_indices == swap_from)[0][0]] = swap_to
        
#         medioids[np.where(medioids == data_mat[swap_from])] = data_mat[swap_to]
    
#     else:
#         break
    
#     plt.scatter(samples_0[:,0],samples_0[:,1], c='b')
#     plt.scatter(samples_1[:,0],samples_1[:,1], c='r')
#     plt.scatter(samples_2[:,0],samples_2[:,1], c='g')
#     plt.scatter(medioids[:,0], medioids[:,1], c = 'k')    
#     plt.show()

    
#%%

# =============================================================================
#     init_clusters_SA
# =============================================================================

data_mat = traffic_matrix
k = 4
T_start = 50
T_end = 1
alpha = 0.5
iter_time = 5

n = len(data_mat)

T = T_start

labels = np.random.randint(low = 0, high = k, size = n)

centroids = np.empty(shape=(k, data_mat.shape[1]))    

furthest_neighbors = np.empty(shape=(k,k))

inner_cluster_sum = 0
for i in range(k):
    cluster = data_mat[np.where(labels == i)]
    centroids[i,:] = np.mean(cluster, axis = 0)
    inner_distances = np.empty(len(cluster))
    
    for j, vec in enumerate(cluster):
        inner_distances[j] = dtw.dtw(vec, centroids[i])[1]
    
    furthest_neighbors[i,:] = np.argpartition(inner_distances, -k)[-k:]
    
    inner_cluster_sum += np.sum(inner_distances)

inter_cluster_sum = 0
for i in range(k-1):
    cluster_i = data_mat[np.where(labels == i)]
    
    for j in range(i+1,k):
        distances = np.empty(len(cluster_i))
        for l, vec in enumerate(cluster_i):
            distances[l] = dtw.dtw(vec, centroids[j])[1]
        
        inter_cluster_sum += np.mean(distances)

E_obj = inter_cluster_sum/inner_cluster_sum

E_best = E_obj

counter = 0
while T > T_end:
    pre = time.time()
    num = 0
    while num < iter_time:
        
        labels_new = labels.copy()
        
        for i in range(k):
            for neighbor in furthest_neighbors[i]:
                new_labels = [j for j in range(k) if j != i]
                labels_new[int(neighbor)] = np.random.choice(new_labels)
        
        centroids_new = np.empty(shape=(k, data_mat.shape[1]))    
        furthest_neighbors_new = np.empty(shape=(k,k))
        
        inner_cluster_sum = 0
        for i in range(k):
            cluster = data_mat[np.where(labels_new == i)]
            centroids_new[i,:] = np.mean(cluster, axis = 0)
            inner_distances = np.empty(len(cluster))
            
            for j, vec in enumerate(cluster):
                inner_distances[j] = dtw.dtw(vec, centroids_new[i])[1]
            
            furthest_neighbors_new[i,:] = np.argpartition(inner_distances, -k)[-k:]
            
            inner_cluster_sum += np.sum(inner_distances)
        
        inter_cluster_sum = 0
        for i in range(k-1):
            cluster_i = data_mat[np.where(labels_new == i)]
            
            for j in range(i+1,k):
                distances = np.empty(len(cluster_i))
                for l, vec in enumerate(cluster_i):
                    distances[l] = dtw.dtw(vec, centroids_new[j])[1]
                
                inter_cluster_sum += np.mean(distances)
        
        E_new = inter_cluster_sum/inner_cluster_sum
        
        dE = E_obj - E_new
        
        if dE > 0:
            labels = labels_new
            centroids = centroids_new
            furthest_neighbors = furthest_neighbors_new
            E_obj = E_new
            
            if E_best > E_new:
                E_best = E_new
                print(E_best)
                num = 0
            else:
                num += 1
                print(num)
        
        else:
            if np.random.random_sample() <= min(1,np.exp(-dE/T)):
                labels = labels_new
                centroids = centroids_new
                furthest_neighbors = furthest_neighbors_new
    
    print(f'Iteration {counter} done. Runtime: {time.time()-pre}, num = {num}, Error = {E_obj}, T = {T}, new T = {alpha*T}')
    
    T = alpha*T
    







