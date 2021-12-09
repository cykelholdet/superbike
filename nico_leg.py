# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:41:18 2021

@author: Cykelholdet
"""

import os
import calendar
import pickle
import numpy as np
# import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
# import contextily as ctx
import bikeshare as bs
import interactive_plot_utils as ipu
import simpledtw as dtw
# import plotting
import time
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


#%% Load data

city = 'nyc'
year = 2019
month = 9
period = 'b' # 'b' = business days or 'w' = weekends
holidays = False
min_trips = 100

# if city == 'nyc':
#     gov_stations = [3254, 3182, 3479]
#     data = bs.Data(city, year, month, blacklist=gov_stations)

data = bs.Data(city,year,month)

if period == 'b':
    traffic_matrix = data.pickle_daily_traffic(holidays=holidays)[0]
    x_trips = 'b_trips'
elif period == 'w':
    traffic_matrix = data.pickle_daily_traffic(holidays=holidays)[1]
    x_trips = 'w_trips'

station_df = ipu.make_station_df(data, holidays = holidays)
mask = station_df[x_trips] > min_trips
station_df = station_df[mask]
traffic_matrix = traffic_matrix[mask]

#%% Make toy data

np.random.seed(10)

data_type = 'sphere'

means = np.array([[12,5.5],
                  [4,6],
                  [8,12]])

# means = np.array([[12,7],
#                   [5,9]])

cluster_sizes = [150,200,250]

n = np.sum(cluster_sizes)

if data_type == 'sphere':
    samples_0 = np.random.multivariate_normal(means[0], np.array([[2,0],[0,5]]), size = cluster_sizes[0])
    samples_1 = np.random.multivariate_normal(means[1], np.array([[2,0],[0,3]]), size = cluster_sizes[1])
    samples_2 = np.random.multivariate_normal(means[2], 2*np.identity(2), size = cluster_sizes[2])

    samples = np.append(samples_0, samples_1, axis = 0)
    samples = np.append(samples, samples_2, axis = 0)


elif data_type == 'multi_sphere':

    means = np.array([[12,11],
                      [4,14],
                      [8,20],
                      [21,45],
                      [31,17],
                      [28,46],
                      [35,11]])
    
    cluster_sizes = [200,350,150,250,180,220,210]
        
    n = np.sum(cluster_sizes)
    
    samples_0 = np.random.multivariate_normal(means[0], 2*np.identity(2), size = cluster_sizes[0])
    samples_1 = np.random.multivariate_normal(means[1], 2*np.identity(2), size = cluster_sizes[1])
    samples_2 = np.random.multivariate_normal(means[2], 2*np.identity(2), size = cluster_sizes[2])
    
    samples_3 = np.random.multivariate_normal(means[3], 2*np.identity(2), size = cluster_sizes[3])
    samples_4 = np.random.multivariate_normal(means[4], 2*np.identity(2), size = cluster_sizes[4])
    
    samples_5 = np.random.multivariate_normal(means[5], 2*np.identity(2), size = cluster_sizes[5])
    samples_6 = np.random.multivariate_normal(means[6], 2*np.identity(2), size = cluster_sizes[6])

    samples = np.append(samples_0, samples_1, axis = 0)
    samples = np.append(samples, samples_2, axis = 0)
    samples = np.append(samples, samples_3, axis = 0)
    samples = np.append(samples, samples_4, axis = 0)
    samples = np.append(samples, samples_5, axis = 0)
    samples = np.append(samples, samples_6, axis = 0)
    


elif data_type == 'elon':
    samples_0 = np.random.multivariate_normal(means[0], np.array([[2,1],[4,10]]), size = cluster_sizes[0])
    # samples_1 = np.random.multivariate_normal(means[1], np.array([[4,1],[9,2]]), size = cluster_sizes[1])
    # samples_2 = np.random.multivariate_normal(means[2], np.array([[4,1],[9,2]]), size = cluster_sizes[2])

    samples = np.append(samples_0, samples_1, axis = 0)
    samples = np.append(samples, samples_2, axis = 0)


# gm = GaussianMixture(n_components=2, verbose = True).fit(samples)

# rgb = [[i[0], 0, i[1]] for i in gm.predict_proba(samples)]


# plt.xlim(0,16)
# plt.ylim(0,16)
plt.scatter(samples[:,0],samples[:,1], c='b')
# plt.scatter(samples_1[:,0],samples_1[:,1], c='b')
# plt.scatter(samples_2[:,0],samples_2[:,1], c='g')

#%% toy test

# means = np.array([[5,5],
#                   [3,10]])


# cluster_sizes = [150,200]

# n = np.sum(cluster_sizes)

# samples_0 = np.random.multivariate_normal(means[0], np.identity(2), size = cluster_sizes[0])
# samples_1 = np.random.multivariate_normal(means[1], np.identity(2), size = cluster_sizes[1])
    
# samples = np.append(samples_0, samples_1, axis = 0)

clf = bs.Classifier(dist_func = 'norm')

# clf.k_medoids(samples, 2)
clf.k_means(samples, 1)
labels = clf.mass_predict(samples)


color_dict = {0: 'r', 1 : 'b'}

color_map = [color_dict[label] for label in labels]

plt.subplot(221)
plt.scatter(samples_0[:,0],samples_0[:,1], c='b')
plt.scatter(samples_1[:,0],samples_1[:,1], c='r')
plt.title('Generated data')
plt.xticks([])

plt.subplot(222)
plt.scatter(samples[:,0], samples[:,1], c = color_map)
plt.scatter(clf.centroids[:,0],clf.centroids[:,1], c = 'k')
plt.yticks([])
plt.xticks([])
plt.title('$k$-medoids clustering')

samples_0 = np.random.multivariate_normal(means[0], np.array([[4,1],[9,2]]), size = cluster_sizes[0])
samples_1 = np.random.multivariate_normal(means[1], np.array([[4,1],[9,2]]), size = cluster_sizes[1])
    
samples = np.append(samples_0, samples_1, axis = 0)

clf = bs.Classifier(dist_func = 'norm')

clf.k_medoids(samples, 2)
# clf.k_means(samples, 2)
labels = clf.mass_predict(samples)


color_dict = {0: 'r', 1 : 'b'}

color_map = [color_dict[label] for label in labels]

plt.subplot(223)
plt.scatter(samples_0[:,0],samples_0[:,1], c='b')
plt.scatter(samples_1[:,0],samples_1[:,1], c='r')


plt.subplot(224)
plt.scatter(samples[:,0], samples[:,1], c = color_map)
plt.scatter(clf.centroids[:,0],clf.centroids[:,1], c = 'k')
plt.yticks([])

plt.savefig('./figures/k_means_toy_test.pdf')

#%% Outlier test

np.random.seed(42)
samples = np.random.multivariate_normal([5,5], np.identity(2)/100, size = 5)
outlier = np.array([[10,7]])

samples = np.append(samples, outlier, axis = 0)

mean = np.mean(samples, axis = 0)

plt.subplot(211)
plt.scatter(samples[:,0], samples[:,1])
plt.scatter(mean[0],mean[1], c = 'k')
plt.xticks([])

plt.subplot(212)
plt.scatter(samples[:,0], samples[:,1])

#%% k-test

k_max = 10

traffic_matrix = samples

SSEs = []
DB_indexes = []
D_indexes = []
S_indexes = []

pre = time.time()
for k in range(2,k_max+1):
    
    clf = KMeans(n_clusters = k, random_state = 42)
    clf.fit(samples)
    
    labels = clf.labels_
    centroids = clf.cluster_centers_
    
    
    SSEs.append(clf.inertia_)
    S_indexes.append(bs.silhouette_index(traffic_matrix, labels, centroids, mute = True))
    DB_indexes.append(bs.Davies_Bouldin_index(traffic_matrix, labels, centroids, mute = True))
    D_indexes.append(bs.Dunn_index(traffic_matrix, labels, centroids, mute = True))
    
    print(f'k = {k} done. Current Runtime: {time.time()-pre}s')
    
# plt.plot(range(1,k_max+1), SSEs)
plt.plot(range(2,k_max+1), DB_indexes)
# plt.plot(range(1,k_max+1), D_indexes)
# plt.plot(range(1,k_max+1), S_indexes)

plt.xticks(range(2,k_max+1))

plt.xlabel('$k$')
plt.ylabel('Davies-Bouldin index')
# plt.title('Test for $k$')
# plt.legend(['DB index', 'D index', 'S index'])

#%% Plot clustering for toy test

k = 7

clf = KMeans(n_clusters = k, random_state = 42)
clf.fit(samples)

color_dict = {0 : 'tab:blue', 1 : 'tab:orange', 2 : 'tab:green', 3 : 'tab:red',
              4 : 'tab:purple', 5 : 'tab:brown', 6: 'tab:pink',
              7 : 'tab:gray', 8 : 'tab:olive', 9 : 'tab:cyan'}

color_map = [color_dict[label] for label in clf.labels_]

plt.scatter(samples[:,0],samples[:,1], c= color_map)
plt.scatter(clf.cluster_centers_[:,0], clf.cluster_centers_[:,1],c='k')

#%% Clustering

k = 3

clf = bs.Classifier(dist_func = 'norm')

results_filename = f'./python_variables/h_clustering_{data.city}{data.year}{data.month:02d}_{period}.pickle'
init_distance_filename = f'./python_variables/distance_matrix_{data.city}{data.year}{data.month:02d}_{period}.pickle'
# clf.h_clustering(traffic_matrix, k, results_filename, init_distance_filename)
# clf.k_means(traffic_matrix, k, seed=69)
pre = time.time()
clf.k_medoids(traffic_matrix, k)

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

# print('Adding basemap... ')
# ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain,
#                     attribution='(C) Stamen Design, (C) OpenStreetMap contributors')

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

#%% Voronoi test

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, LineString
# import shapely.geometry
import shapely.ops
import pandas as pd
import geopandas as gpd

data = bs.Data('nyc', 2019, 9)
station_df, land_use_df = ipu.make_station_df(data, return_land_use=True)

service_radius  = 1000

# np.random.seed(42)

# points = np.random.random((10, 2))

points = station_df[['easting', 'northing']].to_numpy()

points_gdf = gpd.GeoDataFrame(geometry = [Point(station_df.iloc[i]['easting'], 
                                              station_df.iloc[i]['northing'])
                       for i in range(len(station_df))], crs='EPSG:3857')

points_gdf['point'] = points_gdf['geometry']
# points_gdf['geometry'] = points_gdf['point']
# points_gdf.set_crs(epsg=3857, inplace=True)

mean_point= np.mean(points, axis=0)
edge_dist = 1000000
edge_points = np.array([[mean_point[0]-edge_dist, mean_point[1]-edge_dist],
                        [mean_point[0]-edge_dist, mean_point[1]+edge_dist],
                        [mean_point[0]+edge_dist, mean_point[1]+edge_dist],
                        [mean_point[0]+edge_dist, mean_point[1]-edge_dist]])
# points = np.concatenate([points, edge_points], axis = 0)

vor = Voronoi(np.concatenate([points, edge_points], axis=0))
# voronoi_plot_2d(vor)

lines = [LineString(vor.vertices[line])
    for line in vor.ridge_vertices
    if -1 not in line
]


poly_gdf = gpd.GeoDataFrame()
poly_gdf['vor_poly'] = [poly for poly in shapely.ops.polygonize(lines)]
poly_gdf['geometry'] = poly_gdf['vor_poly']
poly_gdf.set_crs(epsg=3857, inplace=True)

poly_gdf = gpd.tools.sjoin(points_gdf, poly_gdf, op='within', how='left')
poly_gdf.drop('index_right', axis=1, inplace=True)

poly_gdf['service_area'] = [
    row['vor_poly'].intersection(row['point'].buffer(service_radius))
    for i, row in poly_gdf.iterrows()]

poly_gdf['geometry'] = poly_gdf['service_area']
poly_gdf.set_crs(epsg=3857, inplace=True)
poly_gdf.to_crs(epsg=4326, inplace=True)
poly_gdf['service_area'] = poly_gdf['geometry']

station_df = gpd.tools.sjoin(station_df, poly_gdf, op='within', how='left')
station_df.drop(columns=['index_right', 'vor_poly', 'point'], inplace=True)

# zoning_df = gpd.read_file('./data/other_data/nyc_zoning_data.json')
union = shapely.ops.unary_union(land_use_df.geometry)

station_df['service_area'] = station_df['service_area'].apply(lambda area: area.intersection(union))

service_area_trim = []
for i, row in station_df.iterrows():
    if isinstance(row['service_area'], shapely.geometry.multipolygon.MultiPolygon):
        for poly in row['service_area']:
            if poly.contains(row['coords']):
                service_area_trim.append(poly)
    else:
        service_area_trim.append(row['service_area'])

station_df['service_area'] = service_area_trim
station_df.set_geometry('service_area', inplace=True)

start = time.time()

station_df['service_area'] = station_df['service_area'].to_crs(epsg=3857)
land_use_df['geometry'] = land_use_df['geometry'].to_crs(epsg=3857)
neighborhoods = []
for i, stat in station_df.iterrows():
    
    buffer = stat['service_area'].buffer(1000)
    
    neighborhoods.append([
        [row['geometry'], row['zone_type']] 
        for j, row in land_use_df.iterrows() 
        if row['geometry'].distance(stat['service_area']) == 0])

print(time.time()-start)
            

#%% 

import datetime
from bikeshare import get_cal
from matplotlib.collections import PolyCollection

city = 'nyc'
year = 2019
month = 9
period = 'b' # 'b' = business days or 'w' = weekends
holidays = False
min_trips = 100

stat=247

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

data = bs.Data(city,year)

traffic_dep, traffic_arr = data.daily_traffic(stat_index = stat, normalise=False, period=period, holidays=holidays)


mean_dep = np.mean(traffic_dep, axis=0)
mean_arr = np.mean(traffic_arr, axis=0)

std_dep = np.std(traffic_dep, axis=0)
std_arr = np.std(traffic_arr, axis=0)

var_dep = np.var(traffic_dep, axis=0)
var_arr = np.var(traffic_arr, axis=0)


# bar plots

hours = np.arange(24)

fig = plt.figure()
ax = plt.add_subplot(projectio='3d')
ax.bar()





# Norm plots

# hour_range = np.arange(24)
# x_range = np.linspace(0,250,10**3)
# hours = np.tile(hour_range, (x_range.size,1)).T
# x_tile = np.tile(x_range, (hour_range.size,1))
# z = np.zeros(shape=(hour_range.size, x_range.size))

# for hour in hour_range:
#     for i, x in enumerate(x_range):
#         z[hour,i] = normal_dist(x, mean_arr[hour], std_arr[hour])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_wireframe(hours,x_tile, z, cstride=1000000)




