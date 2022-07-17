#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:48:02 2022

@author: ubuntu
"""

import pickle

import numpy as np
import geopandas as gpd
import pandas as pd
import contextily as cx

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText, AnchoredOffsetbox
from matplotlib import patheffects
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import bikeshare as bs
import interactive_plot_utils as ipu
import optimization
from clustering import get_clusters


data = bs.Data('nyc', 2019, 9)

station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
sub_polygons = gpd.read_file('data/nyc/nyc_expansion_subdivision_2.geojson')

months = [1,2,3,4,5,6,7,8,9]
asdf = optimization.asdf_months(data, months)

pops = []
for polygon in sub_polygons['geometry']:
    intersections = census_df.intersection(polygon)
    selection = ~intersections.is_empty
    census_intersect = census_df.loc[selection, 'pop_density']
    # Area in km²
    areas = intersections[selection].to_crs(data.laea_crs).area/1000000
    population = np.sum(areas * census_intersect)
    pops.append(population)

sub_polygons['population'] = pops

# Number of stations per person
station_density = len(station_df) / station_df['population'].sum()

proportional_n_stations = sub_polygons['population'] * station_density

# Scale up and round to add up to 60
n_stations = np.floor(proportional_n_stations*3.15)

sub_polygons['n_stations'] = n_stations

traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')
cols = ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational',
        'pop_density', 'nearest_subway_dist', 'nearest_railway_dist']
day_type = 'business_days'
min_trips = 8
clustering = 'k_means'
k = 5
seed = 42
triptype = 'b_trips'
asdf, clusters, labels = get_clusters(
    traffic_matrices, asdf, day_type, min_trips, clustering, k, seed)

if data.city in ['helsinki', 'oslo', 'madrid', 'london']:
    df_cols = [col for col in cols if col != 'percent_industrial']
else:
    df_cols = cols

model_results = ipu.linear_regression(asdf, df_cols, triptype)

minima = []
n_per = 50000

rng = np.random.default_rng(42)

for i, polygon in sub_polygons.iterrows():
    print(i)
    int_exp = optimization.get_intersections(polygon['geometry'], data=data)
   
    existing_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')
   
    # existing_stations['lon'] = existing_stations['long']
    # existing_stations['geometry'] = existing_stations['coords']
   
    # n_existing = len(existing_stations)
   
    # print(f"{n_existing} existing, {polygon['n_stations']} total")
   
    # int_exp = pd.concat((existing_stations[['lat', 'lon', 'coords', 'geometry']], int_exp))
    # int_exp = int_exp.reset_index()
    
    # print(f"There are {len(int_exp)} Intersections")
    
    # point_info = get_point_info(data, int_exp, land_use, census_df)
    
    # months = [1,2,3,4,5,6,7,8,9]
    # # asdf = asdf_months(data, months)
    
    # int_proj = int_exp.to_crs(data.laea_crs)
    
    # n_stations = polygon['n_stations']
    
    # n_combinations = binom(len(int_exp), n_stations)
    # fig, ax = plt.subplots()
    # gpd.GeoSeries(polygon['geometry']).plot(ax=ax)
    # int_exp.plot(ax=ax, color='red')
    # ax.set_title(f"{n_stations} stations : {n_combinations} combinations")
    # print(n_combinations)
    
    # n = len(point_info)
    
    # n_select = int(n_stations)
    
    # n_total = n_select + n_existing
    
    # distances = np.zeros((n, n))
    # for i in range(n):
    #     distances[i] = int_proj.distance(int_proj.geometry.loc[i])
        
    
    # pred = model_results.predict(point_info[['const', *df_cols]])
    
    # def obj_fun(x):
    #     return -np.sum(x*pred)
    
    # def condition(x):
    #     xb = x.astype(bool)
    #     return np.min(distances[xb][:,xb][distances[xb][:,xb] != 0])
    
    # x0 = np.zeros(n-n_existing)
    # x0[:n_select] = 1
    # np.random.seed(42)
    # x0 = np.random.permutation(x0)
    
    # n_permutations = np.floor(np.min((n_per, n_combinations*100))).astype(int)
    
    # population = rng.permuted(np.tile(x0, n_permutations).reshape(n_permutations, x0.size), axis=1)
    
    # existing_population = np.ones((n_permutations, n_existing))
    
    # population = np.hstack((existing_population, population))
    
    # score = parallel_apply_along_axis(obj_fun, 1, population)
    # if n_select > 1:
    #     cond = parallel_apply_along_axis(condition, 1, population)
    # else:
    #     cond = np.sum(population, axis=1)*400
    # mask = np.where(cond < 250)
    # if len(score[mask]) == len(score):
    #     print('mask condition not fulfilled, changing to 200')
    #     mask = np.where(cond < 200)
    #     if len(score[mask]) == len(score):
    #         print('mask condition not fulfilled, changing to 100')
    #         mask = np.where(cond < 100)
    
    # score[mask] = 0
    
    # print(f"min: {population[np.argmin(score)]}, score: {np.min(score)}, condition = {cond}")
    
    
    # # minimum = so.minimize(obj_fun, x0=x0, constraints=(sum_constraint), bounds=bounds, method='SLSQP', options={'maxiter': 10})
    # # print(minimum.message)
    # # minima.append(minimum)
    # # selection_idx = np.argpartition(minimum.x, -n_select)[-n_select:]
    # # minima.append([np.min(score[mask]), population[np.argmin(score[mask])]])
    # score[mask] = 0
    # minima.append(population[np.argmin(score)]) 
#%% Results
   
spacings = [100, 150, 200, 250, 300]
   
minima_res = {}
for i in range(len(sub_polygons)):
    with open(f'./python_variables/nyc_expansion_optimization_polygon_{i:02d}.pickle', 'rb') as file:
        minima_res[i] = pickle.load(file)
        
mini = pd.concat(minima_res.values(), keys=minima_res.keys())

maxdist = 250

sps = [spacing for spacing in spacings if spacing <= maxdist]

mini = mini.loc[pd.IndexSlice[:, sps], :]

bestmini = pd.DataFrame()
for i in range(len(sub_polygons)):
    new_row = mini.xs(i, level=0)[~mini.xs(i, level=0)['solution'].isna()].iloc[[-1]]
    new_row = new_row.rename_axis('spacing').reset_index()
    new_row.index = [i]
    bestmini = pd.concat((bestmini, new_row))

results = bestmini['solution']


selected_intersections = []

for i, polygon in sub_polygons.iterrows():
    merge_tolerance=20
    if i == 11:
        merge_tolerance = 21
    int_exp = optimization.get_intersections(polygon['geometry'], data=data, merge_tolerance=merge_tolerance)
    
    existing_stations_sub = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')

    existing_stations_sub['lon'] = existing_stations_sub['long']
    existing_stations_sub['geometry'] = existing_stations_sub['coords']

    n_existing = len(existing_stations_sub)

    print(f"{n_existing} existing, {polygon['n_stations']} total")
    
    int_exp['existing'] = False
    
    existing_stations_sub['existing'] = True

    int_exp = pd.concat((existing_stations_sub[['lat', 'lon', 'coords', 'geometry', 'existing']], int_exp))
    int_exp = int_exp.reset_index()

    selected_intersections.append(int_exp[results[i] == 1])
    
selected_intersections = pd.concat(selected_intersections)

selected_intersections = selected_intersections.reset_index()

#%% Full model September

import full_model



data = bs.Data('nyc', 2019, 9)

station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
selected_point_info = optimization.get_point_info(data, selected_intersections[['geometry', 'coords']], land_use, census_df)



station_df['const'] = 1

variables_list = ['percent_residential', 'percent_commercial',
                  'percent_recreational', 
                  'pop_density', 'nearest_subway_dist',
                  'nearest_railway_dist', 'center_dist']

selected_point_info = selected_point_info[['const', *variables_list]]

# months = [1,2,3,4,5,6,7,8,9]
# asdf = asdf_months(data, months)

# traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')

# asdf, b, c = get_clusters(traffic_matrices, asdf, day_type='business_days', min_trips=8, clustering='k_means', k=5)



data, asdf, traf_mat = full_model.load_city('nyc', 2019, 9)

model = full_model.FullModel(variables_list)
asdf2 = model.fit(asdf, traf_mat)

clusters = []
volume = []
vectors = []


for i, row in selected_point_info.iterrows():
    log_pred = model.logit_model.predict(row)
    log_best = int(np.argmax(log_pred))
    clusters.append(log_best)
    lin_pred = model.linear_model.predict(row)
    volume.append(lin_pred)
    
    vectors.append(model.centers[log_best] * float(lin_pred))

plt.plot(model.centers.T)

vectors = np.array(vectors)

plt.plot(vectors.T)


expansion_area = gpd.read_file('data/nyc/expansion_2019_area.geojson')

polygon = expansion_area.loc[0]

polygon['n_stations'] = 58

# int_exp = get_intersections(polygon['geometry'], data=data)
   
existing_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')
   
existing_stations['lon'] = existing_stations['long']
existing_stations['geometry'] = existing_stations['coords']

existing_stations['existing'] = True
   
extend = (existing_stations['lat'].min(), existing_stations['long'].min(), 
          existing_stations['lat'].max(), existing_stations['long'].max())

ex_pro = existing_stations.to_crs(data.laea_crs)

pol_pro = expansion_area.to_crs(data.laea_crs)

   

old = ex_pro[ex_pro['existing'] == True]

new = ex_pro[ex_pro['existing'] == False]

sept_stations = existing_stations

#%% Full model different month (November)

sept_data = bs.Data('nyc', 2019, 9)

sept_station_df, sept_land_use, sept_census_df = ipu.make_station_df(sept_data, holidays=False, return_land_use=True, return_census=True)   
expansion_area = gpd.read_file('data/nyc/expansion_2019_area.geojson')

polygon = expansion_area.loc[0]

polygon['n_stations'] = 58

# int_exp = get_intersections(polygon['geometry'], data=data)
   
sept_stations = gpd.sjoin(sept_station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')
   
sept_stations['lon'] = sept_stations['long']
sept_stations['geometry'] = sept_stations['coords']

sept_stations['existing'] = True


data = bs.Data('nyc', 2019, 11)

station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
expansion_area = gpd.read_file('data/nyc/expansion_2019_area.geojson')

polygon = expansion_area.loc[0]

polygon['n_stations'] = 58

traffic_matrices = data.pickle_daily_traffic(holidays=False)

station_df, clusters, labels = get_clusters(traffic_matrices, station_df, 
                                                'business_days', 9, 
                                                'k_means', 5, 42)


# int_exp = get_intersections(polygon['geometry'], data=data)
   
real_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')
   
real_stations['lon'] = real_stations['long']
real_stations['geometry'] = real_stations['coords']

real_stations['existing'] = False

real_stations.loc[real_stations['stat_id'].isin(sept_stations['stat_id']), 'existing'] = True


all_stations = pd.concat([sept_stations, real_stations])


selected_point_info = optimization.get_point_info(data, real_stations.reset_index()[['geometry', 'coords']], land_use, census_df)



station_df['const'] = 1

variables_list = ['percent_residential', 'percent_commercial',
                  'percent_recreational', 
                  'pop_density', 'nearest_subway_dist',
                  'nearest_railway_dist', 'center_dist']

selected_point_info = selected_point_info[['const', *variables_list]]

clusters = []
volume = []
vectors = []


for i, row in selected_point_info.iterrows():
    log_pred = model.logit_model.predict(row)
    log_best = int(np.argmax(log_pred))
    clusters.append(log_best)
    lin_pred = model.linear_model.predict(row)
    volume.append(lin_pred)
    
    vectors.append(model.centers[log_best] * float(lin_pred))
 
#%% Plot figure (Select month by running above cell)
fig, ax = plt.subplots(figsize=(4.5, 10))



selected_intersections['cluster'] = clusters

col_dict = {0: 'tab:blue',
            1: 'tab:orange',
            2: 'tab:green',
            3: 'tab:red',
            4: 'tab:purple',
    }
 
colors = [col_dict[clus] for clus in clusters]

sizes = np.array(volume)

sub_polygons.to_crs(data.laea_crs).plot(label='Expansion area', alpha=0.5, marker='s', ax=ax, color='tab:orange', edgecolor="black")
# old.plot(label='Existing stations',ax=ax, color="tab:blue", alpha=1, zorder=1.5)
selected_intersections.to_crs(data.laea_crs).plot(label='Selected intersections', ax=ax, color=colors, alpha=1, markersize=sizes)


legend_elements = [
    Line2D([0], [0], marker='o', linestyle='None', color='tab:blue', label='Reference'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:orange', label='High morning sink'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:green', label='Low morning sink'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:red', label='Low morning source'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:purple', label='High morning source'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:gray', label='Unclustered'),
    Patch(color='tab:orange', label='Expansion area', alpha=0.6),
    ]
   
legend = ax.legend(handles=legend_elements)
# legend = ax.legend()
ax.axis('off')
# patch = mpatches.Patch(color='tab:orange', label='Expansion area', alpha=0.6)

# handles, labels = ax.get_legend_handles_labels()
# # handles is a list, so append manual patch
# handles.append(patch) 

# # plot the legend
# plt.legend(handles=handles)

scalebar = AnchoredSizeBar(ax.transData, 1000, 
                            f'{1} km', 'lower left', 
                            pad=1, color='black', frameon=False, 
                            size_vertical=5, path_effects=[patheffects.withStroke(linewidth=20, foreground="w")])

scalebar.set_path_effects([patheffects.Stroke(linewidth=3, foreground='w'),])
sc = ax.add_artist(scalebar)


cx.add_basemap(ax, crs=data.laea_crs, attribution="")
text = "(C) Stamen Design, (C) OpenStreetMap Contributors"
ax.text(0.005, 0.005, text, transform=ax.transAxes, size=8,
        path_effects=[patheffects.withStroke(linewidth=2, foreground="w")], wrap=True,)
#%%

plt.savefig(f"figures/nyc_exp_selected_intersections_predicted_sizecolor_sep.png", bbox_inches='tight', dpi=150)
# volume = np.array(volume)


#%%


# for cl, vol in zip(clusters, volume):
#     vectors.append(model.centers[cl] * volume)
    
fig, ax = plt.subplots(figsize=(4.5, 10))


volume = real_stations['b_trips']
clusters = real_stations['label']

clusters[clusters.isna()] = 9

# selected_intersections['cluster'] = clusters

col_dict = {0: 'tab:blue',
            1: 'tab:orange',
            2: 'tab:green',
            3: 'tab:red',
            4: 'tab:purple',
            9: 'tab:gray'
    }

colors = [col_dict[clus] for clus in clusters]

sizes = np.array(volume)

sub_polygons.to_crs(data.laea_crs).plot(label='Expansion area', alpha=0.5, marker='s', ax=ax, color='tab:orange', edgecolor="black")
# old.plot(label='Existing stations',ax=ax, color="tab:blue", alpha=1, zorder=1.5)
real_stations.to_crs(data.laea_crs).plot(label='Stations', ax=ax, color=colors, alpha=1, markersize=sizes)

legend_elements = [
    Line2D([0], [0], marker='o', linestyle='None', color='tab:blue', label='Reference'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:orange', label='High morning sink'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:green', label='Low morning sink'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:red', label='Low morning source'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:purple', label='High morning source'),
    Line2D([0], [0], marker='o', linestyle='None', color='tab:gray', label='Unclustered'),
    Patch(color='tab:orange', label='Expansion area', alpha=0.6),
    ]
  
legend = ax.legend(handles=legend_elements)
ax.axis('off')
# patch = mpatches.Patch(color='tab:orange', label='Expansion area', alpha=0.6)

# handles, labels = ax.get_legend_handles_labels()
# # handles is a list, so append manual patch
# handles.append(patch) 

# # plot the legend
# plt.legend(handles=handles)

scalebar = AnchoredSizeBar(ax.transData, 1000, 
                            f'{1} km', 'lower left', 
                            pad=1, color='black', frameon=False, 
                            size_vertical=5, path_effects=[patheffects.withStroke(linewidth=20, foreground="w")])

scalebar.set_path_effects([patheffects.Stroke(linewidth=3, foreground='w'),])
sc = ax.add_artist(scalebar)
  

cx.add_basemap(ax, crs=data.laea_crs, attribution="")
text = "(C) Stamen Design, (C) OpenStreetMap Contributors"
ax.text(0.005, 0.005, text, transform=ax.transAxes, size=8,
        path_effects=[patheffects.withStroke(linewidth=2, foreground="w")], wrap=True,)

plt.savefig(f"figures/nyc_exp_selected_intersections_real_sizecolor_feb.png", bbox_inches='tight', dpi=150)


# zone_columns = variables_list
# other_columns = variables_list
# const = True

# logit_model, X, y = ipu.stations_logistic_regression(
#         asdf[[*variables_list, 'label']], zone_columns, other_columns, 
#         use_points_or_percents='percents', 
#         make_points_by='station land use', 
#         const=const, test_model=False)

# mask = asdf['b_trips'] >= 8
# stat_slice = asdf[mask]
# linear_model = ipu.linear_regression(stat_slice, variables_list, 'b_trips')

# X = [1,1,1,1,1,1,1,1]
# selo = np.hstack((np.ones((len(selected_intersections), 1)), np.array(selected_intersections[variables_list])))
# log_pred = logit_model.predict(selected_point_info[['const', *variables_list]])

# best_log = np.argmax(np.array(log_pred), axis=1)

# lin_pred = linear_model.predict(selected_point_info[['const', *variables_list]])

#%% Capacity
import capacity_opt

T_START = 6
K = 18
tau = 1/6

traf_est = pd.DataFrame()

for stat_id in selected_point_info.index:
    traffic_est = model.predict_daily_traffic(selected_point_info.loc[stat_id],
                                              predict_cluster=True,
                                              plotfig=False,
                                              verbose=True)
    
    traf_est[stat_id] = traffic_est
    

    
def capacity_helper(vec):
    d = vec[:24]
    a = vec[24:]
    d = list(d)[T_START:]
    a = list(a)[T_START:]
    return capacity_opt.min_size(a,d,tau,K,0.9)
    
min_sizes_est = optimization.parallel_apply_along_axis(capacity_helper, 0, traf_est)

print("")
print(min_sizes_est)
print("")
with open(f'./python_variables/min_sizes_est_ours_T_start{T_START}_K{K}_tau{tau}.pickle', 'wb') as file:
    pickle.dump(min_sizes_est, file)




#%% Plot dimensioning
fig, ax = plt.subplots(1, 1, figsize=(4.5, 10))

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="10%", pad=0.1)

selected_intersections['cluster'] = clusters

cmap = 'viridis'

sub_polygons.to_crs(data.laea_crs).plot(label='Expansion area', alpha=0.5, marker='s', ax=ax, color='tab:orange', edgecolor="black")
# old.plot(label='Existing stations',ax=ax, color="tab:blue", alpha=1, zorder=1.5)
intersections_to_plot = selected_intersections.to_crs(data.laea_crs)
intersections_to_plot.plot(label='Selected intersections', ax=ax, c=min_sizes_est, alpha=1, markersize=40, legend=True)

# legend_elements = [
#     Patch(color='tab:orange', label='Expansion area', alpha=0.6),
#     ]
   
# legend = ax.legend(handles=legend_elements)


# legend = ax.legend()
ax.axis('off')
# patch = mpatches.Patch(color='tab:orange', label='Expansion area', alpha=0.6)

# handles, labels = ax.get_legend_handles_labels()
# # handles is a list, so append manual patch
# handles.append(patch) 

# # plot the legend
# plt.legend(handles=handles)

vmin, vmax = min(min_sizes_est), max(min_sizes_est)

fig = ax.get_figure()
# cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
# fake up the array of the scalar mappable. Urgh...
sm._A = []
fig.colorbar(sm, cax=cax, label='Station size')

scalebar = AnchoredSizeBar(ax.transData, 1000, 
                            f'{1} km', 'lower left', 
                            pad=1, color='black', frameon=False, 
                            size_vertical=5, path_effects=[patheffects.withStroke(linewidth=20, foreground="w")])

scalebar.set_path_effects([patheffects.Stroke(linewidth=3, foreground='w'),])
sc = ax.add_artist(scalebar)


cx.add_basemap(ax, crs=data.laea_crs, attribution="")
text = "(C) Stamen Design, (C) OpenStreetMap Contributors"
ax.text(0.005, 0.005, text, transform=ax.transAxes, size=8,
        path_effects=[patheffects.withStroke(linewidth=2, foreground="w")], wrap=True,)
#%%

plt.savefig(f"figures/nyc_exp_selected_dimensioning_sep_T_start{T_START}_K{K}_tau{tau:.3f}.png", bbox_inches='tight', dpi=150)
# volume = np.array(volume)