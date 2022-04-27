#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:18:09 2022

@author: ubuntu
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import bikeshare as bs
import interactive_plot_utils as ipu

data = bs.Data('nyc', 2019, remove_loops=True)

stations = pd.Series(data.stat.id_index.keys()).astype(int)
od = pd.DataFrame(0, index=stations, columns=stations)

morning_df = data.df[data.df['start_dt'].dt.hour.isin([7, 8])]
afternoon_df = data.df[data.df['start_dt'].dt.hour.isin([17, 18])]

cluster_ods = []
cluster_ods_norm = []

for time, df in zip(['morning', 'afternoon'], [morning_df, afternoon_df]):
    od = pd.DataFrame(0, index=stations, columns=stations)
    od = od.add(df[['start_stat_id', 'end_stat_id']].pivot_table(values='end_stat_id', index='start_stat_id', columns='end_stat_id', fill_value=0, aggfunc=len))
    od.fillna(0, inplace=True)
    
    od.rename_axis(index='origin', columns="destination")
    
    graph = nx.DiGraph(od)
    #nx.draw_networkx(graph)
    
    
    overwrite = False
    traffic_matrices = data.pickle_daily_traffic(holidays=False, normalise=True, overwrite=overwrite)
    station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, 
                                                      return_land_use=True, return_census=True, 
                                                      overwrite=overwrite)
    #service_area, service_area_size = ipu.get_service_area(data.city, station_df, land_use, service_radius=500, voronoi=False)
    
    # percent_cols = [column for column in station_df.columns if "percent_" in column]
    
    # station_df = station_df.drop(columns=percent_cols).merge(
    #     neighborhood_percentages(
    #         data.city, station_df, land_use, 
    #         service_radius=500, use_road=False
    #         ),
    #     how='outer', left_index=True, right_index=True)
    
    k = 5
    
    station_df, clusters, labels = ipu.get_clusters(
        traffic_matrices, station_df, day_type='business_days', min_trips=100,
        clustering='k_means', k=k, random_state=42)
    
    
    cluster_od = pd.DataFrame(0, index=range(k), columns=range(k))
    cluster_od_norm = pd.DataFrame(0, index=range(k), columns=range(k))
    
    
    cluster_members = {}
    
    for i in range(k):
        cluster_members[i] = station_df[station_df['label'] == i]['stat_id']
        
    for i in range(k):
        for j in range(k):
            cluster_od.loc[i, j] = od.loc[cluster_members[i], cluster_members[j]].sum().sum()
            cluster_od_norm.loc[i, j] = (od.loc[cluster_members[i], cluster_members[j]].sum().sum())/len(cluster_members[i]*cluster_members[j])
            
    cluster_od = cluster_od[[1,2,0,3,4]]
    cluster_od = cluster_od.reindex([1,2,0,3,4])
    
    cluster_od_norm = cluster_od_norm[[1,2,0,3,4]]
    cluster_od_norm = cluster_od_norm.reindex([1,2,0,3,4])
    
    cluster_ods.append(cluster_od)
    cluster_ods_norm.append(cluster_od_norm)
    
cluster_names = {0:'Reference', 1:'High M. sink', 2:'Mild M. sink', 3:'Mild M. source', 4:'High M. source'}


times = ['Morning', 'Afternoon']
for a in range(2):
    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.matshow(cluster_ods_norm[a], cmap=plt.cm.Blues)
    
    fig.colorbar(im, label='Number of trips')
    
    for i in range(k):
        for j in range(k):
            c = cluster_ods_norm[a].iloc[j,i]
            ax.text(i, j, f'{c:.2f}', va='center', ha='center')
    
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(pd.Series(cluster_ods[a].columns).map(cluster_names), rotation=22.5, ha='left')
    ax.set_yticklabels(pd.Series(cluster_ods[a].index).map(cluster_names))
    # ax.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    # plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
    #      ha="left", va="center",rotation_mode="anchor")
    
    ax.set_xlabel('Destination')
    ax.set_ylabel('Origin')
    ax.set_title(f'OD by cluster type in the {times[a]} for {bs.name_dict[data.city]}')
    plt.tight_layout()
    plt.savefig(f'figures/cluster_od_{times[a]}_{data.city}.pdf')
