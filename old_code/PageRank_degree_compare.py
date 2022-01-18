# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 13:18:26 2021

@author: nweinr
"""

import os
import calendar
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextily as ctx
import bikeshare as bs
import plotting
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Load data

city = 'nyc'
year = 2019
month = 9
period = 'w' # 'b' = business days or 'w' = weekends

if city == 'nyc':
    gov_stations = [3254, 3182, 3479]
    data = bs.Data(city, year, month, blacklist=gov_stations)

else:
    data = bs.Data(city, year, month)

weekdays = [calendar.weekday(year,month,i) for i in range(1,calendar.monthrange(year,month)[1]+1)]

days_b = [date+1 for date, day in enumerate(weekdays) if day <= 4]
days_w = [date+1 for date, day in enumerate(weekdays) if day > 4]

#%% Get PageRank and degrees
    
d_adj_b = bs.diradjacency(data.df, data.city, data.year, data.month,
                        data.d_index, days_b, data.stat, threshold=1,
                        remove_self_loops=True)

d_adj_w = bs.diradjacency(data.df, data.city, data.year, data.month,
                        data.d_index, days_w, data.stat, threshold=1,
                        remove_self_loops=True)

np.random.seed(42)
PageRanks_b = bs.PageRank(d_adj_b)
np.random.seed(42)
PageRanks_w = bs.PageRank(d_adj_w)

busy_stations_b = data.get_busy_stations(days_b, normalise = True, sort = False)
busy_stations_w = data.get_busy_stations(days_w, normalise = True, sort = False)

degrees_b = np.zeros(shape=(data.stat.n_tot,1))
degrees_w = np.zeros(shape=(data.stat.n_tot,1))

for stat_id, station in enumerate(busy_stations_b):
    degrees_b[stat_id,:] = station[1]

for stat_id, station in enumerate(busy_stations_w):
    degrees_w[stat_id,:] = station[1]

#%% Find rankings and compare

def list_rankings(arr):

    arr_sorted = np.sort(arr, axis = None)

    rankings = np.zeros(len(arr))
    
    for i, val in enumerate(arr):
        for j, val_sort in enumerate(arr_sorted):
            if val == val_sort:
                rankings[i] = j
                arr_sorted[j] = -1
                break
    
    return rankings
    
degree_rankings_b = list_rankings(degrees_b)
degree_rankings_w = list_rankings(degrees_w)

PageRank_rankings_b = list_rankings(PageRanks_b)
PageRank_rankings_w = list_rankings(PageRanks_w)

rankings_difference_b = PageRank_rankings_b - degree_rankings_b
rankings_difference_w = PageRank_rankings_w - degree_rankings_w

#%% Plotting

def plot_rank_difference(rankings_difference, data, days, period, filename = 'figure.png', title = True):
    
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
    
    adj = data.adjacency(days, threshold = 1, remove_self_loops=True)
    
    graph = nx.from_numpy_matrix(adj)
    
    min_val = min(rankings_difference)
    max_val = max(rankings_difference)
    
    nx.draw_networkx_nodes(graph, data.stat.loc_merc, node_size=20,
                            node_color=rankings_difference, cmap='jet',
                            vmin = min_val, vmax = max_val)
    
    nx.draw_networkx_edges(graph, data.stat.loc_merc, alpha=0.1,
                            width=0.2, edge_color='black', arrows=False)
    
    print('Adding colorbar...')
    fig.set_figwidth((1+0.24)*fig.get_size_inches()[0])
    
    cmap = mpl.cm.jet
    
    vbound = max(abs(max_val), abs(min_val))
    norm = mpl.colors.Normalize(vmin=-vbound, vmax=vbound)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  orientation='vertical', label='Difference in ranking')
    
    print('Adding basemap... ')
    ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain,
                        attribution='(C) Stamen Design, (C) OpenStreetMap contributors')
    
    print('Adding scalebar...')
    scalebar = AnchoredSizeBar(ax.transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                               pad=0.2, color='black', frameon=False, size_vertical=50)
    
    ax.add_artist(scalebar)
    if title:
        if period == 'b':
            plt.title(f'Ranking comparison for {data.city} in {month_dict[data.month]} {data.year:d} on business days')
        else:
            plt.title(f'Ranking comparison for {data.city} in {month_dict[data.month]} {data.year:d} on weekends')

    
    plt.tight_layout()
    
    print('Saving figure...')
    if not os.path.exists('figures/ranking_comparisons'):
        os.makedirs('figures/ranking_comparisons')
    
    plt.savefig('figures/ranking_comparisons/' + filename, bbox_inches = 'tight', dpi=150)
    
    return fig

# Actual plotting

filename = f'ranking_comparison_{city}_{year:d}_{month:02d}_' + period + '.png'

if period == 'b':
    fig = plot_rank_difference(rankings_difference_b, data, days_b, period, filename)
if period == 'w':
    fig = plot_rank_difference(rankings_difference_w, data, days_w, period, filename)

#%% Plotting for both business days and weekends

def plot_rank_difference_2(rankings_difference_b, rankings_difference_w, data, days_b, days_w, filename = 'figure.png', basemap = True):

    figsize_dict = {'nyc': (10,8),
                'chic': (10,8),
                'london': (16,5),
                'oslo': (13.2,5),
                'sfran': (12,5),
                'washDC': (16,7.7),
                'madrid': (12,8),
                'mexico': (14.4,8),
                'taipei': (14.6,8)}
    
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
        fig, ax = plt.subplots(ncols = 2, figsize=figsize_dict[data.city])
    except KeyError:
        fig, ax = plt.subplots(ncols = 2, figsize=(16,8))
        
    ax[0].axis(extent)
    ax[1].axis(extent)
    
    print('Drawing network...')
    
    adj_b = data.adjacency(days_b, threshold = 1, remove_self_loops=True)
    adj_w = data.adjacency(days_w, threshold = 1, remove_self_loops=True)

    graph_b = nx.from_numpy_matrix(adj_b)
    graph_w = nx.from_numpy_matrix(adj_w)
    
    min_val = min(min(rankings_difference_b), min(rankings_difference_w))
    max_val = max(max(rankings_difference_b), max(rankings_difference_w))
        
    nx.draw_networkx_nodes(graph_b, data.stat.loc_merc, node_size=20,
                        node_color=rankings_difference_b, cmap='jet',
                        vmin = min_val, vmax = max_val, ax=ax[0])

    nx.draw_networkx_nodes(graph_w, data.stat.loc_merc, node_size=20,
                            node_color=rankings_difference_w, cmap='jet',
                            vmin = min_val, vmax = max_val, ax=ax[1])
    
    nx.draw_networkx_edges(graph_b, data.stat.loc_merc, alpha=0.1,
                            width=0.2, edge_color='black', ax=ax[0], arrows=False)
    
    nx.draw_networkx_edges(graph_w, data.stat.loc_merc, alpha=0.1,
                            width=0.2, edge_color='black', ax=ax[1], arrows=False)
    
    fig.set_figheight((1+0.2)*fig.get_size_inches()[1])
    
    ax[0].set_frame_on(True)
    ax[1].set_frame_on(True)
    
    ax[0].axis('off')
    ax[1].axis('off')
    
    ax[0].title.set_text('Business days')
    ax[1].title.set_text('Weekends')
    
    plt.tight_layout()
    
    print('Adding colorbar...')
    
    vbound = max(abs(max_val), abs(min_val))
    
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=-vbound, vmax=vbound)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  orientation='horizontal', label='Difference in ranking', ax = ax,
                  shrink=0.6, pad = 0.01)
    
    if basemap:
        print('Adding basemap... ')
        ctx.add_basemap(ax[0], source=ctx.providers.Stamen.Terrain,
                        attribution='(C) Stamen Design, (C) OpenStreetMap contributors')
        
        ctx.add_basemap(ax[1], source=ctx.providers.Stamen.Terrain,
                        attribution='(C) Stamen Design, (C) OpenStreetMap contributors')
    
    print('Adding scalebar...')
    scalebar1 = AnchoredSizeBar(ax[0].transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                               pad=0.2, color='black', frameon=False, size_vertical=50)
    
    scalebar2 = AnchoredSizeBar(ax[1].transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                               pad=0.2, color='black', frameon=False, size_vertical=50)
    
    ax[0].add_artist(scalebar1)
    ax[1].add_artist(scalebar2)
    
    # if title:
    #     if period == 'b':
    #         plt.title(f'Ranking comparison for {data.city} in {month_dict[data.month]} {data.year:d} on business days')
    #     else:
    #         plt.title(f'Ranking comparison for {data.city} in {month_dict[data.month]} {data.year:d} on weekends')
    
    print('Saving figure...')
    if not os.path.exists('figures/ranking_comparisons'):
        os.makedirs('figures/ranking_comparisons')
    
    plt.savefig('figures/ranking_comparisons/' + filename, bbox_inches = 'tight', dpi=150)
    
    return fig

filename = f'ranking_comparison_{city}_{year:d}_{month:02d}_b&w.pdf'

fig = plot_rank_difference_2(rankings_difference_b, rankings_difference_w, data, days_b, days_w, filename)

#%% Find rankings for each hour and compare

def plot_rank_difference_hour(rankings_difference, data, day, hour, filename = 'figure.png', title = True):

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
    
    adj = data.adjacency_hour(day, hour, threshold = 1, remove_self_loops=True)
    
    graph = nx.from_numpy_matrix(adj)
    
    min_val = min(rankings_difference)
    max_val = max(rankings_difference)
    
    nx.draw_networkx_nodes(graph, data.stat.loc_merc, node_size=20,
                            node_color=rankings_difference, cmap='jet',
                            vmin = min_val, vmax = max_val)
    
    nx.draw_networkx_edges(graph, data.stat.loc_merc, alpha=0.1,
                            width=0.2, edge_color='black', arrows=False)
    
    print('Adding colorbar...')
    fig.set_figwidth((1+0.24)*fig.get_size_inches()[0])
    
    cmap = mpl.cm.jet
    
    vbound = max(abs(max_val), abs(min_val))
    norm = mpl.colors.Normalize(vmin=-vbound, vmax=vbound)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  orientation='vertical', label='Difference in ranking')
    
    print('Adding basemap... ')
    ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain,
                        attribution='(C) Stamen Design, (C) OpenStreetMap contributors')
    
    print('Adding scalebar...')
    scalebar = AnchoredSizeBar(ax.transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                               pad=0.2, color='black', frameon=False, size_vertical=50)
    
    ax.add_artist(scalebar)
    if title:
        plt.title(f'{data.city} at {year:d}/{month:02d}/{day:02d} {hour-1:02d}:00:00 - {hour:02d}:00:00')
    
    plt.tight_layout()
    
    print('Saving figure...')
    if not os.path.exists('figures/ranking_comparisons'):
        os.makedirs('figures/ranking_comparisons')
    
    plt.savefig('figures/ranking_comparisons/' + filename, bbox_inches = 'tight', dpi=150)
    
    return fig

day = 5

for hour in range(1,25):

    d_adj = bs.diradjacency_hour(data, day, hour-1)
    
    np.random.seed(42)
    PageRanks = bs.PageRank(d_adj)
    
    adj = data.adjacency_hour(day = day, hour = hour-1)
    
    degrees = np.sum(adj, axis = 0)
    
    degree_rankings = list_rankings(degrees)
    PageRank_rankings = list_rankings(PageRanks)
    
    rankings_difference = PageRank_rankings - degree_rankings
    
    filename = f'ranking_comparison_{city}_{year:d}_{month:02d}_{day:02d}_{hour:02d}.png'
    
    fig = plot_rank_difference_hour(rankings_difference, data, day, hour, filename)
    plt.close(fig)

