# -*- coding: utf-8 -*-
"""
@author: Mattek Group 3
"""

import os
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

if city == 'nyc':
    gov_stations = [3254, 3182, 3479]
    data = bs.Data(city, year, month, blacklist=gov_stations)

else:
    data = bs.Data(city, year, month)

#%% Plot degrees/PageRank by workdays and weekends

measurement = 'deg' # either 'deg' (degree) or 'PR' (PageRank)
basemap = True

weekdays = (2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,23,24,25,26,27,30)
weekends = (1,7,8,14,15,21,22,28,29)

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

if measurement == 'deg':

    adj_weekdays = data.adjacency(weekdays, threshold = 1, remove_self_loops=True)
    adj_weekends = data.adjacency(weekends, threshold = 1, remove_self_loops=True)

    graph_weekdays = nx.from_numpy_matrix(adj_weekdays)
    graph_weekends = nx.from_numpy_matrix(adj_weekends)

    deg_weekdays = np.sum(adj_weekdays, axis = 0)
    deg_weekends = np.sum(adj_weekends, axis = 0)

    deg_weekdays = deg_weekdays/len(weekdays)
    deg_weekends = deg_weekends/len(weekends)

    min_val = min(min(deg_weekdays), min(deg_weekends))
    max_val = max(max(deg_weekdays), max(deg_weekends))

    node_color_weekdays = deg_weekdays
    node_color_weekends = deg_weekends

    label = 'degree'
    filename = f'figures/degrees_weekdays_vs_weekend_{data.city}.pdf'

elif measurement == 'PR':

    np.random.seed(42)
    d = 0.85
    N = 100

    adj_weekdays = bs.diradjacency(data.df, data.city, data.year, data.month,
                                   data.d_index, weekdays, data.stat,
                                   threshold=1, remove_self_loops=True)

    adj_weekends = bs.diradjacency(data.df, data.city, data.year, data.month,
                                   data.d_index, weekends, data.stat,
                                   threshold=1, remove_self_loops=True)

    graph_weekdays = nx.from_numpy_matrix(adj_weekdays, create_using=nx.DiGraph)
    graph_weekends = nx.from_numpy_matrix(adj_weekends, create_using=nx.DiGraph)

    PRank_weekdays = bs.PageRank(adj_weekdays)
    PRank_weekends = bs.PageRank(adj_weekends)

    PRank_weekdays = PRank_weekdays.reshape(len(PRank_weekdays))
    PRank_weekends = PRank_weekends.reshape(len(PRank_weekends))

    min_val = min(min(PRank_weekdays), min(PRank_weekends))
    max_val = max(max(PRank_weekdays), max(PRank_weekends))

    node_color_weekdays = PRank_weekdays
    node_color_weekends = PRank_weekends

    label = 'PageRank'
    filename = f'figures/PageRank_weekdays_vs_weekend_{data.city}.pdf'

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
nx.draw_networkx_nodes(graph_weekdays, data.stat.loc_merc, node_size=20,
                        node_color=node_color_weekdays, cmap='YlGnBu_r',
                        vmin = min_val, vmax = max_val, ax=ax[0])

nx.draw_networkx_nodes(graph_weekends, data.stat.loc_merc, node_size=20,
                        node_color=node_color_weekends, cmap='YlGnBu_r',
                        vmin = min_val, vmax = max_val, ax=ax[1])

nx.draw_networkx_edges(graph_weekdays, data.stat.loc_merc, alpha=0.1,
                        width=0.2, edge_color='black', ax=ax[0], arrows=False)

nx.draw_networkx_edges(graph_weekends, data.stat.loc_merc, alpha=0.1,
                        width=0.2, edge_color='black', ax=ax[1], arrows=False)

fig.set_figheight((1+0.2)*fig.get_size_inches()[1])

ax[0].set_frame_on(True)
ax[1].set_frame_on(True)

ax[0].axis('off')
ax[1].axis('off')

ax[0].title.set_text('Average Business Day')
ax[1].title.set_text('Average Weekend Day')

plt.tight_layout()

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
    
    filename = filename[:-4] + '_basemap.pdf'
    
print('Adding colorbar...')
cmap = mpl.cm.YlGnBu_r
norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              orientation='horizontal', label=label, ax = ax,
              shrink=0.6, pad = 0.01)

print('Saving figure...')
plt.savefig(filename, bbox_inches = 'tight', dpi=150)

#plt.tight_layout(h_pad = 0.5)

#%% Get the busiest stations


days = (2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,23,24,25,26,27,30)

# days = (1,7,8,14,15,21,22,28,29)

N = 100

busy_stations = data.get_busy_stations(days, normalise = True)

degrees = [station[1] for station in busy_stations]
mean_degree = np.mean(degrees)

count = 0
print('\n{} stations with the highest degrees:\n'.format(N))
for station in busy_stations[:N]:
    print('Rank {} - {}: degree is {}'.format(
        count+1, station[0], np.round(station[1], decimals = 1)))
    count += 1

print('\nAverage degree: {}\n'.format(np.round(mean_degree, decimals = 1)))

plotting.plot_graph(data, days, savefig = False, ext = 'png',
                    normalise_degrees = True, basemap=True)


#%% PageRank

days = [2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,23,24,25,26,27,30]

days = [1,7,8,14,15,21,22,28,29]

max_rank = 5

d_adj = bs.diradjacency(data.df, data.city, data.year, data.month,
                        data.d_index, days, data.stat, threshold=1,
                        remove_self_loops=True)

np.random.seed(42)
P = bs.PageRank(d_adj, d = 0.85, iterations = 100, initialisation = "rdm")

least_popular_stations = np.argsort(P, axis = None)
popular_stations = np.flip(least_popular_stations)

popular_station_names = [data.stat.names[popular_stations[i]] for i in range(max_rank)]

count = 0
print('\nThe {} stations with the highest PageRank:\n'.format(max_rank))
for station in popular_station_names:
    print('Rank {} - {}: PageRank is {}'.format(count+1, station, P[popular_stations[count]][0]))
    count += 1

#%% Find changes in degree from weekdays to weekend

days_ref = (2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,23,24,25,26,27,30)

days_change = (1,7,8,14,15,21,22,28,29)

deg_change = data.compare_degrees(days_ref, days_change)

#%% Find most popular trips in weekdays and weekends and compare

weekdays = (2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,23,24,25,26,27,30)
weekends = (1,7,8,14,15,21,22,28,29)

N = 200

top_trips_weekdays = data.get_busy_trips(weekdays)[:N]
top_trips_weekends = data.get_busy_trips(weekends)[:N]

stats_weekdays = {trip[0][0] for trip in top_trips_weekdays} | {trip[0][1] for trip in top_trips_weekdays}
stats_weekends = {trip[0][0] for trip in top_trips_weekends} | {trip[0][1] for trip in top_trips_weekends}

edgelist_weekdays = [trip[0] for trip in top_trips_weekdays]
edgelist_weekends = [trip[0] for trip in top_trips_weekends]

weights_weekdays = [trip[1] for trip in top_trips_weekdays]
weights_weekends = [trip[1] for trip in top_trips_weekends]

w_min = min(min(weights_weekdays), min(weights_weekends))
w_max = max(max(weights_weekdays), max(weights_weekends))

stats_color_weekdays = np.full(data.stat.n_tot, fill_value = 'black')
stats_color_weekdays[list(stats_weekdays)] = 'red'
stats_color_weekends = np.full(data.stat.n_tot, fill_value = 'black')
stats_color_weekends[list(stats_weekends)] = 'red'

stats_alpha_weekdays = np.full(data.stat.n_tot, fill_value = 0.3)
stats_alpha_weekdays[list(stats_weekdays)] = 1
stats_alpha_weekends = np.full(data.stat.n_tot, fill_value = 0.3)
stats_alpha_weekends[list(stats_weekends)] = 1

figsize_dict = {'nyc': (10,8),
                'chic': (10,8),
                'london': (16,5),
                'oslo': (13.2,5),
                'sfran': (12,5),
                'washDC': (16,7.7),
                'madrid': (12,8),
                'mexico': (14.4,8),
                'taipei': (14.6,8)}

adj_weekdays = data.adjacency(weekdays, threshold = 1, remove_self_loops=True)
adj_weekends = data.adjacency(weekends, threshold = 1, remove_self_loops=True)

graph_weekdays = nx.from_numpy_matrix(adj_weekdays)#, create_using=nx.DiGraph)
graph_weekends = nx.from_numpy_matrix(adj_weekends)#, create_using=nx.DiGraph)

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

cmap = mpl.cm.jet

nx.draw_networkx_nodes(graph_weekdays, data.stat.loc_merc, node_size=20,
                       node_color=stats_color_weekdays,
                       alpha = stats_alpha_weekdays, ax=ax[0])

nx.draw_networkx_nodes(graph_weekends, data.stat.loc_merc, node_size=20,
                       node_color=stats_color_weekends,
                       alpha = stats_alpha_weekends, cmap=cmap,
                       vmin = np.floor(w_min), vmax = np.ceil(w_max), ax=ax[1])

nx.draw_networkx_edges(graph_weekdays, data.stat.loc_merc,
                       edgelist=edgelist_weekdays[::-1], alpha=0.8, width=3,
                       edge_color=weights_weekdays[::-1], edge_cmap=cmap,
                       edge_vmin = np.floor(w_min), edge_vmax = np.ceil(w_max),
                       ax=ax[0])

nx.draw_networkx_edges(graph_weekends, data.stat.loc_merc,
                       edgelist=edgelist_weekends[::-1], alpha=0.8, width=3,
                       edge_color=weights_weekends[::-1], edge_cmap=cmap,
                       edge_vmin = np.floor(w_min), edge_vmax = np.ceil(w_max),
                       ax=ax[1])

fig.set_figheight((1+0.2)*fig.get_size_inches()[1])

ax[0].set_frame_on(False)
ax[1].set_frame_on(False)

ax[0].title.set_text('Weekdays')
ax[1].title.set_text('Weekends')

plt.tight_layout()

print('Adding basemap... ')
ctx.add_basemap(ax[0], source=ctx.providers.Stamen.Terrain,
                attribution='(C) Stamen Design, (C) OpenStreetMap contributors')
ctx.add_basemap(ax[1], source=ctx.providers.Stamen.Terrain,
                attribution='(C) Stamen Design, (C) OpenStreetMap contributors')

print('Adding colorbar...')
norm = mpl.colors.Normalize(vmin=np.floor(w_min), vmax=np.ceil(w_max))
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             orientation='horizontal', label='weight', ax = ax,
             shrink=0.6, pad = 0.01)

print('Adding scalebar...')
scalebars = {'chic': 5000,
             'london': 5000,
             'madrid': 2000,
             'mexico': 2000,
             'nyc':5000,
             'sfran':5000,
             'taipei':5000,
             'washDC':5000}

scalebar1 = AnchoredSizeBar(ax[0].transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                           pad=0.2, color='black', frameon=False, size_vertical=50)

scalebar2 = AnchoredSizeBar(ax[1].transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                           pad=0.2, color='black', frameon=False, size_vertical=50)

ax[0].add_artist(scalebar1)
ax[1].add_artist(scalebar2)

print('Saving figure...')
plt.savefig('figures/busy_trips_weekdays_vs_weekend_{}_N={}.png'.format(data.city, N),
            bbox_inches = 'tight', dpi=150)

#%% Find the most popular trips

days = [2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,23,24,25,26,27,30]
days = (1,7,8,14,15,21,22,28,29)
N = 200

top_trips = data.get_busy_trips(days)[:N]

print(f'\nThe {N} most popular trips:\n')
top_stat_indices = set()
Rank = 1
trip_dists = 0
for trip in top_trips:
    start_stat_name = data.stat.names[trip[0][0]]
    end_stat_name = data.stat.names[trip[0][1]]

    top_stat_indices = top_stat_indices.union({trip[0][0], trip[0][1]})

    print(f'Rank {Rank}: {start_stat_name} - {end_stat_name} --- weight is {trip[1]}')
    Rank +=1

    start_stat_lat = data.stat.locations[trip[0][0]][0]
    start_stat_lon = data.stat.locations[trip[0][0]][1]
    end_stat_lat = data.stat.locations[trip[0][1]][0]
    end_stat_lon = data.stat.locations[trip[0][1]][1]

    trip_dists += bs.distance(start_stat_lat,
                           start_stat_lon,
                           end_stat_lat,
                           end_stat_lon)

trip_dist_mean = np.round(trip_dists/N, 1)

print(f'\nMean trip distance: {trip_dist_mean}m')

#%% Plot the most popular trips

stats = np.full(data.stat.n_tot, fill_value = 'black')
stats[list(top_stat_indices)] = 'red'

stats_alpha = np.full(data.stat.n_tot, fill_value = 0.3)
stats_alpha[list(top_stat_indices)] = 1

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

graph = nx.from_numpy_matrix(data.adjacency(days))#, create_using = nx.DiGraph)

lat = [data.stat.loc_merc[i][0] for i in range(data.stat.n_tot)]
long = [data.stat.loc_merc[i][1] for i in range(data.stat.n_tot)]

extent = np.array([np.min(lat)-1000, np.max(lat)+1000, np.min(long)-1000, np.max(long)+1000])
try:
    fig, ax = plt.subplots(figsize=figsize_dict[data.city])
except KeyError:
    fig, ax = plt.subplots(figsize=(8,8))
ax.axis(extent)

nx.draw_networkx_nodes(graph, data.stat.loc_merc, node_size=20,
                       node_color=stats, alpha = stats_alpha, ax=ax)

edge_list = [trip[0] for trip in top_trips]

weights = [trip[1] for trip in top_trips]

w_max = top_trips[0][1]
w_min = top_trips[-1][1]

cmap = mpl.cm.jet

nx.draw_networkx_edges(graph, data.stat.loc_merc, edgelist = edge_list[::-1],
                        width = 3, edge_color=weights[::-1], edge_vmin = w_min,
                        edge_vmax=w_max, edge_cmap= cmap)

ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain,
                attribution='(C) Stamen Design, (C) OpenStreetMap contributors')
fig.set_figwidth((1+0.24)*fig.get_size_inches()[0])

norm = mpl.colors.Normalize(vmin=np.floor(w_min), vmax=np.ceil(w_max))
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
              orientation='vertical', label='weight')

scalebar = AnchoredSizeBar(ax.transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                           pad=0.2, color='black', frameon=False, size_vertical=50)

ax.add_artist(scalebar)

#%% Degree by hour

if not os.path.exists('figures/hour_graphs'):
    os.makedirs('figures/hour_graphs')

# Clean up files
for file in os.listdir('figures/hour_graphs'):
    os.remove('figures/hour_graphs/' + file)

#days = (2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,23,24,25,26,27,30)
day = 4
time_per_frame = 500 # in milliseconds, used in GIMP

figsize_dict = {'nyc': (5,8),
                'chic': (5,8),
                'london': (8,5),
                'oslo': (6.6,5),
                'sfran': (6,5)}

lat = [data.stat.loc_merc[i][0] for i in range(data.stat.n_tot)]
long = [data.stat.loc_merc[i][1] for i in range(data.stat.n_tot)]

extent = np.array([np.min(lat)-1000, np.max(lat)+1000, np.min(long)-1000, np.max(long)+1000])

adj_dict = dict()
max_deg = 0

for hour in range(24):

    adj_matrix = np.zeros(shape=(data.stat.n_tot, data.stat.n_tot))
    for day in days:
        adj_matrix += data.adjacency_hour(day, hour, threshold = 0)
        #print(f'adjacency {day:02d} {hour:02d} computed')
    print(f'adjacency {hour:02d} computed')

    adj_dict[hour] = adj_matrix

    degrees = np.sum(adj_matrix, axis = 0)

    if max_deg < np.max(degrees):
        max_deg = np.max(degrees)

for hour in range(24):

    adj_matrix = adj_dict[hour]

    degrees = np.sum(adj_matrix, axis = 0)

    graph = nx.from_numpy_matrix(adj_matrix)

    fig, ax = plt.subplots(figsize=figsize_dict[data.city], facecolor = '#78797B')
    ax.axis(extent)

    plt.tight_layout()

    nx.draw_networkx_nodes(graph, data.stat.loc_merc, node_size=30,
                           node_color=degrees, cmap='YlGnBu_r', ax=ax,
                           vmin = 0, vmax = max_deg)

    nx.draw_networkx_edges(graph, data.stat.loc_merc, alpha=0.8,
                           width=0.2, edge_color='black', ax=ax)

    plt.subplots_adjust(top = 0.9, right = 0.9)

    #cax = plt.axes([0.85, 0.1, 0.03, 0.8])
    norm = mpl.colors.Normalize(vmin = 0, vmax = max_deg)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='YlGnBu_r'),
                          orientation='vertical', label='degree')

    plt.grid(False)
    _ = plt.axis('off')

    plt.title(f'NYC network in \n {data.year}-{data.month:02d} weekdays {hour:02d}:00:00 - {hour:02d}:59:59')

    plt.savefig(f'figures/hour_graphs/hour_{hour:02d}({time_per_frame}ms)', dpi=150)
    print(f'Hour {hour:02d} saved')
    plt.close()


#%% Degree by hour compared

if not os.path.exists('figures/hour_compare_graphs'):
    os.makedirs('figures/hour_compare_graphs')

# Clean up files
for file in os.listdir('figures/hour_compare_graphs'):
    os.remove('figures/hour_compare_graphs/' + file)

day = 4
time_per_frame = 500 # in milliseconds

figsize_dict = {'nyc': (5,8),
                'chic': (5,8),
                'london': (8,5),
                'oslo': (6.6,5),
                'sfran': (6,5)}

lat = [data.stat.loc_merc[i][0] for i in range(data.stat.n_tot)]
long = [data.stat.loc_merc[i][1] for i in range(data.stat.n_tot)]

extent = np.array([np.min(lat)-1000, np.max(lat)+1000, np.min(long)-1000, np.max(long)+1000])

adj_ref = data.adjacency_hour(
        day = day-1, hour = 23, threshold=0, remove_self_loops=False)

ref_degrees = np.sum(adj_ref, axis = 0) + 1 # +1 to work around division by zero

adj_change = data.adjacency_hour(
    day = day, hour = 0, remove_self_loops=False)

deg_change = np.sum(adj_change, axis = 0)

deg_compare = deg_change / ref_degrees

graph = nx.from_numpy_matrix(adj_ref)

fig, ax = plt.subplots(figsize=figsize_dict[city], facecolor = '#78797B')
ax.axis(extent)

nx.draw_networkx_nodes(graph, data.stat.loc_merc, node_size=30,
                       node_color=deg_compare, cmap='jet',
                       ax=ax, vmin = 0, vmax = 1.5)

nx.draw_networkx_edges(graph, data.stat.loc_merc, alpha=0.8, width=0.2,
                       edge_color='black', ax=ax)

plt.subplots_adjust(bottom = 0.1, top = 0.9, right = 0.8)

cax = plt.axes([0.85, 0.1, 0.03, 0.8])
norm = mpl.colors.Normalize(vmin = 0, vmax = 1.5)
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'),
                     orientation='vertical', label='degree', cax=cax)


plt.grid(False)
_ = plt.axis('off')

plt.title(f'London in {year}-{month:02d}-{day:02d} 0:00:00 - 0:59:59 compared to previous hour')

plt.savefig(f'figures/hour_compare_graphs/hour_0({time_per_frame}ms)', dpi=150)
plt.close()

for hour in range(1,24):

    adj_ref = data.adjacency_hour(
        day = day, hour = hour-1, threshold=0, remove_self_loops=False)

    ref_degrees = np.sum(adj_ref, axis = 0) + 1 # +1 to work around division by zero

    adj_change = data.adjacency_hour(
        day = day, hour = hour, remove_self_loops=False)

    deg_change = np.sum(adj_change, axis = 0)

    deg_compare = deg_change / ref_degrees

    graph = nx.from_numpy_matrix(adj_ref)

    fig, ax = plt.subplots(figsize=figsize_dict[data.city], facecolor = '#78797B')
    ax.axis(extent)

    plt.tight_layout()

    plt.grid(False)
    plt.axis(False)

    nx.draw_networkx_nodes(
        graph, data.stat.loc_merc, node_size=20, node_color=deg_compare,
        cmap='jet', vmin = 0, vmax = 1.5)

    nx.draw_networkx_edges(
        graph, data.stat.loc_merc,
        alpha=0.8, width=0.2, edge_color='black')

    plt.subplots_adjust(bottom = 0.1, top = 0.9, right = 0.8)

    cax = plt.axes([0.85, 0.1, 0.03, 0.8])
    norm = mpl.colors.Normalize(vmin = 0, vmax = 1.5)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'),
                         orientation='vertical', label='degree', cax=cax)

    plt.title(f'NYC network in {data.year}-{data.month:02d}-{day:02d} {hour:02d}:00:00 - {hour:02d}:59:59 compared to previous hour')

    plt.savefig(f'figures/hour_compare_graphs/hour_{hour:02d}({time_per_frame}ms)', dpi=150)
    plt.close()
