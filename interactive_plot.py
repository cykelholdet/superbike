#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:36:11 2021

@author: dbvd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import holoviews as hv
import hvplot.pandas
hv.extension('bokeh', logo=False)
import panel as pn
import param

import bikeshare as bs
import time

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids

from holoviews.element.tiles import OSM


year = 2019
month = 4
data = bs.Data('nyc', year, month)
df = data.df

locations = pd.DataFrame(data.stat.locations).T.rename(columns={0: 'long', 1: 'lat'}, index=data.stat.inverse)

locations['easting'], locations['northing'] = hv.util.transform.lon_lat_to_easting_northing(locations['long'], locations['lat'])

df['easting'], df['northing'] = hv.util.transform.lon_lat_to_easting_northing(df['start_stat_long'], df['start_stat_lat'])



station_df = locations.copy()
station_df['name'] = data.stat.names.values()

month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}



# day = 2


# station_df['n_departures'] = data.df[data.df['start_dt'].dt.day == day]['start_stat_id'].value_counts()
# station_df['n_arrivals'] = data.df[data.df['end_dt'].dt.day == day]['end_stat_id'].value_counts()

# station_df['n_departures'].fillna(0, inplace=True)
# station_df['n_arrivals'].fillna(0, inplace=True)

# min_slider = pn.widgets.IntSlider(name='min', start=1, end=300, value=2)
# st_dfi = station_df.interactive()

# def update_data(attrname, old, new):

#     # Get the current slider values
#     subset = st_dfi[station_df['n_departures'] >= min_slider.value]

# subset = st_dfi[station_df['n_departures'] >= min_slider.value]
# hv_plot = st_dfi.hvplot(kind='points', x='easting', y='northing', c='n_departures', cnorm='log', clim=(1, np.nan), s=50, hover_cols=['name'], title=f'NYC {year:d}-{month:02d}-{day:02d}', tiles='StamenTerrainRetina', line_color='black', width=800, height=800)

# #hv_plot = station_df.hvplot(x='easting', y='northing', c='n_departures', kind='points', cnorm='log', clim=(1, np.nan), s=50, hover_cols=['name'], title=f'NYC {year:d}-{month:02d}-{day:02d}', tiles='StamenTerrainRetina', line_color='black', width=800, height=800)

# # display graph in browser
# # a bokeh server is automatically started
# min_slider.on_change('value', update_data)

# bokeh_server = pn.Row(hv_plot, min_slider, min_slider.value).show(port=12345, threaded=True)

# #%%
# def plot_stations(trip_type='n_arrivals', day_type='day', cnorm='linear', min_trips=0, day=1):
#     if day_type == 'weekend':
#         days = np.where(np.array(data.weekdays) >= 5)[0] + 1
#         station_df['n_departures'] = data.df[data.df['start_dt'].dt.day.isin(days)]['start_stat_id'].value_counts()
#         station_df['n_arrivals'] = data.df[data.df['end_dt'].dt.day.isin(days)]['end_stat_id'].value_counts()
#     elif day_type == 'business_days':
#         days = np.where(np.array(data.weekdays) < 5)[0] + 1
#         station_df['n_departures'] = data.df[data.df['start_dt'].dt.day.isin(days)]['start_stat_id'].value_counts()
#         station_df['n_arrivals'] = data.df[data.df['end_dt'].dt.day.isin(days)]['end_stat_id'].value_counts()
#     elif day_type == 'day':
#         station_df['n_departures'] = data.df[data.df['start_dt'].dt.day == day]['start_stat_id'].value_counts()
#         station_df['n_arrivals'] = data.df[data.df['end_dt'].dt.day == day]['end_stat_id'].value_counts()
    
#     station_df['n_departures'].fillna(0, inplace=True)
#     station_df['n_arrivals'].fillna(0, inplace=True)
#     subset = station_df[station_df[trip_type] >= min_trips]
#     if day_type == 'day':
#         title = f'NYC {year:d}-{month:02d}-{day:02d}'
#     else:
#         title = f'NYC {year:d}-{month:02d} {day_type}'
#     subset_plot = subset.hvplot.points(x='easting', y='northing', c=trip_type, cnorm=cnorm, clim=(1, np.nan), s=75, hover_cols=['name'], title=title, tiles='StamenTerrainRetina', line_color='black', width=800, height=800)
#     return subset_plot

# pn.extension()

# kw = dict(min_trips=(0,300), day=(1,30), trip_type=['n_departures', 'n_arrivals'], day_type=['weekend', 'business_days', 'day'], cnorm=['linear', 'log'])


# panel = pn.interact(plot_stations, **kw)
# text = '#Bikesharing'
# panel2 = pn.Row(panel[1][0], pn.Column(text, panel[0][0], panel[0][1], panel[0][2], panel[0][3], panel[0][4]))
# bokeh_server = panel2.show(port=12345)

# #%%

# class BikeParameters(param.Parameterized):
#     trip_type = param.Selector(objects=['n_departures', 'n_arrivals'])
#     day_type = param.Selector(objects=['weekend', 'business_days', 'day'])
#     cnorm = param.Selector(objects=['linear', 'log'])
#     min_trips = param.Integer(default=0, bounds=(0, 600))
#     day = param.Integer(default=1, bounds=(1, data.num_days))
    
#     def view(self):
#         return plot_stations(self.trip_type, self.day_type, self.cnorm, self.min_trips, self.day)

# bike_params = BikeParameters()

# panel_param = pn.Row(bike_params.param, bike_params.view)
# text = '#Bikesharing'
# bokeh_server = panel_param.show(port=12345)


#%%
activity_dict = {'departures': 'start', 'arrivals': 'end', 'd': 'start', 'a': 'end', 'start': 'start', 'end': 'end'}
day_type_dict = {'weekend': 'w', 'business_days': 'b'}


def plot_stations2(station_df, df, activity_type='departures', cnorm='linear', min_trips=0):
    colname = f'n_{activity_type}'
    stdf = station_df
    stdf[colname] = df[f'{activity_dict[activity_type]}_stat_id'].value_counts()
    stdf[colname].fillna(0, inplace=True)
    
    subset = stdf[stdf[colname] >= min_trips]
    title = 'NYC'
    subset_plot = subset.hvplot.points(x='easting', y='northing', c=colname, cnorm=cnorm, clim=(1, np.nan), s=100, hover_cols=['name'], title=title, line_color='black')
    return subset_plot


def plot_clusters(station_df, day_type, min_trips, clustering, k, dist_func):
    if day_type == 'business_days':
        traffic_matrix = data.pickle_daily_traffic()[0]
    elif day_type == "weekend":
        traffic_matrix = data.pickle_daily_traffic()[1]

    clf = bs.Classifier(dist_func = dist_func)
    
    if clustering == 'k_means':
        #clf.k_means(traffic_matrix, k, seed=69)
        clusters = KMeans(k).fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)

    if clustering == 'k_medoids':
        clusters = KMedoids(k).fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)
        
    if clustering == 'h_clustering':
        clusters = None
        labels = AgglomerativeClustering(k).fit_predict(traffic_matrix)


    color_dict = {0 : 'tab:blue', 1 : 'tab:orange', 2 : 'tab:green', 3 : 'tab:red',
              4 : 'tab:purple', 5 : 'tab:brown', 6: 'tab:pink',
              7 : 'tab:gray', 8 : 'tab:olive', 9 : 'tab:cyan'}
    
    color_dict = {0 : 'blue', 1 : 'orange', 2 : 'green', 3 : 'red',
              4 : 'purple', 5 : 'brown', 6: 'pink',
              7 : 'gray', 8 : 'olive', 9 : 'cyan'}

    station_df['color'] = [color_dict[label] for label in labels]
    plot = station_df.hvplot(kind='points', x='easting', y='northing', c='color', s=100, hover_cols=['name'], title='Clusters',  line_color='black')
    plot.opts(apply_ranges=False)
    return plot, clusters, labels

class BikeParameters2(param.Parameterized):
    trip_type = param.Selector(objects=['departures', 'arrivals', 'all'])
    day_type = param.Selector(objects=['weekend', 'business_days', 'day'])
    cnorm = param.Selector(objects=['linear', 'log'])
    min_trips = param.Integer(default=0, bounds=(0, 600))
    day = param.Integer(default=1, bounds=(1, data.num_days))
    clustering = param.Selector(objects=['none', 'k_means', 'k_medoids', 'h_clustering'], doc="Which clustering to perform")
    k = param.Integer(default=3, bounds=(1, 10))
    dist_func = param.Selector(objects=['norm'])
    #@param.depends('day_type', watch=True)
    # def _update_day(self):
    #     if self.day_type != 'day':
    #         self.param['day'].precedence = -1
    #     else:
    #         self.param['day'].precedence = 1
    
bike_params = BikeParameters2()

params = pn.Param(bike_params.param, widgets={
    'trip_type': pn.widgets.RadioButtonGroup,
    'day_type': pn.widgets.RadioButtonGroup,
    'day': pn.widgets.IntSlider,
    })
   
 
@pn.depends(trip_type=bike_params.param.trip_type,
            day_type=bike_params.param.day_type, 
            day=bike_params.param.day,
            min_trips=bike_params.param.min_trips,
            clustering=bike_params.param.clustering,
            k=bike_params.param.k,
            dist_func=bike_params.param.dist_func,
            cnorm=bike_params.param.cnorm)
def bike_para_view(trip_type, day_type, day, min_trips, clustering, k, dist_func, cnorm):
    if day_type == 'day':
        days = day
    else:
        days = day_type
    df_subset = data.subset(days=days, activity_type=trip_type)
    
    if clustering == 'none':
        plot = plot_stations2(station_df, df_subset, trip_type, cnorm, min_trips)
    else:
        plot, globals()['clusters'], globals()['labels'] = plot_clusters(station_df, day_type, min_trips, clustering, k, dist_func)
        plot.opts(legend_cols=3, show_legend=True)
    # if 'firstplot' in globals():
    #     plot.opts(apply_ranges=False)
    # globals()['firstplot'] = False
    return plot



def line_callback_both(index, day_type):
    print(f"{index=}")
    a = data.daily_traffic_average(index, period=day_type_dict[day_type], return_std=True)
    means = pd.DataFrame(a[:2]).T.rename(columns={0:'departures', 1:'arrivals'})
    stds = pd.DataFrame(a[2:]).T.rename(columns={0:'departures', 1:'arrivals'})
    varea = pd.DataFrame()
    varea['dep_low'] = means['departures'] - stds['departures']
    varea['dep_high'] = means['departures'] + stds['departures']
    varea['arr_low'] = means['arrivals'] - stds['arrivals']
    varea['arr_high'] = means['arrivals'] + stds['arrivals']
    varea['hour'] = np.arange(0,24)
    return varea.hvplot.area(x='hour', y='dep_low', y2='dep_high', alpha=0.5, line_width=0) * varea.hvplot.area(x='hour', y='arr_low', y2='arr_high', alpha=0.5, line_width=0) * means['departures'].hvplot() * means['arrivals'].hvplot()

def lc_plot(index):
    if not index:
        return pd.DataFrame([1,1]).hvplot()
    else:
        index = index[0]
    print(f"{index=}")
    a_plot = data.daily_traffic_average(index, plot=True, return_fig=True)
    a_plot.savefig('nicefig.png')


#params = pn.Column(pn.panel(bike_params))

# bike_view = bike_para_view(bike_params.trip_type, bike_params.day_type, bike_params.day, bike_params.min_trips, bike_params.clustering, bike_params.k, bike_params.dist_func, bike_params.cnorm)
# bike_view = hv.Dataset(station_df).apply(bike_para_view, trip_type=bike_params.trip_type,
#                                          day_type=bike_params.day_type, 
#                                          day=bike_params.day,
#                                          min_trips=bike_params.min_trips,
#                                          clustering=bike_params.clustering,
#                                          k=bike_params.k,
#                                          dist_func=bike_params.dist_func,
#                                          cnorm=bike_params.cnorm)

#extremes = station_df.iloc[[station_df['easting'].argmax(), station_df['easting'].argmin(), station_df['northing'].argmax(), station_df['northing'].argmin()]]

extremes = [station_df['easting'].max(), station_df['easting'].min(), station_df['northing'].max(), station_df['northing'].min()]


#extreme_view = extremes.hvplot.points(x='easting', y='northing')

paraview = hv.DynamicMap(bike_para_view)

# paraview.opts(aspect="equal")
#paraview.opts(data_aspect=1)

# bike_view.opts(height=800, width=800)

tiles = hv.element.tiles.StamenTerrainRetina()
#tiles = gts.StamenTerrainRetina
tiles.opts(height=800, width=800, xlim=(extremes[1], extremes[0]), ylim=(extremes[3], extremes[2]), apply_ranges=False, active_tools=['wheel_zoom'], data_aspect=1)


paraview.opts(tools=['tap'])
paraview.opts(apply_ranges=False, nonselection_alpha=0.3)

selection_stream = hv.streams.Selection1D(source=paraview)

@pn.depends(index=selection_stream.param.index,
            day_type=bike_params.param.day_type)
def plot_daily_traffic(index, day_type):
    if not index:
        return pd.DataFrame([1,1]).hvplot()
    else:
        i = index[0]
    plot = line_callback_both(i, day_type)
    plot.opts(title=f'Average hourly traffic for {data.stat.names[i]}', ylabel='percentage')
    return plot

@pn.depends(index=selection_stream.param.index,
            clustering=bike_params.param.clustering,
            day_type=bike_params.param.day_type,
            k=bike_params.param.k)
def plot_centroid(index, clustering, day_type, k):
    print('hello')
    if clustering == 'none':
        return "No clustering"
    else:
        if not index:
            return "Select a station to plot cluster centroid"
        else:
            i = index[0]
            ccs = clusters.cluster_centers_[labels[i]]
            cc_df = pd.DataFrame([ccs[:24], ccs[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
            cc_plot = cc_df['departures'].hvplot() * cc_df['arrivals'].hvplot()
            cc_plot.opts(title=f"Centroid of cluster {labels[i]}", legend_position='top_right', xlabel='hour', ylabel='percentage')
            return cc_plot
    
    

#lines_dep = hv.DynamicMap(line_callback_dep, streams=[selection_stream])
#lines_arr = hv.DynamicMap(line_callback_arr, streams=[selection_stream])

#laaa = selection_stream.param.watch(lambda a: lc_plot(a.new), 'index')
#lines_lc = hv.DynamicMap(lc_plot, streams=[selection_stream])

# lines_dep = line_callback_dep.apply(selection_stream.index)
# lines_arr = line_callback_arr(selection_stream.index)
linecol = pn.Column(plot_daily_traffic, plot_centroid)

panel_param = pn.Row(bike_params, tiles*paraview, linecol)
text = '#Bikesharing'
bokeh_server = panel_param.show(port=12345)

#%%
# stop the bokeh server (when needed)
bokeh_server.stop()

