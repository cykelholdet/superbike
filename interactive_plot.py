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

from holoviews.element.tiles import OSM

year = 2019
month = 4
data = bs.Data('nyc', year, month)
df = data.df

locations = pd.DataFrame(data.stat.locations).T.rename(columns={0: 'long', 1: 'lat'}, index=data.stat.inverse)

locations['easting'], locations['northing'] = hv.util.transform.lon_lat_to_easting_northing(locations['long'], locations['lat'])

df['easting'], df['northing'] = hv.util.transform.lon_lat_to_easting_northing(df['start_stat_long'], df['start_stat_lat'])

#%%

station_df = locations.copy()
station_df['name'] = data.stat.names.values()

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

#%%
def plot_stations(trip_type='n_arrivals', day_type='day', cnorm='linear', min_trips=0, day=1):
    if day_type == 'weekend':
        days = np.where(np.array(data.weekdays) >= 5)[0] + 1
        station_df['n_departures'] = data.df[data.df['start_dt'].dt.day.isin(days)]['start_stat_id'].value_counts()
        station_df['n_arrivals'] = data.df[data.df['end_dt'].dt.day.isin(days)]['end_stat_id'].value_counts()
    elif day_type == 'business_days':
        days = np.where(np.array(data.weekdays) < 5)[0] + 1
        station_df['n_departures'] = data.df[data.df['start_dt'].dt.day.isin(days)]['start_stat_id'].value_counts()
        station_df['n_arrivals'] = data.df[data.df['end_dt'].dt.day.isin(days)]['end_stat_id'].value_counts()
    elif day_type == 'day':
        station_df['n_departures'] = data.df[data.df['start_dt'].dt.day == day]['start_stat_id'].value_counts()
        station_df['n_arrivals'] = data.df[data.df['end_dt'].dt.day == day]['end_stat_id'].value_counts()
    
    station_df['n_departures'].fillna(0, inplace=True)
    station_df['n_arrivals'].fillna(0, inplace=True)
    subset = station_df[station_df[trip_type] >= min_trips]
    if day_type == 'day':
        title = f'NYC {year:d}-{month:02d}-{day:02d}'
    else:
        title = f'NYC {year:d}-{month:02d} {day_type}'
    subset_plot = subset.hvplot.points(x='easting', y='northing', c=trip_type, cnorm=cnorm, clim=(1, np.nan), s=75, hover_cols=['name'], title=title, tiles='StamenTerrainRetina', line_color='black', width=800, height=800)
    return subset_plot

pn.extension()

kw = dict(min_trips=(0,300), day=(1,30), trip_type=['n_departures', 'n_arrivals'], day_type=['weekend', 'business_days', 'day'], cnorm=['linear', 'log'])


panel = pn.interact(plot_stations, **kw)
text = '#Bikesharing'
panel2 = pn.Row(panel[1][0], pn.Column(text, panel[0][0], panel[0][1], panel[0][2], panel[0][3], panel[0][4]))
bokeh_server = panel2.show(port=12345)

#%%

class BikeParameters(param.Parameterized):
    trip_type = param.Selector(objects=['n_departures', 'n_arrivals'])
    day_type = param.Selector(objects=['weekend', 'business_days', 'day'])
    cnorm = param.Selector(objects=['linear', 'log'])
    min_trips = param.Integer(default=0, bounds=(0, 600))
    day = param.Integer(default=1, bounds=(1, data.num_days))
    
    def view(self):
        return plot_stations(self.trip_type, self.day_type, self.cnorm, self.min_trips, self.day)

bike_params = BikeParameters()

panel_param = pn.Row(bike_params.param, bike_params.view)
text = '#Bikesharing'
bokeh_server = panel_param.show(port=12345)


#%%
activity_dict = {'departures': 'start', 'arrivals': 'end', 'd': 'start', 'a': 'end', 'start': 'start', 'end': 'end'}

def plot_stations2(station_df, df, activity_type='departures', cnorm='linear', min_trips=0):
    colname = f'n_{activity_type}'
    station_df[colname] = df[f'{activity_dict[activity_type]}_stat_id'].value_counts()
    station_df[colname].fillna(0, inplace=True)
    
    subset = station_df[station_df[colname] >= min_trips]
    title = 'NYC'
    subset_plot = subset.hvplot.points(x='easting', y='northing', c=colname, cnorm=cnorm, clim=(1, np.nan), s=75, hover_cols=['name'], title=title, tiles='StamenTerrainRetina', line_color='black', width=800, height=800)
    return subset_plot


class BikeParameters2(param.Parameterized):
    trip_type = param.Selector(objects=['departures', 'arrivals', 'all'])
    day_type = param.Selector(objects=['weekend', 'business_days', 'day'])
    cnorm = param.Selector(objects=['linear', 'log'])
    min_trips = param.Integer(default=0, bounds=(0, 600))
    day = param.Integer(default=1, bounds=(1, data.num_days))
    @param.depends('day_type', watch=True)
    def _update_day(self):
        if self.day_type != 'day':
            self.param['day'].constant = True
        else:
            self.param['day'].constant = False
    
    @param.depends('trip_type', 'day_type', 'cnorm', 'min_trips', 'day')
    def view(self):
        if self.day_type == 'day':
            days = self.day
        else:
            days = self.day_type
        df_subset = data.subset(days=days, activity_type=self.trip_type)
        return plot_stations2(station_df, df_subset, self.trip_type, self.cnorm, self.min_trips)
    
bike_params = BikeParameters2()

panel_param = pn.Row(bike_params.param, bike_params.view)
text = '#Bikesharing'
bokeh_server = panel_param.show(port=12345)

#%%
# stop the bokeh server (when needed)
bokeh_server.stop()

