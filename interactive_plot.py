#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:36:11 2021

@author: dbvd
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import holoviews as hv
import hvplot.pandas
hv.extension('bokeh', logo=False)
import panel as pn
import param
from bokeh.models import HoverTool
import geoviews as gv
import cartopy.crs as ccrs

import simpledtw as dtw

import bikeshare as bs
import interactive_plot_utils as ipu

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

import skimage.color as skcolor
import matplotlib.colors as mpl_colors
from matplotlib import cm


cmap = cm.get_cmap('Blues')

# Load bikeshare data

year = 2019
month = 11
data = bs.Data('nyc', year, month)
df = data.df

#station_df = ipu.make_station_df(data, holidays=False)
station_df, land_use = ipu.make_station_df(data, holidays=False, return_land_use=True)
#station_df.dropna(inplace=True)
#%%

extremes = [station_df['easting'].max(), station_df['easting'].min(), station_df['northing'].max(), station_df['northing'].min()]


month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}



activity_dict = {'departures': 'start', 'arrivals': 'end', 'd': 'start', 'a': 'end', 'start': 'start', 'end': 'end'}
day_type_dict = {'weekend': 'w', 'business_days': 'b'}

color_dict = {0 : 'blue', 1 : 'red', 2 : 'yellow', 3 : 'green', #tab:
              4 : 'purple', 5 : 'brown', 6: 'pink',
              7 : 'cyan', 8 : 'olive', 9 : 'magenta', np.nan: 'gray'}

mpl_color_dict = {i: mpl_colors.to_rgb(color_dict[i]) for i in range(10)}
lab_color_dict = {i: skcolor.rgb2lab(mpl_color_dict[i]) for i in range(10)}
lab_color_list = [lab_color_dict[i] for i in range(10)]


def plot_lines(labels, j, c_centers, title_pre="Mean"):
    cc_df = pd.DataFrame([c_centers[:24], c_centers[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
    cc_plot = cc_df['departures'].hvplot() * cc_df['arrivals'].hvplot()
    n = np.sum(labels == j)
    cc_plot.opts(title=f"{title_pre} of cluster {j} ({color_dict[j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage')
    return cc_plot


class BikeParameters2(param.Parameterized):
    trip_type = param.Selector(objects=['departures', 'arrivals', 'all'])
    day_type = param.Selector(objects=['business_days', 'weekend', 'day'])
    clustering = param.Selector(objects=['k_means', 'k_medoids', 'h_clustering', 'gaussian_mixture', 'none', 'zoning'], doc="Which clustering to perform")
    k = param.Integer(default=3, bounds=(1, 10))
    cnorm = param.Selector(objects=['linear', 'log'])
    day = param.Integer(default=1, bounds=(1, data.num_days))
    dist_func = param.Selector(objects=['norm'])
    plot_all_clusters = param.Selector(objects=['False', 'True'])
    random_state = param.Integer(default=42, bounds=(0, 10000))
    min_trips = param.Integer(default=data.num_days*2, bounds=(0, 800))
    
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.clusters = None
        self.labels = None
        self.index = index
    
    @param.depends('day_type', 'min_trips', 'clustering', 'k', 'random_state', watch=False)
    def plot_clusters_full(self):
        if self.day_type == 'business_days':
            traffic_matrix = data.pickle_daily_traffic(holidays=False)[0]
            x_trips = 'b_trips'
        elif self.day_type == "weekend":
            traffic_matrix = data.pickle_daily_traffic(holidays=False)[1]
            x_trips = 'w_trips'
        
        mask = station_df[x_trips] > self.min_trips
        traffic_matrix = traffic_matrix[mask]
        
        
        if self.clustering == 'k_means':
            self.clusters = KMeans(self.k, random_state=self.random_state).fit(traffic_matrix)
            self.labels = self.clusters.predict(traffic_matrix)
            station_df['label'].loc[mask] = self.labels
            station_df['label'].loc[~mask] = np.nan
            station_df['color'] = station_df['label'].map(color_dict)
    
        elif self.clustering == 'k_medoids':
            self.clusters = KMedoids(self.k, random_state=self.random_state).fit(traffic_matrix)
            self.labels = self.clusters.predict(traffic_matrix)
            station_df['label'].loc[mask] = self.labels
            station_df['label'].loc[~mask] = np.nan
            station_df['color'] = station_df['label'].map(color_dict)
            
        elif self.clustering == 'h_clustering':
            self.clusters = None
            self.labels = AgglomerativeClustering(self.k).fit_predict(traffic_matrix)
            station_df['label'].loc[mask] = self.labels
            station_df['label'].loc[~mask] = np.nan
            station_df['color'] = station_df['label'].map(color_dict)
        
        elif self.clustering == 'gaussian_mixture':
            self.clusters = GaussianMixture(self.k, n_init=10, random_state=self.random_state).fit(traffic_matrix)
            self.labels = self.clusters.predict_proba(traffic_matrix)
            lab_mat = np.array(lab_color_list[:self.k]).T
            lab_cols = [np.sum(self.labels[i] * lab_mat, axis=1) for i in range(len(traffic_matrix))]
            labels_rgb = skcolor.lab2rgb(lab_cols)
            station_df['label'].loc[mask] = pd.Series(list(self.labels), index=mask[mask].index)
            station_df['label'].loc[~mask] = np.nan
            station_df['color'].loc[mask] = ['#%02x%02x%02x' % tuple(label.astype(int)) for label in labels_rgb*255]
            station_df['color'].loc[~mask] = 'gray'
            
        elif self.clustering == 'none':
            self.clusters = None
            self.labels = None
            station_df['label'] = np.nan
            station_df['color'] = station_df[x_trips].tolist()
        
        elif self.clustering == 'zoning':
            self.clusters = None
            self.labels = None
            station_df['label'] = np.nan
            station_df['color'] = [color_dict[zone] for zone in pd.factorize(station_df['zone_type'])[0]]
            
        else:
            self.clusters = None
            self.labels = None
            station_df['label'] = np.nan
            station_df['color'] = None
        if self.clustering == 'none':
            title = f'Number of trips per station in {month_dict[data.month]} {data.year} in {bs.name_dict[data.city]}'
        else:
            title = f"{self.clustering} clustering in {month_dict[data.month]} {data.year} in {bs.name_dict[data.city]}"
        #plot = station_df.hvplot(kind='points', x='easting', y='northing', c='color', s=100, hover_cols=['name', 'n_trips', 'n_departures', 'n_arrivals', 'zone_type'], title=title,  line_color='black', colorbar=False)
        #plot.opts(apply_ranges=False)
        #ds = gv.Dataset(station_df, kdims=['stat_id'], vdims=['long', 'lat', 'color'],)
        #plot = ds.to(gv.Points, ['long', 'lat'], ['stat_id'])
        plot = gv.Points(station_df, kdims=['long', 'lat'], vdims=['stat_id', 'color', 'n_trips', 'b_trips', 'w_trips', 'zone_type', 'name', ])
        plot.opts(gv.opts.Points(fill_color='color', size=10, line_color='black'))
        #plot = gv.Points(station_df, ["easting", "northing"])#.opts(projection=ccrs.GOOGLE_MERCATOR, global_extent=True)
        return plot
    

    @param.depends('day_type', 'clustering', 'k', 'plot_all_clusters', 'min_trips', watch=False)
    def plot_centroid(self, index):
        if self.clustering == 'none':
            return "No clustering"
        elif self.clustering == 'h_clustering':
            if self.plot_all_clusters == 'True':
                if self.day_type == 'business_days':
                    traffic_matrix = data.pickle_daily_traffic(holidays=False)[0]
                    x_trips = 'b_trips'
                elif self.day_type == "weekend":
                    traffic_matrix = data.pickle_daily_traffic(holidays=False)[1]
                    x_trips = 'w_trips'
                mask = station_df[x_trips] > self.min_trips
                traffic_matrix = traffic_matrix[mask]
                cc_plot_list = list()
                for j in range(self.k):
                    mean_vector = np.mean(traffic_matrix[np.where(self.labels == j)], axis=0)
                    cc_plot = plot_lines(self.labels, j, mean_vector)
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to get cluster info"
            else:
                i = index[0]
                if self.day_type == 'business_days':
                    traffic_matrix = data.pickle_daily_traffic(holidays=False)[0]
                    x_trips = 'b_trips'
                elif self.day_type == "weekend":
                    traffic_matrix = data.pickle_daily_traffic(holidays=False)[1]
                    x_trips = 'w_trips'
                mask = station_df[x_trips] > self.min_trips
                traffic_matrix = traffic_matrix[mask]
                if ~np.isnan(station_df['label'][i]):
                    j = int(station_df['label'][i])
                    mean_vector = np.mean(traffic_matrix[np.where(self.labels == j)], axis=0)
                    cc_plot = plot_lines(self.labels, j, mean_vector)
                    return pn.Column(cc_plot, f"Station index {i} is in cluster {j}")
                else:
                    return f"Station index {i} is not in a cluster due to min_trips."
        elif self.clustering == 'gaussian_mixture':
            if self.plot_all_clusters == 'True':
                cc_plot_list = list()
                for j in range(self.k):
                    ccs = self.clusters.means_[j]
                    cc_plot = plot_lines(self.labels.argmax(axis=1), j, ccs)
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to plot cluster centroid"
            else:
                i = index[0]
                if ~np.isnan(station_df['label'][i]):
                    j = station_df['label'][i].argmax()
                    textlist = [f"{j}: {self.labels[i][j]:.2f}\n\n" for j in range(self.k)]
                    ccs = self.clusters.means_[j]
                    cc_plot = plot_lines(self.labels, j, ccs)
                    return pn.Column(cc_plot, f"Station index {i} belongs to cluster \n\n {''.join(textlist)}")
                else:
                    return f"Station index {i} does not belong to a cluster duto to min_trips"
        else: # k-means or k-medoids
            if self.plot_all_clusters == 'True':
                if self.clusters == None:
                    return "Please select Clustering"
                    
                cc_plot_list = list()
                for j in range(self.k):
                    ccs = self.clusters.cluster_centers_[j]
                    cc_plot = plot_lines(self.labels, j, ccs, title_pre="Centroid")
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to plot cluster centroid"
            else:
                i = index[0]
                if ~np.isnan(station_df['label'][i]):
                    j = int(station_df['label'][i])
                    ccs = self.clusters.cluster_centers_[j]
                    cc_plot = plot_lines(station_df['label'], j, ccs, title_pre="Centroid")
                    return pn.Column(cc_plot, f"Station index {i} is in cluster {j}")
                else:
                    return f"Station index {i} is not in a cluster due to min_trips."

    
bike_params = BikeParameters2(None)

params = pn.Param(bike_params.param, widgets={
    'clustering': pn.widgets.RadioBoxGroup,
    'trip_type': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success'},
    'day_type': pn.widgets.RadioButtonGroup,
    'plot_all_clusters': {'widget_type': pn.widgets.RadioButtonGroup, 'title': 'Hello'},
    'day': pn.widgets.IntSlider,
    'random_state': pn.widgets.IntInput,
    'min_trips': pn.widgets.IntInput,
    'k': pn.widgets.IntSlider,
    },
    name="Bikeshare Parameters"
    )

paraview = hv.DynamicMap(bike_params.plot_clusters_full)


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



#extreme_view = extremes.hvplot.points(x='easting', y='northing')


@pn.depends(clustering=bike_params.param.clustering)
def plot_tiles(clustering):
    
    tiles = hv.element.tiles.StamenTerrainRetina()
    tiles = gv.tile_sources.StamenTerrainRetina()
    tiles.opts(height=800, width=800, xlim=(extremes[1], extremes[0]), ylim=(extremes[3], extremes[2]), active_tools=['wheel_zoom'])
    return tiles


tileview = hv.DynamicMap(plot_tiles)



tooltips = [
    ('ID', '@stat_id'),
    ('Name', '@name'),
    ('Cluster', '@color'),
    #('n_departures', '@n_departures'),
    #('n_arrivals', '@n_arrivals'),
    ('n_trips', '@n_trips total'),
    ('b_trips', '@b_trips total'),
    ('w_trips', '@w_trips total'),
    ('land use', '@zone_type')
]
hover = HoverTool(tooltips=tooltips)

paraview.opts(tools=['tap', hover])
paraview.opts(nonselection_alpha=0.3)

selection_stream = hv.streams.Selection1D(source=paraview)


@pn.depends(index=selection_stream.param.index,
            day_type=bike_params.param.day_type)
def plot_daily_traffic(index, day_type):
    if not index:
        return "Select a station to see station traffic"
    else:
        i = index[0]
    plot = line_callback_both(i, day_type)
    if day_type == 'business_days':
        b_trips = station_df['b_trips'].iloc[i]
        plot.opts(title=f'Average hourly traffic for {data.stat.names[i]} (b_trips={b_trips:n})', ylabel='percentage')
    if day_type == 'weekend':
        w_trips = station_df['w_trips'].iloc[i]
        plot.opts(title=f'Average hourly traffic for {data.stat.names[i]} (w_trips={w_trips:n})', ylabel='percentage')
    
    return plot


@pn.depends(index=selection_stream.param.index,
            plot_all_clusters=bike_params.param.plot_all_clusters,
            clustering=bike_params.param.clustering,
            k=bike_params.param.k,
            min_trips=bike_params.param.min_trips)
def plotterino(index, plot_all_clusters, clustering, k, min_trips):
    return bike_params.plot_centroid(index)
    

@pn.depends(pn.state.param.busy)
def indicator(busy):
    return "I'm busy" if busy else "I'm idle"


@pn.depends(clustering=bike_params.param.clustering, watch=True)
def show_widgets(clustering):
    if clustering in ['k_means', 'k_medoids', 'h_clustering', 'gaussian_mixture']:
        params.widgets['k'].visible = True
    else:
        params.widgets['k'].visible = False

@pn.depends(min_trips=bike_params.param.min_trips,
            day_type=bike_params.param.day_type)
def minpercent(min_trips, day_type):
    if day_type == 'business_days':
        n_retained = (station_df.b_trips > min_trips).sum()
    else:
        n_retained = (station_df.w_trips > min_trips).sum()
    
    n_removed = len(station_df) - n_retained
    return f"Removed {n_removed:d} stations, which is {(n_removed/len(station_df))*100:.2f}%"


linecol = pn.Column(plot_daily_traffic, plotterino)

params.layout.insert(8, 'Plot all clusters:')
params.layout.insert(3, 'Clustering method:')

param_column = pn.Column(params.layout, minpercent)

panel_param = pn.Row(param_column, tileview*paraview, linecol)
text = '#Bikesharing Clustering Analysis'
panel_column = pn.Column(text, panel_param, indicator)
panel_column.servable() # Run with: panel serve interactive_plot.py --autoreload

bokeh_server = panel_column.show(port=12345)

#%%
bokeh_server.stop() # Run with: panel serve interactive_plot.py --autoreload
