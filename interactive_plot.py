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
import matplotlib.colors as mpl_colors
import skimage.color as skcolor
from matplotlib import cm

import holoviews as hv
import hvplot.pandas
hv.extension('bokeh', logo=False)
import panel as pn
import param
import geoviews as gv
from bokeh.models import HoverTool
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

import bikeshare as bs
import interactive_plot_utils as ipu


cmap = cm.get_cmap('Blues')

# Load bikeshare data

YEAR = 2019
MONTH = None
CITY = 'nyc'

#station_df = ipu.make_station_df(data, holidays=False)
#station_df, land_use = ipu.make_station_df(data, holidays=False, return_land_use=True)
#station_df.dropna(inplace=True)

def mask_traffic_matrix(traffic_matrices, station_df, day_type, min_trips, holidays=False, return_mask=False):
    """
    Applies a mask to the daily traffic matrix based on the minimum number of 
    trips to include.

    Parameters
    ----------
    day_type : str
        'business_days' or 'weekend'.
    min_trips : int
        the minimum number of trips for a station. If the station has fewer
        trips than this, exclude it.
    holidays : bool, optional
        Whether to include holidays in business days (True) or remove them from
        the business days (False). The default is False.

    Returns
    -------
    np array
        masked traffic matrix, that is, the number of 48-dimensional vectors 
        which constitute the rows of the traffic matrix is reduced.

    """
    if day_type == 'business_days':
        traffic_matrix = traffic_matrices[0]
        x_trips = 'b_trips'
    elif day_type == "weekend":
        traffic_matrix = traffic_matrices[1]
        x_trips = 'w_trips'
    mask = station_df[x_trips] > min_trips
    if return_mask:
        return traffic_matrix[mask], mask, x_trips
    else:
        return traffic_matrix[mask]
    

def plot_center(labels, cluster_j, c_center, title_pre="Mean"):
    """
    Plot a 48-dimensional cluster center vector as two lines in a hvplot entity

    Parameters
    ----------
    labels : np array of int
        the labels of all the vectors.
    cluster_j : int
        the cluster which this center is a member of.
    c_center : 48 dimensional np array
        The cluster center to plot.
    title_pre : str, optional
        The preface in the title. Because medoids are medoids for example. The
        default is "Mean".

    Returns
    -------
    cc_plot : hvplot
        a plot.

    """
    cc_df = pd.DataFrame([c_center[:24], c_center[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
    cc_plot = cc_df['departures'].hvplot() * cc_df['arrivals'].hvplot()
    n = np.sum(labels == cluster_j)
    cc_plot.opts(title=f"{title_pre} of cluster {cluster_j} ({color_dict[cluster_j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage')
    return cc_plot


def plot_dta_with_std(data, index, day_type):
    """
    Plot daily traffic average of station including standard deviation bounds.

    Parameters
    ----------
    index : int
        The index of the station.
    day_type : str
        'weekend' or 'business_days'.

    Returns
    -------
    hvplot
        a plot.

    """
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


def get_clusters(traffic_matrices, station_df, day_type, min_trips, clustering, k, random_state):
    """
    From a station dataframe and associated variables, return the updated 
    station df and clustering and labels

    Parameters
    ----------
    station_df : pandas dataframe
        has each station as a row.
    day_type : str
        'weekend' or 'business_days'.
    min_trips : int
        minimum number of trips.
    clustering : str
        clustering type.
    k : int
        number of clusters.
    random_state : int
        the seed for the random generator.

    Returns
    -------
    station_df : pandas dataframe
        has each station as a row and color and label columns populated.
    clusters : sklearn.clustering cluster
        can be used for stuff later.
    labels : np array
        the labels of the masked traffic matrix.

    """
    traffic_matrix, mask, x_trips = mask_traffic_matrix(
        traffic_matrices, station_df, day_type, min_trips, holidays=False, return_mask=True)
    
    if clustering == 'k_means':
        clusters = KMeans(k, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)
        station_df['label'].loc[mask] = labels
        station_df['label'].loc[~mask] = np.nan
        station_df['color'] = station_df['label'].map(color_dict)

    elif clustering == 'k_medoids':
        clusters = KMedoids(k, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)
        station_df['label'].loc[mask] = labels
        station_df['label'].loc[~mask] = np.nan
        station_df['color'] = station_df['label'].map(color_dict)
        
    elif clustering == 'h_clustering':
        clusters = None
        labels = AgglomerativeClustering(k).fit_predict(traffic_matrix)
        station_df['label'].loc[mask] = labels
        station_df['label'].loc[~mask] = np.nan
        station_df['color'] = station_df['label'].map(color_dict)
    
    elif clustering == 'gaussian_mixture':
        clusters = GaussianMixture(k, n_init=10, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict_proba(traffic_matrix)
        lab_mat = np.array(lab_color_list[:k]).T
        lab_cols = [np.sum(labels[i] * lab_mat, axis=1) for i in range(len(traffic_matrix))]
        labels_rgb = skcolor.lab2rgb(lab_cols)
        station_df['label'].loc[mask] = pd.Series(list(labels), index=mask[mask].index)
        station_df['label'].loc[~mask] = np.nan
        station_df['color'].loc[mask] = ['#%02x%02x%02x' % tuple(label.astype(int)) for label in labels_rgb*255]
        station_df['color'].loc[~mask] = 'gray'
        
    elif clustering == 'none':
        clusters = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = station_df[x_trips].tolist()
    
    elif clustering == 'zoning':
        clusters = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = [color_dict[zone] for zone in pd.factorize(station_df['zone_type'])[0]]
        
    else:
        clusters = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = None
    
    return station_df, clusters, labels

class BikeDash(param.Parameterized):
    """
    Class containing everything for the bikeshare dashboard.
    
    Variables for the dashboard are introduced as param objects with their
    possible values. In addition, the plotting functions are defined.
    """
    city = param.Selector(default=CITY, objects=['nyc', 'chic', 'helsinki', 'madrid', 'edinburgh'])
    if MONTH == None:
        month = param.Selector(default=MONTH, objects=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    else:
        month = param.Integer(default=MONTH, bounds=(1, 12))
    trip_type = param.Selector(objects=['departures', 'arrivals', 'all'])
    day_type = param.Selector(objects=['business_days', 'weekend', 'day'])
    clustering = param.Selector(objects=['k_means', 'k_medoids', 'h_clustering', 'gaussian_mixture', 'none', 'zoning'], doc="Which clustering to perform")
    k = param.Integer(default=3, bounds=(1, 10))
    cnorm = param.Selector(objects=['linear', 'log'])
    day = param.Integer(default=1, bounds=(1, 31))
    dist_func = param.Selector(objects=['norm'])
    plot_all_clusters = param.Selector(objects=['False', 'True'])
    show_land_use = param.Selector(objects=['False', 'True'])
    random_state = param.Integer(default=42, bounds=(0, 10000))
    min_trips = param.Integer(default=100, bounds=(0, 800))
    
    
    boolean_ = param.Boolean(True)
    
    
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)  # Runs init for the superclass param.Parameterized
        self.clusters = None
        self.labels = None
        self.index = index
        self.data = bs.Data(self.city, YEAR, self.month)
        self.station_df, self.land_use = ipu.make_station_df(self.data, holidays=False, return_land_use=True)
        self.traffic_matrices = self.data.pickle_daily_traffic(holidays=False)
        
    @param.depends('month', 'city', watch=True)
    def get_data(self):
        print(f'getting {self.month}')
        self.data = bs.Data(self.city, YEAR, self.month)
        self.station_df, self.land_use = ipu.make_station_df(self.data, holidays=False, return_land_use=True)
        self.traffic_matrices = self.data.pickle_daily_traffic(holidays=False)
        print(len(self.station_df))
        self.boolean_ = not self.boolean_
    
    
    @param.depends('day_type', 'min_trips', 'clustering', 'k', 'random_state', 'boolean_', watch=False)
    def plot_clusters_full(self):
        self.station_df, self.clusters, self.labels = get_clusters(self.traffic_matrices, self.station_df, self.day_type, self.min_trips, self.clustering, self.k, self.random_state)
        
        if self.clustering == 'none':
            title = f'Number of trips per station in {month_dict[self.month]} {YEAR} in {bs.name_dict[self.city]}'
        else:
            title = f"{self.clustering} clustering in {month_dict[self.month]} {YEAR} in {bs.name_dict[self.city]}"

        plot = gv.Points(self.station_df, kdims=['long', 'lat'], vdims=['stat_id', 'color', 'n_trips', 'b_trips', 'w_trips', 'zone_type', 'name', ])
        plot.opts(gv.opts.Points(fill_color='color', size=10, line_color='black'))
        return plot
    

    @param.depends('day_type', 'clustering', 'k', 'plot_all_clusters', 'min_trips', 'boolean_', watch=False)
    def plot_centroid(self, index):
        if self.clustering == 'none':
            return "No clustering"
        elif self.clustering == 'h_clustering':
            if self.plot_all_clusters == 'True':
                traffic_matrix = mask_traffic_matrix(self.traffic_matrices, self.station_df, self.day_type, self.min_trips, holidays=False)
                cc_plot_list = list()
                for j in range(self.k):
                    mean_vector = np.mean(traffic_matrix[np.where(self.labels == j)], axis=0)
                    cc_plot = plot_center(self.labels, j, mean_vector)
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to get cluster info"
            else:
                i = index[0]
                traffic_matrix = mask_traffic_matrix(self.traffic_matrices, self.station_df, self.day_type, self.min_trips, holidays=False)
                if ~np.isnan(self.station_df['label'][i]):
                    j = int(self.station_df['label'][i])
                    mean_vector = np.mean(traffic_matrix[np.where(self.labels == j)], axis=0)
                    cc_plot = plot_center(self.labels, j, mean_vector)
                    return pn.Column(cc_plot, f"Station index {i} is in cluster {j}")
                else:
                    return f"Station index {i} is not in a cluster due to min_trips."

        elif self.clustering == 'gaussian_mixture':
            if self.plot_all_clusters == 'True':
                cc_plot_list = list()
                for j in range(self.k):
                    ccs = self.clusters.means_[j]
                    cc_plot = plot_center(self.labels.argmax(axis=1), j, ccs)
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to plot cluster centroid"
            else:
                i = index[0]
                if ~np.isnan(self.station_df['label'][i]):
                    j = self.station_df['label'][i].argmax()
                    textlist = [f"{j}: {self.labels[i][j]:.2f}\n\n" for j in range(self.k)]
                    ccs = self.clusters.means_[j]
                    cc_plot = plot_center(self.labels, j, ccs)
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
                    cc_plot = plot_center(self.labels, j, ccs, title_pre="Centroid")
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to plot cluster centroid"
            else:
                i = index[0]
                if ~np.isnan(self.station_df['label'][i]):
                    j = int(self.station_df['label'][i])
                    ccs = self.clusters.cluster_centers_[j]
                    cc_plot = plot_center(self.station_df['label'], j, ccs, title_pre="Centroid")
                    return pn.Column(cc_plot, f"Station index {i} is in cluster {j}")
                else:
                    return f"Station index {i} is not in a cluster due to min_trips."


bike_params = BikeDash(None)

params = pn.Param(bike_params.param, widgets={
    'clustering': pn.widgets.RadioBoxGroup,
    'trip_type': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success'},
    'day_type': pn.widgets.RadioButtonGroup,
    'plot_all_clusters': {'widget_type': pn.widgets.RadioButtonGroup, 'title': 'Hello'},
    'day': pn.widgets.IntSlider,
    'random_state': pn.widgets.IntInput,
    'min_trips': pn.widgets.IntInput,
    'k': pn.widgets.IntSlider,
    'show_land_use': pn.widgets.RadioButtonGroup,
    },
    name="Bikeshare Parameters"
    )

paraview = hv.DynamicMap(bike_params.plot_clusters_full)


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
            day_type=bike_params.param.day_type,
            city=bike_params.param.city,
            month=bike_params.param.month)
def plot_daily_traffic(index, day_type, city, month): 
    if not index:
        return "Select a station to see station traffic"
    else:
        i = index[0]
    plot = plot_dta_with_std(bike_params.data, i, day_type)
    if day_type == 'business_days':
        b_trips = bike_params.station_df['b_trips'].iloc[i]
        plot.opts(title=f'Average hourly traffic for {bike_params.data.stat.names[i]} (b_trips={b_trips:n})', ylabel='percentage')
    if day_type == 'weekend':
        w_trips = bike_params.station_df['w_trips'].iloc[i]
        plot.opts(title=f'Average hourly traffic for {bike_params.data.stat.names[i]} (w_trips={w_trips:n})', ylabel='percentage')
    
    return plot


@pn.depends(index=selection_stream.param.index,
            plot_all_clusters=bike_params.param.plot_all_clusters,
            clustering=bike_params.param.clustering,
            k=bike_params.param.k,
            min_trips=bike_params.param.min_trips,
            day_type=bike_params.param.day_type,
            city=bike_params.param.city,
            month=bike_params.param.month)
def plot_centroid_callback(index, plot_all_clusters, clustering, k, min_trips, day_type, city, month):
    return bike_params.plot_centroid(index)
    

@pn.depends(pn.state.param.busy)
def indicator(busy):
    return gif_pane if busy else ""


@pn.depends(clustering=bike_params.param.clustering, watch=True)
def show_widgets(clustering):
    if clustering in ['k_means', 'k_medoids', 'h_clustering', 'gaussian_mixture']:
        params.widgets['k'].visible = True
    else:
        params.widgets['k'].visible = False

@pn.depends(min_trips=bike_params.param.min_trips,
            day_type=bike_params.param.day_type,
            city=bike_params.param.city,
            month=bike_params.param.month)
def minpercent(min_trips, day_type, city, month):
    if day_type == 'business_days':
        n_retained = (bike_params.station_df.b_trips > min_trips).sum()
    else:
        n_retained = (bike_params.station_df.w_trips > min_trips).sum()
    
    n_removed = len(bike_params.station_df) - n_retained
    return f"Removed {n_removed:d} stations, which is {(n_removed/len(bike_params.station_df))*100:.2f}%"

@pn.depends(show_land_use=bike_params.param.show_land_use,
            city=bike_params.param.city)
def land_use_plot(show_land_use, city):
    if show_land_use == "True":
        return gv.Polygons(bike_params.land_use, vdims=['zone_type', 'color']).opts(color='color')
    else:
        return gv.Polygons([])

extremes = [bike_params.station_df['easting'].max(), bike_params.station_df['easting'].min(), 
            bike_params.station_df['northing'].max(), bike_params.station_df['northing'].min()]

month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None'}

activity_dict = {'departures': 'start', 'arrivals': 'end', 'd': 'start', 'a': 'end', 'start': 'start', 'end': 'end'}
day_type_dict = {'weekend': 'w', 'business_days': 'b'}

color_dict = {0 : 'blue', 1 : 'red', 2 : 'yellow', 3 : 'green', #tab:
              4 : 'purple', 5 : 'cyan', 6: 'pink',
              7 : 'brown', 8 : 'olive', 9 : 'magenta', np.nan: 'gray'}

mpl_color_dict = {i: mpl_colors.to_rgb(color_dict[i]) for i in range(10)}
lab_color_dict = {i: skcolor.rgb2lab(mpl_color_dict[i]) for i in range(10)}
lab_color_list = [lab_color_dict[i] for i in range(10)]

gif_pane = pn.pane.GIF('Loading_Key.gif')

zoneview = hv.DynamicMap(land_use_plot)
zoneview.opts(alpha=0.5, apply_ranges=False)

tooltips_zone = [
    ('Zone Type', '@zone_type'),
]
hover_zone = HoverTool(tooltips=tooltips_zone)
zoneview.opts(tools=[hover_zone])
# selection_stream.param.values()['index']
linecol = pn.Column(plot_daily_traffic, plot_centroid_callback)

params.layout.insert(11, 'Show land use:')
params.layout.insert(10, 'Plot all clusters:')
params.layout.insert(5, 'Clustering method:')

param_column = pn.Column(params.layout, minpercent)
param_column[1].width=300

panel_param = pn.Row(param_column, tileview*zoneview*paraview, linecol)
text = '#Bikesharing Clustering Analysis'
title_row = pn.Row(text, indicator)
title_row[0].width=400
panel_column = pn.Column(title_row, panel_param)
panel_column.servable() # Run with: panel serve interactive_plot.py --autoreload

bokeh_server = panel_column.show(port=12345)

#%%
bokeh_server.stop() # Run with: panel serve interactive_plot.py --autoreload
