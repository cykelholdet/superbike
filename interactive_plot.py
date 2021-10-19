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

import bikeshare as bs
import time

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

import skimage.color as skcolor
import matplotlib.colors as mpl_colors
from matplotlib import cm

from holoviews.element.tiles import OSM

from shapely.geometry import Point
from shapely.ops import nearest_points
from geopy.distance import great_circle


cmap = cm.get_cmap('Blues')

# Load bikeshare data

year = 2019
month = 9
data = bs.Data('nyc', year, month)
df = data.df

locations = pd.DataFrame(data.stat.locations).T.rename(columns={0: 'long', 1: 'lat'}, index=data.stat.inverse)

locations['easting'], locations['northing'] = hv.util.transform.lon_lat_to_easting_northing(locations['long'], locations['lat'])

df['easting'], df['northing'] = hv.util.transform.lon_lat_to_easting_northing(df['start_stat_long'], df['start_stat_lat'])



station_df = locations.copy()
station_df['name'] = data.stat.names.values()
station_df['n_arrivals'] = data.df['start_stat_id'].value_counts()
station_df['n_departures'] = data.df['end_stat_id'].value_counts()
station_df['n_arrivals'].fillna(0, inplace=True)
station_df['n_departures'].fillna(0, inplace=True)

station_df['n_trips'] = data.df['start_stat_id'].value_counts().add(data.df['end_stat_id'].value_counts(), fill_value=0)

station_df['coords'] = list(zip(station_df['long'], station_df['lat']))
station_df['coords'] = station_df['coords'].apply(Point)

# Load other data

zoning_df = gpd.read_file(f'./data/nyc_zoning_data.json')
zoning_df = zoning_df[['ZONEDIST', 'geometry']]

station_df = gpd.GeoDataFrame(station_df, geometry = 'coords', crs = zoning_df.crs)
station_df = gpd.tools.sjoin(station_df, zoning_df, op='within', how='left')
station_df.drop('index_right', axis = 1, inplace=True)

CTracts_df = gpd.read_file('./data/nyc_CT_data.json')
CTracts_df = CTracts_df[['BoroCT2020', 'geometry', 'Shape__Area']]
CTracts_df.rename({'Shape__Area':'CT_area'}, axis=1, inplace=True)

station_df = gpd.tools.sjoin(station_df, CTracts_df, op='within', how='left')
station_df['BoroCT2020'] = station_df['BoroCT2020'].apply(int)
station_df.drop('index_right', axis = 1, inplace=True)

census_df = pd.read_excel('./data/nyc_census_data.xlsx', sheet_name=1)
census_df = census_df[['Unnamed: 4', '2020 Data']]
census_df.rename({'Unnamed: 4':'BoroCT2020','2020 Data':'pop'}, axis=1, inplace=True)

# census_df['2020 Data'] = census_df['Index'].apply(lambda i: census_df['2020'])

station_df = pd.merge(station_df, census_df, on = 'BoroCT2020')
station_df['pop_density'] = station_df.apply(lambda stat: stat['pop']/stat['CT_area'], axis=1)

subways_df = gpd.read_file('./data/nyc_subways_data.geojson')

station_df['nearest_subway'] = station_df.apply(lambda stat: nearest_points(stat['coords'], subways_df.geometry.unary_union)[1], axis=1)
station_df['nearest_subway_dist'] = station_df.apply(lambda stat: great_circle(stat['coords'].coords[0][::-1], stat['nearest_subway'].coords[0][::-1]), axis=1)



#%%

extremes = [station_df['easting'].max(), station_df['easting'].min(), station_df['northing'].max(), station_df['northing'].min()]


month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
name_dict = {'chic': 'Chicago',
             'london': 'London',
             'madrid': 'Madrid',
             'mexico': 'Mexico City',
             'nyc': 'New York City',
             'sfran': 'San Francisco',
             'taipei': 'Taipei',
             'washDC': 'Washington DC',
             'oslo': 'Oslo',
             'bergen': 'Bergen',
             'trondheim': 'Trondheim',
             'edinburgh': 'Edinburgh'}

activity_dict = {'departures': 'start', 'arrivals': 'end', 'd': 'start', 'a': 'end', 'start': 'start', 'end': 'end'}
day_type_dict = {'weekend': 'w', 'business_days': 'b'}

color_dict = {0 : 'blue', 1 : 'red', 2 : 'yellow', 3 : 'green', #tab:
              4 : 'purple', 5 : 'brown', 6: 'pink',
              7 : 'gray', 8 : 'olive', 9 : 'cyan'}

mpl_color_dict = {i: mpl_colors.to_rgb(color_dict[i]) for i in range(10)}
lab_color_dict = {i: skcolor.rgb2lab(mpl_color_dict[i]) for i in range(10)}
lab_color_list = [lab_color_dict[i] for i in range(10)]


class BikeParameters2(param.Parameterized):
    trip_type = param.Selector(objects=['departures', 'arrivals', 'all'])
    day_type = param.Selector(objects=['business_days', 'weekend', 'day'])
    clustering = param.Selector(objects=['k_means', 'k_medoids', 'h_clustering', 'gaussian_mixture', 'none'], doc="Which clustering to perform")
    k = param.Integer(default=3, bounds=(1, 10))
    cnorm = param.Selector(objects=['linear', 'log'])
    min_trips = param.Integer(default=0, bounds=(0, 600))
    day = param.Integer(default=1, bounds=(1, data.num_days))
    dist_func = param.Selector(objects=['norm'])
    plot_all_clusters = param.Selector(objects=['False', 'True'])
    random_state = param.Integer(default=42, bounds=(0, 2000))
    
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.clusters = None
        self.labels = None
        self.index = index
    
    @param.depends('day_type', 'min_trips', 'clustering', 'k', 'random_state', watch=False)
    def plot_clusters_full(self):
        if self.day_type == 'business_days':
            traffic_matrix = data.pickle_daily_traffic()[0]
        elif self.day_type == "weekend":
            traffic_matrix = data.pickle_daily_traffic()[1]
    
        if self.clustering == 'k_means':
            self.clusters = KMeans(self.k, random_state=self.random_state).fit(traffic_matrix)
            self.labels = self.clusters.predict(traffic_matrix)
            station_df['color'] = [color_dict[label] for label in self.labels]
    
        elif self.clustering == 'k_medoids':
            self.clusters = KMedoids(self.k, random_state=self.random_state).fit(traffic_matrix)
            self.labels = self.clusters.predict(traffic_matrix)
            station_df['color'] = [color_dict[label] for label in self.labels]
            
        elif self.clustering == 'h_clustering':
            self.clusters = None
            self.labels = AgglomerativeClustering(self.k).fit_predict(traffic_matrix)
            station_df['color'] = [color_dict[label] for label in self.labels]
        
        elif self.clustering == 'gaussian_mixture':
            self.clusters = GaussianMixture(self.k, n_init=10, random_state=self.random_state).fit(traffic_matrix)
            self.labels = self.clusters.predict_proba(traffic_matrix)
            lab_mat = np.array(lab_color_list[:self.k]).T
            lab_cols = [np.sum(self.labels[i] * lab_mat, axis=1) for i in range(len(traffic_matrix))]
            labels_rgb = skcolor.lab2rgb(lab_cols)
            station_df['color'] = ['#%02x%02x%02x' % tuple(label.astype(int)) for label in labels_rgb*255]
        
        elif self.clustering == 'none':
            self.clusters = None
            self.labels = None
            station_df['color'] = station_df['n_trips'].tolist()
            
        else:
            self.clusters = None
            self.labels = None
        if self.clustering == 'none':
            title = f'Number of trips per station in {month_dict[data.month]} {data.year} in {name_dict[data.city]}'
        else:
            title = f"{self.clustering} clustering in {month_dict[data.month]} {data.year} in {name_dict[data.city]}"
        plot = station_df.hvplot(kind='points', x='easting', y='northing', c='color', s=100, hover_cols=['name', 'n_trips', 'n_departures', 'n_arrivals'], title=title,  line_color='black', colorbar=False)
        plot.opts(apply_ranges=False)
        return plot
    

    @param.depends('day_type', 'clustering', 'k', 'plot_all_clusters', watch=False)
    def plot_centroid(self, index):
        if self.clustering == 'none':
            return "No clustering"
        elif self.clustering == 'h_clustering':
            if self.plot_all_clusters == 'True':
                if self.day_type == 'business_days':
                    traffic_matrix = data.pickle_daily_traffic()[0]
                elif self.day_type == "weekend":
                    traffic_matrix = data.pickle_daily_traffic()[1]
                cc_plot_list = list()
                for j in range(self.k):
                    mean_vector = np.mean(traffic_matrix[np.where(self.labels == j)], axis=0)
                    cc_df = pd.DataFrame([mean_vector[:24], mean_vector[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
                    cc_plot = cc_df['departures'].hvplot() * cc_df['arrivals'].hvplot()
                    n = np.sum(self.labels == j)
                    cc_plot.opts(title=f"Mean of cluster {j} ({color_dict[j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage')
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to get cluster info"
            else:
                i = index[0]
                if self.day_type == 'business_days':
                    traffic_matrix = data.pickle_daily_traffic()[0]
                elif self.day_type == "weekend":
                    traffic_matrix = data.pickle_daily_traffic()[1]
                j = self.labels[i]
                mean_vector = np.mean(traffic_matrix[np.where(self.labels == j)], axis=0)
                cc_df = pd.DataFrame([mean_vector[:24], mean_vector[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
                cc_plot = cc_df['departures'].hvplot() * cc_df['arrivals'].hvplot()
                n = np.sum(self.labels == j)
                cc_plot.opts(title=f"Mean of cluster {j} ({color_dict[j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage')
                return pn.Column(cc_plot, f"Station index {i} is in cluster {j}")
        elif self.clustering == 'gaussian_mixture':
            if self.plot_all_clusters == 'True':
                cc_plot_list = list()
                for j in range(self.k):
                    ccs = self.clusters.means_[j]
                    cc_df = pd.DataFrame([ccs[:24], ccs[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
                    cc_plot = cc_df['departures'].hvplot() * cc_df['arrivals'].hvplot()
                    n = np.sum(self.labels.argmax(axis=1) == j)
                    cc_plot.opts(title=f"Mean of cluster {j} ({color_dict[j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage')
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to plot cluster centroid"
            else:
                i = index[0]
                j = self.labels[i].argmax()
                textlist = [f"{j}: {self.labels[i][j]:.2f}\n\n" for j in range(self.k)]
                ccs = self.clusters.means_[j]
                cc_df = pd.DataFrame([ccs[:24], ccs[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
                cc_plot = cc_df['departures'].hvplot() * cc_df['arrivals'].hvplot()
                n = np.sum(self.labels.argmax(axis=1) == j)
                cc_plot.opts(title=f"Mean of cluster {j} ({color_dict[j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage')
                return pn.Column(cc_plot, f"Station index {i} belongs to cluster \n\n {''.join(textlist)}")
        else:
            if self.plot_all_clusters == 'True':
                cc_plot_list = list()
                for j in range(self.k):
                    ccs = self.clusters.cluster_centers_[j]
                    cc_df = pd.DataFrame([ccs[:24], ccs[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
                    cc_plot = cc_df['departures'].hvplot() * cc_df['arrivals'].hvplot()
                    n = np.sum(self.labels == j)
                    cc_plot.opts(title=f"Mean of cluster {j} ({color_dict[j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage')
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to plot cluster centroid"
            else:
                i = index[0]
                j = self.labels[i]
                ccs = self.clusters.cluster_centers_[j]
                cc_df = pd.DataFrame([ccs[:24], ccs[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
                cc_plot = cc_df['departures'].hvplot() * cc_df['arrivals'].hvplot()
                n = np.sum(self.labels == j)
                cc_plot.opts(title=f"Centroid of cluster {j} ({color_dict[j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage')
                return pn.Column(cc_plot, f"Station index {i} is in cluster {j}")

    
bike_params = BikeParameters2(None)

params = pn.Param(bike_params.param, widgets={
    'clustering': pn.widgets.RadioBoxGroup,
    'trip_type': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success'},
    'day_type': pn.widgets.RadioButtonGroup,
    'day': pn.widgets.IntSlider,
    'random_state': pn.widgets.IntInput,
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
    #tiles = gts.StamenTerrainRetina
    tiles.opts(height=800, width=800, xlim=(extremes[1], extremes[0]), ylim=(extremes[3], extremes[2]), active_tools=['wheel_zoom'])
    return tiles


tileview = hv.DynamicMap(plot_tiles)



tooltips = [
    ('Name', '@name'),
    ('Cluster', '@color'),
    ('n_departures', '@n_departures'),
    ('n_arrivals', '@n_arrivals'),
    ('n_trips', '@n_trips total'),
]
hover = HoverTool(tooltips=tooltips)

paraview.opts(tools=['tap', hover])
paraview.opts(apply_ranges=False, nonselection_alpha=0.4)

selection_stream = hv.streams.Selection1D(source=paraview)


@pn.depends(index=selection_stream.param.index,
            day_type=bike_params.param.day_type)
def plot_daily_traffic(index, day_type):
    if not index:
        return "Select a station to see station traffic"
    else:
        i = index[0]
    plot = line_callback_both(i, day_type)
    n_trips = station_df['n_trips'].iloc[i]
    plot.opts(title=f'Average hourly traffic for {data.stat.names[i]} (n_trips={n_trips:n})', ylabel='percentage')
    return plot


@pn.depends(index=selection_stream.param.index,
            plot_all_clusters=bike_params.param.plot_all_clusters,
            clustering=bike_params.param.clustering,
            k=bike_params.param.k,)
def plotterino(index, plot_all_clusters, clustering, k):
    return bike_params.plot_centroid(index)
    

linecol = pn.Column(plot_daily_traffic, plotterino)

param_column = pn.Column(params.widgets)

panel_param = pn.Row(params, tileview*paraview, linecol)
text = '#Bikesharing'
panel_column = pn.Column(text, panel_param) 
# bokeh_server = panel_param.show(port=22345)
bokeh_server = panel_column.servable()

#%%
# stop the bokeh server (when needed)
# bokeh_server.stop()

