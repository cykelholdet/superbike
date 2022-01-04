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
from matplotlib import cm
from scipy.spatial import Voronoi
import shapely.ops
from shapely.geometry import Point, LineString

import holoviews as hv
import hvplot.pandas
hv.extension('bokeh', logo=False)
import panel as pn
import param
import geoviews as gv
from bokeh.models import HoverTool

import bikeshare as bs
import interactive_plot_utils as ipu

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

import pickle

cmap = cm.get_cmap('Blues')

# Load bikeshare data

YEAR = 2019
MONTH = 9
CITY = 'chic'

#station_df = ipu.make_station_df(data, holidays=False)
#station_df, land_use = ipu.make_station_df(data, holidays=False, return_land_use=True)
#station_df.dropna(inplace=True)
    

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
    cc_plot.opts(title=f"{title_pre} of cluster {cluster_j} ({ipu.cluster_color_dict[cluster_j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage')
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


class BikeDash(param.Parameterized):
    """
    Class containing everything for the bikeshare dashboard.
    
    Variables for the dashboard are introduced as param objects with their
    possible values. In addition, the plotting functions are defined.
    """
    city = param.Selector(default=CITY, objects=['nyc', 'chic', 'washDC', 'london', 'helsinki', 'madrid', 'edinburgh'])
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
    show_service_area = param.Selector(objects=['False', 'True'])
    service_radius = param.Integer(default=500, bounds=(0,1000))
    service_area_color = param.Selector(objects=['residential', 'commercial', 'recreational'])
    use_road = param.Selector(objects=['False', 'True'])
    random_state = param.Integer(default=42, bounds=(0, 10000))
    min_trips = param.Integer(default=100, bounds=(0, 800))
    
    # LR params
    
    use_points_or_percents = param.Selector(objects=['points', 'percents'])
    make_points_by = param.Selector(objects=['station location', 'station land use'])
    
    residential = param.Boolean(True)
    commercial = param.Boolean(True)
    manufacturing = param.Boolean(True)
    recreational = param.Boolean(True)
    mixed = param.Boolean(True)
    road= param.Boolean(True)
    transportation = param.Boolean(True)
    n_trips = param.Boolean(True)
    pop_density = param.Boolean(True)
    nearest_subway_dist = param.Boolean(True)
    const = param.Boolean(False)
    
    # LR_indicator = param.Boolean(True)
    # boolean_ = param.Boolean(True)
    
    
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)  # Runs init for the superclass param.Parameterized
        self.clusters = None
        self.labels = None
        self.labels_dict = None
        self.index = index
        self.data = bs.Data(self.city, YEAR, self.month)
        self.station_df, self.land_use = ipu.make_station_df(self.data, holidays=False, return_land_use=True)
        self.traffic_matrices = self.data.pickle_daily_traffic(holidays=False)
        
        self.plot_clusters_full()
        self.make_service_areas()
        self.make_logistic_regression()
        
    @param.depends('month', 'city', watch=True)
    def get_data(self):
        print(f'getting {self.month}')
        self.data = bs.Data(self.city, YEAR, self.month)
        self.station_df, self.land_use = ipu.make_station_df(self.data, holidays=False, return_land_use=True)
        self.traffic_matrices = self.data.pickle_daily_traffic(holidays=False)
        print(len(self.station_df))
        
        self.plot_clusters_full()
        self.make_service_areas()
        self.make_logistic_regression()
        # self.boolean_ = not self.boolean_
    
    
    @param.depends('day_type', 'min_trips', 'clustering', 'k', 'random_state', watch=False)
    def plot_clusters_full(self):
        self.station_df, self.clusters, self.labels = ipu.get_clusters(self.traffic_matrices, self.station_df, self.day_type, self.min_trips, self.clustering, self.k, self.random_state)
        
        # print(self.station_df.label.iloc[0])
        
        if self.day_type == 'business_days':
            mean = np.mean(self.traffic_matrices[0][self.station_df.index], axis=0)
        elif self.day_type == 'weekend':
            mean = np.mean(self.traffic_matrices[1][self.station_df.index], axis=0)
        
        mean = mean/np.max(mean)
        
        dist1_list = []
        for center in self.clusters.cluster_centers_:
            dist_from_mean = np.linalg.norm(center/np.max(center)-mean)
            dist1_list.append(dist_from_mean)
        
        avg_candidates = np.argsort(dist1_list)[:2]
        
        dist2_list=[]
        for candidate_label in avg_candidates:
            center = self.clusters.cluster_centers_[candidate_label]
            dist_from_zero = np.linalg.norm(center[:24]-center[24:])
            dist2_list.append(dist_from_zero)
        
        avg_label = avg_candidates[np.argmin(dist2_list)]
        
        new_labels = [avg_label]
        for i in range(1,self.k):
            if i == avg_label:
                new_labels.append(0)
            else:
                new_labels.append(i)
        
        self.labels_dict = dict(zip(range(len(new_labels)), new_labels))
        
        self.station_df = self.station_df.replace({'label' : self.labels_dict})
        self.labels = np.array(self.station_df['label'])
        
        # print(self.station_df.label.iloc[0])
        
        month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None'}
        
        if self.clustering == 'none':
            title = f'Number of trips per station in {month_dict[self.month]} {YEAR} in {bs.name_dict[self.city]}'
        else:
            title = f"{self.clustering} clustering in {month_dict[self.month]} {YEAR} in {bs.name_dict[self.city]}"

        plot = gv.Points(self.station_df, kdims=['long', 'lat'], vdims=['stat_id', 'color', 'n_trips', 'b_trips', 'w_trips', 'zone_type', 'name', ])
        plot.opts(gv.opts.Points(fill_color='color', size=10, line_color='black'))
        
        # self.LR_indicator = not self.LR_indicator
        
        return plot


    @param.depends('day_type', 'clustering', 'k', 'plot_all_clusters', 'min_trips', watch=False)
    def plot_centroid(self, index):
        if self.clustering == 'none':
            return "No clustering"
        elif self.clustering == 'h_clustering':
            if self.plot_all_clusters == 'True':
                traffic_matrix = ipu.mask_traffic_matrix(self.traffic_matrices, self.station_df, self.day_type, self.min_trips, holidays=False)
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
                traffic_matrix = ipu.mask_traffic_matrix(self.traffic_matrices, self.station_df, self.day_type, self.min_trips, holidays=False)
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
                    
                    # print(self.labels[0])
                    
                    ccs = self.clusters.cluster_centers_[self.labels_dict[j]]
                    cc_plot = plot_center(self.labels, j, ccs, title_pre="Centroid")
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to plot cluster centroid"
            else:
                i = index[0]
                if ~np.isnan(self.station_df['label'][i]):
                    j = int(self.station_df['label'][i])
                    ccs = self.clusters.cluster_centers_[self.labels_dict[j]]
                    cc_plot = plot_center(self.station_df['label'], self.labels_dict[j], ccs, title_pre="Centroid")
                    return pn.Column(cc_plot, f"Station index {i} is in cluster {j}")
                else:
                    return f"Station index {i} is not in a cluster due to min_trips."
    
    
    @param.depends('service_radius', 'use_road')
    def make_service_areas(self):
        
        if 'service_area' in self.station_df.columns:
            self.station_df.drop(columns='service_area', inplace=True)
            self.station_df.set_geometry('coords', inplace=True)
        
        points = self.station_df[['easting', 'northing']].to_numpy()
        
        points_gdf = gpd.GeoDataFrame(geometry = [Point(self.station_df.iloc[i]['easting'], 
                                                      self.station_df.iloc[i]['northing'])
                               for i in range(len(self.station_df))], crs='EPSG:3857')
        
        points_gdf['point'] = points_gdf['geometry']
        
        mean_point= np.mean(points, axis=0)
        edge_dist = 1000000
        edge_points = np.array([[mean_point[0]-edge_dist, mean_point[1]-edge_dist],
                                [mean_point[0]-edge_dist, mean_point[1]+edge_dist],
                                [mean_point[0]+edge_dist, mean_point[1]+edge_dist],
                                [mean_point[0]+edge_dist, mean_point[1]-edge_dist]])
        
        vor = Voronoi(np.concatenate([points, edge_points], axis=0))
        
        lines = [LineString(vor.vertices[line])
            for line in vor.ridge_vertices
            if -1 not in line]
        
        poly_gdf = gpd.GeoDataFrame()
        poly_gdf['vor_poly'] = [poly for poly in shapely.ops.polygonize(lines)]
        poly_gdf['geometry'] = poly_gdf['vor_poly']
        poly_gdf.set_crs(epsg=3857, inplace=True)
        
        poly_gdf = gpd.tools.sjoin(points_gdf, poly_gdf, op='within', how='left')
        poly_gdf.drop('index_right', axis=1, inplace=True)
        
        poly_gdf['service_area'] = [
            row['vor_poly'].intersection(row['point'].buffer(self.service_radius))
            for i, row in poly_gdf.iterrows()]
        
        poly_gdf['geometry'] = poly_gdf['service_area']
        poly_gdf.set_crs(epsg=3857, inplace=True)
        poly_gdf.to_crs(epsg=4326, inplace=True)
        poly_gdf['service_area'] = poly_gdf['geometry']
        
        self.station_df = gpd.tools.sjoin(self.station_df, poly_gdf, op='within', how='left')
        self.station_df.drop(columns=['index_right', 'vor_poly', 'point'], inplace=True)
        
        try:
            with open(f'./python_variables/union_{self.city}.pickle', 'rb') as file:
                union = pickle.load(file)
        except FileNotFoundError:
            print(f'No union for {self.city} found. Pickling union...')
            union = shapely.ops.unary_union(self.land_use.geometry)
            with open(f'./python_variables/union_{self.city}.pickle', 'wb') as file:
                pickle.dump(union, file)
            print('Pickling done')
            
        self.station_df['service_area'] = self.station_df['service_area'].apply(lambda area: area.intersection(union))
        
        service_area_trim = []
        for i, row in self.station_df.iterrows():
            if isinstance(row['service_area'], shapely.geometry.multipolygon.MultiPolygon):
                for poly in row['service_area']:
                    if poly.contains(row['coords']):
                        service_area_trim.append(poly)
            else:
                service_area_trim.append(row['service_area'])
        
        self.station_df['service_area'] = service_area_trim
        self.station_df.set_geometry('service_area', inplace=True)
        
        
        self.station_df['service_area'] = self.station_df['service_area'].to_crs(epsg=3857)
        self.land_use['geometry'] = self.land_use['geometry'].to_crs(epsg=3857)
        
        if 'road' in self.station_df['zone_type'].unique() and self.use_road == 'False':
            
            self.station_df['service_area_no_road'] = [
                stat['service_area'].difference(stat['neighborhood_road'])
                for i, stat in self.station_df.iterrows()]
            
            zone_types = [zone for zone in self.station_df['zone_type'].unique() 
                          if zone != 'road']
            
            for zone_type in zone_types:
            
                zone_percents = np.zeros(len(self.station_df))
                
                for i, stat in self.station_df.iterrows():
                    
                    if stat[f'neighborhood_{zone_type}']:
                    
                        zone_percents[i] = stat['service_area_no_road'].buffer(0).intersection(stat[f'neighborhood_{zone_type}']).area/stat['service_area_no_road'].area*100
                    
                self.station_df[f'percent_{zone_type}'] = zone_percents
            
        else:
                
            for zone_type in self.station_df['zone_type'].unique():
                
                zone_percents = np.zeros(len(self.station_df))
                
                for i, stat in self.station_df.iterrows():
                    
                    if stat[f'neighborhood_{zone_type}']:
                    
                        zone_percents[i] = stat['service_area'].buffer(0).intersection(stat[f'neighborhood_{zone_type}']).area/stat['service_area'].area*100
                    
                self.station_df[f'percent_{zone_type}'] = zone_percents
        
        self.station_df['service_area_size'] = self.station_df['service_area'].apply(lambda area: area.area/1000000)
        
        self.station_df['service_area'] = self.station_df['service_area'].to_crs(epsg=4326)
        self.land_use['geometry'] = self.land_use['geometry'].to_crs(epsg=4326)
        
        self.station_df['geometry'] = self.station_df['service_area']
        
        # self.LR_indicator = not self.LR_indicator
    
    @param.depends('day_type', 'min_trips', 'clustering', 'k', 'random_state', 
                   'service_radius', 'use_road', 'LR_indicator', 'use_points_or_percents',
                   'make_points_by', 'residential', 'commercial',
                   'manufacturing', 'recreational', 'mixed',
                   'road', 'transportation', 'n_trips',
                   'pop_density', 'nearest_subway_dist', watch=False)
    def make_logistic_regression(self):
        
        df = self.station_df.copy()
        df = df[~df['label'].isna()]
        
        X=pd.DataFrame()
        
        p_columns = [column for column in df.columns 
                      if 'percent_' in column]
        
        if self.use_points_or_percents == 'points':
            
            if self.make_points_by == 'station location':
                X = pd.get_dummies(df['zone_type'])
                
            elif self.make_points_by == 'station land use':
                
                p_df = df[p_columns]
                
                # get the zonetype with the largest percentage
                zone_types = [p_df.iloc[i].index[p_df.iloc[i].argmax()][8:] 
                              for i in range(len(p_df))]
        
                df['zone_type_by_percent'] = zone_types
            
                X = pd.get_dummies(df['zone_type_by_percent'])
        
        elif self.use_points_or_percents == 'percents':
            X = df[p_columns]
        
        zones_params = [('residential', self.residential),
                        ('commercial', self.commercial),
                        ('manufacturing', self.manufacturing),
                        ('recreational', self.recreational),
                        ('mixed', self.mixed),
                        ('road', self.road),
                        ('transportation', self.transportation)]
        
        drop_columns = []
        
        for zone_type in zones_params:
            if not zone_type[1]:
                
                if zone_type[0] in X.columns:
                    drop_columns.append(zone_type[0])
                    
                elif 'percent_'+zone_type[0] in X.columns:
                    drop_columns.append('percent_'+zone_type[0])
        
        
        other_params = [('n_trips', self.n_trips),
                        ('pop_density', self.pop_density),
                        ('nearest_subway_dist', self.nearest_subway_dist)]
        
        add_columns = [param[0] for param in other_params if param[1]]
        
        X.drop(columns=drop_columns, inplace=True)
       
        for column in add_columns:
            try:
                X[column] = df[column]
            except KeyError:
                pass
        
        if self.const:
            X = sm.add_constant(X)
        
        y = df[~df['label'].isna()]['label']
        
        param_names = {'percent_manufacturing' : '% manufacturing',
                       'percent_commercial' : '% commercial',
                       'percent_residential' : '% residential',
                       'percent_recreational' : '% recreational',
                       'percent_mixed' : '% mixed',
                       'pop_density' : 'pop density',
                       'nearest_subway_dist' : 'nearest subway dist'}
        
        
        
        LR_model = MNLogit(y, X.rename(param_names))
        
        LR_results = LR_model.fit_regularized(maxiter=10000)
        
        return LR_results, X, y
    
    
bike_params = BikeDash(None)

@pn.depends(clustering=bike_params.param.clustering)
def plot_tiles(clustering):
    
    tiles = hv.element.tiles.StamenTerrainRetina()
    tiles = gv.tile_sources.StamenTerrainRetina()
    tiles.opts(height=800, width=800, xlim=(extremes[1], extremes[0]), ylim=(extremes[3], extremes[2]), active_tools=['wheel_zoom'],  apply_ranges=False)
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

pointview = hv.DynamicMap(bike_params.plot_clusters_full)

pointview.opts(tools=['tap', hover])
pointview.opts(nonselection_alpha=0.3)

selection_stream = hv.streams.Selection1D(source=pointview)

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


@pn.depends(show_service_area=bike_params.param.show_service_area,
            service_radius=bike_params.param.service_radius,
            service_area_color=bike_params.param.service_area_color,
            city=bike_params.param.city,
            use_road=bike_params.param.use_road)
def service_area_plot(show_service_area, service_radius, service_area_color, city, use_road):
    bike_params.make_service_areas()
    if show_service_area == 'True':
        zone_percents_columns = [column for column in bike_params.station_df.columns
                                 if 'percent_' in column]
        
        # bike_params.station_df['service_color'] = ['#808080' for i in range(len(bike_params.station_df))]
        
        vdims = zone_percents_columns
        vdims.append('service_area_size')
        
        print(vdims)
        if service_area_color == 'residential':
            return gv.Polygons(bike_params.station_df, vdims=vdims).opts(color='percent_residential')
        elif service_area_color == 'commercial':
            return gv.Polygons(bike_params.station_df, vdims=vdims).opts(color='percent_commercial')
        elif service_area_color == 'recreational':
            return gv.Polygons(bike_params.station_df, vdims=vdims).opts(color='percent_recreational')
        elif service_area_color == 'pop density':
            if 'pop_density' in bike_params.station_df.columns:
                return gv.Polygons(bike_params.station_df, vdims=vdims).opts(color='pop_density')
            else:
                return gv.Polygons([])
        
    else:
        return gv.Polygons([])


@pn.depends(service_radius=bike_params.param.service_radius,
            use_road=bike_params.param.use_road,
            clustering=bike_params.param.clustering,
            k=bike_params.param.k,
            min_trips=bike_params.param.min_trips,
            day_type=bike_params.param.day_type,
            city=bike_params.param.city,
            month=bike_params.param.month,
            use_points_or_percents=bike_params.param.use_points_or_percents,
            make_points_by=bike_params.param.make_points_by,
            residential=bike_params.param.residential,
            commercial=bike_params.param.commercial,
            manufacturing=bike_params.param.manufacturing,
            recreational=bike_params.param.recreational,
            mixed=bike_params.param.mixed,
            road=bike_params.param.road,
            transportation=bike_params.param.transportation,
            n_trips=bike_params.param.n_trips,
            pop_density=bike_params.param.pop_density,
            nearest_subway_dist=bike_params.param.nearest_subway_dist,
            const=bike_params.param.const
            )
def print_logistic_regression(service_radius, use_road, clustering, k, 
                              min_trips, day_type, city, month, residential,
                              commercial, manufacturing, recreational,
                              mixed, road, transportation, n_trips, 
                              pop_density, nearest_subway_dist, const, 
                              use_points_or_percents, make_points_by):
    res, X, y = bike_params.make_logistic_regression()
    return res.summary().as_html()
    

@pn.depends(city=bike_params.param.city, watch=True)
def update_extent(city):
    extremes = [bike_params.station_df['easting'].max(), bike_params.station_df['easting'].min(), 
                bike_params.station_df['northing'].max(), bike_params.station_df['northing'].min()]
    tileview.opts(xlim=(extremes[1], extremes[0]), ylim=(extremes[3], extremes[2]))
    panel_column[1][1].object.data[()].Points.I.data = bike_params.station_df
    print(f"update city = {city}")


def hook(plot, element):
    print('plot.state:   ', plot.state)
    print('plot.handles: ', sorted(plot.handles.keys()))
    plot.handles['xaxis'].axis_label_text_color = 'blue'
    plot.handles['yaxis'].axis_label_text_color = 'red'
    plot.handles['xaxis'].axis_label = extremes[1] - extremes[0]
    plot.handles['yaxis'].axis_label = extremes[3] - extremes[2]
    #plot.handles['x_range'] = [extremes[1], extremes[0]]
    #plot.handles['y_range'] = [extremes[3], extremes[2]]


# =============================================================================
# Misc
# =============================================================================

extremes = [bike_params.station_df['easting'].max(), bike_params.station_df['easting'].min(), 
            bike_params.station_df['northing'].max(), bike_params.station_df['northing'].min()]

month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None'}

activity_dict = {'departures': 'start', 'arrivals': 'end', 'd': 'start', 'a': 'end', 'start': 'start', 'end': 'end'}
day_type_dict = {'weekend': 'w', 'business_days': 'b'}

gif_pane = pn.pane.GIF('Loading_Key.gif')


# =============================================================================
# Parameters
# =============================================================================

param_split = 17 # add to this number if additional parameters are added in the first column

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
    'show_service_area': pn.widgets.RadioButtonGroup,
    'service_radius': pn.widgets.IntInput,
    'use_road': pn.widgets.RadioButtonGroup,
    'use_points_or_percents': pn.widgets.RadioButtonGroup,
    'make_points_by': pn.widgets.RadioButtonGroup
    },
    name="Bikeshare Parameters",
    )

params.parameters = params.parameters[:param_split]

params.layout.insert(15, 'Use roads:')
params.layout.insert(12, 'Show service area:')
params.layout.insert(11, 'Show land use:')
params.layout.insert(10, 'Plot all clusters:')
params.layout.insert(5, 'Clustering method:')

LR_params = pn.Param(bike_params.param, widgets={
    'use_points_or_percents': pn.widgets.RadioButtonGroup,
    'make_points_by': pn.widgets.RadioButtonGroup
    },
    name="Logistic Regression Parameters",
    )

LR_params.parameters = LR_params.parameters[param_split:]

LR_params.layout.insert(2, 'Make points by:')
LR_params.layout.insert(1, 'Use station points or percents:')


# =============================================================================
# Views
# =============================================================================



zoneview = hv.DynamicMap(land_use_plot)
zoneview.opts(alpha=0.5, apply_ranges=False)

service_area_view = hv.DynamicMap(service_area_plot)
service_area_view.opts(alpha=0.5, apply_ranges=False)

#tileview.opts(framewise=True, apply_ranges=False)

tooltips_zone = [
    ('Zone Type', '@zone_type'),
]

tooltips_service_area = [
    ('area (km^2)', '@service_area_size'),
    ('% residential', '@percent_residential'),
    ('% commercial', '@percent_commercial'),
    ('% manufacturing', '@percent_manufacturing'),
    ('% recreational', '@percent_recreational'),
    ('% mixed', '@percent_mixed'),
    ('% road', '@percent_road')]

hover_zone = HoverTool(tooltips=tooltips_zone)
hover_service_area = HoverTool(tooltips=tooltips_service_area)
zoneview.opts(tools=[hover_zone])
service_area_view.opts(tools=[hover_service_area])
# selection_stream.param.values()['index']

views = tileview*zoneview*service_area_view*pointview


# =============================================================================
# Dashboard layout
# =============================================================================

param_column = pn.Column(params.layout, minpercent)
param_column[1].width=300

LR_row = pn.Row(LR_params.layout, print_logistic_regression)

mid_column = pn.Column(views, LR_row)

linecol = pn.Column(plot_daily_traffic, plot_centroid_callback)

panel_param = pn.Row(param_column, mid_column, linecol)
text = '#Bikesharing Clustering Analysis'
title_row = pn.Row(text, indicator)
title_row[0].width=400
panel_column = pn.Column(title_row, panel_param)
panel_column.servable() # Run with: panel serve interactive_plot.py --autoreload

# bokeh_server = panel_column.show(port=12345)

#%%
# bokeh_server.stop() # Run with: panel serve interactive_plot.py --autoreload
