#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:36:11 2021

@author: dbvd
"""

# TODO: Remove panel buttons which do nothing
#       Change pop_density to a log scale when plotting census tracts
#       Change defaults in LR
import numpy as np
import pandas as pd
from matplotlib import cm

import holoviews as hv
import hvplot.pandas

import panel as pn
import param
import geoviews as gv
from bokeh.models import HoverTool
from bokeh.themes.theme import Theme

import bikeshare as bs
import interactive_plot_utils as ipu

hv.extension('bokeh', logo=False)

cmap = cm.get_cmap('Blues')

# Load bikeshare data

YEAR = 2019
MONTH = None
CITY = 'boston'

#station_df = ipu.make_station_df(data, holidays=False)
#station_df, land_use = ipu.make_station_df(data, holidays=False, return_land_use=True)
#station_df.dropna(inplace=True)
    

def plot_center(traffic_matrix, labels, cluster_j, c_center, plot_traf_or_diff,
                title_pre="Mean", plot_std=True):
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
    
    labels = labels[~labels.isna()].to_numpy()
    n = np.sum(labels == cluster_j)
    
    if plot_traf_or_diff == 'Traffic':
        
        c_center = np.mean(traffic_matrix[np.where(labels==cluster_j)], axis=0)
        
        dep_mean, arr_mean = c_center[:24], c_center[24:]
        
        dep_stds = np.std(traffic_matrix[:,:24][np.where(labels==cluster_j)], axis=0)
        arr_stds = np.std(traffic_matrix[:,24:][np.where(labels==cluster_j)], axis=0)
        
        cc_df = pd.DataFrame([dep_mean, arr_mean]).T.rename(
            columns={0:'departures', 1:'arrivals'})
        
        varea = pd.DataFrame()
        varea['dep_low'] = cc_df['departures'] - dep_stds
        varea['dep_high'] = cc_df['departures'] + dep_stds
        varea['arr_low'] = cc_df['arrivals'] - arr_stds
        varea['arr_high'] = cc_df['arrivals'] + arr_stds
        varea['hour'] = np.arange(0,24)
        
        cc_plot = varea.hvplot.area(x='hour', y='dep_low', y2='dep_high', alpha=0.5, line_width=0) * varea.hvplot.area(x='hour', y='arr_low', y2='arr_high', alpha=0.5, line_width=0) * cc_df['departures'].hvplot(xticks=list(zip(range(24),range(24)))) * cc_df['arrivals'].hvplot(xticks=list(zip(range(24),range(24))))
        cc_plot.opts(title=f"{title_pre} of cluster {cluster_j} ({ipu.cluster_color_dict[cluster_j]}) (n={n})", legend_position='top_right', xlabel='hour', ylabel='percentage of all traffic')
    
    else:
        
        traffic_difference = traffic_matrix[:,:24]-traffic_matrix[:,24:]
        
        cc_df = pd.Series(c_center, name='Relative difference')
        
        stds = np.std(traffic_difference[np.where(labels==cluster_j)], axis=0)
        varea = pd.DataFrame()
        varea['std_low'] = cc_df - stds
        varea['std_high'] = cc_df + stds
        varea['hour'] = np.arange(0,24)
        
        cc_plot = varea.hvplot.area(x='hour', y='std_low', y2='std_high', alpha=0.5, line_width=0) * cc_df.hvplot(legend=False, xticks=list(zip(range(24),range(24))))
        cc_plot.opts(title=f"{title_pre} of cluster {cluster_j} ({ipu.cluster_color_dict[cluster_j]}) (n={n})", xlabel='hour', ylabel='relative difference')
    
    return cc_plot


def plot_dta_with_std(data, index, day_type, plot_traf_or_diff, user_type='all'):
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
    
    if plot_traf_or_diff == 'Traffic':
        
        print(f"{index=}. Plotting traffic...", end='')
        a = data.daily_traffic_average(index, period=day_type_dict[day_type], 
                                       return_std=True, user_type=user_type)
        means = pd.DataFrame(a[:2]).T.rename(columns={0:'departures', 1:'arrivals'})
        stds = pd.DataFrame(a[2:]).T.rename(columns={0:'departures', 1:'arrivals'})
        varea = pd.DataFrame()
        varea['dep_low'] = means['departures'] - stds['departures']
        varea['dep_high'] = means['departures'] + stds['departures']
        varea['arr_low'] = means['arrivals'] - stds['arrivals']
        varea['arr_high'] = means['arrivals'] + stds['arrivals']
        varea['hour'] = np.arange(0,24)
        print('Done')
        return varea.hvplot.area(x='hour', y='dep_low', y2='dep_high', alpha=0.5, line_width=0) * varea.hvplot.area(x='hour', y='arr_low', y2='arr_high', alpha=0.5, line_width=0) * means['departures'].hvplot() * means['arrivals'].hvplot(xticks=list(zip(range(24),range(24))))

    else:    
        print(f"{index=}. Plotting difference...", end='')
        a = data.daily_traffic_average(index, period=day_type_dict[day_type], 
                                       return_std=False, user_type=user_type)
        diff = pd.Series(a[0] - a[1], name='Relative difference')
        print('Done')
        return diff.hvplot(xticks=list(zip(range(24),range(24))))
        

class BikeDash(param.Parameterized):
    """
    Class containing everything for the bikeshare dashboard.
    
    Variables for the dashboard are introduced as param objects with their
    possible values. In addition, the plotting functions are defined.
    """
    city = param.Selector(default=CITY, objects=['nyc', 'chicago', 'washdc', 'minneapolis', 'boston', 'london', 'helsinki', 'madrid', 'edinburgh', 'oslo'])
    #if MONTH == None:
    month = param.Selector(default=MONTH, objects=[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    #else:
    #    month = param.Integer(default=MONTH, bounds=(1, 12))
    trip_type = param.Selector(objects=['departures', 'arrivals', 'all'])
    day_type = param.Selector(objects=['business_days', 'weekend',])
    clustering = param.Selector(objects=['k_means', 'k_medoids', 'h_clustering', 'gaussian_mixture', 'none', 'zoning'], doc="Which clustering to perform")
    k = param.Integer(default=3, bounds=(1, 10))
    cnorm = param.Selector(objects=['linear', 'log'])
    day = param.Integer(default=1, bounds=(1, 31))
    dist_func = param.Selector(objects=['norm'])
    plot_traf_or_diff = param.Selector(objects=['Traffic', 'Difference'])
    plot_all_clusters = param.Selector(objects=['False', 'True'])
    keep_furthest_or_closest = param.Selector(objects=['Furthest', 'Closest'])
    dist_to_center = param.Integer(default=100, bounds=(0,100))
    show_land_use = param.Selector(objects=['False', 'True'])
    show_census = param.Selector(objects=['False', 'True'])
    show_service_area = param.Selector(objects=['False', 'True'])
    service_radius = param.Integer(default=500, bounds=(0,1000))
    service_area_color = param.Selector(objects=['residential', 'commercial', 'recreational', 'pop_density'])
    use_road = param.Selector(objects=['False', 'True'])
    random_state = param.Integer(default=42, bounds=(0, 10000))
    min_trips = param.Integer(default=100, bounds=(0, 800))
    user_type = param.Selector(objects=['all', 'Subscriber', 'Customer'])
    
    # LR params
    
    use_points_or_percents = param.Selector(objects=['points', 'percents'])
    make_points_by = param.Selector(objects=['station location', 'station land use'])
    
    residential = param.Boolean(True)
    commercial = param.Boolean(True)
    industrial = param.Boolean(True)
    recreational = param.Boolean(True)
    mixed = param.Boolean(False)
    road = param.Boolean(False)
    transportation = param.Boolean(False)
    unknown = param.Boolean(False)
    n_trips = param.Boolean(False)
    pop_density = param.Boolean(True)
    nearest_subway_dist = param.Boolean(True)
    const = param.Boolean(True)
    
    LR_indicator = param.Boolean(True)
    
    # boolean_ = param.Boolean(True)
    
    
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)  # Runs init for the superclass param.Parameterized
        self.clusters = None
        self.labels = None
        self.labels_dict = None
        self.index = index
        self.data = bs.Data(self.city, YEAR, self.month)
        self.station_df, self.land_use, self.census_df = ipu.make_station_df(self.data, holidays=False, return_land_use=True, return_census=True)
        self.traffic_matrices = self.data.pickle_daily_traffic(holidays=False)
        
        self.service_area_modified = False
        
        self.plot_clusters_full()
        self.make_service_areas()
        print("Make logi")
        self.make_logistic_regression()
        
    @param.depends('month', 'city', watch=True)
    def get_data(self):
        print(f'getting {self.month}')
        self.data = bs.Data(self.city, YEAR, self.month)
        self.station_df, self.land_use, self.census_df = ipu.make_station_df(self.data, holidays=False, return_land_use=True, return_census=True)
        self.traffic_matrices = self.data.pickle_daily_traffic(holidays=False)
        # print(len(self.station_df))
        
        self.plot_clusters_full()
        self.make_service_areas()
        self.make_logistic_regression()
        # self.boolean_ = not self.boolean_
    
    
    @param.depends('day_type', 'min_trips', 'clustering', 'k', 'random_state', 
                   'city', 'month', 'user_type', 'dist_to_center', 
                   'keep_furthest_or_closest', watch=False)
    def plot_clusters_full(self):
        print("Plotting Clusters")
        self.station_df, self.clusters, self.labels = ipu.get_clusters(self.traffic_matrices, self.station_df, self.day_type, self.min_trips, self.clustering, self.k, self.random_state)
        # print(self.station_df.label.iloc[0])
        
        
        # print(self.station_df.label.iloc[0])
        
        month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
              7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None'}
        
        if self.clustering == 'none':
            title = f'Number of trips per station in {month_dict[self.month]} {YEAR} in {bs.name_dict[self.city]}'
        else:
            title = f"{self.clustering} clustering in {month_dict[self.month]} {YEAR} in {bs.name_dict[self.city]}"
        
        stat_df = self.station_df.copy()
        
        # Rank the stations based on distance to its cluster center
        dist_to_center_argsort = np.argsort(stat_df['dist_to_center'].to_numpy())
        placement_dict = dict(zip(dist_to_center_argsort, range(len(stat_df))))
        stat_df['placement'] = stat_df.index.to_numpy()
        stat_df['placement'] = stat_df['placement'].replace(placement_dict)
        
        # Make stations invisible based on their ranking
        if self.keep_furthest_or_closest == 'Furthest':
            if self.dist_to_center != 100:
                print(f'Plotting the {self.dist_to_center}% farthest stations...', end='')
            cut_off_placement = int(len(stat_df)-np.round(len(stat_df)*self.dist_to_center/100))
            point_sizes = np.zeros(len(stat_df))
            point_sizes[stat_df['placement']>=cut_off_placement] = 10
            stat_df['point_size'] = point_sizes
            if self.dist_to_center != 100:
                print('Done')
            
            # stat_df = stat_df[stat_df['placement'] >= cut_off_placement]
        else:
            if self.dist_to_center != 100:
                print(f'Plotting the {self.dist_to_center}% closest stations...', end='')
            cut_off_placement = int(np.round(len(stat_df)*self.dist_to_center/100))
            point_sizes = np.zeros(len(stat_df))
            point_sizes[stat_df['placement']<=cut_off_placement] = 10
            stat_df['point_size'] = point_sizes
            if self.dist_to_center != 100:
                print('Done')
            # stat_df = stat_df[stat_df['placement'] <= cut_off_placement]
        
        # Plot remaining stations
        plot = gv.Points(stat_df, 
                         kdims=['long', 'lat'], 
                         vdims=['stat_id', 'color', 'n_trips', 'b_trips', 
                                'w_trips', 'zone_type', 'name', 'point_size', 
                                'dist_to_center'])
        plot.opts(gv.opts.Points(fill_color='color', size='point_size', line_color='black'))
        
        # self.LR_indicator = not self.LR_indicator
        
        return plot


    @param.depends('day_type', 'clustering', 'k', 'plot_traf_or_diff',
                   'plot_all_clusters', 'min_trips', watch=False)
    def plot_centroid(self, index):
        if self.clustering == 'none':
            return "No clustering"
        elif self.clustering == 'h_clustering':
            if self.plot_all_clusters == 'True':
                traffic_matrix = ipu.mask_traffic_matrix(self.traffic_matrices, self.station_df, self.day_type, self.min_trips, holidays=False)
                cc_plot_list = list()
                for j in range(self.k):
                    #mean_vector = np.mean(traffic_matrix[np.where(self.labels == j)], axis=0)
                    mean_vector = self.clusters[j]
                    cc_plot = plot_center(traffic_matrix, self.labels, j, 
                                          mean_vector, self.plot_traf_or_diff)
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
                    cc_plot = plot_center(traffic_matrix, self.labels, j, 
                                          mean_vector, self.plot_traf_or_diff)
                    return pn.Column(cc_plot, f"Station index {i} is in cluster {j}")
                else:
                    return f"Station index {i} is not in a cluster due to min_trips."

        elif self.clustering == 'gaussian_mixture':
            if self.plot_all_clusters == 'True':
                cc_plot_list = list()
                for j in range(self.k):
                    ccs = self.clusters[j]
                    cc_plot = plot_center(traffic_matrix, self.labels, j, ccs, 
                                          self.plot_traf_or_diff)
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to plot cluster centroid"
            else:
                i = index[0]
                if ~np.isnan(self.station_df['label'][i]):
                    j = self.station_df['label'][i].argmax()
                    textlist = [f"{j}: {self.labels[i][j]:.2f}\n\n" for j in range(self.k)]
                    ccs = self.clusters[j]
                    cc_plot = plot_center(traffic_matrix, self.labels, j, ccs,
                                          self.plot_traf_or_diff)
                    return pn.Column(cc_plot, f"Station index {i} belongs to cluster \n\n {''.join(textlist)}")
                else:
                    return f"Station index {i} does not belong to a cluster duto to min_trips"

        else: # k-means or k-medoids
            traffic_matrix = ipu.mask_traffic_matrix(self.traffic_matrices, self.station_df, self.day_type, self.min_trips, holidays=False)
            if self.plot_all_clusters == 'True':
                # if self.clusters == None:
                #     return "Please select Clustering"
                    
                cc_plot_list = list()
                for j in range(self.k):
                    
                    # print(self.labels[0])
                    
                    ccs = self.clusters[j]
                    cc_plot = plot_center(traffic_matrix, self.labels, j, ccs,
                                          self.plot_traf_or_diff,
                                          title_pre="Centroid")
                    cc_plot_list.append(cc_plot)
                return pn.Column(*cc_plot_list)
            if not index:
                return "Select a station to plot cluster centroid"
            else:
                i = index[0]
                if ~np.isnan(self.station_df['label'][i]):
                    j = int(self.station_df['label'][i])
                    ccs = self.clusters[j]
                    cc_plot = plot_center(traffic_matrix, self.station_df['label'], 
                                          j, ccs, self.plot_traf_or_diff,
                                          title_pre="Centroid")
                    return pn.Column(cc_plot, f"Station index {i} is in cluster {j}")
                else:
                    return f"Station index {i} is not in a cluster due to min_trips."
    
    
    @param.depends('service_radius', 'use_road', watch=True)
    def make_service_areas(self):
        print("Making Service Areas")
        
        if self.service_radius != 500 or self.service_area_modified == True:
            self.station_df['service_area'], self.station_df['service_area_size'] = ipu.get_service_area(self.data.city, self.station_df, self.land_use, self.service_radius)
            self.service_area_modified = True
        
        percent_cols = [column for column in self.station_df.columns if "percent_" in column]

        self.station_df = self.station_df.drop(columns=percent_cols).merge(
            ipu.neighborhood_percentages(
                self.data.city, self.station_df, self.land_use, 
                self.service_radius,self.use_road
                ),
            how='outer', left_index=True, right_index=True)
            
        
        self.LR_indicator = not self.LR_indicator

    
    @param.depends('day_type', 'min_trips', 'clustering', 'k', 'random_state', 
                   'service_radius', 'use_road', 'LR_indicator', 'use_points_or_percents',
                   'make_points_by', 'residential', 'commercial',
                   'industrial', 'recreational', 'mixed',
                   'road', 'transportation', 'unknown', 'n_trips',
                   'pop_density', 'nearest_subway_dist', watch=False)
    def make_logistic_regression(self):
        print("Making Logistic Regression")
        
        zones_params = {'percent_residential': self.residential,
                        'percent_commercial': self.commercial,
                        'percent_industrial': self.industrial,
                        'percent_recreational': self.recreational,
                        'percent_mixed': self.mixed,
                        'percent_road': self.road,
                        'percent_transportation': self.transportation,
                        'percent_UNKNOWN': self.unknown,
                        'percent_educational': False}
        
        zone_columns = [column for column in self.station_df.columns 
                        if (('percent_' in column) and (zones_params[column]))]
        
        p_columns = [column for column in self.station_df.columns 
                         if 'percent_' in column]
        
        other_params = [('n_trips', self.n_trips),
                        ('pop_density', self.pop_density),
                        ('nearest_subway_dist', self.nearest_subway_dist)]
        
        other_columns = [param[0] for param in other_params if param[1] and (param[0] in self.station_df.columns)]
        
        return ipu.stations_logistic_regression(
            self.station_df, zone_columns, other_columns,
            use_points_or_percents=self.use_points_or_percents,
            make_points_by=self.make_points_by, const=self.const)
    
    @param.depends('user_type', watch=True)
    def get_traffic_matrix(self):
        self.traffic_matrices = self.data.pickle_daily_traffic(holidays=False, user_type=self.user_type)
        self.plot_clusters_full()
        self.make_service_areas()
        self.make_logistic_regression()
        
    
    
bike_params = BikeDash(None)

@pn.depends(clustering=bike_params.param.clustering)
def plot_tiles(clustering):
    
    tiles = hv.element.tiles.StamenTerrainRetina()
    tiles = gv.tile_sources.StamenTerrainRetina()
    tiles.opts(
        height=800, width=800, xlim=(extremes[1], extremes[0]),
        ylim=(extremes[3], extremes[2]), active_tools=['wheel_zoom'],
        apply_ranges=False)
    
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
    ('Land use', '@zone_type'),
    ('Dist to center', '@dist_to_center')
]
hover = HoverTool(tooltips=tooltips)

pointview = hv.DynamicMap(bike_params.plot_clusters_full)

pointview.opts(tools=['tap', hover])
pointview.opts(nonselection_alpha=0.3)

selection_stream = hv.streams.Selection1D(source=pointview)

@pn.depends(index=selection_stream.param.index,
            day_type=bike_params.param.day_type,
            city=bike_params.param.city,
            month=bike_params.param.month,
            user_type=bike_params.param.user_type,
            plot_traf_or_diff=bike_params.param.plot_traf_or_diff)
def plot_daily_traffic(index, day_type, city, month, user_type, plot_traf_or_diff): 
    if not index:
        return "Select a station to see station traffic"
    else:
        i = index[0]
    
    plot = plot_dta_with_std(bike_params.data, i, day_type, 
                             plot_traf_or_diff, user_type)
    
    if user_type == 'all':
        tripstring = ''
    elif user_type == 'Subscriber':
        tripstring = 'sub_'
    elif user_type == 'Customer':
        tripstring = 'cus_'
    
    if plot_traf_or_diff == 'Traffic':
        
        if day_type == 'business_days':
            b_trips = bike_params.station_df[f'{tripstring}b_departures'].iloc[i] + bike_params.station_df[f'{tripstring}b_arrivals'].iloc[i]
            plot.opts(title=f'Average hourly traffic for {bike_params.data.stat.names[i]} (b_trips={b_trips:n})', ylabel='percentage of all traffic')
        elif day_type == 'weekend':
            w_trips = bike_params.station_df[f'{tripstring}w_departures'].iloc[i] + bike_params.station_df[f'{tripstring}w_arrivals'].iloc[i]
            plot.opts(title=f'Average hourly traffic for {bike_params.data.stat.names[i]} (w_trips={w_trips:n})', ylabel='percentage of all traffic')
    
    else:
        if day_type == 'business_days':
            b_trips = bike_params.station_df[f'{tripstring}b_departures'].iloc[i] + bike_params.station_df[f'{tripstring}b_arrivals'].iloc[i]
            plot.opts(title=f'Average relative difference for {bike_params.data.stat.names[i]} (b_trips={b_trips:n})', ylabel='relative difference')
        elif day_type == 'weekend':
            w_trips = bike_params.station_df[f'{tripstring}w_departures'].iloc[i] + bike_params.station_df[f'{tripstring}w_arrivals'].iloc[i]
            plot.opts(title=f'Average relative difference for {bike_params.data.stat.names[i]} (w_trips={w_trips:n})', ylabel='relative difference')
    
    
    return plot


@pn.depends(index=selection_stream.param.index,
            plot_traf_or_diff=bike_params.param.plot_traf_or_diff,
            plot_all_clusters=bike_params.param.plot_all_clusters,
            clustering=bike_params.param.clustering,
            k=bike_params.param.k,
            min_trips=bike_params.param.min_trips,
            day_type=bike_params.param.day_type,
            city=bike_params.param.city,
            month=bike_params.param.month,
            user_type=bike_params.param.user_type)
def plot_centroid_callback(index, plot_traf_or_diff, plot_all_clusters, 
                           clustering, k, min_trips, day_type, city, month, 
                           user_type): 
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
            use_road=bike_params.param.use_road,
            LR_indicator=bike_params.param.LR_indicator)
def service_area_plot(show_service_area, service_radius, service_area_color, city, use_road, LR_indicator):
    # bike_params.make_service_areas()
    
    if show_service_area == 'True':
        zone_percents_columns = [column for column in bike_params.station_df.columns
                                  if 'percent_' in column]
        
        # bike_params.station_df['service_color'] = ['#808080' for i in range(len(bike_params.station_df))]
        
        vdims = zone_percents_columns
        vdims.append('service_area_size')
        sdf = bike_params.station_df.set_geometry('service_area')
        sdf['geometry'] = sdf.geometry
        print(f"vdims={vdims}")
        if service_area_color == 'residential':
            return gv.Polygons(sdf, vdims=vdims).opts(color='percent_residential')
        elif service_area_color == 'commercial':
            return gv.Polygons(sdf, vdims=vdims).opts(color='percent_commercial')
        elif service_area_color == 'recreational':
            return gv.Polygons(sdf, vdims=vdims).opts(color='percent_recreational')
        elif service_area_color == 'pop density':
            if 'pop_density' in bike_params.station_df.columns:
                return gv.Polygons(sdf, vdims=vdims).opts(color='pop_density')
            else:
                return gv.Polygons([])
        
    
    return gv.Polygons([])

@pn.depends(show_census=bike_params.param.show_census,
            city=bike_params.param.city)
def census_plot(show_census, city):
    if show_census == 'True' and city in ['nyc', 'chicago', 'washdc', 'boston', 'minneapolis']:
        return gv.Polygons(bike_params.census_df).opts(color='pop_density', cmap='YlGnBu')
    else:
        return gv.Polygons([])


@pn.depends(#service_radius=bike_params.param.service_radius,
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
            industrial=bike_params.param.industrial,
            recreational=bike_params.param.recreational,
            mixed=bike_params.param.mixed,
            road=bike_params.param.road,
            transportation=bike_params.param.transportation,
            unknown=bike_params.param.unknown,
            n_trips=bike_params.param.n_trips,
            pop_density=bike_params.param.pop_density,
            nearest_subway_dist=bike_params.param.nearest_subway_dist,
            const=bike_params.param.const,
            LR_indicator=bike_params.param.LR_indicator
            )
def print_logistic_regression(use_road, clustering, k, 
                              min_trips, day_type, city, month, residential,
                              commercial, industrial, recreational,
                              mixed, road, transportation, unknown, n_trips, 
                              pop_density, nearest_subway_dist, const, 
                              use_points_or_percents, make_points_by, LR_indicator):
    res, X, y = bike_params.make_logistic_regression()
    return res.summary().as_html() if res != None else "Singular"
    

@pn.depends(city=bike_params.param.city, watch=True)
def update_extent(city):
    extremes = [bike_params.station_df['easting'].max(), bike_params.station_df['easting'].min(), 
                bike_params.station_df['northing'].max(), bike_params.station_df['northing'].min()]
    tileview.opts(xlim=(extremes[1], extremes[0]), ylim=(extremes[3], extremes[2]))
    panel_column[1][1][0].object.data[()].Points.I.data = bike_params.station_df
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

rev_month_dict = {v:k for k,v in month_dict.items()}

activity_dict = {'departures': 'start', 'arrivals': 'end', 'd': 'start', 'a': 'end', 'start': 'start', 'end': 'end'}
day_type_dict = {'weekend': 'w', 'business_days': 'b'}

gif_pane = pn.pane.GIF('Loading_Key.gif')

city_list = ['boston', 'chicago', 'minneapolis', 'nyc', 'washdc', 'edinburgh', 'helsinki', 'london', 'madrid', 'oslo']

city_dict = dict(zip([bs.name_dict[city] for city in city_list], city_list))

# =============================================================================
# Parameters
# =============================================================================

param_split = 22 # add to this number if additional parameters/titles are added in the first column

params = pn.Param(bike_params.param, widgets={
    'city': {'widget_type': pn.widgets.Select, 'options': city_dict},
    'clustering': pn.widgets.RadioBoxGroup,
    'month': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success', 'css_classes': ['month-button']},
    'trip_type': {'widget_type': pn.widgets.RadioButtonGroup, 'button_type': 'success'},
    'day_type': pn.widgets.RadioButtonGroup,
    'plot_traf_or_diff': pn.widgets.RadioButtonGroup,
    'plot_all_clusters': {'widget_type': pn.widgets.RadioButtonGroup, 'title': 'Hello'},
    'keep_furthest_or_closest' : pn.widgets.RadioButtonGroup,
    'dist_to_center': {'widget_type' : pn.widgets.IntSlider, 'name' : '% of stations furthest/closest to cluster center'},
    'day': pn.widgets.IntSlider,
    'random_state': pn.widgets.IntInput,
    'min_trips': pn.widgets.IntInput,
    'k': pn.widgets.IntSlider,
    'show_land_use': pn.widgets.RadioButtonGroup,
    'show_census': pn.widgets.RadioButtonGroup,
    'show_service_area': pn.widgets.RadioButtonGroup,
    'service_radius': pn.widgets.IntInput,
    'use_road': pn.widgets.RadioButtonGroup,
    'use_points_or_percents': pn.widgets.RadioButtonGroup,
    'make_points_by': pn.widgets.RadioButtonGroup
    },
    name="Bikeshare Parameters",
    )

css = '''
.bk.bk-btn.bk-btn-success {
    padding: 6pt 0pt 6pt 0pt;    
}
.missing_labels p{
    line-height: 1;
    padding-left: 5px;
}
''' # Reduce padding


# .bk-root {
#     background-color: #2f2f2f;
#     border-color: #2f2f2f;
#     }

pn.extension(raw_css=[css])

params.parameters = params.parameters[:param_split]

# Change None to Year in month selector
params.layout[1].options = city_dict

params.layout[2].options.pop('None')
params.layout[2].options['Year'] = None


# params.layout.insert(15, 'Use roads:')
# params.layout[15].css_classes=['missing_labels'] 
# params.layout[15].margin = (5,5,-12,5)
# params.layout.insert(13, 'Show service area:')
# params.layout[13].css_classes=['missing_labels'] 
# params.layout[13].margin = (5,5,-12,5)
# params.layout.insert(12, 'Show census:')
# params.layout[12].css_classes=['missing_labels'] 
# params.layout[12].margin = (5,5,-12,5)
# params.layout.insert(11, 'Show land use:')
# params.layout[11].css_classes=['missing_labels'] 
# params.layout[11].margin = (5,5,-12,5)
# params.layout.insert(10, 'Plot all clusters:')
# params.layout[10].css_classes=['missing_labels'] 
# params.layout[10].margin = (5,5,-12,5)
# params.layout.insert(5, 'Clustering method:')
# params.layout[5].css_classes=['missing_labels'] 
# params.layout[5].margin = (5,5,-12,5)
# params.layout.insert(2, 'Month:')
# #params.layout.select(lambda x: x.name=='Month')
# params.layout[2].css_classes=['missing_labels'] 
# params.layout[2].margin = (5,5,-12,5)


parameter_strings = [
    ['Month:', 'Month_title'],
    ['Clustering method:', 'Clustering_title'],
    ['Plot traffic or relative difference (clustering is still based on relative difference):', 'traf_or_diff_title'],
    ['Plot all cluster centers:', 'Plot all_title'],
    ['Show land use (can be slow):', 'Show land_title'],
    ['Show census:', 'Show census_title'],
    ['Show service area:', 'Show service_title'],
    ['Use road:', 'Use road_title'],
    ['Keep stations furthest or closest to cluster center:', 'Keep_fur_or_clo_title']
    ]

for parstr in parameter_strings:
    pane = pn.pane.Markdown(parstr[0], name=parstr[1])
    pane.css_classes=['missing_labels']
    pane.margin = (5,5,-12,5)
    params.layout.append(pane)

parameter_names = [
    'City', 'Month_title', 'Month', 'Day type', 'Clustering_title', 
    'Clustering', 'K', 'traf_or_diff_title','Plot traf or diff','Plot all_title', 'Plot all clusters', 'Keep_fur_or_clo_title', 
    'Keep furthest or closest','% of stations furthest/closest to cluster center', 'Show land_title', 'Show land use', 
    'Show census_title', 'Show census', 'Show service_title', 'Show service area', 'Service radius', 'Service area color',
    'Use road_title', 'Use road', 'Random state', 'Min trips', 'User type']

parameter_column = []

for name in parameter_names:
    parameter_column.append(params.layout.select(lambda x: x.name==name)[0])

parameter_column = pn.Column(*parameter_column)

LR_params = pn.Param(bike_params.param, widgets={
    'use_points_or_percents': pn.widgets.RadioButtonGroup,
    'make_points_by': pn.widgets.RadioButtonGroup
    },
    name="Logistic Regression Parameters",
    )

LR_params.parameters = LR_params.parameters[param_split:-1]

LR_params.layout.insert(2, 'Make points by:')
LR_params.layout[2].css_classes=['missing_labels'] 
LR_params.layout[2].margin = (5,5,-12,5)
LR_params.layout.insert(1, 'Use station points or percents:')
LR_params.layout[1].css_classes=['missing_labels'] 
LR_params.layout[1].margin = (5,5,-12,5)

# =============================================================================
# Themes
# =============================================================================

theme = Theme(
    json={
    'attrs': {'Axis': {'major_tick_in': 0,
   'major_tick_out': 3,
   'major_tick_line_alpha': 0.25,
   'major_tick_line_color': '#5B5B5B',
   'minor_tick_line_alpha': 0.25,
   'minor_tick_line_color': '#5B5B5B',
   'axis_line_alpha': 1,
   'axis_line_color': '#5B5B5B',
   'major_label_text_color': '#5B5B5B',
   'major_label_text_font': 'Calibri Light',
   'major_label_text_font_size': '0.95em',
   'major_label_text_font_style': 'bold',
   'axis_label_standoff': 10,
   'axis_label_text_color': '#5B5B5B',
   'axis_label_text_font': 'Calibri Light',
   'axis_label_text_font_size': '1.15em',
   'axis_label_text_font_style': 'bold'},
  'Grid': {'grid_line_dash': [6, 4],
           'grid_line_alpha': 1},
  'Legend': {'spacing': 8,
   'glyph_width': 15,
   'label_standoff': 8,
   'label_text_color': '#5B5B5B',
   'label_text_font': 'Calibri Light',
   'label_text_font_size': '0.95em',
   'label_text_font_style': 'bold',
   'border_line_alpha': 0,
   'background_fill_alpha': 0.25},
  'ColorBar': {'title_text_color': '#5B5B5B',
   'title_text_font': 'Calibri Light',
   'title_text_font_size': '1.15em',
   'title_text_font_style': 'bold',
   'major_label_text_color': '#5B5B5B',
   'major_label_text_font': 'Calibri Light',
   'major_label_text_font_size': '0.95em',
   'major_label_text_font_style': 'bold',
   'major_tick_line_alpha': 0,
   'bar_line_alpha': 0},
  'Title': {'text_color': '#5B5B5B',
   'text_font': 'Calibri Light',
   'text_font_size': '1.25em',
   'text_font_style': 'bold'}}})

# hv.renderer('bokeh').theme = 'dark_minimal'

# =============================================================================
# Views
# =============================================================================

zoneview = hv.DynamicMap(land_use_plot)
zoneview.opts(alpha=0.5, apply_ranges=False)

service_area_view = hv.DynamicMap(service_area_plot)
service_area_view.opts(alpha=0.5, apply_ranges=False)

census_view = hv.DynamicMap(census_plot)
census_view.opts(alpha=0.5, apply_ranges=False)


#tileview.opts(framewise=True, apply_ranges=False)

tooltips_zone = [
    ('Zone Type', '@zone_type'),
]

tooltips_service_area = [
    ('area (km^2)', '@service_area_size'),
    ('% residential', '@percent_residential'),
    ('% commercial', '@percent_commercial'),
    ('% industrial', '@percent_industrial'),
    ('% recreational', '@percent_recreational'),
    ('% mixed', '@percent_mixed'),
    ('% road', '@percent_road'),
    ('% unknown', '@percent_UNKNOWN')]

tooltips_census = [
    ('population', '@population'),
    ('area (km^2)', '@census_area'),
    ('pop_density', '@pop_density')
]

hover_zone = HoverTool(tooltips=tooltips_zone)
hover_service_area = HoverTool(tooltips=tooltips_service_area)
hover_census = HoverTool(tooltips=tooltips_census)

zoneview.opts(tools=[hover_zone])
service_area_view.opts(tools=[hover_service_area])
census_view.opts(tools=[hover_census])

# selection_stream.param.values()['index']

views = tileview*zoneview*census_view*service_area_view*pointview


# =============================================================================
# Dashboard layout
# =============================================================================

param_column = pn.Column(parameter_column, minpercent)
param_column[1].width=300

LR_row = pn.Row(LR_params.layout, print_logistic_regression)

mid_column = pn.Column(views, LR_row)

line_column = pn.Column(plot_daily_traffic, plot_centroid_callback)

business_row = pn.Row(param_column, mid_column, line_column)
text = '#Bikesharing Clustering Analysis'
title_row = pn.Row(text, indicator)
title_row[0].width=400
panel_column = pn.Column(title_row, business_row)
panel_column.servable() # Run with: panel serve interactive_plot.py --autoreload

"""
bokeh_server = panel_column.show(port=12345)
"""

# plot_daily_traffic(5, bike_params.param.day_type, bike_params.param.city, 
#                    bike_params.param.month, bike_params.param.user_type,
#                    bike_params.param.plot_traf_or_diff)

# bike_params.make_service_areas()

#%%
"""
bokeh_server.stop() # Run with: panel serve interactive_plot.py --autoreload
"""
