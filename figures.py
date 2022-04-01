# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:41:40 2022

@author: Nicolai
"""
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import geoviews as gv
import holoviews as hv
import smopy as sm
import matplotlib.pyplot as plt
import shapely
import calendar
import pickle
from sklearn.model_selection import train_test_split
from holoviews import opts

from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.offsetbox import AnchoredText, AnchoredOffsetbox
from geopy.distance import geodesic

import bikeshare as bs
import interactive_plot_utils as ipu
from logistic_table import lr_coefficients

def service_area_figure(data, stat_df, land_use, return_fig=False):
    """
    Makes a figure of the stations and their service areas in a network.

    Parameters
    ----------
    data : bikeshare.Data object
        Data object containing trip data for the network.
    stat_df : Dataframe
        DataFrame containing information of the stations in the network 
        including their service areas, obtained from
        interactive_plot_utils.make_station_df().
    land_use : DataFrame
        DataFrame containing the land use data, obtained from
        interactive_plot_utils.make_station_df().
    return_fig : bool, optional
        Returns the figure and ax object if True. The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure object
        Object containing the figure.
    ax : matplotlib.axes._subplots.AxesSubplot object
        Object ontaining the ax.

    """
    
    extend = (stat_df['lat'].min(), stat_df['long'].min(), 
          stat_df['lat'].max(), stat_df['long'].max())
    
    tileserver = 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg' # Stamen Terrain
    # tileserver = 'http://a.tile.stamen.com/toner/{z}/{x}/{y}.png' # Stamen Toner
    # tileserver = 'http://c.tile.stamen.com/watercolor/{z}/{x}/{y}.png' # Stamen Watercolor
    # tileserver = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png' # OSM Default
    
    m = sm.Map(extend, tileserver=tileserver)
    
    fig, ax = plt.subplots(figsize=(7,10))
    
    m.show_mpl(ax=ax)
    # fig.add_axes(ax)
    
    for i, stat in stat_df.iterrows():
        x, y = m.to_pixels(stat['lat'], stat['long'])
        ax.plot(x, y, 'ob', ms=2, mew=1.5)
        
        if isinstance(stat['service_area'], shapely.geometry.multipolygon.MultiPolygon):
            
            for poly in stat['service_area']:
                coords = np.zeros(shape=(len(poly.exterior.xy[0]),2))    
                
                for j in range(len(coords)):
                    coords[j,0] = poly.exterior.xy[1][j]
                    coords[j,1] = poly.exterior.xy[0][j]
                
                pixels = m.to_pixels(coords)
                
                p = Polygon(list(zip(pixels[:,0], pixels[:,1])), 
                            alpha=0.5, facecolor='tab:cyan', edgecolor='k')
            
                ax.add_artist(p)
                
                # ax.plot(pixels[:,0], pixels[:,1], c='k')
                
        else:
            coords = np.zeros(shape=(len(stat['service_area'].exterior.xy[0]),2))    
            # coords = []    
            
            for j in range(len(stat['service_area'].exterior.xy[0])):
                coords[j,0] = stat['service_area'].exterior.xy[1][j]
                coords[j,1] = stat['service_area'].exterior.xy[0][j]
                
                # coords.append((stat['service_area'].exterior.xy[0][j], 
                #                stat['service_area'].exterior.xy[1][j]))
            
            pixels = m.to_pixels(coords)
            
            p = Polygon(list(zip(pixels[:,0], pixels[:,1])), 
                        alpha=0.5, facecolor='tab:cyan', edgecolor='k')
            
            ax.add_artist(p)
            
            # ax.fill(pixels[:,0], pixels[:,1], c='b')
            # ax.plot(pixels[:,0], pixels[:,1], c='k')
            
    xlim_dict = {'nyc' : (150, 600)}
    ylim_dict = {'nyc' : (767.5,100)}
    
    
    if data.city in xlim_dict.keys():
        ax.set_xlim(xlim_dict[data.city])
    
    if data.city in ylim_dict.keys():
        ax.set_ylim(ylim_dict[data.city])
    
    ax.plot(0, 0, 'ob', ms=2, mew=1.5, label='Station')
    p0 = Polygon([(0,0), (0,1), (1,0)], alpha=0.5, 
                 facecolor='tab:cyan', edgecolor='k', label='Service Area')
    ax.add_patch(p0)
    ax.legend()
    
    scalebar_size_km = 5
    
    c0 = (stat_df.iloc[0].easting, stat_df.iloc[0].northing)
    c1 = (stat_df.iloc[1].easting, stat_df.iloc[1].northing)
    
    geo_dist = np.linalg.norm(np.array(c0) - np.array(c1))
    
    pix0 = m.to_pixels(stat_df.iloc[0].lat, stat_df.iloc[0].long)
    pix1 = m.to_pixels(stat_df.iloc[1].lat, stat_df.iloc[1].long)
    
    pix_dist = np.linalg.norm(np.array(pix0) - np.array(pix1))
    
    scalebar_size = pix_dist/geo_dist*1000*scalebar_size_km
    
    
    scalebar = AnchoredSizeBar(ax.transData, scalebar_size, 
                               f'{scalebar_size_km} km', 'lower right', 
                               pad=0.2, color='black', frameon=False, 
                               size_vertical=2)
    ax.add_artist(scalebar)
    attr = AnchoredText("(C) Stamen Design. (C) OpenStreetMap contributors.",
                       loc = 'lower left', frameon=True, pad=0.1, borderpad=0)
    attr.patch.set_edgecolor('white')
    ax.add_artist(attr)
    
    ax.axis('off')    
    
    plt.tight_layout()
    plt.savefig(f'./figures/paper_figures/service_areas_{data.city}_{data.year}{data.month:02d}.pdf')
        
    if return_fig:
        return fig, ax


def daily_traffic_figure(data, stat_id,  period='b', normalise=True, user_type='all', return_fig=False):
    """
    Makes a figure of the average daily traffic for a station.

    Parameters
    ----------
    data : bikeshare.Data object
        Data object containing trip data for the network.
    stat_id : int
        ID of the station.
    period : str, optional
        Either 'b' for business days or 'w' for weekends. The default is 'b'.
    normalise : bool, optional
        Normalises the traffic with respect to the total number of trips before
        averaging if True. The default is True.
    user_type : str, optional
        Either 'Subscribers' for subscription trips, 'Casual' for casual trips
        or 'all' for both. The default is 'all'.
    return_fig : bool, optional
        Returns the figure object if True. The default is False.

    Returns
    -------
    None.

    """
    traffic = data.daily_traffic_average(data.stat.id_index[stat_id], period=period, normalise=normalise, user_type=user_type)    
    
    departures = traffic[0]
    arrivals = traffic[1]
    
    plt.style.use('seaborn-darkgrid')
    
    fig, ax = plt.subplots(figsize=(10,5))

    if normalise:
        ax.plot(np.arange(24), departures*100, label='departures')
        ax.plot(np.arange(24), arrivals*100, label='arrivals')

        ax.set_ylabel('% of total trips')

    else:
        ax.plot(np.arange(24), departures, label='departures')
        ax.plot(np.arange(24), arrivals, label='arrivals')

        ax.set_ylabel('# trips')

    ax.set_xticks(np.arange(24))
    plt.legend()
    ax.set_xlabel('Hour')
    
    plt.tight_layout()
    
    if return_fig:
        return fig, ax


def stat_df_day(day, city, year, month, columns):
    data_day = bs.Data(city, year, month, day, day_type='business_days', user_type='Subscriber')
    if len(data_day.df) > 0: # Avoid the issue of days with no traffic. E.g. Oslo 2019-04-01
        stat_df = ipu.make_station_df(data_day, holidays=False, overwrite=True)
    else:
        stat_df = pd.DataFrame(columns=columns)
    return stat_df[stat_df.columns & columns]
        

def make_summary_statistics_table(cities=None, variables=None, year=2019, print_only=False):
    """
    Makes a table containing the summary statistics of the variables used in
    the model for all cities. Also calculates the variables for each station
    averaged over all days in which the station was used. 
    WARNING: VERY SLOW, IT WILL TAKE DAYS TO RUN THIS.

    Parameters
    ----------
    cities : iterable, optional
        Iterable containing the cities in which to make summary statistics for. 
        Does all cities if set to None. The default is None.
    variables : iterable, optional
        Iterable containing the variables to make summary statistics for. 
        Does all variables if None. The default is None.
    year : int, optional
        The year to make the tabl for. The default is 2019.
    print_only : bool, optional
        Only prints the table and skips the calculations if True. The default 
        is False.

    Returns
    -------
    tab_df : pandas.DataFrame
        DataFrame conaining the summary statistics table.

    """
    if cities is None:
        cities = ['nyc', 'chicago', 'washdc', 'boston', 
                  'london', 'helsinki', 'oslo', 'madrid']
        
    # variables = ['Share of residential use', 'Share of commercial use',
    #              'Share of recreational use', 'Share of industrial use', 
    #              'Share of transportational use', 'Share of mixed use',
    #              'Population density', 'Distance to nearest subway/railway', 
    #              'Number of  trips']
    
    if variables is None:
        variables = ['percent_residential', 'percent_commercial',
                     'percent_recreational', 'percent_industrial',
                     'percent_mixed', 'percent_transportation', 
                     'percent_educational', 'percent_road', 'percent_UNKNOWN',
                     'pop_density', 'nearest_subway_dist', 'nearest_railway_dist',
                     'nearest_transit_dist', 'n_trips', 'b_trips', 'w_trips']

    if not print_only:
        
        for city in cities:
            data_city = bs.Data(city, year)
            
            stat_ids = list(data_city.stat.id_index.keys())
            
            var_dfs = dict()
            
            for var in variables:
                var_df = pd.DataFrame()
                var_df['stat_id'] = stat_ids
                
                var_dfs[var] = var_df
            
            stat_dfs = dict()
            
            # for month in bs.get_valid_months(city, year):
            #     for day in range(1, calendar.monthrange(year, month)[1]+1):
            #         data_day = bs.Data(city, year, month, day, day_type='business_days', user_type='Subscriber')
            #         if len(data_day.df) > 0: # Avoid the issue of days with no traffic. E.g. Oslo 2019-04-01
            #             stat_df = ipu.make_station_df(data_day, holidays=False, overwrite=True)
                        
                        # for var in variables:
                        #     if var in stat_df.columns:
                        #         var_dfs[var] = var_dfs[var].merge(stat_df[['stat_id', var]], on='stat_id', how='outer')
                        #         var_dfs[var].rename({var: f'{year}-{month:02d}-{day:02d}'}, axis=1, inplace=True)
            
            with mp.Pool(mp.cpu_count()) as pool: # multiprocessing version            
                for month in bs.get_valid_months(city, year):
                    stat_df_day_part = partial(stat_df_day, city=city, year=year, month=month, columns=variables + ['stat_id'])
                    days = range(1, calendar.monthrange(year, month)[1]+1)
                    stat_dfs[month] = pool.map(stat_df_day_part, days)
            
            print(stat_dfs)
            
            for month in bs.get_valid_months(city, year):
                for day in range(1, calendar.monthrange(year, month)[1]+1):
                    stat_df = stat_dfs[month][day-1]
                    for var in variables:
                        if var in stat_df.columns:
                            var_dfs[var] = var_dfs[var].merge(stat_df[['stat_id', var]], on='stat_id', how='outer')
                            var_dfs[var].rename({var: f'{year}-{month:02d}-{day:02d}'}, axis=1, inplace=True)
            
            avg_stat_df = pd.DataFrame()
            avg_stat_df['stat_id'] = stat_ids
            for var in variables:
                if len(var_dfs[var].columns) > 1:
                    avg_stat_df[var] = var_dfs[var][var_dfs[var].columns[1:]].mean(axis=1)
            
            with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'wb') as file:
                pickle.dump(avg_stat_df, file)
        
    
    multiindex = pd.MultiIndex.from_product((cities, 
                                             ['Mean', 'Std. Dev.', 'Min.', 'Max.']))   
    
    tab_df = pd.DataFrame(index=variables, columns = multiindex)
    
    
    
    for city_index, city in enumerate(cities):
        
        with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                avg_stat_df = pickle.load(file)
        
        
        avg_stat_df['pop_density'] = avg_stat_df['pop_density']/10000 # convert to population per 100 m^2
        avg_stat_df['nearest_subway_dist']  = avg_stat_df['nearest_subway_dist']/1000 # convert to km
        avg_stat_df['nearest_railway_dist']  = avg_stat_df['nearest_railway_dist']/1000 # convert to km
        # avg_stat_df['n_trips'] = avg_stat_df['n_trips']/avg_stat_df['n_trips'].sum() # convert to percentage
        # avg_stat_df['b_trips'] = avg_stat_df['b_trips']/avg_stat_df['b_trips'].sum() # convert to percentage
        # avg_stat_df['w_trips'] = avg_stat_df['w_trips']/avg_stat_df['w_trips'].sum() # convert to percentage
        
        
        # city_df = pd.DataFrame(columns=['Variable', 'Mean', 
        #                                 'Std. Dev.', 'Min', 'Max'],
        #                        index=variables)
        # city_df['City'] = bs.name_dict[city]
        # city_df['Variable'] = variables
        tab_df[(city, 'Mean')] = avg_stat_df.mean()
        tab_df[(city, 'Std. Dev.')] = avg_stat_df.std()
        tab_df[(city, 'Min.')] = avg_stat_df.min()
        tab_df[(city, 'Max.')] = avg_stat_df.max()
        
        # tab_df = pd.concat([tab_df, city_df])
    
    var_renames = {'percent_residential' : 'Share of residential use',
                   'percent_commercial' : 'Share of commercial use',
                   'percent_industrial' : 'Share of industrial use',
                   'percent_recreational' : 'Share of recreational use',
                   'percent_mixed' : 'Share of mixed use',
                   'percent_transportation' : 'Share of transportation use',
                   'percent_educational' : 'Share of educational use',
                   'percent_road' : 'Share of road use',
                   'percent_UNKNOWN' : 'Share of unknown use',
                   'n_trips' : 'Number of daily trips',
                   'b_trips' : 'Number of daily business trips',
                   'w_trips' : 'Number of daily weekend trips',
                   'pop_density' : 'Population density [per 100 sq. m]',
                   'nearest_subway_dist' : 'Distance to nearest subway [km]',
                   'nearest_railway_dist' : 'Distance to nearest railway [km]'}
    # tab_df = tab_df.replace(var_renames)
    
    # city_names = [bs.name_dict[city] for city in cities]
    
    tab_df = tab_df.rename(index=var_renames, columns=bs.name_dict)
    
    print(tab_df.to_latex(column_format='@{}l'+('r'*len(tab_df.columns)) + '@{}',
                          index=True, na_rep = '--', float_format='%.2f',
                          multirow=True, multicolumn=True, multicolumn_format='c'))
    
    return tab_df


def make_LR_table(year=2019, k=3):
    """
    Makes a table containing the coefficients of the Logistics Regression 
    model for all cities.

    Parameters
    ----------
    year : int
        Year to train the model for.
    k : int, optional
        Number of clusters in the k-means custering. The default is 3.

    Returns
    -------
    tuple_table : TYPE
        DESCRIPTION.

    """
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
                  'london', 'helsinki', 'oslo', 'madrid']
    
    city_lists = [(['nyc', 'chicago', 'washdc', 'boston'], 'USA'),
                      (['london', 'helsinki', 'oslo', 'madrid'], 'EUR')]
    
    
    percent_index_dict = {
        'percent_UNKNOWN': 'Share of unknown use',
        'percent_residential': 'Share of residential use',
        'percent_commercial': 'Share of commercial use',
        'percent_industrial': 'Share of industrial use',
        'percent_recreational': 'Share of recreational use',
        'percent_educational': 'Share of educational use',
        'percent_mixed': 'Share of mixed use',
        'percent_road': 'Share of road use',
        'percent_transportation': 'Share of transportational use',
        'pop_density': 'Population density [per 100 sq. m]',
        'nearest_subway_dist': 'Distance to nearest subway [km]',
        'nearest_railway_dist': 'Distance to nearest railway [km]',
        'n_trips': 'Share of daily trips',
        'b_trips': 'Share of business trips',
        'w_trips': 'Share of business trips',
        }
    
    omit_columns = {
        'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
        'chicago': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
        'nyc': ['percent_mixed', 'n_trips'],
        'washdc': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
        'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
        'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips', 'percent_industrial'],
        'madrid': ['n_trips', 'percent_industrial'],
        'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips', 'percent_mixed'],
        'USA': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'percent_mixed'],
        'EUR': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'percent_mixed'],
        'All': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'percent_mixed'],
        }
    
    month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
          7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None'}

    
    table = pd.DataFrame([])
    
    for city in cities:
        # print(city)
        with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
            asdf = pickle.load(file)
        
        data = bs.Data(city, year)
        
        # traf_mats = data.pickle_daily_traffic(holidays=False, 
        #                                       user_type='Subscriber',
        #                                       overwrite=True)
        
        traf_df_b = data.daily_traffic_average_all(period='b', holidays=False, 
                                             user_type='Subscriber')
        
        traf_mat_b = np.concatenate((traf_df_b[0].to_numpy(), 
                                     traf_df_b[1].to_numpy()),
                                    axis=1)
        
        traf_df_w = data.daily_traffic_average_all(period='w', holidays=False, 
                                             user_type='Subscriber')
        
        traf_mat_w = np.concatenate((traf_df_w[0].to_numpy(), 
                                     traf_df_w[1].to_numpy()),
                                    axis=1)
        
        
        traf_mats = (traf_mat_b, traf_mat_w)
        
        casual_stations = set(list(asdf.stat_id.unique())) - (set(list(traf_df_b[0].index)) | set(list(traf_df_b[1].index)))
        
        for station in list(casual_stations):
            asdf = asdf[asdf['stat_id'] != station]
        
        asdf = asdf.reset_index()
        
        asdf, clusters, labels = ipu.get_clusters(traf_mats, asdf, 'business_days', 10, 'k_means', k, 42)
        
        zone_columns = ['percent_residential', 'percent_commercial',
                        'percent_recreational', 'percent_industrial']
        
        for column in omit_columns[data.city]:
            if column in zone_columns:
                zone_columns.remove(column)
    
        other_columns = ['pop_density', 'nearest_subway_dist', 
                         'nearest_railway_dist', 'b_trips']
        
        for column in omit_columns[data.city]:
            if column in other_columns:
                other_columns.remove(column)
            
        lr_results, X, y, _ = ipu.stations_logistic_regression(
            asdf, zone_columns, other_columns, 
            use_points_or_percents='percents', 
            make_points_by='station land use', const=False,
            test_model=True, test_ratio=0.2, test_seed=42,
            )
        
        # print(lr_results)
        print(lr_results.summary())

        single_index = lr_results.params[0].index

        parameters = np.concatenate([lr_results.params[i] for i in range(0, k-1)])
        stdev = np.concatenate([lr_results.bse[i] for i in range(0, k-1)])
        pvalues = np.concatenate([lr_results.pvalues[i] for i in range(0, k-1)])
        
        index = np.concatenate([lr_results.params.index for i in range(0, k-1)])
    
        multiindex = pd.MultiIndex.from_product([range(1,k), single_index], names=['Cluster', 'Coef. name'])
    
        pars = pd.Series(parameters, index=multiindex, name='coef')
        sts = pd.Series(stdev, index=multiindex, name='stdev')
        pvs = pd.Series(pvalues, index=multiindex, name='pvalues')
        
        coefs = pd.DataFrame(pars).join((sts, pvs))
            

        lr_coefs = pd.concat({data.city: coefs}, names="", axis=1)

        table = pd.concat([table, lr_coefs], axis=1)        
    
    
    signif_table = table.xs('pvalues', level=1, axis=1) < 0.05
    
    coeftable = table.xs('coef', level=1, axis=1)
    
    tuple_table = pd.concat([coeftable,signif_table]).stack(dropna=False).groupby(level=[0,1,2]).apply(tuple).unstack()
    
    index_list = list(percent_index_dict.keys())
    
    index_renamer = percent_index_dict
    column_renamer = dict(zip(cities, [bs.name_dict[city] for city in cities]))
    
    index_list.insert(0, 'const')
    
    #index_list = set(index_list).intersection(set(table.index.get_level_values(1)))
    
    index_list = [x for x in index_list if x in table.index.get_level_values(1)]
    
    tables = dict()
    for i in range(1, k):
        tables[i] = tuple_table.loc[i].loc[index_list] # Reorder according to index_dict
    
    
    tuple_table = pd.concat(tables, names=['Cluster', 'Coef. name'])
    tuple_table = tuple_table.reindex(columns=cities)
    tuple_table = tuple_table.rename(index=index_renamer, columns=column_renamer)
    
    if k == 3:
        tuple_table = tuple_table.rename(index={1: 'Morning Sink', 
                                                2: 'Morning Source'})
    
    latex_table = tuple_table.to_latex(column_format='@{}ll'+('r'*len(tuple_table.columns)) + '@{}', multirow=True, formatters = [tuple_formatter]*len(tuple_table.columns), escape=False)
    print(latex_table)
    
        
    return tuple_table


def formatter(x):
    if x == np.inf:
        return "inf"
    elif np.abs(x) > 10000 or np.abs(x) < 0.001:
        return f"$\\num{{{x:.2e}}}$"
    else:
        return f"${x:.4f}$"
    
def tuple_formatter(tup):
    x, bold = tup
    if x == np.inf:
        out = "inf"
    elif np.isnan(x):
        out = "--"
    elif np.abs(x) > 10000 or np.abs(x) < 0.001:
        if bold:
            out = f"$\\mathbf{{\\num{{{x:.2e}}}}}$"
        else:
            out = f"$\\num{{{x:.2e}}}$"
    else:
        if bold:
            out = f"$\\mathbf{{{x:.3f}}}$"
        else:
            out = f"${x:.3f}$"
    
    return out

def city_tests(year=2019, cities=None, k=3, test_ratio=0.2, test_seed=42, 
               res='success_rates'):
    """
    
    1. for each city: split data into training and test sets and train LR model.
    2. test eac model on the 8 test sets
    3. show scces rates in a nice figure
    
    bonus: redefine succes rate as rate at which a model does not confuse
           morning sinks and morning sources.
    
    
    
    """
    
    omit_columns = {
        'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
        'chicago': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
        'nyc': ['percent_mixed', 'n_trips'],
        'washdc': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
        'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
        'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips', 'percent_industrial'],
        'madrid': ['n_trips', 'percent_industrial'],
        'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips', 'percent_mixed'],
        'USA': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'percent_mixed'],
        'EUR': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'percent_mixed'],
        'All': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'percent_mixed'],
        }
   
    
    if cities is None:
        cities = ['nyc', 'chicago', 'washdc', 'boston', 
                  'london', 'helsinki', 'oslo', 'madrid']
    
    if test_ratio < 0 or test_ratio > 1:
            raise ValueError("test_ratio must be between 0 and 1")
        
    X = dict()
    y = dict()
    
    X_train_sets = dict()
    y_train_sets = dict()
    
    X_test_sets = dict()
    y_test_sets = dict()
    
    # Split data into training and test data
    
    for city in cities:
            
        with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                asdf = pickle.load(file)
            
        data = bs.Data(city, year)
        
        # traf_mats = data.pickle_daily_traffic(holidays=False, 
        #                                       user_type='Subscriber',
        #                                       overwrite=True)
        
        traf_df_b = data.daily_traffic_average_all(period='b', holidays=False, 
                                             user_type='Subscriber')
        
        traf_mat_b = np.concatenate((traf_df_b[0].to_numpy(), 
                                     traf_df_b[1].to_numpy()),
                                    axis=1)
        
        traf_df_w = data.daily_traffic_average_all(period='w', holidays=False, 
                                             user_type='Subscriber')
        
        traf_mat_w = np.concatenate((traf_df_w[0].to_numpy(), 
                                     traf_df_w[1].to_numpy()),
                                    axis=1)
        
        
        traf_mats = (traf_mat_b, traf_mat_w)
        
        casual_stations = set(list(asdf.stat_id.unique())) - (set(list(traf_df_b[0].index)) | set(list(traf_df_b[1].index)))
        
        for station in list(casual_stations):
            asdf = asdf[asdf['stat_id'] != station]
        
        asdf = asdf.reset_index()
        
        asdf = ipu.get_clusters(traf_mats, asdf, 'business_days', 10, 'k_means', k, 42)[0]
        
        zone_columns = ['percent_residential', 'percent_commercial',
                        'percent_recreational', 'percent_industrial']
        
        for column in omit_columns[data.city]:
            if column in zone_columns:
                zone_columns.remove(column)
    
        other_columns = ['pop_density', 'nearest_transit_dist', 'b_trips']
        
        for column in omit_columns[data.city]:
            if column in other_columns:
                other_columns.remove(column)
        
        lr_results, X[city], y[city] = ipu.stations_logistic_regression(
            asdf, zone_columns, other_columns, 
            use_points_or_percents='percents', 
            make_points_by='station land use', const=False, return_scaled=True)


        if test_seed:
            if isinstance(test_seed, int):
                np.random.seed(test_seed)
            else:
                raise ValueError("test_seed must be an integer")
        

        mask = np.random.rand(len(X[city])) < test_ratio
        
        X_test_sets[city] = X[city][mask]
        y_test_sets[city] = y[city][mask]
        
        X_train_sets[city] = X[city][~mask]
        y_train_sets[city] = y[city][~mask]
        
    # Train and test models
    
    sr_mat = np.zeros(shape=(len(cities), len(cities)))
    
    for i, city_train in enumerate(cities):
        for j, city_test in enumerate(cities):
            
            if (city := city_train) == city_test:
                
                X_train = X_train_sets[city]
                y_train = y_train_sets[city]
                
                X_test = X_test_sets[city]
                y_test = y_test_sets[city]
                
            else:
                X_train = X[city_train]
                y_train = y[city_train]
                
                X_test = X[city_test]
                y_test = y[city_test]
            
            
            
            if res == 'success_rates':
                sr_mat[i,j] = ipu.logistic_regression_test(X_train, y_train, 
                                                           X_test, y_test, 
                                                           plot_cm=False)[0]
            
            elif res == 'confusion':
                cm = ipu.logistic_regression_test(X_train, y_train, 
                                                           X_test, y_test, 
                                                           plot_cm=False,
                                                           normalise_cm='all')[1]
                sr_mat[i,j] = 1 - cm[1,2] - cm[2,1]
    
    plt.style.use('default')
    
    fig, ax = plt.subplots()
    
    im = ax.matshow(sr_mat, cmap=plt.cm.Blues, vmin=0, vmax=1)
    
    fig.colorbar(im, label='Success rate')
    
    for i in range(len(cities)):
        for j in range(len(cities)):
            c = sr_mat[j,i]
                
            if res == 'confusion':
                ax.text(i, j, f'{c:.2f}', va='center', ha='center', c='white')
            else:
                ax.text(i, j, f'{c:.2f}', va='center', ha='center')
    
    city_names = [bs.name_dict[city] for city in cities]
    
    ax.set_xticks(range(len(cities)))
    ax.set_yticks(range(len(cities)))
    ax.set_xticklabels(city_names)
    ax.set_yticklabels(city_names)
    ax.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="left", va="center",rotation_mode="anchor")
    
    ax.set_xlabel('Test city')
    ax.set_ylabel('Train city')
    plt.tight_layout()
    
    if res == 'confusion':
        plt.savefig('./figures/paper_figures/city_tests_conf.pdf')
    
    else:
        plt.savefig('./figures/paper_figures/city_tests.pdf')
    
    return sr_mat

def n_table_formatter(x):
    
    if isinstance(x, tuple):
        n, p = x
        return f"\\multirow{{2}}{{*}}{{\\shortstack{{${n}$\\\$({p}\%)$}}}}"
    
    elif isinstance(x, str):
        return f"\\multirow{{2}}{{*}}{{{x}}}"
    
    else:
        return ""

def plot_cluster_centers(city, k=3, year=2019, month=None, day=None, n_table=False):
    if city != 'all':
        with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
            asdf = pickle.load(file)
        
        data = bs.Data(city, year, month, day)
    
        traf_df_b = data.daily_traffic_average_all(period='b', holidays=False, 
                                             user_type='Subscriber')
        
        traf_mat_b = np.concatenate((traf_df_b[0].to_numpy(), 
                                     traf_df_b[1].to_numpy()),
                                    axis=1)
        
        traf_df_w = data.daily_traffic_average_all(period='w', holidays=False, 
                                             user_type='Subscriber')
        
        traf_mat_w = np.concatenate((traf_df_w[0].to_numpy(), 
                                     traf_df_w[1].to_numpy()),
                                    axis=1)
        
        
        traf_mats = (traf_mat_b, traf_mat_w)
        
        casual_stations = set(list(asdf.stat_id.unique())) - (set(list(traf_df_b[0].index)) | set(list(traf_df_b[1].index)))
        
        for station in list(casual_stations):
            asdf = asdf[asdf['stat_id'] != station]
        
        asdf = asdf.reset_index()
        
        asdf, clusters, labels = ipu.get_clusters(traf_mats, asdf, 'business_days', 10, 'k_means', k, 42)
        
        plt.style.use('seaborn-darkgrid')
        
        fig, ax = plt.subplots()
        
        for i in range(k):
            n = (labels==i).sum()
            ax.plot(clusters.cluster_centers_[i], label=f'Cluster {i} (n={n})')
        ax.set_xticks(range(24))
        ax.set_xlabel('Hour')
        ax.set_xlim(0,23)
        ax.set_ylim(-0.125,0.125)
        ax.set_ylabel('Relative difference')
        ax.legend()
        
        plt.savefig(f'./figures/paper_figures/{city}_clusters.pdf')
        
        plt.style.use('default')
    
        return clusters

    else:
        
        cities = np.array([['nyc', 'chicago'], 
                           ['washdc', 'boston'], 
                           ['london', 'helsinki'], 
                           ['oslo', 'madrid']])
        
        cluster_name_dict = {0 : 'Cluster 0', 
                             1 : 'Cluster 1', 
                             2 : 'Cluster 2',
                             3 : 'Cluster 3',
                             4 : 'Cluster 4',
                             5 : 'Cluster 5',
                             6 : 'Cluster 6',
                             7 : 'Cluster 7',
                             8 : 'Cluster 8',
                             9 : 'Cluster 9',
                             10 : 'Cluster 10',}
       
        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10,10))
        plt.style.use('seaborn-darkgrid')
        
        clusters_dict = dict()
        
        multiindex = pd.MultiIndex.from_product((list(cities.flatten()), ['n', 'p']), names=['city', 'number']) 
        n_df = pd.DataFrame(index=multiindex, columns=['city'] +list(range(k)))
        
        for row in range(4):
            for col in range(2):
        
                city = cities[row,col]
                
                with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                    asdf = pickle.load(file)
                
                data = bs.Data(city, year, month, day)
            
                traf_df_b = data.daily_traffic_average_all(period='b', holidays=False, 
                                                     user_type='Subscriber')
                
                traf_mat_b = np.concatenate((traf_df_b[0].to_numpy(), 
                                             traf_df_b[1].to_numpy()),
                                            axis=1)
                
                traf_df_w = data.daily_traffic_average_all(period='w', holidays=False, 
                                                     user_type='Subscriber')
                
                traf_mat_w = np.concatenate((traf_df_w[0].to_numpy(), 
                                             traf_df_w[1].to_numpy()),
                                            axis=1)
                
                
                traf_mats = (traf_mat_b, traf_mat_w)
                
                casual_stations = set(list(asdf.stat_id.unique())) - (set(list(traf_df_b[0].index)) | set(list(traf_df_b[1].index)))
                
                for station in list(casual_stations):
                    asdf = asdf[asdf['stat_id'] != station]
                
                asdf = asdf.reset_index()
                
                asdf, clusters, labels = ipu.get_clusters(traf_mats, asdf, 'business_days', 10, 'k_means', k, 42)
                
                clusters_dict[city] = clusters
                
                # Make figure
                
                for i in range(k):
                    ax[row,col].plot(clusters[i], label=cluster_name_dict[i])
                
                ax[row,col].set_xticks(range(24))
                ax[row,col].set_xlim(0,23)
                ax[row,col].set_ylim(-0.15,0.15)
                
                # if row != 3:
                #     ax[row,col].xaxis.set_ticklabels([])
                # else:
                #     ax[row,col].set_xlabel('Hour')
                
                if row == 3:
                    ax[row,col].set_xlabel('Hour')
                
                if col == 1:
                    ax[row,col].yaxis.set_ticklabels([])
                else:
                    ax[row,col].set_ylabel('Relative difference')
                
                ax[row,col].set_title(bs.name_dict[city])
                
                # Update n_df
                
                n_total = (~labels.isna()).sum()
                n_df.loc[(city, 'n'), 'city'] = bs.name_dict[city]
                n_df.loc[(city, 'p'), 'city'] = ''
                for i in range(k):
                    n = (labels==i).sum()
                    n_df.loc[(city, 'n'), i] = (n, np.round(n/n_total*100, 1))
                    n_df.loc[(city, 'p'), i] = ''
                    
                
                    
        # Print figure        
                
        plt.tight_layout(pad=2)
        ax[3,0].legend(loc='upper center', bbox_to_anchor=(1,-0.2), ncol=len(ax[3,0].get_lines()))
        
        try:
            plt.savefig(f'./figures/paper_figures/clusters_all_cities_k={k}.pdf')
        except PermissionError:
            print('Permission Denied. Continuing...')
        
        # plt.style.use('default')
        
        if n_table:
    
            # Print n_df
            # n_df = n_df.assign(help='').set_index('help',append=True)
            # n_df = n_df.droplevel(level=1)
            latex_table = n_df.to_latex(column_format='@{}l'+('r'*(len(n_df.columns)-1)) + '@{}', 
                                           index=False, 
                                           formatters = [n_table_formatter]*len(n_df.columns), 
                                           escape=False)
        
            print(latex_table)
        
            return clusters_dict, n_df
        
        else:
            return clusters_dict


def k_test_table(cities=None, year=2019, month=None, k_min=2, k_max=10, 
                 cluster_seed=42, plot_figures=False, overwrite=False):
    
    if cities is None:
            cities = ['nyc', 'chicago', 'washdc', 'boston', 
                      'london', 'helsinki', 'oslo', 'madrid'] 
    
    metrics = ['DB', 'D', 'S', 'SS']
        
    k_list = [i for i in range(k_min, k_max+1)]
    
    if not overwrite:
        try:
            with open('./python_variables/k_table.pickle', 'rb') as file:
                res_table = pickle.load(file)
        
        except FileNotFoundError:
            print('No pickle found. Making pickle...')
            res_table = k_test_table(cities=cities, year=year, month=month, 
                                     k_min=k_min, k_max=k_max, 
                                     cluster_seed=cluster_seed, plot_figures=plot_figures, 
                                     overwrite=True)
        
    else:
        
        multiindex = pd.MultiIndex.from_product((cities, metrics))  
        
        res_table = pd.DataFrame(index=k_list, columns=multiindex)
        
        for city in cities:
            
            with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                asdf = pickle.load(file)
            
            data = bs.Data(city, year, month)
            
            traf_df_b = data.daily_traffic_average_all(period='b', holidays=False, 
                                                 user_type='Subscriber')
            
            traf_mat_b = np.concatenate((traf_df_b[0].to_numpy(), 
                                         traf_df_b[1].to_numpy()),
                                        axis=1)
            
            traf_df_w = data.daily_traffic_average_all(period='w', holidays=False, 
                                                 user_type='Subscriber')
            
            traf_mat_w = np.concatenate((traf_df_w[0].to_numpy(), 
                                         traf_df_w[1].to_numpy()),
                                        axis=1)
            
            
            traf_mats = (traf_mat_b, traf_mat_w)
            
            casual_stations = set(list(asdf.stat_id.unique())) - (set(list(traf_df_b[0].index)) | set(list(traf_df_b[1].index)))
            
            for station in list(casual_stations):
                asdf = asdf[asdf['stat_id'] != station]
            
            asdf = asdf.reset_index()
            
            DB_list = []
            D_list = []
            S_list = []
            SS_list = []
            
            for k in k_list:
                
                print(f'\nCalculating for k={k}...\n')
                
                asdf, clusters, labels = ipu.get_clusters(traf_mats, asdf, 'business_days', 10, 'k_means', k, cluster_seed)
                
                mask = ~labels.isna()
                
                labels = labels.to_numpy()[mask]
                
                data_mat = (traf_mat_b[:,:24] - traf_mat_b[:,24:])[mask]
                
                DB_list.append(ipu.get_Davies_Bouldin_index(data_mat, 
                                                            clusters.cluster_centers_,
                                                            labels,
                                                            verbose=True))
                
                D_list.append(ipu.get_Dunn_index(data_mat, 
                                                 clusters.cluster_centers_,
                                                 labels,
                                                 verbose=True))
                
                S_list.append(ipu.get_silhouette_index(data_mat, 
                                                       clusters.cluster_centers_,
                                                       labels,
                                                       verbose=True))
                
                SS_list.append(clusters.inertia_)
                
            res_table[(city, 'DB')] = DB_list
            res_table[(city, 'D')] = D_list
            res_table[(city, 'S')] = S_list
            res_table[(city, 'SS')] = SS_list
        
        res_table = res_table.rename(columns=bs.name_dict)
        
        with open('./python_variables/k_table.pickle', 'wb') as file:
            pickle.dump(res_table, file)
        
    print(res_table.to_latex(column_format='@{}l'+('r'*len(res_table.columns)) + '@{}',
                             index=True, na_rep = '--', float_format='%.3f',
                             multirow=True, multicolumn=True, multicolumn_format='c'))
    
    if plot_figures:
        plt.style.use('seaborn-darkgrid')
        
        metrics_dict = {
                        'D' : 'Dunn Index (higher is better)',
                        'S' : 'Silhouette Index (higher is better)',
                        'DB' : 'Davies-Bouldin index (lower is better)',
                        'SS' : 'Sum of Squares (lower is better)'}
        
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
        
        count=0
        for row in range(2):
            for col in range(2):
                for city in cities:
                    ax[row, col].plot(res_table[(bs.name_dict[city], 
                                                 list(metrics_dict.keys())[count])],
                                      label=bs.name_dict[city])
                    ax[row,col].set_title(metrics_dict[
                        list(metrics_dict.keys())[count]])
                    
                    # if row == 0:
                    #     ax[row,col].xaxis.set_ticklabels([])
                    
                    # else:
                    ax[row,col].set_xlabel('k')
                count+=1
                
        plt.tight_layout(pad=2)
        ax[1,0].legend(loc='upper center', bbox_to_anchor=(1.05,-0.1), ncol=len(ax[0,0].get_lines()))
        
        plt.savefig('figures/paper_figures/k_test_figures.pdf')
        
    return res_table
        
if __name__ == "__main__":
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
              'london', 'helsinki', 'oslo', 'madrid']
    
    # sum_stat_table=make_summary_statistics_table(print_only=True)
    # LR_table=make_LR_table(2019)
    # k_table = k_test_table(plot_figures=True)
    # sr = city_tests(k=5)
    
    clusters, n_table = plot_cluster_centers('all',k=3, n_table=True)
    
    # clusters_list = []
    # for k in [2,3,4,5,6,7,8,9,10]:
        # clusters_list.append(plot_cluster_centers('all', k=k))
    #     plt.close()
    
   
    
    
