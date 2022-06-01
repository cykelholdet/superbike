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
import time
import pickle
import contextily as cx
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from holoviews import opts

from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.offsetbox import AnchoredText, AnchoredOffsetbox
from matplotlib import patheffects
from geopy.distance import geodesic

import bikeshare as bs
import interactive_plot_utils as ipu
from logistic_table import lr_coefficients
from clustering import get_clusters

def service_area_figure(city, year, month, day, return_fig=False):
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
    
    data = bs.Data(city, year, month, day)
    
    stat_df= ipu.make_station_df(data, holidays=False)
    
    # extend = (stat_df['lat'].min(), stat_df['long'].min(), 
    #           stat_df['lat'].max(), stat_df['long'].max())
    
    # tileserver = 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg' # Stamen Terrain
    # # tileserver = 'http://a.tile.stamen.com/toner/{z}/{x}/{y}.png' # Stamen Toner
    # # tileserver = 'http://c.tile.stamen.com/watercolor/{z}/{x}/{y}.png' # Stamen Watercolor
    # # tileserver = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png' # OSM Default
    
    # m = sm.Map(extend, tileserver=tileserver)
    
    fig, ax = plt.subplots(figsize=(7,10))
    
    # m.show_mpl(ax=ax)
    # fig.add_axes(ax)
    ax.axis('off')
    
    stat_df['service_area'].to_crs(data.laea_crs).plot(ax=ax, alpha=0.5, facecolor='tab:cyan', edgecolor='k', label='banana')
    stat_df.to_crs(data.laea_crs).plot(color='b', ax=ax,  markersize=2.5, label='Station')
    
    scalebar = AnchoredSizeBar(ax.transData, 1000, 
                                f'{1} km', 'lower right', 
                                pad=1, color='black', frameon=False, 
                                size_vertical=50)
    ax.add_artist(scalebar)
    
    cx.add_basemap(ax, crs=data.laea_crs,
                   attribution="(C) Stamen Design, (C) OpenStreetMap Contributors",
                   )
    
    p0 = Polygon([(0,0), (0,1), (1,0)], alpha=0.5, 
                  facecolor='tab:cyan', edgecolor='k', label='Service Area')
    ax.add_patch(p0)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'./figures/paper_figures/service_areas_{data.city}_{data.year}{data.month:02d}.pdf', dpi=300)
      
    
    # for i, stat in stat_df.iterrows():
    #     x, y = m.to_pixels(stat['lat'], stat['long'])
    #     ax.plot(x, y, 'ob', ms=2, mew=1.5)
        
    #     if isinstance(stat['service_area'], shapely.geometry.multipolygon.MultiPolygon):
            
    #         for poly in stat['service_area']:
    #             coords = np.zeros(shape=(len(poly.exterior.xy[0]),2))    
                
    #             for j in range(len(coords)):
    #                 coords[j,0] = poly.exterior.xy[1][j]
    #                 coords[j,1] = poly.exterior.xy[0][j]
                
    #             pixels = m.to_pixels(coords)
                
    #             p = Polygon(list(zip(pixels[:,0], pixels[:,1])), 
    #                         alpha=0.5, facecolor='tab:cyan', edgecolor='k')
            
    #             ax.add_artist(p)
                
    #             # ax.plot(pixels[:,0], pixels[:,1], c='k')
                
    #     else:
    #         coords = np.zeros(shape=(len(stat['service_area'].exterior.xy[0]),2))    
    #         # coords = []    
            
    #         for j in range(len(stat['service_area'].exterior.xy[0])):
    #             coords[j,0] = stat['service_area'].exterior.xy[1][j]
    #             coords[j,1] = stat['service_area'].exterior.xy[0][j]
                
    #             # coords.append((stat['service_area'].exterior.xy[0][j], 
    #             #                stat['service_area'].exterior.xy[1][j]))
            
    #         pixels = m.to_pixels(coords)
            
    #         p = Polygon(list(zip(pixels[:,0], pixels[:,1])), 
    #                     alpha=0.5, facecolor='tab:cyan', edgecolor='k')
            
    #         ax.add_artist(p)
            
    #         # ax.fill(pixels[:,0], pixels[:,1], c='b')
    #         # ax.plot(pixels[:,0], pixels[:,1], c='k')
            
    # xlim_dict = {'nyc' : (150, 600)}
    # ylim_dict = {'nyc' : (767.5,100)}
    
    
    # if data.city in xlim_dict.keys():
    #     ax.set_xlim(xlim_dict[data.city])
    
    # if data.city in ylim_dict.keys():
    #     ax.set_ylim(ylim_dict[data.city])
    
    # ax.plot(0, 0, 'ob', ms=2, mew=1.5, label='Station')
    # p0 = Polygon([(0,0), (0,1), (1,0)], alpha=0.5, 
    #               facecolor='tab:cyan', edgecolor='k', label='Service Area')
    # ax.add_patch(p0)
    # ax.legend()
    
    # scalebar_size_km = 5
    
    # c0 = (stat_df.iloc[0].easting, stat_df.iloc[0].northing)
    # c1 = (stat_df.iloc[1].easting, stat_df.iloc[1].northing)
    
    # geo_dist = np.linalg.norm(np.array(c0) - np.array(c1))
    
    # pix0 = m.to_pixels(stat_df.iloc[0].lat, stat_df.iloc[0].long)
    # pix1 = m.to_pixels(stat_df.iloc[1].lat, stat_df.iloc[1].long)
    
    # pix_dist = np.linalg.norm(np.array(pix0) - np.array(pix1))
    
    # scalebar_size = pix_dist/geo_dist*1000*scalebar_size_km
    
    
    # scalebar = AnchoredSizeBar(ax.transData, scalebar_size, 
    #                             f'{scalebar_size_km} km', 'lower right', 
    #                             pad=0.2, color='black', frameon=False, 
    #                             size_vertical=2)
    # ax.add_artist(scalebar)
    # attr = AnchoredText("(C) Stamen Design. (C) OpenStreetMap contributors.",
    #                     loc = 'lower left', frameon=True, pad=0.1, borderpad=0)
    # attr.patch.set_edgecolor('white')
    # ax.add_artist(attr)
    
    # ax.axis('off')    
    
    # plt.tight_layout()
    # plt.savefig(f'./figures/paper_figures/service_areas_{data.city}_{data.year}{data.month:02d}.pdf')
        
    if return_fig:
        return fig, ax


def daily_traffic_figure(stat_id, city, year, month=None, day=None, traffic_type = 'traffic',
                         day_type='business_days', normalise=True, user_type='Subscriber', std=True,
                         return_fig=False, savefig=False):
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
    
    data = bs.Data(city, year, month, day)
    
    
    if std:
        traf_mat, std_mat = data.pickle_daily_traffic(holidays=False, 
                                                    user_type=user_type,
                                                    day_type=day_type,
                                                    overwrite=False,
                                                    normalise=normalise,
                                                    return_std = std)
    else:
        traf_mat = data.pickle_daily_traffic(holidays=False, 
                                              user_type=user_type,
                                              day_type=day_type,
                                              overwrite=False,
                                              normalise=normalise,
                                              return_std = std)
        
    if traffic_type == 'traffic':
        
        departures = traf_mat[data.stat.id_index[stat_id]][:24]
        arrivals = traf_mat[data.stat.id_index[stat_id]][24:]
        
        plt.style.use('seaborn-darkgrid')
        
        fig, ax = plt.subplots(figsize=(10,5))
        
        if normalise:
            ax.plot(np.arange(24), departures*100, label='departures')
            ax.plot(np.arange(24), arrivals*100, label='arrivals')
            
            if std:
                ax.fill_between(np.arange(24), 
                                departures*100-std_mat[data.stat.id_index[stat_id]][:24]*100,
                                departures*100+std_mat[data.stat.id_index[stat_id]][:24]*100,
                                facecolor='tab:blue', alpha=0.2)
                ax.fill_between(np.arange(24), 
                                arrivals*100-std_mat[data.stat.id_index[stat_id]][:24]*100,
                                arrivals*100+std_mat[data.stat.id_index[stat_id]][:24]*100,
                                facecolor='tab:orange', alpha=0.2)
            
            ax.set_ylabel('% of total trips')

        else:
            ax.plot(np.arange(24), departures, label='departures')
            ax.plot(np.arange(24), arrivals, label='arrivals')
            
            if std:
                ax.fill_between(np.arange(24), 
                                departures-std_mat[data.stat.id_index[stat_id]][:24],
                                departures+std_mat[data.stat.id_index[stat_id]][:24],
                                facecolor='tab:blue', alpha=0.2)
                ax.fill_between(np.arange(24), 
                                arrivals-std_mat[data.stat.id_index[stat_id]][:24],
                                arrivals+std_mat[data.stat.id_index[stat_id]][:24],
                                facecolor='tab:orange', alpha=0.2)
            
            ax.set_ylabel('# trips')
        
        ax.legend()
        
    elif traffic_type == 'difference':
        
        departures = traf_mat[data.stat.id_index[stat_id]][:24]
        arrivals = traf_mat[data.stat.id_index[stat_id]][24:]
        
        diff = departures-arrivals
        
        plt.style.use('seaborn-darkgrid')
        fig, ax = plt.subplots(figsize=(10,5))
        
        if normalise:
            ax.plot(np.arange(24), diff)
            ax.set_ylabel('Relative difference')

        else:
            ax.plot(np.arange(24), diff)
            ax.set_ylabel('Absolute difference')  
        
    else:
        raise ValueError("Please provide either 'traffic' or 'difference' as traffic_type")
    
    ax.set_xticks(np.arange(24))
    ax.set_xlabel('Hour')
    
    ax.set_title(f'Average daily traffic for {data.stat.names[data.stat.id_index[stat_id]]} (ID: {stat_id})')
    
    plt.tight_layout()
    
    
    if savefig:
        
        if month is None and day is None:
            filestr = f'./figures/{city}{year}_{stat_id}_average_daily_traffic.pdf'
        elif day is None:
            filestr = f'./figures/{city}{year}{month:02d}_{stat_id}_average_daily_traffic.pdf'
        else:
            filestr = f'./figures/{city}{year}{month:02d}{day:02d}_{stat_id}_average_daily_traffic.pdf'
    
        plt.savefig(filestr)
    
    if return_fig:
        return fig, ax
        

def make_summary_statistics_table(cities=None, variables=None, year=2019):
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
    
    if variables is None:
        variables = variables_list
    
    multiindex = pd.MultiIndex.from_product((cities, 
                                             ['Mean', 'Std. Dev.', 'Min.', 'Max.']))   
    
    tab_df = pd.DataFrame(index=variables, columns = multiindex)
    
    for city_index, city in enumerate(cities):
        try:
            with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                asdf = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')
        
        asdf=asdf[asdf.b_trips >=8]
        
        # asdf['pop_density'] =  asdf['pop_density']/10000 # convert to population per 100 m^2
        # asdf['nearest_subway_dist']  =  asdf['nearest_subway_dist']/1000 # convert to km
        # asdf['nearest_railway_dist']  =  asdf['nearest_railway_dist']/1000 # convert to km
        
        # asdf['center_dist']  =  asdf['center_dist']/1000 # convert to km
        # avg_stat_df['n_trips'] = avg_stat_df['n_trips']/avg_stat_df['n_trips'].sum() # convert to percentage
        # avg_stat_df['b_trips'] = avg_stat_df['b_trips']/avg_stat_df['b_trips'].sum() # convert to percentage
        # avg_stat_df['w_trips'] = avg_stat_df['w_trips']/avg_stat_df['w_trips'].sum() # convert to percentage
        
        
        # city_df = pd.DataFrame(columns=['Variable', 'Mean', 
        #                                 'Std. Dev.', 'Min', 'Max'],
        #                        index=variables)
        # city_df['City'] = bs.name_dict[city]
        # city_df['Variable'] = variables
        tab_df[(city, 'Mean')] =  asdf.mean()
        tab_df[(city, 'Std. Dev.')] =  asdf.std()
        tab_df[(city, 'Min.')] =  asdf.min()
        tab_df[(city, 'Max.')] =  asdf.max()
        
        # tab_df = pd.concat([tab_df, city_df])
    
    
    # tab_df = tab_df.replace(var_renames)
    
    # city_names = [bs.name_dict[city] for city in cities]
    
    tab_df = tab_df.rename(index=variables_dict, columns=bs.name_dict)
    
    print(tab_df.to_latex(column_format='@{}l'+('r'*len(tab_df.columns)) + '@{}',
                          index=True, na_rep = '--', float_format='%.2f',
                          multirow=True, multicolumn=True, multicolumn_format='c'))
    
    return tab_df


def make_LR_table(year=2019, k=5, const=True, method = 'LR'):
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
    
    omit_columns = {
        'boston': ['percent_educational', 'percent_UNKNOWN', 'n_trips', 'b_trips'],
        'chicago': ['percent_transportation', 'percent_UNKNOWN', 'n_trips', 'b_trips'],
        'nyc': ['percent_mixed', 'n_trips', 'b_trips'],
        'washdc': ['percent_transportation', 'percent_UNKNOWN', 'n_trips', 'b_trips'],
        'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'n_trips', 'b_trips'],
        'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips', 'b_trips'],
        'madrid': ['percent_industrial', 'n_trips', 'b_trips'],
        'oslo': ['percent_transportation', 'percent_UNKNOWN', 'n_trips', 'b_trips'],
        'USA': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'b_trips',],
        'EUR': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'b_trips',],
        'All': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'b_trips',],
        }
    
    month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
          7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None'}
    
    if method not in (method_list := ['LR', 'OLS']):
        raise ValueError(f'Please provide a method from {method_list}')
    
    table = pd.DataFrame([])
    
    for city in cities:
        # print(city)
        try:
            with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                asdf = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
    
        data = bs.Data(city, year)
        
        traf_mat = data.pickle_daily_traffic(holidays=False,
                                              # user_type='Subscriber',
                                              day_type='business_days',
                                              overwrite=False)
                
        mask = ~asdf['n_trips'].isna()
        
        asdf = asdf[mask]
        asdf = asdf.reset_index(drop=True)
        
        try:
            traf_mat = traf_mat[mask]
        except IndexError:
            pass
        
        asdf, clusters, labels = get_clusters(traf_mat, asdf, 'business_days', 8, 'k_means', k, 42)
        
        zone_columns = [var for var in variables_list if 'percent' in var]
        
        for column in omit_columns[data.city]:
            if column in zone_columns:
                zone_columns.remove(column)
        
        other_columns = [var for var in variables_list if 'percent' not in var]
        
        for column in omit_columns[data.city]:
            if column in other_columns:
                other_columns.remove(column)
        
        if method == 'LR':
            lr_results, X, y, _, cm = ipu.stations_logistic_regression(
                asdf, zone_columns, other_columns, 
                use_points_or_percents='percents', 
                make_points_by='station land use', const=const,
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
        
            

        elif method == 'OLS':
            lr_results = ipu.linear_regression(asdf, [*zone_columns, *other_columns], 'b_trips')
            print(lr_results.summary())
            
            parameters = lr_results.params
            stdev = lr_results.bse
            pvalues = lr_results.pvalues
            
            multiindex = lr_results.params.index
            
            pars = pd.Series(parameters, index=multiindex, name='coef')
            sts = pd.Series(stdev, index=multiindex, name='stdev')
            pvs = pd.Series(pvalues, index=multiindex, name='pvalues')
            
            coefs = pd.DataFrame(pars).join((sts, pvs))
                
    
            lr_coefs = pd.concat({data.city: coefs}, names="", axis=1)
    
            table = pd.concat([table, lr_coefs], axis=1)        
        
    # Make table nice
    
    if method == 'LR':
        
        signif_table = table.xs('pvalues', level=1, axis=1) < 0.05
        
        coeftable = table.xs('coef', level=1, axis=1)
        
        tuple_table = pd.concat([coeftable,signif_table]).stack(dropna=False).groupby(level=[0,1,2]).apply(tuple).unstack()

        index_list = list(variables_dict.keys())
        
        index_renamer = variables_dict
        column_renamer = dict(zip(cities, [bs.name_dict[city] for city in cities]))
        
        if const:
            index_list.insert(0, 'Const.')
        
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
    
    elif method == 'OLS':
        
        signif_table = table.xs('pvalues', level=1, axis=1) < 0.05
        
        coeftable = table.xs('coef', level=1, axis=1)

        tuple_table = pd.concat([coeftable,signif_table]).stack(dropna=False).groupby(level=[0,1]).apply(tuple).unstack()

        index_list = list(variables_dict.keys())
        
        index_renamer = variables_dict
        column_renamer = dict(zip(cities, [bs.name_dict[city] for city in cities]))

        if const:
            index_list.insert(0, 'Const.')

        index_list = [x for x in index_list if x in table.index.get_level_values(0)]
        
        tuple_table = tuple_table.loc[index_list]
    
        tuple_table = tuple_table.reindex(columns=cities)
        tuple_table = tuple_table.rename(index=index_renamer, columns=column_renamer)
        tuple_table.index = tuple_table.index.rename('Coef. name')
        
        latex_table = tuple_table.to_latex(column_format='@{}l'+('r'*len(tuple_table.columns)) + '@{}', formatters = [tuple_formatter]*len(tuple_table.columns), escape=False)
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
    

def city_tests(year=2019, cities=None, k=5, test_ratio=0.2, test_seed=42,
               res='success_rates'):
    """
    
    1. for each city: split data into training and test sets and train LR model.
    2. test eachh model on the 8 test sets
    3. show succes rates in a nice figure
    
    bonus: redefine succes rate as rate at which a model does not confuse
           morning sinks and morning sources.
    
    
    
    """
    
    omit_columns = {
        'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips', 'b_trips'],
        'chicago': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips', 'b_trips'],
        'nyc': ['percent_mixed', 'n_trips', 'b_trips'],
        'washdc': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips', 'b_trips'],
        'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips', 'b_trips'],
        'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips', 'percent_industrial', 'b_trips'],
        'madrid': ['n_trips', 'percent_industrial', 'b_trips'],
        'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips', 'b_trips', 'percent_mixed'],
        'USA': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'b_trips', 'percent_mixed'],
        'EUR': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'b_trips', 'percent_mixed'],
        'All': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'b_trips', 'percent_mixed'],
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
        try:
            with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                asdf = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
    
            
        data = bs.Data(city, year)
        
        traf_mat = data.pickle_daily_traffic(holidays=False, 
                                              user_type='Subscriber',
                                              day_type='business_days',
                                              overwrite=False)
                
        mask = ~asdf['n_trips'].isna()
        
        asdf = asdf[mask]
        asdf = asdf.reset_index(drop=True)
        
        try:
            traf_mat = traf_mat[mask]
        except IndexError:
            pass
        
        asdf = get_clusters(traf_mat, asdf, 'business_days', 8, 'k_means', 
                                k, random_state=42, use_dtw=True, city=city)[0]
        
        zone_columns = [var for var in variables_list if 'percent' in var]
        
        for column in omit_columns[data.city]:
            if column in zone_columns:
                zone_columns.remove(column)
    
        other_columns = [var for var in variables_list if 'percent' not in var]
        
        for column in omit_columns[data.city]:
            if column in other_columns:
                other_columns.remove(column)
        
        lr_results, X[city], y[city] = ipu.stations_logistic_regression(
            asdf, zone_columns, other_columns, 
            use_points_or_percents='percents', 
            make_points_by='station land use', 
            plot_cm=True, const=True, return_scaled=True)


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
                
            # if res == 'confusion':
            #     ax.text(i, j, f'{c:.2f}', va='center', ha='center', c='white')
            # else:
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

def train_test_cm(city_train, city_test, k=5, year=2019, const=True, test_seed=42, savefig=True):
    
    omit_columns = {
        'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips', 'b_trips'],
        'chicago': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips', 'b_trips'],
        'nyc': ['percent_mixed', 'n_trips', 'b_trips'],
        'washdc': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips', 'b_trips'],
        'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips', 'b_trips'],
        'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips', 'percent_industrial', 'b_trips'],
        'madrid': ['n_trips', 'percent_industrial', 'b_trips'],
        'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips', 'b_trips', 'percent_mixed'],
        'USA': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'b_trips', 'percent_mixed'],
        'EUR': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'b_trips', 'percent_mixed'],
        'All': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'b_trips', 'percent_mixed'],
        }
    if city_train == city_test:
        city = city_train
        
        try:
            with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                asdf = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
    
            
        data = bs.Data(city, year)
        
        traf_mat = data.pickle_daily_traffic(holidays=False, 
                                              user_type='Subscriber',
                                              day_type='business_days',
                                              overwrite=False)
                
        mask = ~asdf['n_trips'].isna()
        
        asdf = asdf[mask]
        asdf = asdf.reset_index(drop=True)
        
        try:
            traf_mat = traf_mat[mask]
        except IndexError:
            pass
        
        asdf = get_clusters(traf_mat, asdf, 'business_days', 8, 'k_means', k, 0)[0]
        
        zone_columns = [var for var in variables_list if 'percent' in var]
        
        for column in omit_columns[data.city]:
            if column in zone_columns:
                zone_columns.remove(column)
    
        other_columns = [var for var in variables_list if 'percent' not in var]
        
        for column in omit_columns[data.city]:
            if column in other_columns:
                other_columns.remove(column)
        
        lr_results, X, y, predictions, cm = ipu.stations_logistic_regression(
            asdf, zone_columns, other_columns, 
            use_points_or_percents='percents', 
            make_points_by='station land use', 
            test_model=True, test_seed=test_seed, plot_cm=True, normalise_cm='true',
            const=const, return_scaled=True)
        
        if savefig:
            plt.savefig(f'./figures/paper_figures/cm_{year}_{city}_{city}.pdf')
        
    else:
        
        X = dict()
        y = dict()
        
        for city in [city_train, city_test]:
            try:
                with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                    asdf = pickle.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
        
            data = bs.Data(city, year)
            
            traf_mat = data.pickle_daily_traffic(holidays=False, 
                                                  user_type='Subscriber',
                                                  day_type='business_days',
                                                  overwrite=False)
                    
            mask = ~asdf['n_trips'].isna()
            
            asdf = asdf[mask]
            asdf = asdf.reset_index(drop=True)
            
            try:
                traf_mat = traf_mat[mask]
            except IndexError:
                pass
            
            asdf = get_clusters(traf_mat, asdf, 'business_days', 8, 'k_means', k, 0)[0]
            
            zone_columns = [var for var in variables_list if 'percent' in var]
            
            for column in omit_columns[data.city]:
                if column in zone_columns:
                    zone_columns.remove(column)
        
            other_columns = [var for var in variables_list if 'percent' not in var]
            
            for column in omit_columns[data.city]:
                if column in other_columns:
                    other_columns.remove(column)
            
            lr_results, X[city], y[city] = ipu.stations_logistic_regression(
                asdf, zone_columns, other_columns, 
                use_points_or_percents='percents', 
                make_points_by='station land use', 
                test_model=False, const=const, return_scaled=True)
    
        success_rate, cm, predictions = ipu.logistic_regression_test(
            X[city_train], y[city_train], X[city_test], y[city_test], 
            plot_cm=True, normalise_cm='true')
        
        if savefig:
            plt.savefig(f'./figures/paper_figures/cm_{year}_{city_train}_{city_test}.pdf')
        
    return cm

def pre_processing_table(cities=None, year=2019, month=None, min_trips=8):
    
    if cities is None:
        cities = ['nyc', 'chicago', 'washdc', 'boston', 
                  'london', 'helsinki', 'oslo', 'madrid'] 

    table_index = [['City', 'Pre-cleaning', 'Pre-cleaning', 'h1', 
                    'Post-cleaning', 'Post-cleaning', 'h2', 
                    'Data Retained (%)', 'Data Retained (%)'],
                   ['','Trips','Stations', '', 'Trips','Stations','',
                    'Trips','Stations']]

    table = pd.DataFrame(index=cities, 
                         columns=pd.MultiIndex.from_arrays(table_index))
    
    table[('City','')] = [bs.name_dict[city] for city in cities]
    table[('h1','')], table[('h2','')] = '', ''
    
    for city in cities:
        
        try:
            with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                asdf = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
            
        data = bs.Data(city, year, month, day_type='business_days',
                       user_type='Subscribers', remove_loops=False,
                       overwrite=True)
        
        table.loc[city, ('Pre-cleaning', 'Trips')] = len(data.df)
        table.loc[city, ('Pre-cleaning', 'Stations')] = len(data.stat.id_index)
        
        data = bs.Data(city, year, month, day_type='business_days', 
                       user_type='Subscriber', remove_loops=True, 
                       overwrite=True)
        
        asdf = asdf[asdf['b_trips'] >= min_trips]
        
        table.loc[city, ('Post-cleaning', 'Trips')] = (
            data.df['start_stat_id'].isin(asdf['stat_id']) | 
                     data.df['end_stat_id'].isin(asdf['stat_id'])).sum()
        table.loc[city, ('Post-cleaning', 'Stations')] = len(asdf)
        
        table.loc[city, ('Data Retained (%)', 'Trips')] = table.loc[city,('Post-cleaning', 'Trips')]/table.loc[city,('Pre-cleaning', 'Trips')]*100
        table.loc[city, ('Data Retained (%)', 'Stations')] = table.loc[city,('Post-cleaning', 'Stations')]/table.loc[city,('Pre-cleaning', 'Stations')]*100
        
        print(table.to_latex(column_format='@{}l'+('r'*(len(table.columns)-1)) + '@{}',
                             index=False, multicolumn=True, multicolumn_format='c',
                             float_format=lambda x: f'{x:.2f}'))
        
    return table

def chaining_effect_fig():
    
    np.random.seed(8)
    
    means = [(1.5,2.5), (3.55,2.45)]
    
    cluster_1 = np.random.multivariate_normal(means[0], 0.1*np.identity(2), 100)
    cluster_2 = np.random.multivariate_normal(means[1], 0.1*np.identity(2), 100)
    
    plt.plot([2.70057, 2.28188], [2.4436, 2.45356], linestyle='--', c='k', zorder=1)
    
    plt.scatter(cluster_1[:,0], cluster_1[:,1], c='r', zorder=2)
    plt.scatter(cluster_2[:,0], cluster_2[:,1], c='b', zorder=2)
    
    
    # plt.scatter(2.70057, 2.4436, c='k')
    # plt.scatter(2.28188, 2.45356, c='k')
    
    
    plt.scatter(4.5,1.5, c='orange')
    plt.scatter(1,4, c='purple')
    
    
    plt.xlim(0,5)
    plt.ylim(0,5)
    
    plt.tight_layout()
    
    plt.savefig('./figures/chaining_effect.pdf')
    
    # plt.axis('off')

def cm_mean_fig(city_train, city_test, plotfig=True, savefig=True):
    
    clusters = ['Reference', 'High morning sink', 'Low morning sink',
                'Low morning source', 'High morning source']
    
    if city_train==city_test:
        seed_range=range(50,75)
        cm = np.zeros(shape=(5,5))
        for seed in seed_range:
            cm += train_test_cm(city_train, city_test, test_seed=seed)
        cm_mean = cm/len(seed_range)
    else:
        cm_mean = train_test_cm(city_train, city_test)
    
    if plotfig:
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_mean)
        disp.plot()
        
        disp.ax_.set_xlabel('Predicted cluster')
        disp.ax_.set_ylabel('True cluster')
        
        disp.ax_.set_xticklabels(clusters)
        disp.ax_.set_yticklabels(clusters)
        
        disp.ax_.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
        plt.setp([tick.label2 for tick in disp.ax_.xaxis.get_major_ticks()], rotation=45,
             ha="left", va="center",rotation_mode="anchor")
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(f'./figures/cm_plots/cm_mean_2019_{city_train}_{city_test}.png')
        
    return cm_mean

variables_list = ['percent_residential', 'percent_commercial',
                  'percent_recreational',
                  'pop_density', 'nearest_subway_dist',
                  'nearest_railway_dist', 'center_dist']

variables_dict = {'percent_residential' : 'Share of residential use',
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
                  'nearest_railway_dist' : 'Distance to nearest railway [km]',
                  'center_dist' : 'Distance to city center [km]'}

if __name__ == "__main__":
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
              'london', 'helsinki', 'oslo', 'madrid']
    
    # chaining_effect_fig()
    
    # fig, ax = daily_traffic_figure(519, 'nyc', 2019, normalise=True, 
    #                                traffic_type='traffic', std=False,
    #                                return_fig=True)
    
    # sum_stat_table=make_summary_statistics_table()
    # LR_table=make_LR_table(2019, k=5, const=True, method='LR')
    
    # sr = city_tests(k=5, test_seed=6)
    
    # fig, ax = service_area_figure('nyc', 2019, 9, 5, return_fig=True)
    
    # seed_range = range(50,75)
    # sr = np.zeros(shape=(8,8))
    # for seed in seed_range:
    #     sr += city_tests(k=5, test_seed=seed)
    # sr_mean = sr/len(seed_range)
    
    for city_train in cities:
        for city_test in cities:
            cm_mean = cm_mean_fig(city_train, city_test)
    
    service_area_figure('nyc', 2019, 10, 23, return_fig=False)
    # pre_process_table = pre_processing_table()
   
    # for city_train in cities:
    #     for city_test in cities:
    #         cm = train_test_cm(city_train, city_test, k=5, const=False)
    #         plt.savefig(f'./figures/cm_plots/{city_train}_{city_test}.png')
            
            
    
