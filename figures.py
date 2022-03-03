# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:41:40 2022

@author: Nicolai
"""

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


def make_summary_statistics_table(cities=None, variables=None, year=2019, print_only=False):
    
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
                     'pop_density', 'nearest_subway_dist', 'nearest_railway_dist'
                     'n_trips', 'b_trips', 'w_trips']

    if not print_only:
        
        for city in cities:
            data_city = bs.Data(city, year)
            
            stat_ids = list(data_city.stat.id_index.keys())
            
            var_dfs = dict()
            
            for var in variables:
                var_df = pd.DataFrame()
                var_df['stat_id'] = stat_ids
                
                var_dfs[var] = var_df
            
            for month in bs.get_valid_months(city, year):
                for day in range(1, calendar.monthrange(year, month)[1]+1):
                    data_day = bs.Data(city, year, month, day)
                    if len(data_day.df) > 0: # Avoid the issue of days with no traffic. E.g. Oslo 2019-04-01
                        stat_df = ipu.make_station_df(data_day, holidays=False, overwrite=True)
                        
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
                   'n_trips' : 'Number of daily trips',
                   'pop_density' : 'Population density [per sq. km]',
                   'nearest_subway_dist' : 'Distance to nearest subway/railway [m]'}
    # tab_df = tab_df.replace(var_renames)
    
    # city_names = [bs.name_dict[city] for city in cities]
    
    tab_df = tab_df.rename(index=var_renames, columns=bs.name_dict)
    
    print(tab_df.to_latex(column_format='@{}l'+('r'*len(tab_df.columns)) + '@{}',
                          index=True, na_rep = '--', float_format='%.2f',
                          multirow=True, multicolumn=True, multicolumn_format='c'))
    
    return tab_df


def make_LR_table(year, k=3):
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
                  'london', 'helsinki', 'oslo', 'madrid']
    
    city_lists = [(['nyc', 'chicago', 'washdc', 'boston'], 'USA'),
                      (['london', 'helsinki', 'oslo', 'madrid'], 'EUR')]
    
    
    percent_index_dict = {
        'percent_UNKNOWN': 'Share of unknown use',
        'percent_residential': 'Share of residential use',
        'percent_commercial': 'Share of commercial use',
        'percent_industrial': 'Share of industrial use',
        'percent_recreational': 'Share of recreatinal use',
        'percent_educational': 'Share of educational use',
        'percent_mixed': 'Share of mixed use',
        'percent_road': 'Share of road use',
        'percent_transportation': 'Share of transportational use',
        'pop_density': 'Population density [per sq. km]',
        'nearest_subway_dist': 'Distance to nearest subway/railway [m]',
        'n_trips': 'Number of daily trips',
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
        
        asdf = ipu.get_clusters(traf_mats, asdf, 'business_days', 10, 'k_means', k)[0]
        
        zone_columns = ['percent_residential', 'percent_commercial',
                        'percent_recreational', 'percent_industrial']
        
        for column in omit_columns[data.city]:
            if column in zone_columns:
                zone_columns.remove(column)
    
        other_columns = ['pop_density', 'nearest_subway_dist']
        
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
            out = f"$\\num[math-rm=\\mathbf]{{{x:.2e}}}$"
        else:
            out = f"$\\num{{{x:.2e}}}$"
    else:
        if bold:
            out = f"$\\mathbf{{{x:.3f}}}$"
        else:
            out = f"${x:.3f}$"
    
    return out





if __name__ == "__main__":
    
    # table=make_summary_statistics_table(print_only=True)
    table=make_LR_table(2019)
    
    
    
    
    
    
    
    
    
    
    
    
    
