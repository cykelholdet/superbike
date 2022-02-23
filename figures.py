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

def service_area_figure(data, stat_df, land_use):
    
    
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
    
    ax.set_xlim(xlim_dict[data.city])
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
        
    return fig, ax

def make_summary_statistics_table(cities=None, variables=None, year=2019, print_only=False):
    
    if cities is None:
        cities = ['nyc', 'chic', 'washDC', 'boston', 
                  'london', 'helsinki', 'oslo', 'madrid']
        
    # variables = ['Share of residential use', 'Share of commercial use',
    #              'Share of recreational use', 'Share of industrial use', 
    #              'Share of transportational use', 'Share of mixed use',
    #              'Population density', 'Distance to nearest subway/railway', 
    #              'Number of  trips']
    
    variables is None:
        variables = ['percent_residential', 'percent_commercial',
                     'percent_recreational', 'percent_industrial', 
                     'percent_transportation', 'percent_mixed',
                     'pop_density', 'nearest_subway_dist', 'n_trips']

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
                    stat_df = ipu.make_station_df(data_day)
                    
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
                picke.dump(avg_stat_df, file)
        
        
    tab_df = pd.DataFrame(columns = ['city', 'Variable', 'Mean', 'Std. Dev.', 'Min', 'Max'])
    
    for city in cities:
        
        with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                avg_stat_df = picke.load(file)
        
        city_df = pd.DataFrame(columns=['city', 'Variable', 'Mean', 
                                        'Std. Dev.', 'Min', 'Max'],
                               index=variables)
        city_df['city'] = city
        city_df['Variable'] = variables
        city_df['Mean'] = avg_stat_df.mean()
        city_df['Std. Dev.'] = avg_stat_df.std()
        city_df['Min'] = avg_stat_df.min()
        city_df['Max'] = avg_stat_df.max()
        
        tab_df = pd.concat([tab_df, city_df])
    
    var_renames = {'percent_residential' : 'Share of residential use',
                   'percent_commercial' : 'Share of commercial use',
                   'percent_industrial' : 'Share of industrial use',
                   'percent_recreational' : 'Share of recreational use',
                   'percent_mixed' : 'Share of mixed use',
                   'percent_transportation' : 'Share of transportation use',
                   'n_trips' : 'Number of daily trips',
                   'pop_density' : 'Population density',
                   'nearest_subway_dist' : 'Distance to nearest subway/railway'}
    tab_df = tab_df.replace(var_renames)
    print(tab_df.to_latex(index=False, na_rep = '--', float_format='%.2f'))
    
    tab_df = pd.DataFrame(columns = ['city', 'Variable', 'Mean', 'Std. Dev.', 'Min', 'Max'])
    
    return tab_df



if __name__ == "__main__":
    # data = bs.Data('oslo', 2019, 9, 30)
    # stat_df, land_use, census_df = ipu.make_station_df(data, return_land_use=True, return_census=True, overwrite=False)
    
    avg_stat_df = make_summary_statistics_table()
    
    # fig, ax = service_area_figure(data, stat_df, land_use)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
