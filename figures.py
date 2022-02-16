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
from sklearn.model_selection import train_test_split
from holoviews import opts
from holoviews.operation.datashader import datashade
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.offsetbox import AnchoredText, AnchoredOffsetbox
from geopy.distance import geodesic

import bikeshare as bs
import interactive_plot_utils as ipu
from logistic_table import lr_coefficients

def service_area_figure(data, stat_df, land_use):
    
    stat_df = ipu.service_areas(data.city, stat_df, land_use)
    
    extend = (stat_df['lat'].min(), stat_df['long'].min(), 
          stat_df['lat'].max(), stat_df['long'].max())
    
    
    m = sm.Map(extend, tileserver='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg', z=15)
    
    
    
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
    plt.savefig('./figures/paper_figures/service_areas.pdf')
        
    return fig, ax


if __name__ == "__main__":
    data = bs.Data('nyc', 2019, 9)
    stat_df, land_use, census_df = ipu.make_station_df(data, return_land_use=True, return_census=True, overwrite=False)
    
    fig, ax = service_area_figure(data, stat_df, land_use)
    
    
    
    
    
    
    
    
    
    
    
    
    
    