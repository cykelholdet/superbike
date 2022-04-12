#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 08:37:54 2022

@author: ubuntu
"""
import pickle

import numpy as np
import pandas as pd
import geopandas as gpd
import smopy as sm
import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon

from OSMPythonTools.overpass import Overpass, overpassQueryBuilder



from OSMPythonTools.nominatim import Nominatim


import bikeshare as bs
import interactive_plot_utils as ipu


gpd.options.use_pygeos = False


overpass = Overpass(endpoint='https://overpass.kumi.systems/api/')

# nominatim = Nominatim()
# nyc = nominatim.query('NYC')

# query = overpassQueryBuilder(bbox=[48.1, 16.3, 48.3, 16.5], elementType='node', selector='"highway"="bus_stop"', conditions='count_tags() > 6', out='body')

# query = '[bbox:40.726109,-73.9845088,40.7369623,-73.974269][timeout:300][maxsize:10000000];way[highway~"^(motorway|trunk|primary|secondary|tertiary|residential)$"]   [~"^(name|ref)$"~"."] -> .allways;foreach.allways -> .currentway(    (.allways; - .currentway;)->.otherways_unfiltered;    way.otherways_unfiltered(if:t["name"] != currentway.u(t["name"]) || t["ref"] != currentway.u(t["ref"])) -> .otherways;  	node(w.currentway)->.currentwaynodes;  	node(w.otherways)->.otherwaynodes;  	node.currentwaynodes.otherwaynodes;  	out;    // output ways at intersection  /*    way(bn);    out geom;  */);'

# query='way[highway~"^(motorway|trunk|primary|secondary|tertiary|residential)$"][~"^(name|ref)$"~"."](40.726109,-73.9845088,40.7369623,-73.974269) -> .allways;foreach.allways -> .currentway( (.allways; - .currentway;)->.otherways_unfiltered;);way.otherways_unfiltered(if:t["name"] != currentway.u(t["name"]) || t["ref"] != currentway.u(t["ref"])) -> .otherways; node(w.currentway)->.currentwaynodes; node(w.otherways)->.otherwaynodes; node.currentwaynodes.otherwaynodes;out;'

# query='way[highway~"^(motorway|trunk|primary|secondary|tertiary|residential)$"][~"^(name|ref)$"~"."](40.726109,-73.9845088,40.7369623,-73.974269) -> .allways;foreach.allways -> .currentway( (.allways; - .currentway;)->.otherways_unfiltered;)way.otherways_unfiltered(if:t["name"] != currentway.u(t["name"]) || t["ref"] != currentway.u(t["ref"])) -> .otherways; node(w.currentway)->.currentwaynodes; node(w.otherways)->.otherwaynodes; node.currentwaynodes.otherwaynodes;out;'

# query='(way[highway~"^(motorway|trunk|primary|secondary|tertiary|residential)$"][~"^(name|ref)$"~"."](40.726109,-73.9845088,40.7369623,-73.974269) -> .allways;foreach.allways -> .currentway( (.allways; - .currentway;)->.otherways_unfiltered;way.otherways_unfiltered(if:t["name"] != currentway.u(t["name"]) || t["ref"] != currentway.u(t["ref"])) -> .otherways; node(w.currentway)->.currentwaynodes; node(w.otherways)->.otherwaynodes; node.currentwaynodes.otherwaynodes;););out;'

# query='(way[highway~"^(motorway|trunk|primary|secondary|tertiary|residential)$"][~"^(name|ref)$"~"."](40.726109,-73.9845088,40.7369623,-73.974269) -> .allways;foreach.allways -> .currentway( (.allways; - .currentway;)->.otherways_unfiltered;way.otherways_unfiltered(if:t["name"] != currentway.u(t["name"]) || t["ref"] != currentway.u(t["ref"])) -> .otherways; node(w.currentway)->.currentwaynodes; node(w.otherways)->.otherwaynodes; node.currentwaynodes.otherwaynodes;out;);out;);out;'


def heatmap_grid(bounds, resolution):
    latmin, lonmin, latmax, lonmax = bounds
    grid_points = []
    for lat in np.arange(latmin, latmax, resolution):
        for lon in np.arange(lonmin, lonmax, resolution):
            grid_points.append(Point((round(lat,4), round(lon,4))))
    
    
    grid_points = gpd.GeoDataFrame(geometry=grid_points, crs='epsg:3857')
    
    return grid_points


# city = 'chicago'
cities = ['boston', 'chicago', 'nyc', 'washdc', 'helsinki', 'london', 'madrid', 'oslo']

for city in cities:
    year = 2019
    month = None
    
    data = bs.Data(city, year, month)
    
    station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)
    
    polygon1 = Polygon(
        [(station_df['easting'].min()-1000, station_df['northing'].min()-1000),
         (station_df['easting'].min()-1000, station_df['northing'].max()),
         (station_df['easting'].max(), station_df['northing'].max()),
         (station_df['easting'].max(), station_df['northing'].min()-1000)])
    
    latmin, lonmin, latmax, lonmax = polygon1.bounds
    
    grid1 = heatmap_grid(polygon1.bounds, 1000)
    grid1 = grid1.to_crs(epsg=4326)
    
    polygon2 = Polygon(
        [(station_df['easting'].min(), station_df['northing'].min()),
         (station_df['easting'].min(), station_df['northing'].max()+1000),
         (station_df['easting'].max()+1000, station_df['northing'].max()+1000),
         (station_df['easting'].max()+1000, station_df['northing'].min())])
    
    grid2 = heatmap_grid(polygon2.bounds, 1000)
    grid2 = grid2.to_crs(epsg=4326)
    
    fig, ax = plt.subplots()
    grid1.plot(ax=ax)
    grid2.plot(color='red', ax=ax)
    
    inters = gpd.GeoDataFrame()
    
    for p1, p2 in zip(grid1['geometry'], grid2['geometry']):
        
        query=f'way[highway~"^(motorway|trunk|primary|secondary|tertiary|residential)$"][~"^(name|ref)$"~"."]({p1.coords[0][1]},{p1.coords[0][0]},{p2.coords[0][1]},{p2.coords[0][0]}) -> .allways;foreach.allways -> .currentway( (.allways; - .currentway;)->.otherways_unfiltered;way.otherways_unfiltered(if:t["name"] != currentway.u(t["name"]) || t["ref"] != currentway.u(t["ref"])) -> .otherways; node(w.currentway)->.currentwaynodes; node(w.otherways)->.otherwaynodes; node.currentwaynodes.otherwaynodes;out;);'
    
    
        result = overpass.query(query, timeout=6000)
    
        print(len(result.elements()))
        
        if len(result.elements()) > 0:
            result_json = result.toJSON()
        
            result_df = pd.DataFrame(result_json['elements'])
            node_df = result_df[result_df['type'] == 'node']
        
            node_df['coords'] = list(zip(node_df['lon'], node_df['lat']))
            node_df['coords'] = node_df['coords'].apply(Point)
        
            node_df = gpd.GeoDataFrame(node_df, geometry='coords', crs='epsg:4326')
            inters = inters.append(node_df)
    
    
    inters = inters.set_geometry('coords')
    inters = inters.set_crs(epsg=4326)
    inters.plot()
    # result.ways()
    # result_json = result.toJSON()
    
    # result_df = pd.DataFrame(result_json['elements'])
    
    # node_df = result_df[result_df['type'] == 'node']
    
    # node_df['coords'] = list(zip(node_df['lon'], node_df['lat']))
    # node_df['coords'] = node_df['coords'].apply(Point)
    
    # node_df = gpd.GeoDataFrame(node_df, geometry='coords', crs='epsg:4326')
    # node_df.plot()
    
    with open(f'./python_variables/union_{city}.pickle', 'rb') as file:
        union = pickle.load(file)
    
    union = gpd.GeoDataFrame(geometry=union, crs='epsg:4326')
    
    node_df = inters.reset_index(drop=True)
    
    node_df = gpd.overlay(node_df, union, how='intersection')
    
    with open(f'./python_variables/intersections_{city}.pickle', 'wb') as file:
        pickle.dump(node_df, file)
        
    #%%
    
    # with open(f'./python_variables/intersections_{city}.pickle', 'rb') as file:
    #     inters = pickle.load(file)
    
    extend = (node_df['lat'].min(), node_df['lon'].min(), 
          node_df['lat'].max(), node_df['lon'].max())
    
    tileserver = 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg' # Stamen Terrain
    # tileserver = 'http://a.tile.stamen.com/toner/{z}/{x}/{y}.png' # Stamen Toner
    # tileserver = 'http://c.tile.stamen.com/watercolor/{z}/{x}/{y}.png' # Stamen Watercolor
    # tileserver = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png' # OSM Default
    
    m = sm.Map(extend, tileserver=tileserver)
    
    fig, ax = plt.subplots(figsize=(7,10))
    
    m.show_mpl(ax=ax)
    
    x, y = m.to_pixels(node_df['lat'], node_df['lon'])
    ax.plot(x, y, 'ob', ms=2, mew=1.5)
    
    plt.savefig(f'figures/intersections_{city}.pdf', dpi=300, bbox_inches='tight')
