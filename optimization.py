#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 07:35:21 2022

@author: ubuntu
"""

import osmnx
import numpy as np
import pandas as pd
import geopandas as gpd
import geoviews as gv
import panel as pn
from bokeh.models import HoverTool

import interactive_plot_utils as ipu
import bikeshare as bs


def get_intersections(data, station_df, merge_tolerance=20, custom_filter=None):
    
    extent = (station_df['lat'].max(), station_df['lat'].min(), 
          station_df['long'].max(), station_df['long'].min())
    
    if custom_filter == None:
        custom_filter = (
            '["highway"]["area"!~"yes"]["access"!~"private"]'
            '["highway"!~"abandoned|bus_guideway|construction|corridor|elevator|escalator|footway|'
            'motor|planned|platform|proposed|raceway|steps|service|motorway|motorway_link|track|path"]'
            '["bicycle"!~"no"]["service"!~"private"]'
            )
    
    gra = osmnx.graph.graph_from_bbox(
        *extent, custom_filter=custom_filter, retain_all=True)
    gra_projected = osmnx.projection.project_graph(gra)
# Tolerance for distance between points in m (defualt 10m)
    
    # gra_projected_simplified = osmnx.simplification.consolidate_intersections(gra_projected, tolerance=tol)
    # gra_simplified = osmnx.projection.project_graph(gra_projected_simplified, to_crs='epsg:4326')
    
    # gra_only_intersect = gra
    
    nodes = osmnx.simplification.consolidate_intersections(gra_projected, tolerance=merge_tolerance, rebuild_graph=False, dead_ends=False)
    nodes_gdf = gpd.GeoDataFrame(geometry=nodes.to_crs(epsg=4326))
    nodes_gdf['lon'] = nodes_gdf['geometry'].x
    nodes_gdf['lat'] = nodes_gdf['geometry'].y
    
    nodes_gdf['coords'] = nodes_gdf['geometry']
    
    nodes_gdf.set_geometry('coords', inplace=True)
    
    return nodes_gdf


def plot_intersections(nodes, nodes2=None, websocket_origin=None):

    tiles = gv.tile_sources.StamenTerrainRetina()
    tiles.opts(height=800, width=1600, active_tools=['wheel_zoom'])
    
    if 'highway' in nodes.columns:
        plot = gv.Points(nodes[['lon', 'lat', 'highway', 'street_count']],
                         kdims=['lon', 'lat'],
                         vdims=['highway', 'street_count'])
    else:
        plot = gv.Points(nodes[['lon', 'lat']],
                         kdims=['lon', 'lat'],)
    plot.opts(fill_color='blue', line_color='black', size=8)

    if nodes2 != None:
        plot2 = gv.Points(nodes2[['lon', 'lat']],
                         kdims=['lon', 'lat'],)
        plot2.opts(fill_color='blue', line_color='black', size=8)

        panelplot = pn.Column(tiles*plot, tiles*plot2)
    else:
        panelplot = pn.Column(tiles*plot)

    tooltips = [
        ('highway', '@highway'),
        ('street count', '@street_count'),
    ]

    hover = HoverTool(tooltips=tooltips)
    plot.opts(tools=[hover])

    bokeh_plot = panelplot.show(port=3000, websocket_origin=websocket_origin)
    
    return bokeh_plot


if __name__ == "__main__":
    city = 'nyc'
    year = 2019
    
    data = bs.Data(city, year, None)
    station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)
    
    intersections = get_intersections(data, station_df)
    
    
    
    neighborhoods = ipu.point_neighborhoods(intersections['geometry'], land_use)

    intersections = intersections.join(neighborhoods)

    service_area, service_area_size = ipu.get_service_area(city, intersections, land_use, voronoi=False)
    
    intersections['service_area'] = service_area
    
    percentages = ipu.neighborhood_percentages(city, intersections, land_use)
    pop_density = ipu.pop_density_in_service_area(intersections, census_df)
    nearest_subway = ipu.nearest_transit(city, intersections)

    point_info = pd.DataFrame(index=percentages.index)
    point_info['const'] = 1.0
    point_info[['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational']] = percentages[['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational']]
    point_info['pop_density'] = np.array(pop_density)/10000
    point_info['nearest_subway_dist'] = nearest_subway['nearest_subway_dist']/1000
    point_info['nearest_railway_dist'] = nearest_subway['nearest_railway_dist']/1000
    
    bk = plot_intersections(intersections, websocket_origin=('130.225.39.60'))
    '''
    bk.stop()
    '''
