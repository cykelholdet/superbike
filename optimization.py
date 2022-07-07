#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 07:35:21 2022

@author: ubuntu
"""

import multiprocessing
import itertools
import time
import pickle

import osmnx
import numpy as np
import pandas as pd
import geopandas as gpd
import geoviews as gv
import panel as pn
from bokeh.models import HoverTool
import scipy.optimize as so
from scipy.special import binom
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.offsetbox import AnchoredText, AnchoredOffsetbox
from matplotlib import patheffects
import contextily as cx
from more_itertools import distinct_permutations

from sympy.utilities.iterables import multiset_permutations
    
from shapely.geometry import Polygon

import interactive_plot_utils as ipu
import bikeshare as bs
from clustering import get_clusters


def get_intersections(polygon=None, data=None, station_df=None, merge_tolerance=20, custom_filter=None):
    """
    Obtain intersections in polygon. If polygon is none, get intersections in
    rectangular bounding box around stations in station_df.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    station_df : TYPE
        DESCRIPTION.
    merge_tolerance : TYPE, optional
        DESCRIPTION. The default is 20.
    custom_filter : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    nodes_gdf : TYPE
        DESCRIPTION.

    """
    # Tolerance for distance between points in m (defualt 10m)
    if custom_filter == None:
        custom_filter = (
            '["highway"]["area"!~"yes"]["access"!~"private"]'
            '["highway"!~"abandoned|bus_guideway|construction|corridor|elevator|escalator|footway|'
            'motor|planned|platform|proposed|raceway|steps|service|motorway|motorway_link|track|path"]'
            '["bicycle"!~"no"]["service"!~"private"]'
            )
        
    
    if polygon == None:
        extent = (station_df['lat'].max(), station_df['lat'].min(), 
              station_df['long'].max(), station_df['long'].min())
        
        gra = osmnx.graph.graph_from_bbox(
            *extent, custom_filter=custom_filter, retain_all=True)
    
    else:
        gra = osmnx.graph.graph_from_polygon(
            polygon=polygon, custom_filter=custom_filter, retain_all=True)
        
    
    if data != None:
        crs = data.laea_crs
    else:
        crs = None
    
    gra_projected = osmnx.projection.project_graph(gra, to_crs=crs)
    
    
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


def get_point_info(data, nodes, land_use, census_df):
    """
    Get info of points with column 'geometry' and 'coords' both in epsg 4326

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    nodes : TYPE
        DESCRIPTION.
    land_use : TYPE
        DESCRIPTION.
    census_df : TYPE
        DESCRIPTION.

    Returns
    -------
    point_info : TYPE
        DESCRIPTION.

    """
    neighborhoods = ipu.point_neighborhoods(nodes['geometry'], land_use)

    nodes = nodes.join(neighborhoods)

    service_area, service_area_size = ipu.get_service_area(data, nodes, land_use, voronoi=False)
    
    nodes['service_area'] = service_area
    
    percentages = ipu.neighborhood_percentages(data, nodes, land_use)
    pop_density = ipu.pop_density_in_service_area(nodes, census_df)
    nearest_subway = ipu.nearest_transit(data.city, nodes)

    point_info = pd.DataFrame(index=percentages.index)
    point_info['const'] = 1.0
    point_info[['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational']] = percentages[['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational']]
    point_info['pop_density'] = np.array(pop_density)
    point_info['nearest_subway_dist'] = nearest_subway['nearest_subway_dist']
    point_info['nearest_railway_dist'] = nearest_subway['nearest_railway_dist']
    point_info['center_dist'] = ipu.geodesic_distance(nodes, bs.city_center_dict[data.city])
    
    return point_info


def asdf_months(data, months, variables=None):
    
    if variables is None:
        variables = ['percent_residential', 'percent_commercial',
                     'percent_recreational', 'percent_industrial',
                     'percent_mixed', 'percent_transportation', 
                     'percent_educational', 'percent_road', 'percent_UNKNOWN',
                     'pop_density', 'nearest_subway_dist', 'nearest_railway_dist',
                     'nearest_transit_dist', 'center_dist', 'n_trips', 'b_trips', 'w_trips']
    
    stat_ids = list(data.stat.id_index.keys())
    
    asdfs = dict()
    counts = dict()
    
    avg_stat_df_year = pd.DataFrame()
    avg_stat_df_year['stat_id'] = stat_ids
    
    # make/load asdfs for each month
    for month in months:
        asdfs[month], counts[month] = ipu.pickle_asdf_month(
            data.city, data.year, month, variables=variables, 
            return_counts=True, overwrite=False,)
    
    for var in variables:
        
        counts_df = pd.DataFrame()
        counts_df['stat_id'] = stat_ids
        
        var_df = pd.DataFrame()
        var_df['stat_id'] = stat_ids
        
        
        for month in months:
            asdf, count = asdfs[month], counts[month]
            
            if var in asdf.columns:
                var_df = var_df.merge(asdf[['stat_id', var]], 
                                      on='stat_id', how='outer')
                var_df.rename({var: month}, axis=1, inplace=True)

                counts_df = counts_df.merge(count[['stat_id', var]],
                                            on='stat_id', how='outer')
                counts_df.rename({var: month}, axis=1, inplace=True)

        var_df = var_df.drop('stat_id', axis=1)
        counts_df = counts_df.drop('stat_id', axis=1)
        var_df = var_df.mul(counts_df)
        
        avg_stat_df_year[var] = var_df.sum(axis=1)/counts_df.sum(axis=1)
    
    return avg_stat_df_year
    

def plot_intersections(nodes, nodes2=None, websocket_origin=None, polygons=None, vdims=None, return_panel=False):

    tiles = gv.tile_sources.StamenTerrainRetina()
    tiles.opts(height=800, width=1600, active_tools=['wheel_zoom'])
    
    if 'highway' in nodes.columns:
        plot = gv.Points(nodes[['lon', 'lat', 'highway', 'street_count']],
                         kdims=['lon', 'lat'],
                         vdims=['highway', 'street_count'])
        plot.opts(fill_color='blue', line_color='black', size=8)
    elif vdims != None:
        plot = gv.Points(nodes[['lon', 'lat', *vdims]],
                         kdims=['lon', 'lat'],
                         vdims=vdims)
        plot.opts(gv.opts.Points(fill_color=vdims[0], size=8, line_color='black'))
    else:
        plot = gv.Points(nodes[['lon', 'lat']],
                         kdims=['lon', 'lat'],)
        plot.opts(fill_color='blue', line_color='black', size=8)

    if nodes2 is not None:
        plot2 = gv.Points(nodes2[['lon', 'lat']],
                         kdims=['lon', 'lat'],)
        plot2.opts(fill_color='blue', line_color='black', size=8)
        combined_plot = tiles*plot
        panelplot = pn.Column(combined_plot, tiles*plot2)
    elif polygons is not None:
        plot2 = gv.Polygons(polygons['geometry'])
        plot2.opts(alpha=0.7)
        combined_plot = tiles*plot2*plot
        panelplot = pn.Column(combined_plot)
    else:
        combined_plot = tiles*plot
        panelplot = pn.Column(combined_plot)
    

    tooltips = [
        ('highway', '@highway'),
        ('street count', '@street_count'),
        ('existing', '@existing'),
        ('stat_id', '@stat_id')
    ]

    hover = HoverTool(tooltips=tooltips)
    plot.opts(tools=[hover])
    
    if return_panel == True:
        bokeh_plot = combined_plot
    else:
        bokeh_plot = panelplot.show(port=3000, websocket_origin=websocket_origin)
    
    return bokeh_plot


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    """
    Like numpy.apply_along_axis(), but with arguments in a tuple
    instead.

    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    """
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    From https://docs.python.org/3.9/library/itertools.html#itertools-recipes
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


# #%%

if __name__ == "__main__":
    city = 'nyc'
#     year = 2019
    
#     data = bs.Data(city, year, None)
#     station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)
    
#     intersections = get_intersections(data=data, station_df=station_df)
    
    
    
#     neighborhoods = ipu.point_neighborhoods(intersections['geometry'], land_use)

#     intersections = intersections.join(neighborhoods)

#     service_area, service_area_size = ipu.get_service_area(data, intersections, land_use, voronoi=False)
    
#     intersections['service_area'] = service_area
    
#     percentages = ipu.neighborhood_percentages(data, intersections, land_use)
#     pop_density = ipu.pop_density_in_service_area(intersections, census_df)
#     nearest_subway = ipu.nearest_transit(city, intersections)

#     point_info = pd.DataFrame(index=percentages.index)
#     point_info['const'] = 1.0
#     point_info[['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational']] = percentages[['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational']]
#     point_info['pop_density'] = np.array(pop_density)
#     point_info['nearest_subway_dist'] = nearest_subway['nearest_subway_dist']
#     point_info['nearest_railway_dist'] = nearest_subway['nearest_railway_dist']
    
# #     import scipy.optimize as so
# #     import cvxpy
    
# #     def obj_fun(C):
# #         d = np.array([0.5, 0.6, 0.8, 0.9])
# #         return -np.sum(C*d)
    
# #     def con_fun(C):
# #         return np.sum(C)
    
# #     def con_val(C):
# #         return ((C == [1,1,1,1]) + (C == [0,0,0,0])).astype(int)
    
# #     sum_constraint = so.LinearConstraint(np.array([1,1,1,1]), 2, 2)
    
# #     sum_constraint = so.NonlinearConstraint(con_fun, 2, 2)
    
# #     val_constraint = so.Bounds([0, 0, 0, 0], [1, 1, 1, 1])
    
# #     so.minimize(obj_fun, x0=np.array([0, 1, 0, 1]), constraints=(sum_constraint), bounds=val_constraint)
    
    
# #     data = np.array([0.5, 0.6, 0.8, 0.9])
    
# #     selection = cvxpy.Variable(shape=4, boolean=True)
    
# #     constraint = cvxpy.sum(selection) == 2

# #     cost = cvxpy.sum(cvxpy.multiply(selection, data))
    
# #     problem = cvxpy.Problem(cvxpy.Maximize(cost), constraints=[constraint])
    
# #     score = problem.solve(solver=cvxpy.GLPK_MI)
    
#     import pickle
#     import logistic_regression
#     import shapely
    
#     cols = ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational',
#             'pop_density', 'nearest_subway_dist', 'nearest_railway_dist']
#     day_type = 'business_days'
#     min_trips = 8
#     clustering = 'k_means'
#     k = 5
#     seed = 42
#     triptype = 'b_trips'
    
#     data = bs.Data(city, year, None)

#     # station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)
#     traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')
#     # station_df, clusters, labels = get_clusters(
#     #     traffic_matrices, station_df, day_type, min_trips, clustering, k, seed)
    
#     # asdf, clusters, labels = get_clusters(traf_mats, asdf, 'business_days', 10, 'k_means', k, 42)
#     try:
#         with open(f'./python_variables/{data.city}{year}_avg_stat_df.pickle', 'rb') as file:
#             asdf = pickle.load(file)
#     except FileNotFoundError:
#         raise FileNotFoundError(f'The average station DataFrame for {data.city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
        
#     # mask = ~asdf['n_trips'].isna()
    
#     # asdf = asdf[mask]
#     # asdf = asdf.reset_index(drop=True)
    
#     asdf, clusters, labels = get_clusters(
#         traffic_matrices, asdf, day_type, min_trips, clustering, k, seed)
    
#     if city in ['helsinki', 'oslo', 'madrid', 'london']:
#         df_cols = [col for col in cols if col != 'percent_industrial']
#     else:
#         df_cols = cols
    
#     model_results = ipu.linear_regression(asdf, df_cols, triptype)
    
#     pred = model_results.predict(point_info[['const', *df_cols]])
    
#     #%%
    
#     n = len(point_info)
    
#     n_selected = 100
    
#     data = pred
    
#     int_proj = intersections.to_crs(epsg=3857)
    
#     distances = np.zeros((n, n))
    
#     for i in range(n):
#         distances[i] = int_proj.distance(int_proj.geometry.loc[i])
    
#     for i in range(n):
#         distances[i, i] = 0
        
#     # distances[np.where(distances < 500)] = 1000000
    
#     dist_matrix = cvxpy.Constant(distances)
    
#     sa = intersections['service_area'].to_crs(epsg=3857)
#     shapely.ops.unary_union(sa).area
    
#     saa = sa.area

#     selection = cvxpy.Variable(shape=n, boolean=True)
    
#     constraint = cvxpy.sum(selection) == n_selected
    
#     # distance_constraint = cvxpy.min(distances[selection == 1][:, selection == 1]) >= 500
#     disto = cvxpy.max(dist_matrix @ selection)
#     disto = cvxpy.diag(selection) @ dist_matrix @ cvxpy.diag(selection)
#     disto = cvxpy.sum(cvxpy.diag(selection) @ dist_matrix)
        
#     distance_constraint = disto <= 100000000
    
#     cost = cvxpy.sum(cvxpy.multiply(selection, pred))
    
#     problem = cvxpy.Problem(cvxpy.Maximize(cost), constraints=[constraint])
    
#     score = problem.solve(solver=cvxpy.GLPK_MI)
    
#     print(selection.value)
    
    
#     #%% SO opti. Too slow. 2 iterations takes many hours with SLSQP solver
#     n = len(point_info)
    
#     # pred = pred[:1000]
    
#     # n = 1000
    
#     n_select = 100
    
#     def obj_fun(x):
#         return -np.sum(x*pred)
    
#     sum_constraint = so.LinearConstraint(np.array([[1]*n]), n_select, n_select)
#     sum_constraint = so.NonlinearConstraint(np.sum, 0, n_select)
#     bounds = so.Bounds([0]*n, [1]*n)
    
#     x0 = np.zeros(n)
#     x0[:n_select] = 1
#     np.random.seed(42)
#     x0 = np.random.permutation(x0)
    
#     minimum = so.minimize(obj_fun, x0=x0, constraints=(sum_constraint), bounds=bounds, method='SLSQP', options={'maxiter': 2})
    
#     selection_idx = np.argpartition(minimum.x, -n_select)[-n_select:]
    
#     selection_so = np.zeros(n)
#     selection_so[selection_idx] = 1

    
#     #%% linprog works and within a reasonable time but can only use linear constraints
#     A_eq = np.array([[1]*n])
    
#     lim = so.linprog(-pred, A_eq=A_eq, b_eq=n_select, bounds=(0,1), options={'maxiter': 10})
    
#     selection_idx = np.argpartition(lim.x, -n_select)[-n_select:]
    
#     selection_so = np.zeros(n)
#     selection_so[selection_idx] = 1
    
#     #%% gekko
    
#     n = len(point_info)

#     n_select = 100
    
#     import time
    
#     from gekko import GEKKO
    
#     t_pre = time.time()
#     m = GEKKO()
    
#     # help(m)
    
    
    
#     c = [m.Const(pred_i) for pred_i in pred]
        
#     # x = [m.Var(lb=0, ub=1) for i in range(n)]
#     x = m.Array(m.Var, n, lb=0, ub=1, integer=True)
    
#     m.Equation(m.sum(x) == 100)
    
#     # m.Equation(x @ distances @ x  > 100)

#     m.Maximize(m.sum([x_i*pred_i for x_i, pred_i in zip(x, pred)]))
    
#     m.solve()    
    
#     solution_gekko = np.array([x_i.value for x_i in x]).reshape(-1)

#     selection_idx = np.argpartition(solution_gekko, -n_select)[-n_select:]
    
#     selection_gekko = np.zeros(n)
#     selection_gekko[selection_idx] = 1
    
#     print(f"time taken: {time.time() - t_pre}")
    
#     #%%
    
#     def condition(x):
#         return np.min(distances[x][:,x][distances[x][:,x] != 0])
    
#     x0 = np.zeros(n, dtype=bool)
#     x0[:n_select] = 1
#     np.random.seed(42)
#     x0 = np.random.permutation(x0)
    
#     rng = np.random.default_rng(42)
   
#     n_per = 2000000
    
#     perms = rng.permuted(np.tile(x0, n_per).reshape(n_per, x0.size), axis=1)
    
    #%% multi
    


# result = parallel_apply_along_axis(condition, 1, perms)

# spaced_idx = np.where(result > 100)[0]
# spaced_candidates = perms[spaced_idx]

    def obj_fun(x):
        return -np.sum(x*pred)
    
    # scores = parallel_apply_along_axis(obj_fun, 1, spaced_candidates)
    
    # selection_score = spaced_candidates[np.argmin(scores)]
    
    #%% DIY GA
    
    # n = len(point_info)

    # n_select = 100
    
    # batch_size = 1000
    # n_iters = 100
    # elite_percentage = 0.2
    # random_percentage = 0.2
    # children_percentage = 0.2
    # mutated_percentage = 0.4
    
    # mutation_bits = 1
    
    # def condition(x):
    #     return np.min(distances[x][:,x][distances[x][:,x] != 0])
    
    
    # x0 = np.zeros(n, dtype=bool)
    # x0[:n_select] = 1
    # np.random.seed(42)
    # x0 = np.random.permutation(x0)
    
    # rng = np.random.default_rng(42)
   
    # n_per = batch_size
    
    # n_elite = int(np.floor(batch_size*elite_percentage))
    # n_random = int(np.floor(batch_size*random_percentage))
    # n_children = int(np.floor(batch_size*children_percentage))
    # n_mutated = int(np.floor(batch_size*mutated_percentage))
    
    
    # population = rng.permuted(np.tile(x0, n_per).reshape(n_per, x0.size), axis=1)
    
    # best_score = 0
    
    
    # for i in range(n_iters):
    #     score = parallel_apply_along_axis(obj_fun, 1, population)
    #     best = np.min(score)
    #     print(f"Best score: {best} (iteration {i})")
    #     if best < best_score:
    #         best_index = np.argmin(score)
    #         best_genes = population[best_index]
    #         print(f"index: {best_index}")
    #         print(np.where(population[0])[0])
    #         best_score = best
            
            
        
    #     # elite = population[np.argpartition(score, n_elite)[:n_elite]]  # Take the top n_elite
        
    #     # random = rng.permuted(np.tile(x0, n_random).reshape(n_random, x0.size), axis=1)
        
    #     score_a = score + 300
        
    #     probabilities = -1*score_a / (-1 * score_a.sum())
        
    #     mating_pool = rng.choice(population, n_per, p=probabilities)
        
    #     # mating_pool = rng.permutation(mating_pool)
        
    #     previous_population = population.copy()
        
    #     n_children = 400
        
        
        
    #     # Create children
    #     # parents = rng.permutation(mating_pool)
    #     parents = rng.choice(mating_pool, n_children*2)
    #     # parents = mating_pool[:2*n_children]
    #     parent1 = parents[:n_children]
    #     parent2 = parents[n_children:2*n_children]
        
    #     rows, cols = np.where(parent1)
    #     index = cols.reshape((n_children, -1))
    #     idx = rng.random(index.shape).argsort(0)
    #     genes1 = rng.choice(idx, size=(50), axis=1, replace=False)

    #     rows2, cols2 = np.where(parent2)
    #     index2 = cols2.reshape((n_children, -1))
    #     idx2 = rng.random(index2.shape).argsort(0)
    #     genes2 = rng.choice(idx2, size=(50), axis=1, replace=False)
        
    #     children_idx = np.hstack((genes1, genes2))
    #     children_cols = children_idx.flatten()
    #     children = np.zeros((n_children, n), dtype=bool)
    #     children[rows, children_cols] = True
        
    #     n_add_child_genes = n_select - children.sum(axis=1)
    #     add_rows, add_cols = np.where(~children)
        
    #     for n_add, row in zip(n_add_child_genes, children):
    #         cols = np.where(~row)[0]
    #         row[rng.choice(cols, size=(n_add), replace=False)] = True
        
    #     copies = rng.choice(mating_pool, n_per - n_children)
        
    #     population = np.vstack((children, copies))
        
    #     mutate_idx = rng.integers(0, n_per, size=50)
        
    #     mutated = population[mutate_idx].astype(bool)
        
    #     for row in mutated:
    #         true_cols = np.where(row)[0]
    #         false_cols = np.where(~row)[0]
            
    #         row[rng.choice(true_cols, size=(mutation_bits), replace=False)] = False
    #         row[rng.choice(false_cols, size=(mutation_bits), replace=False)] = True
    
    #     population[mutate_idx] = mutated
    #     # population = np.vstack((elite, random, children, mutated))
    
        
    
    # #%% GA
    
    # # from geneticalgorithm import geneticalgorithm as ga
    
    # # def obj_fun(x):
    # #     return -np.sum(x*pred)
    
    
    # # model=ga(function=obj_fun,dimension=n,variable_type='bool')
    
    # # model.run()
    
    
    # #%% Expansion area.
    
    # city = 'nyc'
    
    # data = bs.Data(city, 2019, 9)
    # station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)
    # expansion_area = gpd.read_file('data/nyc/expansion_2019_area.geojson')
    # int_exp = get_intersections(expansion_area.loc[0, 'geometry'], data=data)
    # point_info = get_point_info(data, int_exp, land_use, census_df)
    
    # months = [1,2,3,4,5,6,7,8,9]
    # asdf = asdf_months(data, months)
    
    # int_proj = int_exp.to_crs(data.laea_crs)
    
    # n = len(int_proj)
    
    # distances = np.zeros((n, n))
    # for i in range(n):
    #     distances[i] = int_proj.distance(int_proj.geometry.loc[i])


    # traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')
    # cols = ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational',
    #         'pop_density', 'nearest_subway_dist', 'nearest_railway_dist']
    # day_type = 'business_days'
    # min_trips = 8
    # clustering = 'k_means'
    # k = 5
    # seed = 42
    # triptype = 'b_trips'
    # asdf, clusters, labels = get_clusters(
    #     traffic_matrices, asdf, day_type, min_trips, clustering, k, seed)

    # if city in ['helsinki', 'oslo', 'madrid', 'london']:
    #     df_cols = [col for col in cols if col != 'percent_industrial']
    # else:
    #     df_cols = cols

    # model_results = ipu.linear_regression(asdf, df_cols, triptype)

    # pred = model_results.predict(point_info[['const', *df_cols]])
    
    
    # n = len(point_info)
    
    # n_select = 10
    
    # def obj_fun(x):
    #     return -np.sum(x*pred)
    
    # def condition(x):
    #     xb = x.astype(bool)
    #     return np.min(distances[xb][:,xb][distances[xb][:,xb] != 0])
    
    # sum_constraint = so.LinearConstraint(np.array([[1]*n]), n_select, n_select)
    # sum_constraint = so.NonlinearConstraint(condition, 200, n_select)
    # bounds = so.Bounds([0]*n, [1]*n)
    
    # x0 = np.zeros(n)
    # x0[:n_select] = 1
    # np.random.seed(42)
    # x0 = np.random.permutation(x0)
    
    # minimum = so.minimize(obj_fun, x0=x0, constraints=(sum_constraint), bounds=bounds, method='SLSQP', options={'maxiter': 10})
    
    # selection_idx = np.argpartition(minimum.x, -n_select)[-n_select:]
    
    # selection_so = np.zeros(n)
    # selection_so[selection_idx] = 1
    
    
    #%% Expansion subdivision
    
    # First step: Determine how many stations to place in each subpolygon.
    
    data = bs.Data('nyc', 2019, 9)
    
    station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
    sub_polygons = gpd.read_file('data/nyc/nyc_expansion_subdivision_2.geojson')
    
    months = [1,2,3,4,5,6,7,8,9]
    asdf = asdf_months(data, months)
    
    pops = []
    for polygon in sub_polygons['geometry']:
        intersections = census_df.intersection(polygon)
        selection = ~intersections.is_empty
        census_intersect = census_df.loc[selection, 'pop_density']
        # Area in km²
        areas = intersections[selection].to_crs(data.laea_crs).area/1000000
        population = np.sum(areas * census_intersect)
        pops.append(population)
    
    sub_polygons['population'] = pops
    
    # Number of stations per person
    station_density = len(station_df) / station_df['population'].sum()
    
    proportional_n_stations = sub_polygons['population'] * station_density
    
    # Scale up and round to add up to 60
    n_stations = np.floor(proportional_n_stations*3.15)
    
    sub_polygons['n_stations'] = n_stations
    
    traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')
    cols = ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational',
            'pop_density', 'nearest_subway_dist', 'nearest_railway_dist']
    day_type = 'business_days'
    min_trips = 8
    clustering = 'k_means'
    k = 5
    seed = 42
    triptype = 'b_trips'
    asdf, clusters, labels = get_clusters(
        traffic_matrices, asdf, day_type, min_trips, clustering, k, seed)

    if data.city in ['helsinki', 'oslo', 'madrid', 'london']:
        df_cols = [col for col in cols if col != 'percent_industrial']
    else:
        df_cols = cols

    model_results = ipu.linear_regression(asdf, df_cols, triptype)
    
    minima = {}
    # scores = [[]]*len(sub_polygons)
    
    n_per = 10000
    
    # rng = np.random.default_rng(42)
    
    spacings = [100, 150, 200, 250, 300]
    
    for i, polygon in sub_polygons.loc[[11]].iterrows():
        t_start = time.time()
        int_exp = get_intersections(polygon['geometry'], data=data)
       
        existing_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')
       
        existing_stations['lon'] = existing_stations['long']
        existing_stations['geometry'] = existing_stations['coords']
       
        n_existing = len(existing_stations)
       
        print(f"{n_existing} existing, {polygon['n_stations']} total")
       
        int_exp = pd.concat((existing_stations[['lat', 'lon', 'coords', 'geometry']], int_exp))
        int_exp = int_exp.reset_index()
        
        print(f"There are {len(int_exp)} Intersections")
        
        point_info = get_point_info(data, int_exp, land_use, census_df)
        
        months = [1,2,3,4,5,6,7,8,9]
        # asdf = asdf_months(data, months)
        
        int_proj = int_exp.to_crs(data.laea_crs)
        
        n_stations = polygon['n_stations']
        
        n_combinations = binom(len(int_exp) - n_existing, n_stations)
        fig, ax = plt.subplots()
        gpd.GeoSeries(polygon['geometry']).plot(ax=ax)
        int_exp.plot(ax=ax, color='red')
        ax.set_title(f"{n_stations} stations : {n_combinations} combinations")
        print(n_combinations)
        
        n = len(point_info)
        
        n_select = int(n_stations)
        
        n_total = n_select + n_existing
        
        distances = np.zeros((n, n))
        for j in range(n):
            distances[j] = int_proj.distance(int_proj.geometry.loc[j])
            
        
        pred = model_results.predict(point_info[['const', *df_cols]])
        
        def obj_fun(x):
            return -np.sum(x*pred)
        
        def condition(x):
            xb = x.astype(bool)
            return np.min(distances[xb][:,xb][distances[xb][:,xb] != 0])
        
        sum_constraint = so.LinearConstraint(np.array([[1]*n]), n_select, n_select)
        sum_constraint = so.NonlinearConstraint(condition, 200, n_select)
        bounds = so.Bounds([0]*n, [1]*n)
        
        x0 = np.zeros(n-n_existing)
        x0[:n_select] = 1
        # np.random.seed(42)
        # x0 = np.random.permutation(x0)
        
        # n_permutations = np.floor(np.min((n_per, n_combinations*100))).astype(int)
        
        perm_generator = distinct_permutations(x0)
        grouped_perms = grouper(perm_generator, n_per, fillvalue=tuple(x0))
        
        print(n_combinations/n_per)
        
        best_scores = {}
        
        for spacing in spacings:
            minima[(i, spacing)] = (None,0)
            
        
        for population in grouped_perms:
        
            # population = rng.permuted(np.tile(x0, n_permutations).reshape(n_permutations, x0.size), axis=1)
            
            existing_population = np.ones((n_per, n_existing))
            
            population = np.hstack((existing_population, population))
            
            score = parallel_apply_along_axis(obj_fun, 1, population)
            if n_select > 1:
                cond = parallel_apply_along_axis(condition, 1, population)
            else:
                cond = np.sum(population, axis=1)*400
            
            # spacing = 250
            # mask = np.where(cond < 250)
            # if len(score[mask]) == len(score):
            #     print('mask condition not fulfilled, changing to 200')
            #     spacing = 200
            #     mask = np.where(cond < 200)
            #     if len(score[mask]) == len(score):
            #         print('mask condition not fulfilled, changing to 150')
            #         spacing = 150
            #         mask = np.where(cond < 150)
            #         if len(score[mask]) == len(score):
            #             print('mask condition not fulfilled, changing to 100')
            #             spacing = 100
            #             mask = np.where(cond < 100)
            #             if len(score[mask]) == len(score):
            #                 print('mask condition not fulfilled, changing to 80')
            #                 spacing = 80
            #                 mask = np.where(cond < 80)
            
            
            
            for spacing in spacings:
                mask = np.where(cond < spacing)
                
                score[mask] = 0
                
                best = population[np.argmin(score)]
                min_score = np.min(score)
                
                if spacing == 250:
                    print(f"min: {best}, score: {min_score}")
            
            
                # minimum = so.minimize(obj_fun, x0=x0, constraints=(sum_constraint), bounds=bounds, method='SLSQP', options={'maxiter': 10})
                # print(minimum.message)
                # minima.append(minimum)
                # selection_idx = np.argpartition(minimum.x, -n_select)[-n_select:]
                # minima.append([np.min(score[mask]), population[np.argmin(score[mask])]])
                
                
                
                if min_score < minima[(i, spacing)][1]:
                    minima[(i,spacing)] = (best, min_score)

        print(minima)
        print(f"time taken: {(time.time() - t_start)/60:.1f} min.")
        
        save_minima = [minima[(i, spacing)] for spacing in spacings]
        save_minima = pd.DataFrame(save_minima, index=spacings, columns=('solution', 'score'))
        with open(f'./python_variables/nyc_expansion_optimization_polygon_{i:02d}.pickle', 'wb') as file:
            pickle.dump(save_minima, file)
    #%% Results
    
    minima_res = {}
    for i in range(len(sub_polygons)):
        with open(f'./python_variables/nyc_expansion_optimization_polygon_{i:02d}.pickle', 'rb') as file:
            minima_res[i] = pickle.load(file)
            
    mini = pd.concat(minima_res.values(), keys=minima_res.keys())
    
    maxdist = 100
    
    sps = [spacing for spacing in spacings if spacing <= maxdist]
    
    mini = mini.loc[pd.IndexSlice[:, sps], :]
    
    bestmini = pd.DataFrame()
    for i in range(len(sub_polygons)):
        new_row = mini.xs(i, level=0)[~mini.xs(i, level=0)['solution'].isna()].iloc[[-1]]
        new_row = new_row.rename_axis('spacing').reset_index()
        new_row.index = [i]
        bestmini = pd.concat((bestmini, new_row))
    
    results = bestmini['solution']
    
    
    selected_intersections = []
    
    for i, polygon in sub_polygons.iterrows():
        int_exp = get_intersections(polygon['geometry'], data=data)
        
        existing_stations_sub = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')

        existing_stations_sub['lon'] = existing_stations_sub['long']
        existing_stations_sub['geometry'] = existing_stations_sub['coords']

        n_existing = len(existing_stations_sub)

        print(f"{n_existing} existing, {polygon['n_stations']} total")
        
        int_exp['existing'] = False
        
        existing_stations_sub['existing'] = True

        int_exp = pd.concat((existing_stations_sub[['lat', 'lon', 'coords', 'geometry', 'existing']], int_exp))
        int_exp = int_exp.reset_index()

        selected_intersections.append(int_exp[results[i] == 1])
        
    selected_intersections = pd.concat(selected_intersections)
    
    #%%
    # int_exp[selection_so == 1]
    bk = plot_intersections(selected_intersections, websocket_origin=['130.225.39.60', 'localhost:3000'], polygons=sub_polygons, vdims=['existing'])
    '''
    bk.stop()
    '''


    #%% Complete area random samples
    
    data = bs.Data('nyc', 2019, 9)
    
    station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
    expansion_area = gpd.read_file('data/nyc/expansion_2019_area.geojson')
    int_exp = get_intersections(expansion_area.loc[0, 'geometry'], data=data)
    point_info = get_point_info(data, int_exp, land_use, census_df)
    
    months = [1,2,3,4,5,6,7,8,9]
    asdf = asdf_months(data, months)
    
    traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')
    cols = ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational',
            'pop_density', 'nearest_subway_dist', 'nearest_railway_dist']
    day_type = 'business_days'
    min_trips = 8
    clustering = 'k_means'
    k = 5
    seed = 42
    triptype = 'b_trips'
    asdf, clusters, labels = get_clusters(
        traffic_matrices, asdf, day_type, min_trips, clustering, k, seed)

    if data.city in ['helsinki', 'oslo', 'madrid', 'london']:
        df_cols = [col for col in cols if col != 'percent_industrial']
    else:
        df_cols = cols

    model_results = ipu.linear_regression(asdf, df_cols, triptype)
    
    minima = []
    # n_per = 100000
    
    rng = np.random.default_rng(42)
    
    # set polygon
    
    batch_size = 1000000
    n_iters = 100
    elite_percentage = 0.2
    random_percentage = 0.2
    children_percentage = 0.0
    mutated_percentage = 0.6
    
    mutation_bits = 1
    
    n_elite = int(batch_size*elite_percentage)
    n_random = int(batch_size*random_percentage)
    n_mutated = int(batch_size*mutated_percentage)
    
    polygon = expansion_area.loc[0]
    
    polygon['n_stations'] = 58
    
    int_exp = get_intersections(polygon['geometry'], data=data)
   
    existing_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')
   
    existing_stations['lon'] = existing_stations['long']
    existing_stations['geometry'] = existing_stations['coords']
   
    n_existing = len(existing_stations)
   
    print(f"{n_existing} existing, {polygon['n_stations']} total")
   
    int_exp = pd.concat((existing_stations[['lat', 'lon', 'coords', 'geometry']], int_exp))
    int_exp = int_exp.reset_index()

    
    point_info = get_point_info(data, int_exp, land_use, census_df)
    
    months = [1,2,3,4,5,6,7,8,9]
    # asdf = asdf_months(data, months)
    
    int_proj = int_exp.to_crs(data.laea_crs)
    
    n_stations = polygon['n_stations']
    
    n_combinations = binom(len(int_exp), n_stations)
    fig, ax = plt.subplots()
    gpd.GeoSeries(polygon['geometry']).plot(ax=ax)
    int_exp.plot(ax=ax, color='red')
    ax.set_title(f"{n_stations} stations : {n_combinations} combinations")
    print(n_combinations)
    
    n = len(point_info)
    
    n_select = int(n_stations)
    
    n_total = n_select + n_existing
    
    distances = np.zeros((n, n))
    for i in range(n):
        distances[i] = int_proj.distance(int_proj.geometry.loc[i])
        
    
    pred = model_results.predict(point_info[['const', *df_cols]])
    
    def obj_fun(x):
        return -np.sum(x*pred)
    
    def condition(x):
        xb = x.astype(bool)
        return np.min(distances[xb][:,xb][distances[xb][:,xb] != 0])
    
    
    x0 = np.zeros(n-n_existing)
    x0[:n_select] = 1
    np.random.seed(42)
    x0 = np.random.permutation(x0)
    
    # n_permutations = np.floor(np.min((n_per, n_combinations*100))).astype(int)
    
    population = rng.permuted(np.tile(x0, batch_size).reshape(batch_size, x0.size), axis=1)
    
    existing_population = np.ones((batch_size, n_existing))
    
    population = np.hstack((existing_population, population))
    
    existing_population_for_random = np.ones((n_random, n_existing))
    
    population_new = population
    
    best_score = 0
    
    for i in range(n_iters):
        population = population_new
        
        score = parallel_apply_along_axis(obj_fun, 1, population)
        if n_select > 1:
            cond = parallel_apply_along_axis(condition, 1, population)
        else:
            cond = np.sum(population, axis=1)*400
        mask = np.where(cond < 250)
        if len(score[mask]) == len(score):
            print('mask condition not fulfilled, changing to 200')
            mask = np.where(cond < 200)
            if len(score[mask]) == len(score):
                print('mask condition not fulfilled, changing to 100')
                mask = np.where(cond < 100)
                if len(score[mask]) == len(score):
                    print('mask condition not fulfilled, changing to 80')
                    mask = np.where(cond < 80)
                    if len(score[mask]) == len(score):
                        print('mask condition not fulfilled, changing to 50')
                        mask = np.where(cond < 50)
                    
        
        score[mask] = 0
        
        # print(f"min: {population[np.argmin(score)]}, score: {np.min(score)}, condition = {cond}")
        
        best = np.min(score)
        print(f"Best score: {best} (iteration {i})")
        if best < best_score:
            best_index = np.argmin(score)
            best_genes = population[best_index]
            print(f"index: {best_index}")
            print(np.where(population[0])[0])
            best_score = best
        
        
        # Generate new population
        elite = population[np.argpartition(score, n_elite)[:n_elite]]  # Take the top n_elite
        
        random = rng.permuted(np.tile(x0, n_random).reshape(n_random, x0.size), axis=1)
        random = np.hstack((existing_population_for_random, random))
        
        mutated = rng.choice(elite, n_mutated).astype(bool)
        for row in mutated:
            r = row[n_existing:]
            true_cols = np.where(r)[0]
            false_cols = np.where(~r)[0]
            row[rng.choice(true_cols, size=(mutation_bits), replace=False)] = False
            row[rng.choice(false_cols, size=(mutation_bits), replace=False)] = True
            
        population_new = np.concatenate((elite, random, mutated))
        

        
    
    minima = population[np.argmin(score)]

    # results = minima 
    
    int_exp = get_intersections(polygon['geometry'], data=data)
    
    existing_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')

    existing_stations['lon'] = existing_stations['long']
    existing_stations['geometry'] = existing_stations['coords']

    n_existing = len(existing_stations)

    print(f"{n_existing} existing, {polygon['n_stations']} total")
    
    int_exp['existing'] = False
    
    existing_stations['existing'] = True

    int_exp = pd.concat((existing_stations[['lat', 'lon', 'coords', 'geometry', 'existing']], int_exp))
    int_exp = int_exp.reset_index()

    selected_intersections = int_exp[minima == 1]
        
    
    #%%
    # int_exp[selection_so == 1]
    bk = plot_intersections(selected_intersections, websocket_origin=['130.225.39.60', 'localhost:3000'], polygons=polygon, vdims=['existing'])
    '''
    bk.stop()
    '''


    #%% Prior existing stations
    
    data = bs.Data('nyc', 2019, 9)
    
    station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
    expansion_area = gpd.read_file('data/nyc/expansion_2019_area.geojson')
    
    polygon = expansion_area.loc[0]
    
    polygon['n_stations'] = 58
    
    # int_exp = get_intersections(polygon['geometry'], data=data)
   
    existing_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')
   
    existing_stations['lon'] = existing_stations['long']
    existing_stations['geometry'] = existing_stations['coords']
    
    existing_stations['existing'] = True

    sept_stations = existing_stations

    #%% Real stations
    
    data = bs.Data('nyc', 2019, 11)
    
    station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
    expansion_area = gpd.read_file('data/nyc/expansion_2019_area.geojson')
    
    polygon = expansion_area.loc[0]
    
    polygon['n_stations'] = 58
    
    # int_exp = get_intersections(polygon['geometry'], data=data)
   
    existing_stations = gpd.sjoin(station_df, gpd.GeoDataFrame(geometry=[polygon['geometry']], crs='epsg:4326'), op='within')
   
    existing_stations['lon'] = existing_stations['long']
    existing_stations['geometry'] = existing_stations['coords']
    
    existing_stations['existing'] = False
    
    existing_stations.loc[existing_stations['stat_id'].isin(sept_stations['stat_id']), 'existing'] = True

    
    all_stations = pd.concat([sept_stations, existing_stations])
    
    # point_info = get_point_info(data, existing_stations, land_use, census_df)
    
    months = [1,2,3,4,5,6,7,8,9, 10, 11]
    asdf = asdf_months(data, months)
    
    traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')
    cols = ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational',
            'pop_density', 'nearest_subway_dist', 'nearest_railway_dist']
    day_type = 'business_days'
    min_trips = 8
    clustering = 'k_means'
    k = 5
    seed = 42
    triptype = 'b_trips'
    asdf, clusters, labels = get_clusters(
        traffic_matrices, asdf, day_type, min_trips, clustering, k, seed)

    if data.city in ['helsinki', 'oslo', 'madrid', 'london']:
        df_cols = [col for col in cols if col != 'percent_industrial']
    else:
        df_cols = cols

    model_results = ipu.linear_regression(asdf, df_cols, triptype)
    
    existing_stations['const'] = 1
    
    pred = model_results.predict(existing_stations[['const', *df_cols]])
    
    def obj_fun(x):
        return -np.sum(x*pred)
    
    score = -np.sum(pred)
    print(f"{score=}")
    '''
    bk = plot_intersections(existing_stations, websocket_origin=['130.225.39.60', 'localhost:3000'], polygons=polygon, vdims=['existing', 'stat_id'])

    bk.stop()
    '''
    #%% Figure existing and new stations
    
    extend = (existing_stations['lat'].min(), existing_stations['long'].min(), 
              existing_stations['lat'].max(), existing_stations['long'].max())
    
    ex_pro = existing_stations.to_crs(data.laea_crs)
    
    pol_pro = expansion_area.to_crs(data.laea_crs)
    
    old = ex_pro[ex_pro['existing'] == True]
    new = ex_pro[ex_pro['existing'] == False]
    
    
    fig, ax = plt.subplots(figsize=(4.5, 10))
    
    pol_pro.plot(label='Expansion area', alpha=0.5, marker='s', ax=ax, color='tab:orange', edgecolor="tab:orange")
    old.plot(label='Existing stations',ax=ax, color="tab:blue")
    new.plot(label='New stations', ax=ax, color="tab:red")
    
    legend = ax.legend()
    ax.axis('off')
    patch = mpatches.Patch(color='tab:orange', label='Expansion Area', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    # handles is a list, so append manual patch
    handles.append(patch) 
    
    # plot the legend
    plt.legend(handles=handles)
    
    scalebar = AnchoredSizeBar(ax.transData, 1000, 
                                f'{1} km', 'lower left', 
                                pad=1, color='black', frameon=False, 
                                size_vertical=5)
    ax.add_artist(scalebar)
    
    cx.add_basemap(ax, crs=data.laea_crs, attribution="")
    text = "(C) Stamen Design, (C) OpenStreetMap Contributors"
    ax.text(0.005, 0.005, text, transform=ax.transAxes, size=8,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="w")], wrap=True,)
    plt.savefig("figures/nyc existing and new expansion.png", bbox_inches='tight', dpi=150)
    
    #%% Figure expansion area in whole city

    
    
    
    corner_points = Polygon(
            [(station_df['long'].min(), station_df['lat'].min()),
             (station_df['long'].min(), station_df['lat'].max()),
             (station_df['long'].max(), station_df['lat'].max()),
             (station_df['long'].max(), station_df['lat'].min())])
    
    cpoints = gpd.GeoDataFrame(geometry=[corner_points], crs="epsg:4326").to_crs(data.laea_crs)
    
    data = bs.Data('nyc', 2019, 9)
    
    station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
    
    fig, ax = plt.subplots(figsize=(7, 10))
    
    cpoints.plot(alpha=0, ax=ax)
    pol_pro.plot(label='Expansion area', alpha=0.5, marker='s', ax=ax, color='tab:orange', edgecolor="tab:orange")
    station_df.to_crs(data.laea_crs).plot(ax=ax, label='Existing stations Sept. 2019', markersize=15)
    # old.plot(label='Existing stations',ax=ax, color="tab:orange")
    # new.plot(label='New stations', ax=ax, color="tab:brown")
    
    ax.legend()
    ax.axis('off')
    patch = mpatches.Patch(color='tab:orange', label='Expansion Area', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    # handles is a list, so append manual patch
    handles.append(patch) 
    
    # plot the legend
    plt.legend(handles=handles, loc="upper right")
    
    scalebar = AnchoredSizeBar(ax.transData, 1000, 
                                f'{1} km', 'lower left', 
                                pad=1, color='black', frameon=False, 
                                size_vertical=5)
    ax.add_artist(scalebar)
    
    cx.add_basemap(ax, crs=data.laea_crs, attribution="")
    text = "(C) Stamen Design, (C) OpenStreetMap Contributors"
    ax.text(0.005, 0.005, text, transform=ax.transAxes, size=8,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="w")], wrap=True,)
    plt.savefig("figures/nyc_expansion_area.png", bbox_inches='tight', dpi=150)
    
    #%% Figure intersections
    
    extend = (existing_stations['lat'].min(), existing_stations['long'].min(), 
              existing_stations['lat'].max(), existing_stations['long'].max())
    
    ex_pro = existing_stations.to_crs(data.laea_crs)
    
    pol_pro = expansion_area.to_crs(data.laea_crs)
    
    old = ex_pro[ex_pro['existing'] == True]
    
    expansion_area = gpd.read_file('data/nyc/expansion_2019_area.geojson')
    int_exp = get_intersections(expansion_area.loc[0, 'geometry'], data=data)
    

    
    fig, ax = plt.subplots(figsize=(4.5, 10))
    
    sub_polygons.to_crs(data.laea_crs).plot(label='Expansion area', alpha=0.5, marker='s', ax=ax, color='tab:orange', edgecolor="black", linewidth=1)
    # pol_pro.plot(label='Expansion area', alpha=0.5, marker='s', ax=ax, color='tab:orange', edgecolor="tab:orange")
    old.plot(label='Existing stations',ax=ax, color="tab:blue", alpha=1, zorder=1.5)
    int_exp.to_crs(data.laea_crs).plot(label='Intersections', ax=ax, color="tab:red", alpha=1, markersize=10)

    legend = ax.legend()
    ax.axis('off')
    patch = mpatches.Patch(color='tab:orange', label='Expansion Area', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    # handles is a list, so append manual patch
    handles.append(patch) 
    
    # plot the legend
    plt.legend(handles=handles)
    
    scalebar = AnchoredSizeBar(ax.transData, 1000, 
                                f'{1} km', 'lower left', 
                                pad=1, color='black', frameon=False, 
                                size_vertical=5)
    ax.add_artist(scalebar)

    
    cx.add_basemap(ax, crs=data.laea_crs, attribution="")
    text = "(C) Stamen Design, (C) OpenStreetMap Contributors"
    ax.text(0.005, 0.005, text, transform=ax.transAxes, size=8,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="w")], wrap=True,)
    plt.savefig("figures/nyc_exp_intersections.png", bbox_inches='tight', dpi=150)
    
    #%% Figure subdivision (run again with better settings)
    
    extend = (existing_stations['lat'].min(), existing_stations['long'].min(), 
              existing_stations['lat'].max(), existing_stations['long'].max())
    
    ex_pro = existing_stations.to_crs(data.laea_crs)
    
    pol_pro = expansion_area.to_crs(data.laea_crs)
    
    old = ex_pro[ex_pro['existing'] == True]
    
 
    
    fig, ax = plt.subplots(figsize=(4.5, 10))
    
    sub_polygons.to_crs(data.laea_crs).plot(label='Expansion area', alpha=0.5, marker='s', ax=ax, color='tab:orange', edgecolor="black")
    old.plot(label='Existing stations',ax=ax, color="tab:blue", alpha=1, zorder=1.5)
    selected_intersections.to_crs(data.laea_crs).plot(label='Selected intersections', ax=ax, color="tab:red", alpha=1, markersize=30)

    legend = ax.legend()
    ax.axis('off')
    patch = mpatches.Patch(color='tab:orange', label='Expansion area', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    # handles is a list, so append manual patch
    handles.append(patch) 
    
    # plot the legend
    plt.legend(handles=handles)
    
    scalebar = AnchoredSizeBar(ax.transData, 1000, 
                                f'{1} km', 'lower left', 
                                pad=1, color='black', frameon=False, 
                                size_vertical=5, path_effects=[patheffects.withStroke(linewidth=20, foreground="w")])
    
    scalebar.set_path_effects([patheffects.Stroke(linewidth=3, foreground='w'),])
    sc = ax.add_artist(scalebar)

    
    cx.add_basemap(ax, crs=data.laea_crs, attribution="")
    text = "(C) Stamen Design, (C) OpenStreetMap Contributors"
    ax.text(0.005, 0.005, text, transform=ax.transAxes, size=8,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="w")], wrap=True,)
    
    plt.savefig(f"figures/nyc_exp_selected_intersections_{maxdist}.png", bbox_inches='tight', dpi=150)
    
    #%% Figure subdivision real (run existing_stations with november)
    
    extend = (existing_stations['lat'].min(), existing_stations['long'].min(), 
              existing_stations['lat'].max(), existing_stations['long'].max())
    
    ex_pro = existing_stations.to_crs(data.laea_crs)
    
    pol_pro = expansion_area.to_crs(data.laea_crs)
    
    old = ex_pro[ex_pro['existing'] == True]
    new = ex_pro[ex_pro['existing'] == False]
 
    
    fig, ax = plt.subplots(figsize=(4.5, 10))
    
    sub_polygons.to_crs(data.laea_crs).plot(label='Expansion area', alpha=0.5, marker='s', ax=ax, color='tab:orange', edgecolor="black")
    old.plot(label='Existing stations',ax=ax, color="tab:blue", alpha=1, zorder=1.5)
    # selected_intersections.to_crs(data.laea_crs).plot(label='Selected intersections', ax=ax, color="tab:red", alpha=1, markersize=30)
    new.plot(label='Real new stations',ax=ax, color="tab:red", alpha=1, zorder=1.5)
    legend = ax.legend()
    ax.axis('off')
    patch = mpatches.Patch(color='tab:orange', label='Expansion area', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    # handles is a list, so append manual patch
    handles.append(patch) 
    
    # plot the legend
    plt.legend(handles=handles)
    
    scalebar = AnchoredSizeBar(ax.transData, 1000, 
                                f'{1} km', 'lower left', 
                                pad=1, color='black', frameon=False, 
                                size_vertical=5, )
    ax.add_artist(scalebar)

    
    cx.add_basemap(ax, crs=data.laea_crs, attribution="")
    text = "(C) Stamen Design, (C) OpenStreetMap Contributors"
    ax.text(0.005, 0.005, text, transform=ax.transAxes, size=8,
            path_effects=[patheffects.withStroke(linewidth=2, foreground="w")], wrap=True,)
    plt.savefig("figures/nyc_exp_selected_real.png", bbox_inches='tight', dpi=150)
    
    #%% Full model predictions of placed station traffic
    
    import full_model
    
    selected_intersections = selected_intersections.reset_index()
    
    data = bs.Data('nyc', 2019, 9)
    
    station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)   
    selected_point_info = get_point_info(data, selected_intersections[['geometry', 'coords']], land_use, census_df)
    
    
    variables_list = ['percent_residential', 'percent_commercial',
                      'percent_recreational', 
                      'pop_density', 'nearest_subway_dist',
                      'nearest_railway_dist', 'center_dist']
    
    months = [1,2,3,4,5,6,7,8,9]
    asdf = asdf_months(data, months)
    
    traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')
    
    asdf, b, c = get_clusters(traffic_matrices, asdf, day_type='business_days', min_trips=8, clustering='k_means', k=5)
    
    
    
    # data, asdf, traf_mat = full_model.load_city('nyc')

    model = full_model.FullModel(variables_list)
    asdf2 = model.fit(asdf, traffic_matrices)
    
    #%% Compute objective function value
    
    pi = get_point_info(data, selected_intersections.reset_index(), land_use, census_df)
    
    solution_score = model_results.predict(pi[['const', *df_cols]])
    
    print(sum(solution_score))
