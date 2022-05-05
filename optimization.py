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

    service_area, service_area_size = ipu.get_service_area(data.city, intersections, land_use, voronoi=False)
    
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
    
    import scipy.optimize as so
    import cvxpy
    
    def obj_fun(C):
        d = np.array([0.5, 0.6, 0.8, 0.9])
        return -np.sum(C*d)
    
    def con_fun(C):
        return np.sum(C)
    
    def con_val(C):
        return ((C == [1,1,1,1]) + (C == [0,0,0,0])).astype(int)
    
    sum_constraint = so.LinearConstraint(np.array([1,1,1,1]), 2, 2)
    
    sum_constraint = so.NonlinearConstraint(con_fun, 2, 2)
    
    val_constraint = so.Bounds([0, 0, 0, 0], [1, 1, 1, 1])
    
    so.minimize(obj_fun, x0=np.array([0, 1, 0, 1]), constraints=(sum_constraint), bounds=val_constraint)
    
    
    data = np.array([0.5, 0.6, 0.8, 0.9])
    
    selection = cvxpy.Variable(shape=4, boolean=True)
    
    constraint = cvxpy.sum(selection) == 2

    cost = cvxpy.sum(cvxpy.multiply(selection, data))
    
    problem = cvxpy.Problem(cvxpy.Maximize(cost), constraints=[constraint])
    
    score = problem.solve(solver=cvxpy.GLPK_MI)
    
    import pickle
    import logistic_regression
    import shapely
    
    cols = ['percent_residential', 'percent_commercial', 'percent_industrial', 'percent_recreational',
            'pop_density', 'nearest_subway_dist', 'nearest_railway_dist']
    day_type = 'business_days'
    min_trips = 8
    clustering = 'k_means'
    k = 5
    seed = 42
    triptype = 'b_trips'
    
    data = bs.Data(city, year, None)

    # station_df, land_use, census_df = ipu.make_station_df(data, holidays=False, return_land_use=True, return_census=True)
    traffic_matrices = data.pickle_daily_traffic(holidays=False, user_type='Subscriber')
    # station_df, clusters, labels = ipu.get_clusters(
    #     traffic_matrices, station_df, day_type, min_trips, clustering, k, seed)
    
    # asdf, clusters, labels = ipu.get_clusters(traf_mats, asdf, 'business_days', 10, 'k_means', k, 42)
    try:
        with open(f'./python_variables/{data.city}{year}_avg_stat_df.pickle', 'rb') as file:
            asdf = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f'The average station DataFrame for {data.city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
        
    # mask = ~asdf['n_trips'].isna()
    
    # asdf = asdf[mask]
    # asdf = asdf.reset_index(drop=True)
    
    asdf, clusters, labels = ipu.get_clusters(
        traffic_matrices, asdf, day_type, min_trips, clustering, k, seed)
    
    if city in ['helsinki', 'oslo', 'madrid', 'london']:
        df_cols = [col for col in cols if col != 'percent_industrial']
    else:
        df_cols = cols
    
    model_results = logistic_regression.linear_regression(asdf, df_cols, triptype)
    
    pred = model_results.predict(point_info[['const', *df_cols]])
    
    #%%
    
    n = len(point_info)
    
    n_selected = 100
    
    data = pred
    
    int_proj = intersections.to_crs(epsg=3857)
    
    distances = np.zeros((n, n))
    
    for i in range(n):
        distances[i] = int_proj.distance(int_proj.geometry.loc[i])
    
    for i in range(n):
        distances[i, i] = 0
        
    # distances[np.where(distances < 500)] = 1000000
    
    dist_matrix = cvxpy.Constant(distances)
    
    sa = intersections['service_area'].to_crs(epsg=3857)
    shapely.ops.unary_union(sa).area
    
    saa = sa.area

    selection = cvxpy.Variable(shape=n, boolean=True)
    
    constraint = cvxpy.sum(selection) == n_selected
    
    # distance_constraint = cvxpy.min(distances[selection == 1][:, selection == 1]) >= 500
    disto = cvxpy.max(dist_matrix @ selection)
    disto = cvxpy.diag(selection) @ dist_matrix @ cvxpy.diag(selection)
    disto = cvxpy.sum(cvxpy.diag(selection) @ dist_matrix)
        
    distance_constraint = disto <= 100000000
    
    cost = cvxpy.sum(cvxpy.multiply(selection, pred))
    
    problem = cvxpy.Problem(cvxpy.Maximize(cost), constraints=[constraint])
    
    score = problem.solve(solver=cvxpy.GLPK_MI)
    
    print(selection.value)
    
    
    #%% SO opti. Too slow. 2 iterations takes many hours with SLSQP solver
    n = len(point_info)
    
    # pred = pred[:1000]
    
    # n = 1000
    
    n_select = 100
    
    def obj_fun(x):
        return -np.sum(x*pred)
    
    sum_constraint = so.LinearConstraint(np.array([[1]*n]), n_select, n_select)
    sum_constraint = so.NonlinearConstraint(np.sum, 0, n_select)
    bounds = so.Bounds([0]*n, [1]*n)
    
    x0 = np.zeros(n)
    x0[:n_select] = 1
    np.random.seed(42)
    x0 = np.random.permutation(x0)
    
    minimum = so.minimize(obj_fun, x0=x0, constraints=(sum_constraint), bounds=bounds, method='SLSQP', options={'maxiter': 1})
    
    selection_idx = np.argpartition(minimum.x, -n_select)[-n_select:]
    
    selection_so = np.zeros(n)
    selection_so[selection_idx] = 1

    
    #%% linprog works and within a reasonable time but can only use linear constraints
    A_eq = np.array([[1]*n])
    
    lim = so.linprog(-pred, A_eq=A_eq, b_eq=n_select, bounds=(0,1), options={'maxiter': 10})
    
    selection_idx = np.argpartition(lim.x, -n_select)[-n_select:]
    
    selection_so = np.zeros(n)
    selection_so[selection_idx] = 1
    
    #%% gekko
    
    n = len(point_info)

    n_select = 100
    
    import time
    
    from gekko import GEKKO
    
    t_pre = time.time()
    m = GEKKO()
    
    # help(m)
    
    
    
    c = [m.Const(pred_i) for pred_i in pred]
        
    # x = [m.Var(lb=0, ub=1) for i in range(n)]
    x = m.Array(m.Var, n, lb=0, ub=1, integer=True)
    
    m.Equation(m.sum(x) == 100)
    
    # m.Equation(x @ distances @ x  > 100)

    m.Maximize(m.sum([x_i*pred_i for x_i, pred_i in zip(x, pred)]))
    
    m.solve()    
    
    solution_gekko = np.array([x_i.value for x_i in x]).reshape(-1)

    selection_idx = np.argpartition(solution_gekko, -n_select)[-n_select:]
    
    selection_gekko = np.zeros(n)
    selection_gekko[selection_idx] = 1
    
    print(f"time taken: {time.time() - t_pre}")
    
    #%%
    
    def condition(x):
        return np.min(distances[x][:,x][distances[x][:,x] != 0])
    
    x0 = np.zeros(n, dtype=bool)
    x0[:n_select] = 1
    np.random.seed(42)
    x0 = np.random.permutation(x0)
    
    rng = np.random.default_rng(42)
   
    n_per = 2000000
    
    perms = rng.permuted(np.tile(x0, n_per).reshape(n_per, x0.size), axis=1)
    
    #%% multi
    
    import multiprocessing
    
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
    
    # result = parallel_apply_along_axis(condition, 1, perms)
    
    # spaced_idx = np.where(result > 100)[0]
    # spaced_candidates = perms[spaced_idx]
    
    def obj_fun(x):
        return -np.sum(x*pred)
    
    # scores = parallel_apply_along_axis(obj_fun, 1, spaced_candidates)
    
    # selection_score = spaced_candidates[np.argmin(scores)]
    
    #%% DIY GA
    
    n = len(point_info)

    n_select = 100
    
    batch_size = 1000
    n_iters = 100
    elite_percentage = 0.2
    random_percentage = 0.2
    children_percentage = 0.2
    mutated_percentage = 0.4
    
    mutation_bits = 1
    
    def condition(x):
        return np.min(distances[x][:,x][distances[x][:,x] != 0])
    
    
    x0 = np.zeros(n, dtype=bool)
    x0[:n_select] = 1
    np.random.seed(42)
    x0 = np.random.permutation(x0)
    
    rng = np.random.default_rng(42)
   
    n_per = batch_size
    
    n_elite = int(np.floor(batch_size*elite_percentage))
    n_random = int(np.floor(batch_size*random_percentage))
    n_children = int(np.floor(batch_size*children_percentage))
    n_mutated = int(np.floor(batch_size*mutated_percentage))
    
    
    population = rng.permuted(np.tile(x0, n_per).reshape(n_per, x0.size), axis=1)
    
    best_score = 0
    
    
    for i in range(n_iters):
        score = parallel_apply_along_axis(obj_fun, 1, population)
        best = np.min(score)
        print(f"Best score: {best} (iteration {i})")
        if best < best_score:
            best_index = np.argmin(score)
            best_genes = population[best_index]
            print(f"index: {best_index}")
            print(np.where(population[0])[0])
            best_score = best
            
            
        
        # elite = population[np.argpartition(score, n_elite)[:n_elite]]  # Take the top n_elite
        
        # random = rng.permuted(np.tile(x0, n_random).reshape(n_random, x0.size), axis=1)
        
        score_a = score + 300
        
        probabilities = -1*score_a / (-1 * score_a.sum())
        
        mating_pool = rng.choice(population, n_per, p=probabilities)
        
        # mating_pool = rng.permutation(mating_pool)
        
        previous_population = population.copy()
        
        n_children = 400
        
        
        
        # Create children
        # parents = rng.permutation(mating_pool)
        parents = rng.choice(mating_pool, n_children*2)
        # parents = mating_pool[:2*n_children]
        parent1 = parents[:n_children]
        parent2 = parents[n_children:2*n_children]
        
        rows, cols = np.where(parent1)
        index = cols.reshape((n_children, -1))
        idx = rng.random(index.shape).argsort(0)
        genes1 = rng.choice(idx, size=(50), axis=1, replace=False)

        rows2, cols2 = np.where(parent2)
        index2 = cols2.reshape((n_children, -1))
        idx2 = rng.random(index2.shape).argsort(0)
        genes2 = rng.choice(idx2, size=(50), axis=1, replace=False)
        
        children_idx = np.hstack((genes1, genes2))
        children_cols = children_idx.flatten()
        children = np.zeros((n_children, n), dtype=bool)
        children[rows, children_cols] = True
        
        n_add_child_genes = n_select - children.sum(axis=1)
        add_rows, add_cols = np.where(~children)
        
        for n_add, row in zip(n_add_child_genes, children):
            cols = np.where(~row)[0]
            row[rng.choice(cols, size=(n_add), replace=False)] = True
        
        copies = rng.choice(mating_pool, n_per - n_children)
        
        population = np.vstack((children, copies))
        
        mutate_idx = rng.integers(0, n_per, size=50)
        
        mutated = population[mutate_idx]
        
        for row in mutated:
            true_cols = np.where(row)[0]
            false_cols = np.where(~row)[0]
            
            row[rng.choice(true_cols, size=(mutation_bits), replace=False)] = False
            row[rng.choice(false_cols, size=(mutation_bits), replace=False)] = True
    
        population[mutate_idx] = mutated
        # population = np.vstack((elite, random, children, mutated))
    
        
    
    #%% GA
    
    from geneticalgorithm import geneticalgorithm as ga
    
    def obj_fun(x):
        return -np.sum(x*pred)
    
    
    model=ga(function=obj_fun,dimension=n,variable_type='bool')
    
    model.run()
    
    #%%
    
    bk = plot_intersections(intersections[selection_score == 1], websocket_origin=('130.225.39.60'))
    '''
    bk.stop()
    '''
