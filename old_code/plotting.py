# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:52:51 2021

@author: Mattek Group 3
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import contextily as ctx


import bikeshare as bs

def plot_graph(
        data, days, savefig = False, ext = 'pdf', colorbar = True,
        basemap = False, normalise_degrees = False, node_size = 20,
        edge_width = 0.2, edge_color = 'black', edge_alpha = 0.8,
        adj_threshold = 1, remove_self_loops = True):
    """
    Plots the graph of a bike-sharing network using its weighted adjacency
    matrix and degree matrix. Has the ability to add a street map as a base
    map.

    Parameters
    ----------
    data : class
        Data class of the network.
    days : Tuple
        Days in consideration.
    savefig : bool, optional
        Saves the figure if True. The default is False.
    ext : str, optional
        specifies the file extension of the saved figure. The default is 'pdf'.
    colorbar : bool, optional
        Adds a colorbar if True. The default is True.
    basemap : bool, optional
        Adds a basemap if True. The default is False.
    normalise_degrees : bool, optional
        Normalises the degrees with respect to the average degree across all
        nodes if True. The default is False.
    node_size : float, optional
        Sets the size of the nodes on the graph. The default is 10.
    edge_width : float, optional
        Sets the width of the edges on the graph. The default is 0.1.
    edge_color : str, optional
        Sets the color of the edges. The default is 'black'.
    edge_alpha : float, optional
        Sets the alpha/transparency of the edges. The default is 0.8.

    Returns
    -------
    None.

    """
    
    figsize_dict = {'nyc': (5,8),
                        'chic': (5,8),
                        'london': (8,5),
                        'oslo': (6.6,5),
                        'sfran': (6,5),
                        'washDC': (8,7.7),
                        'madrid': (6,8),
                        'mexico': (7.2,8), 
                        'taipei': (7.3,8)}
    
    adj_matrix = data.adjacency(days, threshold = adj_threshold, remove_self_loops=remove_self_loops)
    deg_matrix = data.get_degree_matrix(days)
    
    graph = nx.from_numpy_matrix(adj_matrix)
    
    degrees = [deg_matrix[i,i] for i in range(len(deg_matrix))]
    
    if normalise_degrees:
        print('Normalising degrees...')
        degrees = degrees/np.mean(degrees)
    
    lat = [data.stat.loc_merc[i][0] for i in range(data.stat.n_tot)]
    long = [data.stat.loc_merc[i][1] for i in range(data.stat.n_tot)]
        
    extent = np.array([np.min(lat)-1000, np.max(lat)+1000, np.min(long)-1000, np.max(long)+1000])
    
    # aspect_ratio = (extent[3]-extent[2])/(extent[1]-extent[0])
    # width = 8
    try:
        fig, ax = plt.subplots(figsize=figsize_dict[data.city])
    except KeyError:
        fig, ax = plt.subplots(figsize=(8,8))
    ax.axis(extent)
    
    
    
    print('Drawing network...')
    nx.draw_networkx_nodes(graph, data.stat.loc_merc, node_size=node_size, node_color=degrees, cmap='YlGnBu_r', ax=ax)
    nx.draw_networkx_edges(graph, data.stat.loc_merc, alpha=edge_alpha, width=edge_width, edge_color=edge_color, ax=ax)
    ax.set_aspect(1)
    
    if colorbar:
        print('Adding colorbar...')
        fig.set_figwidth((1+0.24)*fig.get_size_inches()[0])
        
        cmap = mpl.cm.YlGnBu_r
        norm = mpl.colors.Normalize(vmin=np.floor(min(degrees)), vmax=np.ceil(max(degrees)))
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      orientation='vertical', label='degree')
        
    if basemap:
        print("Adding basemap...")
        #ctx.add_basemap(ax)
        ctx.add_basemap(ax, source=ctx.providers.Thunderforest.OpenCycleMap(apikey='7b7ec633799740d18aa9c877e5515b78'))
    
    plt.grid(False)
    _ = plt.axis('off')
    
    plt.tight_layout()
    
    if savefig:
        print('Saving figure...')
        filename = 'figures/graph_{}_{}_to_{}-{}-{}.'.format(data.city, min(days), max(days), data.month, data.year) + ext
        plt.savefig(filename)
    
    return fig, ax
    
if __name__ == '__main__':
    
    DATA = bs.Data("nyc", 2019, 9)
    
    DAYS = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)
    #DAYS = (2,3,4,5,6,9,10,11,12,13,16,17,18,19,20,23,24,25,26,27,30)
    #DAYS = (1,7,8,14,15,21,22,28,29)
    
    fig, ax = plot_graph(DATA, DAYS, savefig = True, ext = 'png', normalise_degrees=True, basemap = False, colorbar=False)
