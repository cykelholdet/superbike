#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:47:27 2022

@author: dbvd
"""
import time
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import skimage.color as skcolor

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import BallTree
from sklearn_extra.cluster import KMedoids

import simpledtw as dtw
import bikeshare as bs
import interactive_plot_utils as ipu

def get_clusters(traffic_matrices, station_df, day_type, min_trips, 
                 clustering, k, random_state=None):
    """
    From a station dataframe and associated variables, return the updated 
    station df and clustering and labels

    Parameters
    ----------
    station_df : pandas dataframe
        has each station as a row.
    day_type : str
        'weekend' or 'business_days'.
    min_trips : int
        minimum number of trips.
    clustering : str
        clustering type.
    k : int
        number of clusters.
    random_state : int
        the seed for the random generator.

    Returns
    -------
    station_df : pandas dataframe
        has each station as a row and color and label columns populated.
    clusters : sklearn.clustering cluster
        can be used for stuff later.
    labels : np array
        the labels of the masked traffic matrix.

    """
    traffic_matrix, mask, x_trips = ipu.mask_traffic_matrix(
        traffic_matrices, station_df, day_type, 
        min_trips, holidays=False, return_mask=True)
    
    traffic_matrix = traffic_matrix[:,:24] - traffic_matrix[:,24:]

    # if cluster_what == 'Relative difference':
    #     traffic_matrix = traffic_matrix[:,:24] - traffic_matrix[:,24:]
    
    # else:
    #     traffic_matrix = traffic_matrix[:,:24] + traffic_matrix[:,24:]
    
    if clustering == 'k_means':
        clusters = KMeans(k, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        # station_df, means, labels = sort_clusters(station_df, means, labels, traffic_matrices, day_type, k)
        # clusters = clusters.cluster_centers_
        # labels = station_df['label']
        station_df, clusters, labels = sort_clusters2(station_df, 
                                                      clusters.cluster_centers_, 
                                                      labels)
        station_df['color'] = station_df['label'].map(cluster_color_dict)

    elif clustering == 'k_medoids':
        clusters = KMedoids(k, random_state=random_state).fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        station_df, clusters, labels = sort_clusters2(station_df, clusters.cluster_centers_, 
                                                      labels)
        station_df['color'] = station_df['label'].map(cluster_color_dict)
        
    elif clustering == 'h_clustering':
        clusters = None
        labels = AgglomerativeClustering(k).fit_predict(traffic_matrix)
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        clusters = cluster_mean(traffic_matrix, station_df, labels, k)
        station_df, clusters, labels = sort_clusters2(station_df, clusters, labels)
        station_df['color'] = station_df['label'].map(cluster_color_dict)
    
    elif clustering == 'gaussian_mixture':
        
        clusters_init = KMeans(k, random_state=random_state).fit(traffic_matrix).cluster_centers_
        
        clusters = GaussianMixture(k, n_init=100, means_init = clusters_init,
                                   random_state=random_state, 
                                   verbose=2).fit(traffic_matrix)
        labels = clusters.predict_proba(traffic_matrix)
        station_df.loc[mask, 'label'] = pd.Series(list(labels), 
                                                  index=mask[mask].index)
        station_df.loc[~mask, 'label'] = np.nan
        # station_df, clusters, labels = sort_clusters2(station_df, clusters.means_, 
        #                                               labels, 
        #                                               cluster_type='gaussian_mixture',
        #                                               mask=mask)
        station_df.loc[mask, 'label_prob'] = station_df['label']
        station_df.loc[~mask, 'label_prob'] = np.nan
        station_df.loc[mask, 'label'] = [np.argmax(x) for x in station_df['label_prob'].loc[mask]]
        station_df.loc[~mask, 'label'] = np.nan
        
        lab_mat = np.array(lab_color_list[:k]).T
        lab_cols = [np.sum(station_df['label_prob'][mask][i] * lab_mat, axis=1) for i in station_df['label_prob'][mask].index]
        labels_rgb = skcolor.lab2rgb(lab_cols)
        station_df.loc[mask, 'color'] = ['#%02x%02x%02x' % tuple(label.astype(int)) for label in labels_rgb*255]
        station_df.loc[~mask, 'color'] = 'gray'
        
        clusters = clusters.means_
        labels = station_df['label']
        
    elif clustering == 'none':
        clusters = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = station_df[x_trips].tolist()
    
    elif clustering == 'zoning':
        clusters = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = [cluster_color_dict[zone] for zone in pd.factorize(station_df['zone_type'])[0]]
        
    else:
        clusters = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = None
    
    dist_to_center = np.full(traffic_matrices[0].shape[0], np.nan)
    if day_type == 'business_days':
        traf_mat = traffic_matrices[0][:,:24] - traffic_matrices[0][:,24:]
    elif day_type == 'weekend':
        traf_mat = traffic_matrices[1][:,:24] - traffic_matrices[1][:,24:]
                 
        dist_to_center[np.where(labels == i)] = np.linalg.norm(
            traf_mat[np.where(labels==i)] - clusters[i], axis=1)
    
    station_df['dist_to_center'] = dist_to_center

    return station_df, clusters, labels

def sort_clusters(station_df, cluster_means, labels, traffic_matrices, day_type, k, cluster_type=None, mask=None):
    # Order the clusters by setting cluster 0 to be closest to the mean traffic.
    
    if day_type == 'business_days':
        
        traffic_matrices = traffic_matrices[0][station_df.index][:,:24] - traffic_matrices[0][station_df.index][:,24:]
        # mean = np.mean(traffic_matrices)
        
        # mean = np.mean(traffic_matrices[0][station_df.index], axis=0)
    # elif day_type == 'weekend':
        # mean = np.mean(traffic_matrices[1][station_df.index], axis=0)
    
    morning_hours = np.array([6,7,8,9,10])
    afternoon_hours = np.array([15,16,17,18,19])
    
    #mean = mean/np.max(mean)
    
    peakiness = [] # Distance to mean
    rush_houriness = [] # difference between arrivals and departures
    for center in cluster_means:
        # dist_from_mean = np.linalg.norm(center-mean)
        dist_from_mean = np.linalg.norm(center)
        peakiness.append(dist_from_mean)
        # rhn = np.sum((center[morning_hours] - center[morning_hours+24]) - (center[afternoon_hours] - center[afternoon_hours+24]))
        
        rhn = np.sum(center[morning_hours] - center[afternoon_hours])
        
        
        rush_houriness.append(rhn)
    
    rush_houriness = np.array(rush_houriness)
    first = np.argmin(np.array(peakiness)*0.5 + np.abs(rush_houriness))
    order = np.argsort(rush_houriness)
    
    order_new = order.copy()
    for i, item in enumerate(order):
        if item == first:
            order_new[i] = order[0]
            order_new[0] = first
    order = order_new
    
    # print(f"first = {first}")
    # print(order)
    # print(f"rush-houriness = {rush_houriness}")
    # print(f"peakiness = {peakiness}")
    
    # for i in range(k):
    #     if abs(rush_houriness[i]) < 0.05:
    #         print(f"small_rushouriness {i}")
    #     if peakiness[i] > 0.1:
    #         print(f'large peakiness {i}')
    
    # for i in range(k):
    #     if abs(rush_houriness[i]) < 0.05 and peakiness[i] > 0.1:
    #         temp = order[i]
    #         order[i] = order[-1]
    #         order[-1] = temp
    #         print(f'swapped {order[-1]} for {order[i]}')
    # print(labels[0:2])
    
        
    
    labels_dict = dict(zip(order, range(len(order))))
    print(labels_dict)
    
    if cluster_type == 'gaussian_mixture':
        values = np.zeros_like(labels)
        values[:,order] = labels[:,range(k)]
        
        station_df['label'].loc[mask] = pd.Series(list(values), index=mask[mask].index)
    else:
        station_df = station_df.replace({'label' : labels_dict})
    labels = station_df['label']
    
    centers = np.zeros_like(cluster_means)
    for i in range(k):
        centers[labels_dict[i]] = cluster_means[i]
    cluster_means = centers
    return station_df, cluster_means, labels


def sort_clusters2(station_df, cluster_means, labels, cluster_type=None, mask=None):
    
    k = len(cluster_means)
    
    max_peak = np.max(np.abs(cluster_means))
    
    morning_hours = np.array([6,7,8,9,10])
    afternoon_hours = np.array([15,16,17,18,19])
    
    morning_source = np.zeros(24)
    morning_source[morning_hours] = max_peak
    # morning_source[afternoon_hours] = -max_peak
    
    morning_sink = np.zeros(24)
    morning_sink[morning_hours] = -max_peak
    # morning_sink[afternoon_hours] = max_peak
    
    order = np.zeros(k)
    
    sinks_and_sources = set() # set containing label of all centers except reference
    
    count = 1
    while count < k:
        
        dist_to_morning_source = np.full(k, np.inf)
        dist_to_morning_sink = np.full(k, np.inf)
        
        for i, center in enumerate(cluster_means):
            if i not in sinks_and_sources: # Don't check previous winners
                center_normed = center/np.max(np.abs(center))*max_peak
                w = 1/np.linalg.norm(center, 1)
                dist_to_morning_source[i] = np.linalg.norm(center-morning_source)
                dist_to_morning_sink[i] = np.linalg.norm(center-morning_sink)
            
        # Compare the two closest and pick the winner
        if np.min(dist_to_morning_source) < np.min(dist_to_morning_sink):
            sinks_and_sources = sinks_and_sources | {np.argmin(dist_to_morning_source)}
        else:
            sinks_and_sources = sinks_and_sources | {np.argmin(dist_to_morning_sink)}
    
        count+=1
    
    # morning_source[afternoon_hours] = -max_peak
    # morning_sink[afternoon_hours] = max_peak
    
    dist_to_sink = np.full(k, np.inf)
    for label, center in enumerate(cluster_means):
        if label not in sinks_and_sources:
            order[0] = label
    
        else:
            dist_to_sink[label] = np.linalg.norm(center-morning_sink)
    
    order[1:] = np.argsort(dist_to_sink)[:k-1]
    
    labels_dict = dict(zip(order, range(len(order))))
    logging.debug(labels_dict)
    
    if cluster_type == 'gaussian_mixture':
        values = np.zeros_like(labels)
        order = [int(i) for i in order] # make elements in order integers
        values[:,order] = labels[:,range(k)]
        
        station_df['label'].loc[mask] = pd.Series(list(values), index=mask[mask].index)
    else:
        station_df = station_df.replace({'label' : labels_dict})
    labels = station_df['label']
    
    centers = np.zeros_like(cluster_means)
    for i in range(k):
        centers[labels_dict[i]] = cluster_means[i]
    cluster_means = centers
    
    return station_df, cluster_means, labels
    
    
def cluster_mean(traffic_matrix, station_df, labels, k):
    mean_vector = np.zeros((k, traffic_matrix.shape[1]))
    for j in range(k):
        mean_vector[j,:] = np.mean(traffic_matrix[np.where(labels == j)], axis=0)
    return mean_vector

def Davies_Bouldin_index(data_mat, centers, labels, verbose=False):
       """
       Calculates the Davies-Bouldin index of clustered data.

       Parameters
       ----------
       data_mat : ndarray
           Array containing the feature vectors.
       labels : itr, optional
           Iterable containg the labels of the feature vectors. If no labels
           are given, they are calculated using the mass_predict method.

       Returns
       -------
       DB_index : float
           Davies-Bouldin index.

       """

       k = len(centers)

       if verbose:
           print('Calculating Davies-Bouldin index...')

       pre = time.time()

       S_scores = np.empty(k)

       for i in range(k):
           data_mat_cluster = data_mat[np.where(labels == i)]
           distances = [np.linalg.norm(row-centers[i])
                        for row in data_mat_cluster]
           S_scores[i] = np.mean(distances)

       R = np.empty(shape=(k, k))
       for i in range(k):
           for j in range(k):
               if i == j:
                   R[i, j] = 0
               else:
                   R[i, j] = (S_scores[i] + S_scores[j]) / \
                       np.linalg.norm(centers[i]-centers[j])

       D = [max(row) for row in R]

       DB_index = np.mean(D)

       if verbose:
           print(f'Done. Time taken: {(time.time()-pre):.1f} s')

       return DB_index

def Dunn_index(data_mat, centers, labels=None, verbose=False):
    """
    Calculates the Dunn index of clustered data. WARNING: VERY SLOW.

    Parameters
    ----------
    data_mat : ndarray
        Array containing the feature vectors.
    labels : itr, optional
        Iterable containg the labels of the feature vectors. If no labels
        are given, they are calculated using the mass_predict method.

    Returns
    -------
    D_index : float
        Dunn index.

    """
    k = len(centers)

    if verbose:
        print('Calculating Dunn Index...')

    pre = time.time()

    intra_cluster_distances = np.empty(k)
    inter_cluster_distances = np.full(shape=(k, k), fill_value=np.inf)

    for i in range(k):
        data_mat_cluster = data_mat[np.where(labels == i)]
        cluster_size = len(data_mat_cluster)
        distances = np.empty(shape=(cluster_size, cluster_size))

        for h in range(cluster_size):
            for j in range(cluster_size):
                distances[h, j] = np.linalg.norm(data_mat[h]-data_mat[j])

        intra_cluster_distances[i] = np.max(distances)

        for j in range(k):
            if j != i:
                data_mat_cluster_j = data_mat[np.where(labels == j)]
                cluster_size_j = len(data_mat_cluster_j)
                between_cluster_distances = np.empty(
                    shape=(cluster_size, cluster_size_j))
                for m in range(cluster_size):
                    for n in range(cluster_size_j):
                        between_cluster_distances[m, n] = np.linalg.norm(
                            data_mat_cluster[m]-data_mat_cluster_j[n])
                inter_cluster_distances[i, j] = np.min(
                    between_cluster_distances)

    D_index = np.min(inter_cluster_distances) / \
        np.max(intra_cluster_distances)

    if verbose:
        print(f'Done. Time taken: {(time.time()-pre):.1f} s')

    return D_index

def silhouette_index(data_mat, centers, labels, verbose=False):
    """
    Calculates the silhouette index of clustered data.

    Parameters
    ----------
    data_mat : ndarray
        Array containing the feature vectors.
    labels : itr, optional
        Iterable containg the labels of the feature vectors. If no labels
        are given, they are calculated using the mass_predict method.

    Returns
    -------
    S_index : float
        Silhouette index.

    """
    k = len(centers)

    if verbose:
        print('Calculating Silhouette index...')

    pre = time.time()

    s_coefs = np.empty(len(data_mat))

    for i, vec1 in enumerate(data_mat):
        
        if (labels==labels[i]).sum() != 1: # if not singleton cluster
            
            in_cluster = np.delete(data_mat, i, axis=0)
            in_cluster = in_cluster[np.where(
                np.delete(labels, i) == labels[i])]
    
            in_cluster_size = len(in_cluster)
    
            in_cluster_distances = np.empty(in_cluster_size)
            for j, vec2 in enumerate(in_cluster):
                in_cluster_distances[j] = np.linalg.norm(vec1-vec2)
            
            ai = np.mean(in_cluster_distances)
        
        else:
            ai = 0
            
        mean_out_cluster_distances = np.full(k, fill_value=np.inf)

        for j in range(k):
            if j != labels[i]:
                out_cluster = data_mat[np.where(labels == j)]
                out_cluster_distances = np.empty(len(out_cluster))

                for l, vec2 in enumerate(out_cluster):
                    out_cluster_distances[l] = np.linalg.norm(vec1-vec2)

                mean_out_cluster_distances[j] = np.mean(
                    out_cluster_distances)

        bi = np.min(mean_out_cluster_distances)

        s_coefs[i] = (bi-ai)/max(ai, bi)

    S_index = np.mean(s_coefs)

    if verbose:
        print(f'Done. Time taken: {(time.time()-pre):.1f} s')

    return S_index

def SSE(data_mat, centers, labels, verbose=False):
    
    k = len(centers)
    
    if verbose:
        print('Calculating SSE...')
    
    pre = time.time()
    
    SSE_arr = np.zeros(k)
    
    for i in range(k):
        cluster = data_mat[labels==i]
        SSE_arr[i] = np.linalg.norm(cluster-centers[i], axis=1).sum()
        
    SSE = SSE_arr.sum()
    
    if verbose:
        print(f'Done. Time taken: {time.time()-pre:.1f} s')

    return SSE

def plot_cluster_centers(city, k=5, year=2019, month=None, day=None, 
                         cluster_seed=42, min_trips=8, n_table=False):
    if city != 'all':
        try:
            with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                asdf = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
    
        
        data = bs.Data(city, year, month, day)
        
        traf_mats = data.pickle_daily_traffic(holidays=False, 
                                              user_type='Subscriber',
                                              overwrite=False)
                
        mask = ~asdf['n_trips'].isna()
        
        asdf = asdf[mask]
        asdf = asdf.reset_index(drop=True)
        
        try:
            traf_mats = (traf_mats[0][mask], traf_mats[1])
        except IndexError:
            pass
                
        
        asdf, clusters, labels = get_clusters(traf_mats, asdf, 
                                              day_type='business_days', 
                                              min_trips=min_trips, 
                                              clustering='k_means', k=k, 
                                              random_state=cluster_seed)
        
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
        
        cluster_name_dict = {0 : 'Reference', 
                             1 : 'High morning sink', 
                             2 : 'Low morning sink',
                             3 : 'Low morning source',
                             4 : 'High morning source',
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
                
                try:
                    with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                        asdf = pickle.load(file)
                except FileNotFoundError:
                    raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        

                data = bs.Data(city, year, month, day)
                
                traf_mats = data.pickle_daily_traffic(holidays=False, 
                                                      user_type='Subscriber',
                                                      normalise=True,
                                                      overwrite=False) 
                
                mask = ~asdf['n_trips'].isna()
                
                asdf = asdf[mask]
                asdf = asdf.reset_index(drop=True)
                
                try:
                    traf_mats = (traf_mats[0][mask], traf_mats[1])
                except IndexError:
                    pass
                
                asdf, clusters, labels = get_clusters(traf_mats, asdf, 
                                                      day_type='business_days', 
                                                      min_trips=min_trips, 
                                                      clustering='k_means', k=k,
                                                      random_state=cluster_seed)
                
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

def n_table_formatter(x):
    
    if isinstance(x, tuple):
        n, p = x
        return f"\\multirow{{2}}{{*}}{{\\shortstack{{${n}$\\\$({p}\%)$}}}}"
    
    elif isinstance(x, str):
        return f"\\multirow{{2}}{{*}}{{{x}}}"
    
    else:
        return ""

def k_test_table(cities=None, year=2019, month=None, k_min=2, k_max=10,
                 cluster_seed=42, savefig=False, overwrite=False):
    
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
                                     cluster_seed=cluster_seed, 
                                     savefig=savefig, 
                                     overwrite=True)
        
    else:
        
        multiindex = pd.MultiIndex.from_product((cities, metrics))  
        
        res_table = pd.DataFrame(index=k_list, columns=multiindex)
        
        for city in cities:
            
            try:
                with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                    asdf = pickle.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
    
            
            data = bs.Data(city, year, month)
            
            traf_mats = data.pickle_daily_traffic(holidays=False, 
                                                  user_type='Subscriber',
                                                  overwrite=False)
                
            mask = ~asdf['n_trips'].isna()
            
            asdf = asdf[mask]
            asdf = asdf.reset_index(drop=True)
            
            try:
                traf_mats = (traf_mats[0][mask], traf_mats[1])
            except IndexError:
                pass
            
            DB_list = []
            D_list = []
            S_list = []
            SS_list = []
            
            for k in k_list:
                
                print(f'\nCalculating for k={k}...\n')
                
                asdf, clusters, labels = get_clusters(traf_mats, asdf, 
                                                      day_type='business_days', 
                                                      min_trips=8, 
                                                      clustering='k_means', k=k, 
                                                      random_state=cluster_seed)
                
                mask = ~labels.isna()
                
                labels = labels.to_numpy()[mask]
                
                data_mat = (traf_mats[0][:,:24] - traf_mats[0][:,24:])[mask]
                
                DB_list.append(Davies_Bouldin_index(data_mat, 
                                                    clusters,
                                                    labels,
                                                    verbose=True))
                
                D_list.append(Dunn_index(data_mat, 
                                         clusters,
                                         labels,
                                         verbose=True))
                
                S_list.append(silhouette_index(data_mat, 
                                               clusters,
                                               labels,
                                               verbose=True))
                
                SS_list.append(SSE(data_mat,
                                   clusters,
                                   labels,
                                   verbose=True))
                
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
    
    if savefig:
        plt.savefig('figures/paper_figures/k_test_figures.pdf')
    
    return res_table

def cluster_algo_test(cities=None, year=2019, month=None, k_min=2, k_max=10,
                      cluster_seed=42, min_trips=8, 
                      savefig=False, overwrite=False):
    
    # clustering_algos = ['k_means', 'k_medoids', 'h_clustering', 'gaussian_mixture']
    clustering_algos = ['gaussian_mixture']
    metrics = ['DB', 'D', 'S', 'SS']
    k_list = [i for i in range(k_min, k_max+1)]
    # k_list = [2]
    
    if isinstance(cities, str):
        city = cities
        
        multiindex = pd.MultiIndex.from_product((clustering_algos, metrics))  
        res_table = pd.DataFrame(index=k_list, columns=multiindex)
        
        try:
            with open(f'./python_variables/{city}{year}_avg_stat_df.pickle', 'rb') as file:
                asdf = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        
        
        data = bs.Data(city, year, month)
        traf_mats = data.pickle_daily_traffic(holidays=False, 
                                              user_type='Subscriber',
                                              overwrite=False)
        mask = ~asdf['n_trips'].isna()
        asdf = asdf[mask]
        asdf = asdf.reset_index(drop=True)
        
        try:
            traf_mats = (traf_mats[0][mask], traf_mats[1])
        except IndexError:
            pass
        
        for clustering_algo in clustering_algos:
            
            
            print(f'\nCalculating for {clustering_algo}...\n')
            
            DB_list = []
            D_list = []
            S_list = []
            SS_list = []
                
            for k in k_list:
                
                print(f'\nCalculating for k={k}...\n')
                
                asdf, clusters, labels = get_clusters(traf_mats, asdf, 
                                                      day_type='business_days', 
                                                      min_trips=min_trips, 
                                                      clustering=clustering_algo, 
                                                      k=k, random_state=cluster_seed)
                
                mask = ~labels.isna()
                
                labels = labels.to_numpy()[mask]
                
                data_mat = (traf_mats[0][:,:24] - traf_mats[0][:,24:])[mask]
                
                # if clustering_algo == 'gaussian_mixture':
                #     labels = pd.Series(np.argmax(labels, axis=1))
                
                DB_list.append(Davies_Bouldin_index(data_mat, 
                                                    clusters,
                                                    labels,
                                                    verbose=True))
                
                # D_list.append(Dunn_index(data_mat, 
                #                          clusters,
                #                          labels,
                #                          verbose=True))
                
                # S_list.append(silhouette_index(data_mat, 
                #                                clusters,
                #                                labels,
                #                                verbose=True))
                
                SS_list.append(SSE(data_mat,
                                   clusters,
                                   labels,
                                   verbose=True))
                
            res_table[(clustering_algo, 'DB')] = DB_list
            # res_table[(clustering_algo, 'D')] = D_list
            # res_table[(clustering_algo, 'S')] = S_list
            res_table[(clustering_algo, 'SS')] = SS_list
        
        
        
        
        
    elif cities is None:
        cities = ['nyc', 'chicago', 'washdc', 'boston', 
                  'london', 'helsinki', 'oslo', 'madrid']
    
    else:
        raise TypeError('Please provide cities as either a string (one city) or None (all cities).')
    
    return res_table
    
cluster_color_dict = {0 : 'blue', 1 : 'red', 2 : 'yellow', 3 : 'green', #tab:
                      4 : 'purple', 5 : 'cyan', 6: 'pink',
                      7 : 'brown', 8 : 'olive', 9 : 'magenta', np.nan: 'gray'}

mpl_color_dict = {i: mpl_colors.to_rgb(cluster_color_dict[i]) for i in range(10)}
lab_color_dict = {i: skcolor.rgb2lab(mpl_color_dict[i]) for i in range(10)}
lab_color_list = [lab_color_dict[i] for i in range(10)]


if __name__ == '__main__':
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
              'london', 'helsinki', 'oslo', 'madrid']
    
    cluster_algo_test_table = cluster_algo_test('nyc')
    
    # k_table = k_test_table(savefig=True, overwrite=True)
    # clusters, n_table = plot_cluster_centers('all', k=5, n_table=True)

    # clusters_list = []
    # for k in [2,3,4,5,6,7,8,9,10]:
    #     clusters_list.append(plot_cluster_centers('all', k=k))
    #     plt.close()




