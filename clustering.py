#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:47:27 2022

@author: dbvd
"""
import time
import pickle
import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import skimage.color as skcolor
import smopy as sm

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import BallTree
from sklearn_extra.cluster import KMedoids

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.offsetbox import AnchoredText, AnchoredOffsetbox

import simpledtw as dtw
import bikeshare as bs
import interactive_plot_utils as ipu

def mask_traffic_matrix(traffic_matrices, station_df, day_type, min_trips, holidays=False, return_mask=False):
    """
    Applies a mask to the daily traffic matrix based on the minimum number of 
    trips to include.

    Parameters
    ----------
    day_type : str
        'business_days' or 'weekend'.
    min_trips : int
        the minimum number of trips for a station. If the station has fewer
        trips than this, exclude it.
    holidays : bool, optional
        Whether to include holidays in business days (True) or remove them from
        the business days (False). The default is False.

    Returns
    -------
    np array
        masked traffic matrix, that is, the number of 48-dimensional vectors 
        which constitute the rows of the traffic matrix is reduced.

    """
    if day_type == 'business_days':
        traffic_matrix = traffic_matrices[0]
        x_trips = 'b_trips'
    elif day_type == "weekend":
        traffic_matrix = traffic_matrices[1]
        x_trips = 'w_trips'
    else:
        raise ValueError("Please enter 'business_days' or 'weekend'.")
    mask = station_df[x_trips] >= min_trips
    if return_mask:
        return traffic_matrix[mask], mask, x_trips
    else:
        return traffic_matrix[mask]

def get_clusters(traffic_matrices, station_df, day_type, min_trips, 
                 clustering, k, random_state=None, use_dtw=False, 
                 linkage='average', city=None):
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
    traffic_matrix, mask, x_trips = mask_traffic_matrix(
        traffic_matrices, station_df, day_type, 
        min_trips, holidays=False, return_mask=True)
    
    traffic_matrix = traffic_matrix[:,:24] - traffic_matrix[:,24:]
    
    if use_dtw:
        try:
            with open(f'./python_variables/{city}_dtw_matrix_min_trips={min_trips}.pickle', 'rb') as file:
                dtw_matrix = pickle.load(file)
        
        except FileNotFoundError:
            print('Calculating dtw matrix...')
            pre = time.time()
            dtw_matrix = np.zeros(shape=(len(traffic_matrix), len(traffic_matrix)))
            for i1, vec1 in enumerate(traffic_matrix):
                for i2, vec2 in enumerate(traffic_matrix[:i1]):
                    dtw_matrix[i2,i1] = dtw.dtw1(vec1, vec2)
            for i1, vec1 in enumerate(traffic_matrix):
                for i2, vec2 in enumerate(traffic_matrix[:i1]):
                    dtw_matrix[i1,i2] = dtw_matrix[i2,i1]
            
            print(f'Done. Time taken: {time.time()-pre} s')
            
            with open(f'./python_variables/{city}_dtw_matrix_min_trips={min_trips}.pickle', 'wb') as file:
                pickle.dump(dtw_matrix, file)
            
    if clustering == 'k_means':
        clusters = KMeans(k, random_state=random_state, algorithm='full').fit(traffic_matrix)
        labels = clusters.predict(traffic_matrix)
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        # station_df, means, labels = sort_clusters(station_df, means, labels, traffic_matrices, day_type, k)
        # clusters = clusters.cluster_centers_
        # labels = station_df['label']
        station_df, centers, labels = sort_clusters2(station_df, 
                                                      clusters.cluster_centers_, 
                                                      labels)
        station_df['color'] = station_df['label'].map(cluster_color_dict)

    elif clustering == 'k_medoids':
        
        if use_dtw:
            clusters = KMedoids(k, metric='precomputed', random_state=random_state).fit(dtw_matrix)
            labels = clusters.labels_
            centers = traffic_matrix[clusters.medoid_indices_]
        else:
            clusters = KMedoids(k, random_state=random_state).fit(traffic_matrix)
            labels = clusters.predict(traffic_matrix)
            centers = clusters.cluster_centers_
        
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        station_df, centers, labels = sort_clusters2(station_df, 
                                                      centers, 
                                                      labels)
        station_df['color'] = station_df['label'].map(cluster_color_dict)
        
    elif clustering == 'h_clustering':
        
        if use_dtw:
            clusters = AgglomerativeClustering(k, affinity='precomputed',
                                             linkage=linkage).fit(dtw_matrix)
            labels = clusters.labels_
        else:
            labels = AgglomerativeClustering(k, linkage=linkage).fit_predict(traffic_matrix)
        
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        centers = cluster_mean(traffic_matrix, station_df, labels, k)
        station_df, centers, labels = sort_clusters2(station_df, centers, labels)
        station_df['color'] = station_df['label'].map(cluster_color_dict)
    
    elif clustering == 'gaussian_mixture':
        
        clusters_init = KMeans(k, random_state=random_state).fit(traffic_matrix).cluster_centers_
        
        clusters = GaussianMixture(k, n_init=100, means_init = clusters_init,
                                   random_state=random_state, 
                                   covariance_type='full').fit(traffic_matrix)
        labels_prob = clusters.predict_proba(traffic_matrix)
        station_df.loc[mask, 'label_prob'] = pd.Series(list(labels_prob), 
                                                  index=mask[mask].index)
        station_df.loc[~mask, 'label_prob'] = np.nan
        labels = [np.argmax(x) for x in station_df['label_prob'].loc[mask]]
        station_df.loc[mask, 'label'] = labels
        station_df.loc[~mask, 'label'] = np.nan
        
        station_df, centers, labels = sort_clusters2(station_df, clusters.means_, 
                                                     labels, 
                                                     cluster_type='gaussian_mixture',
                                                     mask=mask)

        lab_mat = np.array(lab_color_list[:k]).T
        lab_cols = [np.sum(station_df['label_prob'][mask][i] * lab_mat, axis=1) for i in station_df['label_prob'][mask].index]
        labels_rgb = skcolor.lab2rgb(lab_cols)
        station_df.loc[mask, 'color'] = ['#%02x%02x%02x' % tuple(label.astype(int)) for label in labels_rgb*255]
        station_df.loc[~mask, 'color'] = 'gray'
        
        # clusters = clusters.means_
        # labels = station_df['label']
        
    elif clustering == 'none':
        centers = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = station_df[x_trips].tolist()
    
    elif clustering == 'zoning':
        centers = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = [cluster_color_dict[zone] for zone in pd.factorize(station_df['zone_type'])[0]]
        
    else:
        centers = None
        labels = None
        station_df['label'] = np.nan
        station_df['color'] = None
    
    dist_to_center = np.full(traffic_matrices[0].shape[0], np.nan)
    if day_type == 'business_days':
        traf_mat = traffic_matrices[0][:,:24] - traffic_matrices[0][:,24:]
    elif day_type == 'weekend':
        traf_mat = traffic_matrices[1][:,:24] - traffic_matrices[1][:,24:]
                 
        dist_to_center[np.where(labels == i)] = np.linalg.norm(
            traf_mat[np.where(labels==i)] - centers[i], axis=1)
    
    station_df['dist_to_center'] = dist_to_center

    return station_df, centers, labels

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
        # If Gaussian mixture, then also reorder the label probabilities
        
        labels_prob = np.array([
            list(row) for row in station_df['label_prob'][mask].to_numpy()
            ]) # recreate the label probs
        values = np.zeros_like(labels_prob)
        order = [int(i) for i in order] # make elements in order integers
        values[:,order] = labels_prob[:,range(k)]
        
        station_df.loc[mask, 'label_prob'] = pd.Series(list(values), index=mask[mask].index)
        station_df.loc[~mask, 'label_prob'] = np.nan
        
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

def SSE(data_mat, centers, labels, dist='norm', verbose=False):
    
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
                         clustering='k_means', cluster_seed=42, min_trips=8, 
                         n_table=False, use_dtw=False, linkage='average', savefig=True):
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
                
        
        asdf, centers, labels = get_clusters(traf_mats, asdf, 
                                              day_type='business_days', 
                                              min_trips=min_trips, 
                                              clustering=clustering, k=k, 
                                              random_state=cluster_seed,
                                              use_dtw=use_dtw, linkage=linkage,
                                              city=city)
        
        plt.style.use('seaborn-darkgrid')
        
        fig, ax = plt.subplots()
        
        for i in range(k):
            n = (labels==i).sum()
            ax.plot(centers[i], label=f'Cluster {i} (n={n})')
        ax.set_xticks(range(24))
        ax.set_xlabel('Hour')
        ax.set_xlim(0,23)
        ax.set_ylim(-0.125,0.125)
        ax.set_ylabel('Relative difference')
        ax.legend()
        
        if savefig:
            plt.savefig(f'./figures/paper_figures/{city}_clusters.pdf')
        
        plt.style.use('default')
    
        return centers

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
                                                      clustering=clustering, k=k,
                                                      random_state=cluster_seed,
                                                      use_dtw=use_dtw, city=city)
                
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
        
        if savefig:
            plt.savefig(f'./figures/paper_figures/clusters_all_cities_k={k}.pdf')
        
        
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

def k_test_table(cities=None, year=2019, month=None, clustering='k_means', 
                 k_min=2, k_max=10, min_trips=8, cluster_seed=42, 
                 savefig=False, overwrite=False, use_dtw=False):
    
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
                                                  overwrite=overwrite)
                
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
                                                      min_trips=min_trips, 
                                                      clustering=clustering, k=k, 
                                                      random_state=cluster_seed,
                                                      use_dtw=use_dtw, city=city)
                
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
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7,6))
    
    count=0
    for row in range(2):
        for col in range(2):
            for city in cities:
                ax[row, col].plot(k_list, 
                                  res_table[(bs.name_dict[city], 
                                             list(metrics_dict.keys())[count])],
                                  label=bs.name_dict[city])
                ax[row,col].set_title(metrics_dict[
                    list(metrics_dict.keys())[count]])
                
                ax[row, col].set_xticks(k_list)
                ax[row, col].set_xlim(min(k_list)-0.4, max(k_list)+0.4)
                
                ax[row,col].set_xlabel('$k$')
            count+=1
            
    plt.tight_layout(pad=3)
    ax[1,0].legend(loc='upper center', bbox_to_anchor=(1.05,-0.18), ncol=4)#len(ax[0,0].get_lines()))
    
    if savefig:
        plt.savefig('figures/paper_figures/k_test_figures.pdf')
    
    return res_table

def make_cluster_algo_test_figure(res_table, city, year, month, 
                                  clustering_algos, savefig=False):
    
    plt.style.use('seaborn-darkgrid')
    
    metrics_dict = {'D' : 'Dunn Index (higher is better)',
                    'S' : 'Silhouette Index (higher is better)',
                    'DB' : 'Davies-Bouldin index (lower is better)',
                    'SS' : 'Sum of Squares (lower is better)'}
    
    cluster_dict = {'k_means' : '$k$-means',
                    'k_medoids' : '$k$-medoids',
                    'h_clustering' : 'Hierarchical clustering',
                    'gaussian_mixture' : 'EM'}
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    
    count=0
    for row in range(2):
        for col in range(2):
            for cluster_algo in clustering_algos:
                ax[row, col].plot(res_table[(cluster_algo, 
                                             list(metrics_dict.keys())[count])],
                                  label=cluster_dict[cluster_algo])
                ax[row,col].set_title(metrics_dict[
                    list(metrics_dict.keys())[count]])
                
                ax[row,col].set_xlabel('$k$')
            count+=1
            
    plt.tight_layout(pad=2)
    ax[1,0].legend(loc='upper center', bbox_to_anchor=(1.05,-0.1), ncol=len(ax[0,0].get_lines()))
    
    if savefig:
        if 'cluster_algo_tests' not in os.listdir('figures'):
            os.mkdir('./figures/cluster_algo_tests')
        
        if month is None:
            plt.savefig(f'./figures/cluster_algo_tests/{city}{year}.pdf')
        else:
            plt.savefig(f'./figures/cluster_algo_tests/{city}{year}{month:02d}.pdf')
    

def cluster_algo_test(cities=None, year=2019, month=None, k_min=2, k_max=10,
                      cluster_seed=42, min_trips=8, 
                      savefig=False, overwrite=False, use_dtw=False):
    
    clustering_algos = ['k_means', 'k_medoids', 'h_clustering', 'gaussian_mixture']
    # clustering_algos = ['h_clustering']
    metrics = ['D', 'S', 'DB', 'SS']
    k_list = [i for i in range(k_min, k_max+1)]
    
    cluster_dict = {'k_means' : '$k$-means',
                    'k_medoids' : '$k$-medoids',
                    'h_clustering' : 'Hierarchical clustering',
                    'gaussian_mixture' : 'EM'}
    
    metrics_dict = {'D' : 'Dunn Index\n(higher is better)',
                    'S' : 'Silhouette Index\n(higher is better)',
                    'DB' : 'Davies-Bouldin index\n(lower is better)',
                    'SS' : 'Sum of Squares\n(lower is better)'}
    
    if isinstance(cities, str):
        city = cities
        
        if not overwrite:
            try:
                if month is None:  
                    with open(f'./python_variables/cluster_algo_test_{city}{year}.pickle', 'rb') as file:
                        res_table = pickle.load(file)
                else:
                    with open(f'./python_variables/cluster_algo_test_{city}{year}{month:02d}.pickle', 'rb') as file:
                        res_table = pickle.load(file)
                
                make_cluster_algo_test_figure(res_table, city, year, 
                                              month, clustering_algos,
                                              savefig=savefig)
                
                return res_table
                
            except FileNotFoundError:
                print('WARNING: No existing file found. Continuing...')

        
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
                                                      k=k, random_state=cluster_seed,
                                                      use_dtw=use_dtw, city=city)
                
                mask = ~labels.isna()
                
                labels = labels.to_numpy()[mask]
                
                data_mat = (traf_mats[0][:,:24] - traf_mats[0][:,24:])[mask]
                
                # if clustering_algo == 'gaussian_mixture':
                #     labels = pd.Series(np.argmax(labels, axis=1))
                
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
                
            res_table[(clustering_algo, 'DB')] = DB_list
            res_table[(clustering_algo, 'D')] = D_list
            res_table[(clustering_algo, 'S')] = S_list
            res_table[(clustering_algo, 'SS')] = SS_list
        
        make_cluster_algo_test_figure(res_table, city, year, 
                                      month, clustering_algos,
                                      savefig=savefig)
        
        if month is None:  
            with open(f'./python_variables/cluster_algo_test_{city}{year}.pickle', 'wb') as file:
                pickle.dump(res_table, file)
        else:
            with open(f'./python_variables/cluster_algo_test_{city}{year}{month:02d}.pickle', 'wb') as file:
                pickle.dump(res_table, file)
        
        return res_table
        
        
    elif cities is None:
        cities = ['nyc', 'chicago', 'washdc', 'boston', 
                  'london', 'helsinki', 'oslo', 'madrid']
        
        big_res_table = pd.DataFrame()
        for city in cities:
            res_table = cluster_algo_test(city, year=year, month=month, 
                                          k_min=k_min, k_max=k_max,
                                          cluster_seed=cluster_seed, 
                                          min_trips=min_trips, 
                                          savefig=savefig,
                                          overwrite=overwrite)
            
            mindex = pd.MultiIndex.from_product(([city], list(res_table.index)))
            res_table.index = mindex
        
            big_res_table = pd.concat([big_res_table, res_table])
        
        plt.style.use('seaborn-darkgrid')
        
        fig, ax = plt.subplots(nrows=len(cities), 
                               ncols=len(metrics),
                               figsize=(10,13))
        
        for row, city in enumerate(cities): 
            for col, metric in enumerate(metrics):
                
                for cluster_algo in clustering_algos:
                    ax[row, col].plot(k_list, big_res_table.loc[
                        city, (cluster_algo, metric)].values, 
                        label=cluster_dict[cluster_algo])
                
                ax[row, col].set_xticks(k_list)
                ax[row, col].set_xlim(min(k_list)-0.4, max(k_list)+0.4)
                    
                if row == 0:
                    ax[row,col].set_title(metrics_dict[metric])
                elif row == (len(cities)-1):
                    ax[row, col].set_xlabel('$k$')
                    
                
                if col == 0:
                    ax[row,col].set_ylabel(bs.name_dict[city])
        
        plt.tight_layout(pad=2)
        ax[-1,0].legend(loc='upper center', bbox_to_anchor=(2.3,-0.3), ncol=len(ax[0,0].get_lines()))
        
        if savefig:
            if 'cluster_algo_tests' not in os.listdir('figures'):
                os.mkdir('./figures/cluster_algo_tests')
            
            if month is None:
                plt.savefig(f'./figures/cluster_algo_tests/all_cities_{year}.pdf')
            else:
                plt.savefig(f'./figures/cluster_algo_tests/all_cities_{year}{month:02d}.pdf')
        
        return big_res_table
        
    else:
        raise TypeError('Please provide cities as either a string (one city) or None (all cities).')

def make_linkage_test_figure(res_table, city, year, month, 
                             clusterings, savefig=False):
    
    plt.style.use('seaborn-darkgrid')
    
    metrics = ['D', 'S', 'DB', 'SS']
    
    legend = ['$k$-means', 
              '$l_2$-norm, single linkage',
              '$l_2$-norm, complete linkage',
              '$l_2$-norm, average linkage',
              'DTW, single linkage',
              'DTW, complete linkage',
              'DTW, average linkage']
    cluster_dict = dict(zip(clusterings, legend))
    
    metrics_dict = {'D' : 'Dunn Index (higher is better)',
                    'S' : 'Silhouette Index (higher is better)',
                    'DB' : 'Davies-Bouldin index (lower is better)',
                    'SS' : 'Sum of Squares (lower is better)'}
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
    
    count=0
    for row in range(2):
        for col in range(2):
            for clustering in clusterings:
                ax[row, col].plot(res_table[(clustering, 
                                             list(metrics_dict.keys())[count])],
                                  label=cluster_dict[clustering])
                ax[row,col].set_title(metrics_dict[
                    list(metrics_dict.keys())[count]])
                
                ax[row,col].set_xlabel('$k$')
            count+=1
            
    plt.tight_layout(pad=2)
    ax[1,0].legend(loc='upper center', bbox_to_anchor=(1.05,-0.1), ncol=len(ax[0,0].get_lines()))
    
    if savefig:
        if 'linkage_tests' not in os.listdir('figures'):
            os.mkdir('./figures/linkage_tests')
        
        if month is None:
            plt.savefig(f'./figures/linkage_tests/{city}{year}.pdf')
        else:
            plt.savefig(f'./figures/linkage_tests/{city}{year}{month:02d}.pdf')

def linkage_test(cities=None, year=2019, month=None, k_min=2, k_max=10,
                 cluster_seed=42, min_trips=8, 
                 savefig=False, overwrite=False):
    
    # (algo, use_dtw, linkage)
    clusterings = [('k_means', False, 'average'),
                   ('h_clustering', False, 'single'),
                   ('h_clustering', False, 'complete'),
                   ('h_clustering', False, 'average'),
                   ('h_clustering', True, 'single'),
                   ('h_clustering', True, 'complete'),
                   ('h_clustering', True, 'average')]
    
    metrics = ['D', 'S', 'DB', 'SS']
    k_list = [i for i in range(k_min, k_max+1)]
    
    legend = ['$k$-means', 
              '$l_2$-norm, single linkage',
              '$l_2$-norm, complete linkage',
              '$l_2$-norm, average linkage',
              'DTW, single linkage',
              'DTW, complete linkage',
              'DTW, average linkage']
    cluster_dict = dict(zip(clusterings, legend))
    metrics_dict = {'D' : 'Dunn Index (higher is better)',
                    'S' : 'Silhouette Index (higher is better)',
                    'DB' : 'Davies-Bouldin index (lower is better)',
                    'SS' : 'Sum of Squares (lower is better)'}
    
    if isinstance(cities, str):
        city = cities
        
        if not overwrite:
            try:
                if month is None:  
                    with open(f'./python_variables/linkage_test_{city}{year}.pickle', 'rb') as file:
                        res_table = pickle.load(file)
                else:
                    with open(f'./python_variables/linkage_test_{city}{year}{month:02d}.pickle', 'rb') as file:
                        res_table = pickle.load(file)
                
                make_linkage_test_figure(res_table, city, year, 
                                         month, clustering_algos,
                                         savefig=savefig)
                
                return res_table
                
            except FileNotFoundError:
                print('WARNING: No existing file found. Continuing...')

        
        multiindex = pd.MultiIndex.from_product((clusterings, metrics))  
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
        
        for clustering in clusterings:
            
            print(f'\nCalculating for {clustering}...\n')
            
            DB_list = []
            D_list = []
            S_list = []
            SS_list = []
                
            for k in k_list:
                
                print(f'\nCalculating for k={k}...\n')
                
                asdf, clusters, labels = get_clusters(traf_mats, asdf, 
                                                      day_type='business_days', 
                                                      min_trips=min_trips, 
                                                      clustering=clustering[0], 
                                                      k=k, random_state=cluster_seed,
                                                      use_dtw=clustering[1], 
                                                      linkage=clustering[2],
                                                      city=city)
                
                mask = ~labels.isna()
                
                labels = labels.to_numpy()[mask]
                
                data_mat = (traf_mats[0][:,:24] - traf_mats[0][:,24:])[mask]
                
                # if clustering_algo == 'gaussian_mixture':
                #     labels = pd.Series(np.argmax(labels, axis=1))
                
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
                
            res_table[(clustering, 'DB')] = DB_list
            res_table[(clustering, 'D')] = D_list
            res_table[(clustering, 'S')] = S_list
            res_table[(clustering, 'SS')] = SS_list
        
        make_linkage_test_figure(res_table, city, year, 
                                 month, clusterings,
                                 savefig=savefig)
        
        if month is None:  
            with open(f'./python_variables/linkage_test_{city}{year}.pickle', 'wb') as file:
                pickle.dump(res_table, file)
        else:
            with open(f'./python_variables/linkage_test_{city}{year}{month:02d}.pickle', 'wb') as file:
                pickle.dump(res_table, file)
        
        return res_table
        
        
    elif cities is None:
        cities = ['nyc', 'chicago', 'washdc', 'boston', 
                  'london', 'helsinki', 'oslo', 'madrid']
        
        big_res_table = pd.DataFrame()
        for city in cities:
            res_table = linkage_test(city, year=year, month=month, 
                                     k_min=k_min, k_max=k_max,
                                     cluster_seed=cluster_seed, 
                                     min_trips=min_trips, 
                                     savefig=savefig,
                                     overwrite=overwrite)
            
            mindex = pd.MultiIndex.from_product(([city], list(res_table.index)))
            res_table.index = mindex
        
            big_res_table = pd.concat([big_res_table, res_table])
        
        plt.style.use('seaborn-darkgrid')
        
        fig, ax = plt.subplots(nrows=len(cities), 
                               ncols=len(metrics),
                               figsize=(14,18))
        
        for row, city in enumerate(cities): 
            for col, metric in enumerate(metrics):
                
                for clustering in clusterings:
                    ax[row, col].plot(k_list, big_res_table.loc[
                        city, (clustering, metric)].values, 
                        label=cluster_dict[clustering])
                
                ax[row, col].set_xticks(k_list)
                ax[row, col].set_xlim(min(k_list)-0.4, max(k_list)+0.4)
                    
                if row == 0:
                    ax[row,col].set_title(metrics_dict[metric])
                elif row == (len(cities)-1):
                    ax[row, col].set_xlabel('k')
                    
                
                if col == 0:
                    ax[row,col].set_ylabel(bs.name_dict[city])
        
        
        ax[-1,0].legend(loc='upper center', bbox_to_anchor=(2.3,-0.2), ncol=len(ax[0,0].get_lines()))
        
        if savefig:
            if 'linkage_tests' not in os.listdir('figures'):
                os.mkdir('./figures/linkage_tests')
            
            if month is None:
                plt.savefig(f'./figures/linkage_tests/all_cities_{year}.pdf')
            else:
                plt.savefig(f'./figures/linkage_tests/all_cities_{year}{month:02d}.pdf')
        
        return big_res_table
        
    else:
        raise TypeError('Please provide cities as either a string (one city) or None (all cities).')

def plot_stations(city, year=2019, month=None, day=None, 
                  clustering='k_means', k=5, cluster_seed=42, min_trips=8, 
                  use_dtw=False, linkage='average', savefig=True):
    
    new_color_dict = {0 : 'tab:blue', 1 : 'tab:orange', 2 : 'tab:green',
                      3 : 'tab:red', 4: 'tab:purple'}
    
    cluster_name_dict = {0 : 'Reference', 
                         1 : 'High morning sink', 
                         2 : 'Low morning sink',
                         3 : 'Low morning source',
                         4 : 'High morning source'}
    
    try:
        if month is None:
            filestr = f'./python_variables/{city}{year}_avg_stat_df.pickle'
        else:
            filestr = f'./python_variables/{city}{year}{month}_avg_stat_df.pickle'
            
        with open(filestr, 'rb') as file:
            asdf = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f'The average station DataFrame for {city} in {year} was not found. Please make it using interactive_plot_utils.pickle_asdf()')        

    data = bs.Data(city, year, month, day)
    
    stat_df= ipu.make_station_df(data, holidays=False)
    
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
            
    
    asdf, centers, labels = get_clusters(traf_mats, asdf, 
                                         day_type='business_days', 
                                         min_trips=min_trips, 
                                         clustering=clustering, k=k, 
                                         random_state=cluster_seed,
                                         use_dtw=use_dtw, linkage=linkage,
                                         city=city)
    
    asdf['new_color'] = asdf['label'].apply(lambda l: new_color_dict[l] 
                                            if not pd.isna(l) else 'grey')
    
    extend = (stat_df['lat'].min(), stat_df['long'].min(), 
          stat_df['lat'].max(), stat_df['long'].max())
    
    tileserver = 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg' # Stamen Terrain
    # tileserver = 'http://a.tile.stamen.com/toner/{z}/{x}/{y}.png' # Stamen Toner
    # tileserver = 'http://c.tile.stamen.com/watercolor/{z}/{x}/{y}.png' # Stamen Watercolor
    # tileserver = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png' # OSM Default
    
    m = sm.Map(extend, tileserver=tileserver)
    
    xlim_dict = {'nyc' : (150, 600), 'chicago' : (150,600), 
                 'washdc' : (0,390), 'boston' : (0, 300),
                 'london' : (30,800), 'helsinki' : (130,767),
                 'oslo' : (0,600), 'madrid' : (0,511)}
    ylim_dict = {'nyc' : (767.5,100), 'chicago' : (900,150),
                 'washdc' : (580,180), 'boston' : (600,225),
                 'london' : (550,30), 'helsinki' : (560,230),
                 'oslo' : (500,100), 'madrid' : (750, 50)}
   
    ms_dict = {'nyc' : 6, 'chicago' : 6, 'washc' : 6, 'boston' : 6,
               'london' : 3, 'helsinki' : 2, 'oslo' : 3, 'madrid' : 6}
    
    fig, ax = plt.subplots(figsize=(7,10))
    
    m.show_mpl(ax=ax)
    
    if city in ms_dict.keys():
        ms = ms_dict[city]
    else:
        ms=6
    
    for i, stat in asdf.iterrows():
        x, y = m.to_pixels(stat_df[stat_df.stat_id==stat.stat_id]['lat'], 
                           stat_df[stat_df.stat_id==stat.stat_id]['long'])
        ax.plot(x, y, 'o', ms=ms, color = stat['new_color'])
    
    if city in xlim_dict.keys():
        ax.set_xlim(xlim_dict[city])
    
    if city in ylim_dict.keys():
        ax.set_ylim(ylim_dict[city])
    
    ax.axis('off')
    plt.tight_layout()
    
    for key in new_color_dict.keys():
        ax.plot(0,0,'o', ms=6, 
                color=new_color_dict[key], 
                label=cluster_name_dict[key])
    ax.plot(0,0,'o', ms=6, color='grey', label='Unclustered')
    
    # Add legend
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,0), ncol=3, frameon=False)
    
    # Add scalebar
    
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
    
    # Add OSM attribute
    
    attr = AnchoredText("(C) Stamen Design. (C) OpenStreetMap contributors.",
                        loc = 'lower left', frameon=True, pad=0.1, borderpad=0)
    attr.patch.set_edgecolor('white')
    ax.add_artist(attr)
    
    plt.tight_layout()
    
    if savefig:
        fig.savefig(f'./figures/station_cluster_plot_{city}.pdf', bbox_inches='tight')
    
    return fig, ax
    
cluster_color_dict = {0 : 'blue', 1 : 'red', 2 : 'yellow', 3 : 'green', #tab:
                      4 : 'purple', 5 : 'cyan', 6: 'pink',
                      7 : 'brown', 8 : 'olive', 9 : 'magenta', np.nan: 'gray'}

mpl_color_dict = {i: mpl_colors.to_rgb(cluster_color_dict[i]) for i in range(10)}
lab_color_dict = {i: skcolor.rgb2lab(mpl_color_dict[i]) for i in range(10)}
lab_color_list = [lab_color_dict[i] for i in range(10)]


if __name__ == '__main__':
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
              'london', 'helsinki', 'oslo', 'madrid']
    
    # cluster_algo_test_table = cluster_algo_test(cluster_seed=42,
    #                                             savefig=False, overwrite=True, 
    #                                             use_dtw=True)
    
    # linkage_test_table = linkage_test(cluster_seed=42,
    #                                   savefig=True, overwrite=True)
    
    # k_table = k_test_table(clustering='k_means', 
    #                         savefig=True, overwrite=False, use_dtw=True)
    
    clusters, n_table = plot_cluster_centers('all', k=7, clustering='k_medoids',
                                              use_dtw=True, linkage='complete', 
                                              n_table=True, savefig=True)
    
    
    # clusters_list = []
    # for k in [2,3,4,5,6,7,8,9,10]:
    #     clusters_list.append(plot_cluster_centers('all', k=k))
    #     plt.close()




