#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:47:27 2022

@author: dbvd
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import simpledtw as dtw



def dist_norm(vec1, vec2):
    return np.linalg.norm(vec1-vec2)


def dist_dtw(vec1, vec2):
    return dtw.dtw(vec1, vec2)[1]


def Davies_Bouldin_index(data_mat, labels, centroids, dist_func='norm', verbose=False):
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

    k = len(centroids)

    if dist_func == 'norm':
        dist = dist_norm

    elif dist_func == 'dtw':
        dist = dist_dtw

    if verbose:
        print('Calculating Davies-Bouldin index...')

    pre = time.time()

    S_scores = np.empty(k)

    for i in range(k):
        data_mat_cluster = data_mat[np.where(labels == i)]
        distances = [dist(row, centroids[i]) for row in data_mat_cluster]
        S_scores[i] = np.mean(distances)

    R = np.empty(shape=(k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                R[i, j] = 0
            else:
                R[i, j] = (S_scores[i] + S_scores[j]) / \
                    dist(centroids[i], centroids[j])

    D = [max(row) for row in R]

    DB_index = np.mean(D)

    if verbose:
        print(f'Done. Time taken: {(time.time()-pre):.1f} s')

    return DB_index


def Dunn_index(data_mat, labels, centroids, dist_func='norm', verbose=False):
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
    k = len(centroids)

    if dist_func == 'norm':
        dist = dist_norm

    elif dist_func == 'dtw':
        dist = dist_dtw

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
                distances[h, j] = dist(data_mat[h], data_mat[j])

        intra_cluster_distances[i] = np.max(distances)

        for j in range(k):
            if j != i:
                data_mat_cluster_j = data_mat[np.where(labels == j)]
                cluster_size_j = len(data_mat_cluster_j)
                between_cluster_distances = np.empty(
                    shape=(cluster_size, cluster_size_j))
                for m in range(cluster_size):
                    for n in range(cluster_size_j):
                        between_cluster_distances[m, n] = dist(
                            data_mat_cluster[m], data_mat_cluster_j[n])
                inter_cluster_distances[i, j] = np.min(
                    between_cluster_distances)

    D_index = np.min(inter_cluster_distances)/np.max(intra_cluster_distances)

    if verbose:
        print(f'Done. Time taken: {(time.time()-pre):.1f} s')

    return D_index


def silhouette_index(data_mat, labels, centroids, dist_func='norm', verbose=False):
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

    k = len(centroids)

    if dist_func == 'norm':
        dist = dist_norm

    elif dist_func == 'dtw':
        dist = dist_dtw

    if verbose:
        print('Calculating Silhouette index...')

    pre = time.time()

    s_coefs = np.empty(len(data_mat))

    for i, vec1 in enumerate(data_mat):
        in_cluster = np.delete(data_mat, i, axis=0)
        in_cluster = in_cluster[np.where(np.delete(labels, i) == labels[i])]

        in_cluster_size = len(in_cluster)

        if in_cluster_size != 0:

            in_cluster_distances = np.empty(in_cluster_size)
            for j, vec2 in enumerate(in_cluster):
                in_cluster_distances[j] = dist(vec1, vec2)

            mean_out_cluster_distances = np.full(k, fill_value=np.inf)

            for j in range(k):
                if j != labels[i]:
                    out_cluster = data_mat[np.where(labels == j)]
                    out_cluster_distances = np.empty(len(out_cluster))

                    for l, vec2 in enumerate(out_cluster):
                        out_cluster_distances[l] = dist(vec1, vec2)

                    mean_out_cluster_distances[j] = np.mean(out_cluster_distances)

            ai = np.mean(in_cluster_distances)
            bi = np.min(mean_out_cluster_distances)

            s_coefs[i] = (bi-ai)/max(ai, bi)

        else:
            s_coefs[i] = 0

    S_index = np.mean(s_coefs)

    if verbose:
        print(f'Done. Time taken: {(time.time()-pre):.1f} s')

    return S_index


def k_test(data_mat, cluster_func, k_max = 10, random_state = 42,
           tests = 'full', plot = False):

    tests = ['SSE', 'DB', 'D', 'S']

    results = np.zeros(shape=(len(tests),k_max-1))

    # print(f'{f"Test result for {cluster_func}":^{spacing}}')
    # print('-'*spacing)

    print(f'{"k":5}{"SSE":15}{"DB_index":15}{"D_index":15}{"S_index":15}')
    print('-'*60)

    for i, k in enumerate(range(2, k_max+1)):
        clusters = cluster_func(k, random_state = random_state).fit(data_mat)
        labels = clusters.predict(data_mat)
        centroids = clusters.cluster_centers_

        results[0, i] = clusters.inertia_
        results[1, i] = Davies_Bouldin_index(data_mat, labels, centroids)
        results[2, i] = Dunn_index(data_mat, labels, centroids)
        results[3, i] = silhouette_index(data_mat, labels, centroids)

        print(
           f'{k:<5,d}{results[0,i]:<15.8f}{results[1,i]:<15.8f}{results[2,i]:<15.8f}{results[3,i]:<15,.8f}')


    res_df = pd.DataFrame(index = range(2,k_max+1),
                          columns = ['SSE', 'DB', 'D', 'S'])
    res_df.index.rename('k', inplace=True)

    for test_i, test in enumerate(res_df.columns):
        res_df[test] = results[test_i]

    if plot:

        plt.subplot(221)
        plt.plot(range(2,k_max+1), res_df['SSE'])
        plt.xticks(range(2,k_max+1))
        # plt.xlabel('$k$')
        plt.legend(['SSE'])

        plt.subplot(222)
        plt.plot(range(2,k_max+1), res_df['DB'], c='tab:orange')
        plt.xticks(range(2,k_max+1))
        # plt.xlabel('$k$')
        plt.legend(['DB_index'])

        plt.subplot(223)
        plt.plot(range(2,k_max+1), res_df['D'], c='tab:green')
        plt.xticks(range(2,k_max+1))
        plt.xlabel('$k$')
        plt.legend(['D_index'])

        plt.subplot(224)
        plt.plot(range(2,k_max+1), res_df['S'], c='tab:red')
        plt.xticks(range(2,k_max+1))
        plt.xlabel('$k$')
        plt.legend(['S_index'])

    return res_df