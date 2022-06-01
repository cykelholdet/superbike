# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:08:19 2022

@author: Nicolai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError

import bikeshare as bs
import interactive_plot_utils as ipu
from clustering import get_clusters, mask_traffic_matrix


class FullModel:
    def __init__(self, variables, clustering='k_means', k=5, min_trips=8, cluster_seed=42, use_dtw=False,
                 linkage = 'average', user_type='Subscriber', day_type='business_days'):
        
        self.variables = variables
        self.clustering = clustering
        self.k = k
        self.min_trips = min_trips
        self.cluster_seed = cluster_seed
        self.use_dtw = use_dtw
        self.linkage = linkage
        self.user_type = user_type
        if day_type in ['business_days', 'weekend']:
            self.day_type = day_type
        else:
            raise ValueError("Please provide day_type as either 'business_days' or 'weekend")
        self.fitted = False
        self.labels = None
        self.centers = None
        self.centers_traffic = None
        self.logit_model = None
        self.linear_model = None
        
    def fit(self, stat_df, traf_mat, const=True):
        
        for var in self.variables:
            if var not in stat_df.columns:
                raise KeyError(f"{var} not found in stat_df columns")
        
        stat_df, self.centers, self.labels = get_clusters(traf_mat, stat_df, 
                                                          day_type=self.day_type, 
                                                          min_trips=self.min_trips, 
                                                          clustering=self.clustering, k=self.k, 
                                                          random_state=self.cluster_seed,
                                                          use_dtw=self.use_dtw, linkage=self.linkage)
        
        self.centers_traffic = []
        for l in range(self.k):
            cluster = stat_df['label'] == l
            self.centers_traffic.append(np.mean(traf_mat[cluster], axis=0))
            
        
        zone_columns  = [var for var in self.variables if 'percent' in var]
        other_columns = [var for var in self.variables if 'percent' not in var]
        
        self.logit_model, X, y = ipu.stations_logistic_regression(
                stat_df, zone_columns, other_columns, 
                use_points_or_percents='percents', 
                make_points_by='station land use', 
                const=const, test_model=False)
        
        print(self.logit_model.summary())
            
        if (x_trips := 'b_trips' if self.day_type == 'business_days' else 'w_trips') in stat_df.columns:
            
            mask = stat_df[x_trips]>=self.min_trips
            stat_slice = stat_df[mask]
            self.linear_model = ipu.linear_regression(stat_slice, 
                                                      [*zone_columns, *other_columns], 
                                                      x_trips)
        self.fitted = True
        
        return stat_df
    
    def predict(self, stat, verbose=True):
        # give row from stat_df and predict traffic pattern and number of trips
        # and maybe plot pedicted traffic pattern
        
        if not self.fitted:
            raise NotFittedError("This FullModel instance is not fitted yet. Call 'fit' before using this method.")
        
        if isinstance(stat, pd.core.series.Series):
            
            
            var_values = np.zeros(len(self.variables)+1)
            var_values[0] = 1
            for i, var in enumerate(self.variables):
                if var in stat.index:
                    var_values[i+1] = stat[var]
            
            label_predict = self.logit_model.predict(exog=var_values)
            trips_predict = self.linear_model.predict(exog=var_values)
            # trips_predict = np.exp(trips_predict[0])
            
            if verbose:
                print("\nProbabilities of labels:")
                string=''
                for i, p in enumerate(label_predict[0]):
                    string += f'label {i}: {p*100} %\n'
                print(string)
                print(f"Predicted label:   {label_predict.argmax()}")
                print(f"Predicted # trips: {trips_predict}")
            
            return int(label_predict.argmax()), trips_predict
            
        
        elif isinstance(stat, pd.core.frame.DataFrame):
            
            label_predicts = np.zeros(len(stat), dtype=int)
            trips_predicts = np.zeros(len(stat))
            
            for i in range(len(stat)):
                label_predicts[i], trips_predicts[i] = self.predict(
                    stat.iloc[i], verbose=False)
            
            return label_predicts, trips_predicts
    
    def predict_daily_traffic(self, stat_row, predict_cluster=True, 
                              plotfig=True, verbose=True):
        if not self.fitted:
            raise NotFittedError("This FullModel instance is not fitted yet. Call 'fit' before using this method.")
        
        if predict_cluster:
            label, trips = self.predict(stat_row, verbose=verbose)
        
        else:
            label = int(stat_row['label'])
            trips = self.predict(stat_row, verbose=False)[1][0]
        
        traffic = self.centers_traffic[int(label)]*trips
        
        if plotfig:
            plt.style.use('seaborn-darkgrid')
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(range(24), traffic[:24], c='tab:blue', label='departures')
            ax.plot(range(24), traffic[24:], c='tab:red', label='arrivals')
            ax.set_xticks(range(24))
            
            ax.set_xlabel('Hour')
            ax.set_ylabel('Predicted # trips')
            ax.set_title(f'Predicted label: {label} - predicted # trips: {trips}')
            
            ax.legend()
            
        return traffic
    
    
    # def test_model(self, stat_df, traf_mat, error='MSE', predict_cluster=True, verbose=True):
        
    #     if not self.fitted:
    #         raise NotFittedError("This FullModel instance is not fitted yet. Call 'fit' before using this method.")
        
    #     min_trips_mask = stat_df['b_trips'] >= self.min_trips
        
    #     df = stat_df[min_trips_mask].copy()
    #     tm = traf_mat[min_trips_mask]
        
    #     if predict_cluster:
    #         label_est, trips_est = self.predict(df, verbose=verbose)
        
    #     else:
    #         label_est = df['label'].to_list()
    #         trips_est = self.predict(df, verbose=False)[1]
        
    #     label_est = list(map(int, label_est))
        
    #     traf_mat_est = np.zeros_like(tm)
        
    #     for i in range(traf_mat_est.shape[0]):
    #         traf_mat_est[i, :] = self.centers_traffic[label_est[i]]*trips_est[i]
        
    #     traf_err = traf_mat_est-tm
        
    #     if error == 'MSE':
    #         err = np.mean(np.abs(traf_err)**2)
        
    #     elif error == 'MAE':
    #         err = np.mean(np.abs(traf_err))
        
    #     elif error == 'ME':
    #         err = np.mean(traf_err)
        
    #     if verbose:
            
    #         if predict_cluster:
    #             cluster_success_rate = (label_est == df['label']).sum()/len(label_est)
            
    #             print(f'\nTest completed.\nClustering success rate: {cluster_success_rate}%\n{error}: {err}')
            
    #         else:
    #             print(f'\nTest completed.\n{error}: {err}')
            
    #     return err
    
    
def trips_predict_error_plot(stat_df, traf_mat, variables, by_cluster=False,
                             error='residual', show_id=False, plotfig=True, savefig=False):
    
    model = FullModel(variables)
    model.fit(stat_df, traf_mat)
    trip_mask = stat_df['b_trips']>=model.min_trips
    trips_predicted = model.predict(stat_df[trip_mask])[1]
    
    error_list = ['residual', 'absolute', 'relative']
    
    if error == 'residual':
        errors = stat_df[trip_mask]['b_trips'].to_numpy() - trips_predicted
    elif error == 'absolute':
        errors = stat_df[trip_mask]['b_trips'].to_numpy() - trips_predicted
    elif error == 'relative':
        errors = (np.abs(stat_df[trip_mask]['b_trips'].to_numpy() - trips_predicted))/stat_df[trip_mask]['b_trips'].to_numpy()
    
    else:
        raise ValueError(f"Please choose an error from {error_list}")
    
    if by_cluster:
        
        errors_by_cluster = []
        
        for l in range(model.k):
            
            mask = stat_df[trip_mask]['label'] == l
            
            stat_slice = stat_df[trip_mask][mask]
            traf_slice = traf_mat[trip_mask][mask]
                   
            error_df = pd.DataFrame(index=stat_slice['stat_id'])
            error_df['true'] = stat_slice['b_trips']
            error_df['predicted'] = trips_predicted[mask]
            error_df['error'] = errors[mask]
            
            errors_by_cluster.append(error_df)
            
            if plotfig:
                plt.style.use('seaborn-darkgrid')
                fig, ax = plt.subplots(figsize=(10,8))
                ax.scatter(trips_predicted[mask], errors[mask],
                           c=stat_slice['b_trips'], cmap='viridis')
                
                if show_id:
                    for i in range(len(stat_slice)):
                        ax.annotate(str(stat_slice.iloc[i]['stat_id']), 
                                    (trips_predicted[mask][i], errors[mask][i]))
                
                ax.set_xlabel('Predicted # trips')
                ax.set_ylabel(f'Error ({error})')
                ax.set_title(f'Cluster {l}')
            
        return errors_by_cluster
            
    else:
        
        error_df = pd.DataFrame(index=stat_df[trip_mask]['stat_id'])
        error_df['true'] = stat_df[trip_mask]['b_trips'].to_numpy()
        error_df['predicted'] = trips_predicted
        error_df['error'] = errors
        
        if plotfig:
            plt.style.use('seaborn-darkgrid')
            fig, ax = plt.subplots(figsize=(10,8))
            ax.scatter(error_df['predicted'], error_df['error'])
            
            if show_id:
                for i in range(len(stat_df[trip_mask])):
                    ax.annotate(str(stat_df[trip_mask].iloc[i]['stat_id']), (trips_predicted[i], errors[i]))
            
            ax.set_xlabel('Predicted # trips')
            ax.set_ylabel(f'Error ({error})')
            
            
        return error_df

def test_model(stat_df, traf_mat, variables, test_ratio=0.2, test_seed=None):
    stat_df = stat_df[~stat_df['b_trips'].isna()].copy()
    
    # Split data into training and test set
    
    if test_seed:
        if isinstance(test_seed, int):
            np.random.seed(test_seed)
        else:
            raise TypeError('test_seed should be of int type')
    
    mask = np.random.rand(len(stat_df)) < test_ratio
    
    df_train = stat_df.copy()[~mask]
    tm_train = traf_mat[~mask]
    
    df_test = stat_df.copy()[mask]
    tm_test = traf_mat[mask]

    # Train model
    
    model = FullModel(variables)
    df_train=model.fit(df_train, tm_train)
    
    # Predict traffic of test set
    
    tm_test_true = tm_test*df_test['b_trips'][:,None]
    
    tm_test_est = np.zeros_like(tm_test_true)
    
    for i in range(len(tm_test_est)):
        tm_test_est[i,:] = model.predict_daily_traffic(df_test.iloc[i],
                                                       plotfig=False,
                                                       verbose=False)
        
    tm_err = tm_test_est-tm_test_true
    

def test_model_stratisfied(stat_df, traf_mat, variables, test_ratio=0.2, test_seed=None, combine=False):
    
    mask = ~stat_df['b_trips'].isna()
    
    stat_df = stat_df[mask].copy()
    traf_mat = traf_mat[mask]
    
    min_trips_mask = stat_df['b_trips'] >= 8
    
    stat_df = stat_df[min_trips_mask]
    traf_mat = traf_mat[min_trips_mask]
    
    # Split data into training and test set
    
    if test_seed:
        if isinstance(test_seed, int):
            np.random.seed(test_seed)
        else:
            raise TypeError('test_seed should be of int type')
        
    # Cluster the stations into low-traffic, mid-traffic and high-traffic
    
    # trips = stat_df['b_trips'].to_numpy()
    # trip_classifier = KMeans(3).fit(trips.reshape(-1,1))
    # labels = trip_classifier.predict(trips.reshape(-1,1))
    
    traffic_split = [20,60,20]
    
    argsort = np.argsort(stat_df['b_trips'].to_numpy())
    
    low_mid_index = int(np.round(traffic_split[0]/100*len(argsort)))
    mid_high_index = int(np.round(traffic_split[2]/100*len(argsort)))
    
    low_stats_i = argsort[:low_mid_index]
    mid_stats_i = argsort[low_mid_index:-mid_high_index]
    high_stats_i = argsort[-mid_high_index:]
    
    # # Divide up the data set into test sets and training set
    
    n_stats = len(stat_df)
    
    n_test = n_stats*test_ratio
    
    n_low_stats = len(low_stats_i)
    n_mid_stats = len(mid_stats_i)
    n_high_stats = len(high_stats_i)
    
    np.random.shuffle(low_stats_i)
    np.random.shuffle(mid_stats_i)
    np.random.shuffle(high_stats_i)
    
    low_stat_selected = low_stats_i[:int(np.round(n_test*n_low_stats/n_stats))]
    mid_stat_selected = mid_stats_i[:int(np.round(n_test*n_mid_stats/n_stats))]
    high_stat_selected = high_stats_i[:int(np.round(n_test*n_high_stats/n_stats))]
    
    low_stat_mask = stat_df.index.isin(low_stat_selected)
    mid_stat_mask = stat_df.index.isin(mid_stat_selected)
    high_stat_mask = stat_df.index.isin(high_stat_selected)
    
    df_low_test = stat_df[low_stat_mask]
    df_mid_test = stat_df[mid_stat_mask]
    df_high_test = stat_df[high_stat_mask]
    
    test_indices = np.concatenate((low_stat_selected, 
                                  mid_stat_selected,
                                  high_stat_selected))
    
    train_mask = ~stat_df.index.isin(test_indices)
    df_train = stat_df[train_mask]
    tm_train = traf_mat[train_mask]
    
    model = FullModel(variables)
    df_train=model.fit(df_train, tm_train)
    
    if combine:
        
        df_test = stat_df[~train_mask]
        tm_true = traf_mat[~train_mask]*df_test['b_trips'][:,None]
        
        tm_est = np.zeros_like(tm_true)
        for i in range(len(tm_est)):
            tm_est[i,:] = model.predict_daily_traffic(df_test.iloc[i],
                                                      plotfig=False,
                                                      verbose=False)
    
        tm_err = tm_est-tm_true
        
        return tm_err
        
        
    else:
        
        tm_low_test_true = traf_mat[low_stat_mask]*df_low_test['b_trips'][:,None]
        tm_mid_test_true = traf_mat[mid_stat_mask]*df_mid_test['b_trips'][:,None]
        tm_high_test_true = traf_mat[high_stat_mask]*df_high_test['b_trips'][:,None]
        
        tm_low_test_est = np.zeros_like(tm_low_test_true)
        for i in range(len(tm_low_test_est)):
            tm_low_test_est[i,:] = model.predict_daily_traffic(df_low_test.iloc[i],
                                                               plotfig=False,
                                                               verbose=False)
    
        tm_mid_test_est = np.zeros_like(tm_mid_test_true)
        for i in range(len(tm_mid_test_est)):
            tm_mid_test_est[i,:] = model.predict_daily_traffic(df_mid_test.iloc[i],
                                                               plotfig=False,
                                                               verbose=False)
            
        tm_high_test_est = np.zeros_like(tm_high_test_true)
        for i in range(len(tm_high_test_est)):
            tm_high_test_est[i,:] = model.predict_daily_traffic(df_high_test.iloc[i],
                                                                plotfig=False,
                                                                verbose=False)
    
        tm_low_err = tm_low_test_est - tm_low_test_true
        tm_mid_err = tm_mid_test_est - tm_mid_test_true
        tm_high_err = tm_high_test_est - tm_high_test_true
    
        return tm_low_err, tm_mid_err, tm_high_err


def load_city(city, year=2019, month=None, day=None, normalise=True):
    
    # Read asdf
    
    if month is None:
        filestr = f'./python_variables/{city}{year}_avg_stat_df.pickle'
    else:
        filestr = f'./python_variables/{city}{year}{month:02d}_avg_stat_df.pickle'
        
    with open(filestr, 'rb') as file:
        asdf = pickle.load(file)
    
    # Make Data object and traffic matrix
    
    data = bs.Data(city, year, month)
    traf_mat = data.pickle_daily_traffic(holidays=False, 
                                         user_type='Subscriber',
                                         day_type='business_days',
                                         normalise=normalise,
                                         overwrite=False)

    mask = ~asdf['n_trips'].isna()
    
    asdf = asdf[mask]
    asdf = asdf.reset_index(drop=True)
    
    try:
        traf_mat = traf_mat[mask]
    except IndexError:
        pass

    return data, asdf, traf_mat

def table_formatter(x):
    if x == np.inf:
        return "inf"
    elif np.abs(x) > 10000 or np.abs(x) < 0.001:
        return f"$\\num{{{x:.2e}}}$"
    else:
        return f"${x:.4f}$"

    #%% Do data
    
if __name__ == '__main__':
    
    CITY = 'nyc'
    YEAR = 2019
    MONTH = None
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
              'london', 'helsinki', 'oslo', 'madrid']
    
    variables_list = ['percent_residential', 'percent_commercial',
                      'percent_recreational', 
                      'pop_density', 'nearest_subway_dist',
                      'nearest_railway_dist', 'center_dist']
    
    data, asdf, traf_mat = load_city(CITY)
    
    model = FullModel(variables_list)
    asdf=model.fit(asdf, traf_mat)
    
    low_err, mid_err, high_err = test_model_stratisfied(asdf, traf_mat, variables_list, test_seed=42)
    
    #%% residual plots
    
    plt.style.use('seaborn-darkgrid')
    big_fig, big_ax = plt.subplots(figsize=(8,12), nrows=4, ncols=2)
    
    count=0
    for row in range(4):
        for col in range(2):
            city = cities[count]
            data, asdf, traf_mat = load_city(city)
            errors = trips_predict_error_plot(asdf, traf_mat, variables_list, error='residual', 
                                              by_cluster=False, show_id=True)
            big_ax[row,col].scatter(errors['predicted'], errors['error'])
            # big_ax[row,col].scatter(errors['predicted'], errors['error'], 
            #                         c=np.log(errors.true), cmap='viridis')
            
            line_stop = max(big_ax[row,col].get_xlim()[1], big_ax[row,col].get_ylim()[1])
            
            
            if row == 3:
                big_ax[row,col].set_xlabel('Predicted # trips')
            
            if col == 0:
                big_ax[row,col].set_ylabel('Residual')
            
            big_ax[row,col].set_title(f'{bs.name_dict[city]}')
            
            count+=1
    
    big_fig.tight_layout()
    
    #%% Relative error plots
    
    plt.style.use('seaborn-darkgrid')
    big_fig, big_ax = plt.subplots(figsize=(8,12), nrows=4, ncols=2)
    
    count=0
    for row in range(4):
        for col in range(2):
            city = cities[count]
            data, asdf, traf_mat = load_city(city)
            errors = trips_predict_error_plot(asdf, traf_mat, variables_list, error='relative', 
                                              by_cluster=False, show_id=False, plotfig=False)
            big_ax[row,col].scatter(errors['true'], errors['error']), 
                                    # c=np.log(errors['true']), cmap='viridis')
            
            if row == 3:
                big_ax[row,col].set_xlabel('True # trips')
            
            if col == 0:
                # big_ax[row,col].set_ylabel('Absolute error')
                big_ax[row,col].set_ylabel('Relative error')
            
            big_ax[row,col].set_title(f'{bs.name_dict[city]}')
            
            count+=1
    
    plt.tight_layout()
    
    #%% True # trips vs. predicted # trips
    
    plt.style.use('seaborn-darkgrid')
    big_fig, big_ax = plt.subplots(figsize=(8,12), nrows=4, ncols=2)
    
    count=0
    for row in range(4):
        for col in range(2):
            city = cities[count]
            data, asdf, traf_mat = load_city(city)
            errors = trips_predict_error_plot(asdf, traf_mat, variables_list, error='residual', 
                                              by_cluster=False, show_id=True)
            big_ax[row,col].scatter(errors['true'], errors['predicted'])
            # big_ax[row,col].scatter(errors['predicted'], errors['error'], 
            #                         c=np.log(errors.true), cmap='viridis')
            
            line_stop = max(big_ax[row,col].get_xlim()[1], big_ax[row,col].get_ylim()[1])
            big_ax[row,col].plot([0,line_stop], [0,line_stop], linestyle='--', c='k')
            
            big_ax[row,col].set_box_aspect(1)
            
            if row == 3:
                big_ax[row,col].set_xlabel('True # trips')
            
            if col == 0:
                big_ax[row,col].set_ylabel('Predicted # trips')
            
            big_ax[row,col].set_title(f'{bs.name_dict[city]}')
            
            count+=1
    
    big_fig.tight_layout(w_pad=-10)
    
    
    #%% Predict daily traffic
    
    model = FullModel(variables_list)
    asdf=model.fit(asdf, traf_mat)
    
    traf_mat_true = load_city(CITY, normalise=False)[2]
    
    stat_id = 520
    
    traffic_est = model.predict_daily_traffic(asdf[asdf.stat_id==stat_id],
                                              predict_cluster=True,
                                              plotfig=False)
    
    # Compare prediction to actual traffic
    
    traffic_true = traf_mat_true[data.stat.id_index[stat_id]]
    
    plt.style.use('seaborn-darkgrid')
    fig_dep, ax_dep = plt.subplots(figsize=(10,5))
    
    ax_dep.plot(traffic_true[:24], label='True traffic')
    ax_dep.plot(traffic_est[:24], label='Estimated traffic')
    
    ax_dep.set_xlabel('Hour')
    ax_dep.set_ylabel('# Trips')
    ax_dep.set_title(f'Predicted number of departures each hour for {data.stat.names[data.stat.id_index[stat_id]]} (ID: {stat_id})')
    
    ax_dep.legend()
    
    fig_arr, ax_arr = plt.subplots(figsize=(10,5))
    
    ax_arr.plot(traffic_true[24:], label='True traffic')
    ax_arr.plot(traffic_est[24:], label='Estimated traffic')
    
    ax_arr.set_xlabel('Hour')
    ax_arr.set_ylabel('# Trips')
    ax_arr.set_title(f'Predicted number of arrivals each hour for {data.stat.names[data.stat.id_index[stat_id]]} (ID: {stat_id})')
    
    ax_arr.legend()
    
    #%% Test daily traffic prediction
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
              'london', 'helsinki', 'oslo', 'madrid']
    
    variables_list = ['percent_residential', 'percent_commercial',
                      'percent_recreational', 
                      'pop_density', 'nearest_subway_dist',
                      'nearest_railway_dist', 'center_dist']
    
    
    error = 'MAE'
    show_std = True
    
    plt.style.use('seaborn-darkgrid')
    bigfig, bigax = plt.subplots(nrows=4, ncols=2, figsize=(10,10))
    
    count=0
    for row in range(4):
        for col in range(2):
            
            city = cities[count]
            
            data, asdf, traf_mat = load_city(city, normalise=True)
            
            model = FullModel(variables_list)
            asdf=model.fit(asdf, traf_mat)
            
            # labels = asdf.copy()['label']
            
            data, asdf, traf_mat_unnormed = load_city(city, normalise=False)
            # asdf_new['label'] = labels
            
            tm_err = test_model_stratisfied(asdf, traf_mat, variables_list, 
                                            test_seed=42, combine=True)
            
            dep_errors_mean = []
            dep_errors_std = []
            
            arr_errors_mean = []
            arr_errors_std = []
            
            # min_trips_mask = asdf['b_trips'] >= model.min_trips
            # labels_est, trips_est = model.predict(asdf[min_trips_mask])
            # traffic_true = traf_mat_unnormed[min_trips_mask]
            
            # traffic_est = np.zeros_like(traffic_true)
            
            # for i, label in enumerate(labels_est):
            #     traffic_est[i,:] = model.centers_traffic[label]*trips_est[i]
            
            if error == 'ME':
                
                mean_error = np.mean(tm_err, axis=0)
                std_error = np.std(tm_err, axis=0)
                ylabel = 'Mean error'
            
            elif error == 'MAE':
                
                mean_error = np.mean(np.abs(tm_err), axis=0)
                std_error = np.std(np.abs(tm_err), axis=0)
                ylabel = 'Mean absolute error'
                
                
            # elif error == 'MRE':
                
            #     mean_error = np.mean(np.abs(traffic_est-traffic_true)/traffic_true, axis=0)
            #     std_error = np.std(np.abs(traffic_est-traffic_true)/traffic_true, axis=0)
            #     ylabel = 'Mean relative error'
                
            dep_errors_mean = mean_error[:24]
            dep_errors_std = std_error[:24]
            
            arr_errors_mean = mean_error[24:]
            arr_errors_std = std_error[24:]
            
            bigax[row,col].plot(dep_errors_mean, c='tab:blue', label='departures')
            bigax[row,col].plot(arr_errors_mean, c='tab:orange', label='arrivals')
            
            if show_std:
                
                dep_fill_floor = np.zeros_like(dep_errors_std)
                arr_fill_floor = np.zeros_like(arr_errors_std)
                
                for i in range(len(dep_errors_std)):
                    dep_fill_floor[i] = np.max(dep_errors_mean[i]-dep_errors_std[i], 0)
                    arr_fill_floor[i] = np.max(arr_errors_mean[i]-arr_errors_std[i], 0)
                
                
                bigax[row,col].fill_between(np.arange(24), 0,
                                            dep_errors_mean+dep_errors_std,
                                            facecolor='tab:blue', alpha=0.2, label=r'departures $\pm$ std. dev.')
                bigax[row,col].fill_between(np.arange(24), 0,
                                            arr_errors_mean+arr_errors_std,
                                            facecolor='tab:orange', alpha=0.2, label=r'arrivals $\pm$ std. dev.')
            
            bigax[row,col].set_xticks(range(24))
            
            bigax[row,col].set_ylim(0,13)
            bigax[row,col].set_yticks(np.linspace(0,13,14))
            
            if col==0:
                bigax[row,col].set_ylabel(ylabel)
            
            if row==3:
                bigax[row,col].set_xlabel('Hour')
            
            bigax[row,col].set_title(bs.name_dict[city])
            count+=1
            
    bigfig.tight_layout(h_pad=4)
    lgd=bigax[3,0].legend(loc='upper center', bbox_to_anchor=(1,-0.2), ncol=4)
    bigfig.savefig('./figures/FullModel_MAE.pdf', bbox_extra_artsts=(lgd,), bbox_inches='tight')
    
    #%% Test daily traffic prediction by cluster
    
    split_clusters = False
    dep_or_arr = 'arr'
    test_ratio = 0.2
    np.random.seed(42)
    
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
    
    plt.style.use('seaborn-darkgrid')
    bigfig, bigax = plt.subplots(nrows=4, ncols=2, figsize=(10,10))
    
    count=0
    for row in range(4):
        for col in range(2):
            
            city = cities[count]
            
            data, asdf, traf_mat = load_city(city, normalise=True)
            
            min_trips_mask = asdf['b_trips'] >= 8
            df = asdf[min_trips_mask].copy()
            traf_mat = traf_mat[min_trips_mask]
            
            
            # Only get the correct labels
            model = FullModel(variables_list)
            df = model.fit(df, traf_mat)
            labels_true = df['label']
            
            n_stats = len(labels_true)
            n_test = n_stats*test_ratio
            
            test_dfs = []
            test_selected = np.array([])
            for j in range(model.k):
                cluster_mask = labels_true == j
                
                n_cluster = sum(cluster_mask)
                cluster_i = df[cluster_mask].index.to_numpy()
                np.random.shuffle(cluster_i)
                selected_i = cluster_i[:int(np.round(n_test*n_cluster/n_stats))]
                
                test_selected = np.concatenate((test_selected, selected_i))
                test_dfs.append(df[df.index.isin(selected_i)])
                
            # Make train df    
            
            train_mask = ~df.index.isin(test_selected)
            
            df_train = df[train_mask]
            tm_train = traf_mat[train_mask]
            
            df_test = df[~train_mask]
            labels_true = df_test['label'].to_numpy()
            
            model = FullModel(variables_list)
            df_train = model.fit(df_train, tm_train)
            
            data, asdf, tm_true = load_city(city, normalise=False)
            tm_true = tm_true[min_trips_mask]
            tm_true = tm_true[~train_mask]
            
            labels_est, trips_est = model.predict(df_test)
            
            tm_est = np.zeros_like(tm_true)
            for i, label in enumerate(labels_est):
                tm_est[i,:] = model.centers_traffic[label]*trips_est[i]
            
            tm_err = tm_est-tm_true
            # tm_err = tm_true
            
            
            if split_clusters:
                
                for j in range(model.k):
                    mean_err = np.mean(tm_err[labels_true == j], axis=0)
                    
                    if dep_or_arr == 'dep':
                        bigax[row, col].plot(range(24), mean_err[:24], label=cluster_name_dict[j])
                    elif dep_or_arr == 'arr':
                        bigax[row, col].plot(range(24), mean_err[24:], label=cluster_name_dict[j])
            
            else:
                mean_err = np.mean(tm_err, axis=0)
                bigax[row,col].plot(range(24), mean_err[:24], label='departures')
                bigax[row,col].plot(range(24), mean_err[24:], label='arrivals')
            
            bigax[row,col].set_xticks(range(24))
            
            bigax[row,col].set_ylim(-9,4)
            bigax[row,col].set_yticks(np.linspace(-9,4,14))
            
            if col==0:
                bigax[row,col].set_ylabel('Mean error')
            
            if row==3:
                bigax[row,col].set_xlabel('Hour')
            
            bigax[row,col].set_title(bs.name_dict[city])
            count+=1
            
    plt.tight_layout(h_pad=4)
    lgd=bigax[3,0].legend(loc='upper center', bbox_to_anchor=(1,-0.2), ncol=5)
    
    
    #%% Plot predictions of all station vs. their true traffic
    
    model = FullModel(variables_list)
    asdf=model.fit(asdf, traf_mat)
    
    labels = asdf.copy()['label']
    
    data, asdf_new, traf_mat_unnormed = load_city(CITY, normalise=False)
    asdf_new['label'] = labels
    
    clustered = ~asdf['label'].isna()
    
    stat_clustered = asdf_new[clustered]
    traf_mat_clustered = traf_mat_unnormed[clustered]
    
    for i in range(len(stat_clustered)):
        
        traffic_est = model.predict_daily_traffic(stat_clustered.iloc[i],
                                                  predict_cluster=True,
                                                  plotfig=False,
                                                  verbose=False)
        traffic_true = traf_mat_clustered[i]
    
        
        fig, ax = plt.subplots(figsize=(10,4))
        
        ax.plot(traffic_true, label='true')
        ax.plot(traffic_est, label='predicted')
        ax.set_ylabel('# trips')
        ax.legend()

    
    #%% Stratified model test
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
              'london', 'helsinki', 'oslo', 'madrid']
    
    variables_list = ['percent_residential', 'percent_commercial',
                      'percent_recreational', 
                      'pop_density', 'nearest_subway_dist',
                      'nearest_railway_dist', 'center_dist']
    
    dep_or_arr = 'dep'

    plt.style.use('seaborn-darkgrid')
    bigfig, bigax = plt.subplots(nrows=4, ncols=2, figsize=(10,10))
    
    count = 0
    for row in range(4):
        for col in range(2):
            
            city = cities[count]
            # city= 'london'
            
            data, asdf, traf_mat = load_city(city)
            
            low_err, mid_err, high_err = test_model_stratisfied(asdf, traf_mat, variables_list, test_seed=42)
            
            low_err = np.mean(low_err, axis=0)
            mid_err = np.mean(mid_err, axis=0)
            high_err = np.mean(high_err, axis=0)
            
            if dep_or_arr == 'dep':
                bigax[row,col].plot(range(24), low_err[:24], label='Low traffic stations')
                bigax[row,col].plot(range(24), mid_err[:24], label='Mid traffic stations')
                bigax[row,col].plot(range(24), high_err[:24], label='High traffic stations')
            
            elif dep_or_arr == 'arr':
                bigax[row,col].plot(low_err[24:], label='Low traffic stations')
                bigax[row,col].plot(mid_err[24:], label='Mid traffic stations')
                bigax[row,col].plot(high_err[24:], label='High traffic stations')
            
            bigax[row,col].set_xticks(range(24))
            
            bigax[row,col].set_ylim(-7,3)
            bigax[row,col].set_yticks(np.linspace(-7,3,11))
            
            if col==0:
                bigax[row,col].set_ylabel('Mean error')
            
            if row==3:
                bigax[row,col].set_xlabel('Hour')
            
            bigax[row,col].set_title(bs.name_dict[city])
            count+=1
    
    plt.tight_layout(h_pad=4)
    lgd=bigax[3,0].legend(loc='upper center', bbox_to_anchor=(1,-0.2), ncol=4)
    
    if dep_or_arr == 'dep':
        bigfig.savefig('./figures/FullModel_strat_dep.pdf', bbox_extra_artsts=(lgd,), bbox_inches='tight')
    
    elif dep_or_arr == 'arr':
        bigfig.savefig('./figures/FullModel_strat_arr.pdf', bbox_extra_artsts=(lgd,), bbox_inches='tight')
    
    #%% Test demand model between cities
    
    cities = ['nyc', 'chicago', 'washdc', 'boston', 
              'london', 'helsinki', 'oslo', 'madrid']
    
    variables_list = ['percent_residential', 'percent_commercial',
                      'percent_recreational', 
                      'pop_density', 'nearest_subway_dist',
                      'nearest_railway_dist', 'center_dist']
    
    np.random.seed(42)
    
    error = 'MAE'
    
    seed_range = range(50,75)
    
    err_table = pd.DataFrame(index=cities, columns=cities)
    
    for city_train in cities:
        
        data, asdf, traf_mat = load_city(city_train)
        
        for city_test in cities:
            
            if city_train == city_test:
                
                err = 0
                for split_seed in seed_range:
                    
                    np.random.seed(split_seed)
                    
                    # split data randomnly
                    
                    mask = np.random.rand(len(asdf)) < 0.2
                    
                    df_train = asdf[~mask]
                    tm_train = traf_mat[~mask]
                    
                    df_test = asdf[mask]
                    tm_test = traf_mat[mask]
                    
                    model = FullModel(variables_list)
                    model.fit(df_train, tm_train)
                    
                    demand_true = df_test['b_trips'].to_numpy()
                    demand_est = model.predict(df_test)[1]
                    
                    if error =='ME':
                        err += np.mean(demand_est-demand_true)
                    
                    elif error == 'MAE':
                        err += np.mean(np.abs(demand_est-demand_true))
                    
                    elif error == 'MSE':
                        err += np.mean(np.abs(demand_est-demand_true)**2)
                        
                err_table.loc[city_train, city_test] = err/len(seed_range)
                    
            else:
                
                # Get training and test sets
                
                data, df_train, tm_train = data, asdf, traf_mat
                data, df_test, tm_test = load_city(city_test)
                    
                # train model
                
                model = FullModel(variables_list)
                model.fit(df_train, tm_train)
                
                # Predict demand
                
                demand_true = df_test['b_trips'].to_numpy()
                demand_est = model.predict(df_test)[1]
                
                if error =='ME':
                    err = np.mean(demand_est-demand_true)
                
                elif error == 'MAE':
                    err = np.mean(np.abs(demand_est-demand_true))
                
                elif error == 'MSE':
                    err = np.mean(np.abs(demand_est-demand_true)**2)
        
                err_table.loc[city_train, city_test] = err
        
    err_table = err_table.rename(index=bs.name_dict, columns=bs.name_dict)
    print(err_table.to_latex(column_format='@{}l'+('r'*len(err_table.columns)) + '@{}', float_format="{:0.1f}".format, escape=False))
    
    
    err_mat = err_table.to_numpy()
    a = np.zeros(shape=(len(cities), len(cities)))
    
    for i in range(len(cities)):     
        for j in range(len(cities)):
            a[i,j] = err_mat[i,j]
    
    plt.style.use('default')
    fig, ax = plt.subplots()
    
    im = ax.matshow(a, cmap=plt.cm.Blues)
    
    fig.colorbar(im, label='Mean Absolute Error')
    
    for i in range(len(cities)):
        for j in range(len(cities)):
            c = a[j,i]
                
            # if res == 'confusion':
            #     ax.text(i, j, f'{c:.2f}', va='center', ha='center', c='white')
            # else:
            ax.text(i, j, f'{c:.1f}', va='center', ha='center')
    
    city_names = [bs.name_dict[city] for city in cities]
    
    ax.set_xticks(range(len(cities)))
    ax.set_yticks(range(len(cities)))
    ax.set_xticklabels(city_names)
    ax.set_yticklabels(city_names)
    ax.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    plt.setp([tick.label2 for tick in ax.xaxis.get_major_ticks()], rotation=45,
         ha="left", va="center",rotation_mode="anchor")
    
    ax.set_xlabel('Test city')
    ax.set_ylabel('Train city')
    plt.tight_layout()
    








