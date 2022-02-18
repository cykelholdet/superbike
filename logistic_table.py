#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 08:52:14 2022

@author: dbvd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import bikeshare as bs
import interactive_plot_utils as ipu


def lr_coefficients(data, name, min_trips=100, clustering='k_means', k=3, 
                    random_state=42, day_type='business_days', 
                    service_radius=500, use_points_or_percents='points', 
                    make_points_by='station_location', add_const=False, 
                    use_road=False, remove_columns=[], title='City', 
                    big_station_df=False, return_model=False):
    
    if big_station_df:
        station_df, traffic_matrices, labels = ipu.big_station_df(data)
    else:
        station_df, land_use = ipu.make_station_df(data, holidays=False, return_land_use=True)
        traffic_matrices = data.pickle_daily_traffic(holidays=False)
    
        station_df, clusters, labels = ipu.get_clusters(traffic_matrices, 
                                                        station_df, 
                                                        day_type, 
                                                        min_trips, 
                                                        clustering, 
                                                        k, 
                                                        random_state=random_state)
        
        percent_cols = [column for column in station_df.columns if "percent_" in column]
        station_df = station_df.drop(columns=percent_cols).merge(
            ipu.neighborhood_percentages(
                data.city, station_df, land_use, 
                service_radius=service_radius, use_road=use_road
                ),
            how='outer', left_index=True, right_index=True)

    zone_columns = [column for column in station_df.columns if 'percent_' in column]
    
    if use_road == False and 'percent_road' in zone_columns:
        zone_columns.remove('percent_road')
    
    for column in remove_columns:
        if column in zone_columns:
            zone_columns.remove(column)

    other_columns = ['n_trips', 'pop_density', 'nearest_subway_dist']

    for column in remove_columns:
        if column in other_columns:
            other_columns.remove(column)

    lr_results, X, y = ipu.stations_logistic_regression(station_df, zone_columns, other_columns, use_points_or_percents=use_points_or_percents, make_points_by=make_points_by, const=add_const)

    print(lr_results)
    print(lr_results.summary())

    traffic_matrix, mask, _ = ipu.mask_traffic_matrix(traffic_matrices, station_df, day_type, min_trips, holidays=False, return_mask=True)
    
    for j in range(k):
        mean_vector = np.mean(traffic_matrix[np.where(labels[mask] == j)], axis=0)
        
        cc_df = pd.DataFrame([mean_vector[:24], mean_vector[24:]]).T.rename(columns={0:'departures', 1:'arrivals'})
        
        if big_station_df:
            cc_df.plot(title=f"{data} cluster {j}")
        else:
            cc_df.plot(title=f"{bs.name_dict[data.city]} {data.year} cluster {j}")

    single_index = lr_results.params[0].index

    parameters = np.concatenate([lr_results.params[i] for i in range(0, k-1)])
    stdev = np.concatenate([lr_results.bse[i] for i in range(0, k-1)])
    pvalues = np.concatenate([lr_results.pvalues[i] for i in range(0, k-1)])
    
    index = np.concatenate([lr_results.params.index for i in range(0, k-1)])

    multiindex = pd.MultiIndex.from_product([range(1,k), single_index], names=['Cluster', 'Coef. name'])

    pars = pd.Series(parameters, index=multiindex, name='coef')
    sts = pd.Series(stdev, index=multiindex, name='stdev')
    pvs = pd.Series(pvalues, index=multiindex, name='pvalues')
    
    coefs = pd.DataFrame(pars).join((sts, pvs))
    
    if return_model:
        return pd.concat({name: coefs}, names=[title], axis=1), lr_results, X, y
    else:
        return pd.concat({name: coefs}, names=[title], axis=1)


def formatter(x):
    if x == np.inf:
        return "inf"
    elif np.abs(x) > 10000 or np.abs(x) < 0.001:
        return f"$\\num{{{x:.2e}}}$"
    else:
        return f"${x:.4f}$"
    
def tuple_formatter(tup):
    x, bold = tup
    if x == np.inf:
        out = "inf"
    elif np.isnan(x):
        out = "--"
    elif np.abs(x) > 10000 or np.abs(x) < 0.001:
        if bold:
            out = f"$\\num[math-rm=\\mathbf]{{{x:.2e}}}$"
        else:
            out = f"$\\num{{{x:.2e}}}$"
    else:
        if bold:
            out = f"$\\mathbf{{{x:.3f}}}$"
        else:
            out = f"${x:.3f}$"
    
    return out



#%%
if __name__ == '__main__':
    
    index_dict = {'UNKNOWN': 'Unknown',
                  'residential': 'Residential',
                  'commercial': 'Commercial',
                  'industrial': 'Industrial',
                  'recreational': 'Recreational',
                  'educational': 'Educational',
                  'mixed': 'Mixed',
                  'road': 'Road',
                  'transportation': 'Transportation',
                  'n_trips': '\# Trips',
                  'nearest_subway_dist': 'Nearest Subway',
                  'pop_density': 'Pop. Density',
                  }
    
    percent_index_dict = {
        'percent_UNKNOWN': '\% Unknown',
        'percent_residential': '\% Residential',
        'percent_commercial': '\% Commercial',
        'percent_industrial': '\% Industrial',
        'percent_recreational': '\% Recreational',
        'percent_educational': '\% Educational',
        'percent_mixed': '\% Mixed',
        'percent_road': '\% Road',
        'percent_transportation': '\% Transportation',
        'n_trips': '\# Trips',
        'nearest_subway_dist': 'Nearest Subway',
        'pop_density': 'Pop. Density',
        }
    
    omit_columns = {
        'boston': ['percent_educational', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
        'chic': ['percent_transportation', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
        'nyc': ['percent_mixed', 'n_trips'],
        'washDC': ['percent_transportation', 'percent_industrial', 'percent_UNKNOWN', 'percent_mixed', 'n_trips'],
        'helsinki': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips'],
        'london': ['percent_transportation', 'percent_UNKNOWN', 'n_trips'],
        'madrid': ['n_trips'],
        'oslo': ['percent_transportation', 'percent_UNKNOWN', 'percent_industrial', 'n_trips', 'percent_mixed'],
        'USA': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'percent_mixed'],
        'EUR': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'percent_mixed'],
        'All': ['percent_transportation', 'percent_UNKNOWN', 'percent_educational', 'n_trips', 'percent_mixed'],
        }
    
    month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
          7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None'}

    CITY = 'chic'
    YEAR = 2019
    MONTH = None
    
    clustering = 'k_means'
    k = 3
    day_type = 'business_days'
    min_trips = 100
    use_points_or_percents = 'percents'
    make_points_by = 'station location'
    random_state = 42
    service_radius = 500
    use_road = False
    add_const = True
    
    table_type = 'only_coef'
# =============================================================================
#   Table types
# =============================================================================
    if table_type == 'month':
        
        table = pd.DataFrame([])
        for month in bs.get_valid_months(CITY, 2019):
            data = bs.Data(CITY, YEAR, month)
            
            table = pd.concat((table, lr_coefficients(
                data, month_dict[data.month], min_trips, clustering, k,
                random_state=random_state,
                day_type=day_type, service_radius=service_radius,
                use_points_or_percents=use_points_or_percents, 
                make_points_by=make_points_by, use_road=use_road,
                add_const=add_const, remove_columns=omit_columns[CITY],
                title=f"{bs.name_dict[CITY]} {YEAR}"
                )
            ), axis=1)
        
        add_year = True
        
        if add_year:
            data = bs.Data(CITY, YEAR, None)
            table = pd.concat((table, lr_coefficients(
                data, data.year, min_trips, clustering, k,
                random_state=random_state,
                day_type=day_type, service_radius=service_radius,
                use_points_or_percents=use_points_or_percents, 
                make_points_by=make_points_by, use_road=use_road,
                add_const=add_const, remove_columns=omit_columns[CITY],
                title=f"{bs.name_dict[CITY]} {YEAR}"
                )
            ), axis=1)
        
        print_type = 'only_coefs'
    
    if table_type == 'year':
        
        
        table = pd.DataFrame([])
        data = bs.Data(CITY, YEAR, None)
        table = pd.concat((table, lr_coefficients(
            data, month_dict[data.month], min_trips, clustering, k,
            random_state=random_state,
            day_type=day_type, service_radius=service_radius,
            use_points_or_percents=use_points_or_percents, 
            make_points_by=make_points_by, use_road=use_road,
            add_const=add_const, remove_columns=omit_columns[CITY],
            title=f"{bs.name_dict[CITY]} {YEAR}"
            )
        ), axis=1)
        
        print_type = 'only_coefs'
    
    elif table_type == 'points_percentage':
        point_names = [#('points', 'station location', 'Station Location'),
                       ('points', 'station land use', 'Service Area Max'),]
                       #('percents', 'station land use', 'Percentage Land Use')]
        
        remove_percent = {
            'percent_UNKNOWN': 'UNKNOWN',
            'percent_residential': 'residential',
            'percent_commercial': 'commercial',
            'percent_industrial': 'industrial',
            'percent_recreational': 'recreational',
            'percent_educational': 'educational',
            'percent_mixed': 'mixed',
            'percent_road': 'road',
            'percent_transportation': 'transportation',
            }
        
        table = pd.DataFrame([])
        data = bs.Data(CITY, YEAR, MONTH)
        
        for pop, make_by, name in point_names:
            
            table = pd.concat((table, lr_coefficients(
                data, name, min_trips, clustering, k,
                random_state=random_state,
                day_type=day_type, service_radius=service_radius,
                use_points_or_percents=pop, 
                make_points_by=make_by, use_road=use_road,
                add_const=add_const, remove_columns=omit_columns[CITY],
                title=f"{bs.name_dict[CITY]} {YEAR} {month_dict[MONTH]}"
                ).rename(index=remove_percent)
            ), axis=1)
        
        print_type = 'pvalues'
        
    elif table_type == 'cluster_type':
        clustering = [('k_means', 'K Means'),
                      ('k_medoids', 'K Medoids'),
                      ('h_clustering', 'Hierarchical Clustering'),
                      ]
        
        
        table = pd.DataFrame([])
        data = bs.Data(CITY, YEAR, MONTH)
        
        for pop, make_by, name in point_names:
            
            table = pd.concat((table, lr_coefficients(
                data, name, min_trips, clustering, k,
                random_state=random_state,
                day_type=day_type, service_radius=service_radius,
                use_points_or_percents=pop, 
                make_points_by=make_by, use_road=use_road,
                )
            ), axis=1)
        
        print_type = 'only_coefs'
            
    elif table_type == 'city':
        city_list = ['nyc', 'boston', 'washDC', 'london']
        
        table = pd.DataFrame([])
        for city in city_list:
            data = bs.Data(city, YEAR, MONTH)
            
            table = pd.concat((table, lr_coefficients(
                data, bs.name_dict[data.city], min_trips, clustering, k,
                random_state=random_state,
                day_type=day_type, service_radius=service_radius,
                use_points_or_percents=use_points_or_percents, 
                make_points_by=make_points_by, use_road=use_road, add_const=add_const, remove_columns=omit_columns[CITY], title="City"
                )
            ), axis=1)
        
        
        tables = dict()
        for i in range(1, k):
            tables[i] = table.loc[i].loc[list(index_dict.keys())] # Reorder according to index_dict
        
        table = pd.concat(tables, names=['Cluster', 'Coef. name'])
        table = table.rename(index=index_dict)
        
        print(table.to_latex(multicolumn_format='c', multirow=True, formatters = [formatter]*len(table.columns), escape=False))
        
        print_type = None
        
    elif table_type == 'only_coef':
        city_list = ['boston', 'chic', 'nyc', 'washDC', 'helsinki', 'london', 'madrid', 'oslo']
        
        table = pd.DataFrame([])
        for city in city_list:
            data = bs.Data(city, YEAR, MONTH)
            
            table = pd.concat((table, lr_coefficients(
                data, bs.name_dict[data.city], min_trips, clustering, k,
                random_state=random_state,
                day_type=day_type, service_radius=service_radius,
                use_points_or_percents=use_points_or_percents, 
                make_points_by=make_points_by, use_road=use_road, add_const=add_const, remove_columns=omit_columns[city], title="City"
                )
            ), axis=1)
        
        
        # signif_table = table.xs('pvalues', level=1, axis=1) < 0.05
        
        # coeftable = table.xs('coef', level=1, axis=1)
        
        # tuple_table = pd.concat([coeftable,signif_table]).stack(dropna=False).groupby(level=[0,1,2]).apply(tuple).unstack()
        
        # if use_points_or_percents == 'percents':
        #     index_list = list(percent_index_dict.keys())
        #     index_renamer = percent_index_dict
        # else:
        #     index_list = list(index_dict.keys())
        #     index_renamer = index_dict
        
        # index_list.insert(0, 'const')
        
        # index_list = [x for x in index_list if x in table.index.get_level_values(1)]
        
        
        # tables = dict()
        # for i in range(1, k):
        #     tables[i] = tuple_table.loc[i].loc[index_list] # Reorder according to index_dict
        
        # tuple_table = pd.concat(tables, names=['Cluster', 'Coef. name'])
        # tuple_table = tuple_table.rename(index=index_renamer)
        
        # if k == 3:
        #     tuple_table = tuple_table.rename(index={1: 'Morning Sink', 2: 'Morning Source'})
        
        # tuple_table = tuple_table.reindex(columns=[bs.name_dict[i] for i in city_list])
        
        # print(tuple_table.to_latex(column_format='ll'+'r'*(len(coeftable.columns)), multirow=True, formatters = [tuple_formatter]*len(coeftable.columns), escape=False))
        
        print_type = 'only_coefs'
    
    
    elif table_type == 'us_eu':
        city_lists = [(['boston', 'chic', 'nyc', 'washDC'], 'USA'),
                      (['helsinki', 'london', 'madrid', 'oslo'], 'EUR'),
                      (['boston', 'chic', 'nyc', 'washDC', 'helsinki', 'london', 'madrid', 'oslo'], 'All')
                      ]
        
        table = pd.DataFrame([])
        for cities, name in city_lists:
            table = pd.concat((table, lr_coefficients(
                cities, name, min_trips, clustering, k,
                random_state=random_state,
                day_type=day_type, service_radius=service_radius,
                use_points_or_percents=use_points_or_percents, 
                make_points_by=make_points_by, use_road=use_road,
                add_const=add_const, remove_columns=omit_columns[name],
                title="", big_station_df=True
                )
            ), axis=1)
        
        print_type = 'only_coefs'
    
    if table_type == 'month_all_cities':
        for city in ['boston', 'chic', 'nyc', 'washDC', 'helsinki', 'london', 'madrid', 'oslo']:
            month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
                  7:'Jul',8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec', None:'None', YEAR: YEAR}
            
            table = pd.DataFrame([])
            for month in bs.get_valid_months(city, YEAR):
                data = bs.Data(city, YEAR, month)
                
                table = pd.concat((table, lr_coefficients(
                    data, month_dict[data.month], min_trips, clustering, k,
                    random_state=random_state,
                    day_type=day_type, service_radius=service_radius,
                    use_points_or_percents=use_points_or_percents, 
                    make_points_by=make_points_by, use_road=use_road, add_const=add_const, remove_columns=omit_columns[city], title=f"{bs.name_dict[city]} {YEAR}"
                    )
                ), axis=1)
            
            add_year = True
            
            if add_year:
                data = bs.Data(city, YEAR, None)
                table = pd.concat((table, lr_coefficients(
                    data, data.year, min_trips, clustering, k,
                    random_state=random_state,
                    day_type=day_type, service_radius=service_radius,
                    use_points_or_percents=use_points_or_percents, 
                    make_points_by=make_points_by, use_road=use_road, add_const=add_const, remove_columns=omit_columns[city], title=f"{bs.name_dict[city]} {YEAR}"
                    )
                ), axis=1)
                
                

            signif_table = table.xs('pvalues', level=1, axis=1) < 0.05
            
            coeftable = table.xs('coef', level=1, axis=1)
            
            tuple_table = pd.concat([coeftable,signif_table]).stack(dropna=False).groupby(level=[0,1,2]).apply(tuple).unstack()
            
            if use_points_or_percents == 'percents':
                index_list = list(percent_index_dict.keys())
                index_renamer = percent_index_dict
            else:
                index_list = list(index_dict.keys())
                index_renamer = index_dict
            
            index_list.insert(0, 'const')
            
            #index_list = set(index_list).intersection(set(table.index.get_level_values(1)))
            
            index_list = [x for x in index_list if x in table.index.get_level_values(1)]
            
            tables = dict()
            for i in range(1, k):
                tables[i] = tuple_table.loc[i].loc[index_list] # Reorder according to index_dict
            
            tuple_table = pd.concat(tables, names=['Cluster', 'Coef. name'])
            tuple_table = tuple_table.rename(index=index_renamer)
            
            column_list = list(bs.get_valid_months(city, YEAR))
            
            if add_year:
                column_list.append(YEAR)
            
            tuple_table = tuple_table.reindex(columns=[month_dict[i] for i in column_list])
            
            if k == 3:
                tuple_table = tuple_table.rename(index={1: 'Morning Sink', 2: 'Morning Source'})
            
            latex_table = tuple_table.to_latex(column_format='ll'+'r'*(len(coeftable.columns)), multirow=True, formatters = [tuple_formatter]*len(coeftable.columns), escape=False)
            print(latex_table)
            with open('figures/coef_table.tex','a') as file:
                file.write(latex_table + "\n")
            
            coeftable.plot()
                
        print_type = None
    
# =============================================================================
#   Printing section
# =============================================================================
    if print_type == "only_coefs":
        
        signif_table = table.xs('pvalues', level=1, axis=1) < 0.05
        
        coeftable = table.xs('coef', level=1, axis=1)
        
        tuple_table = pd.concat([coeftable,signif_table]).stack(dropna=False).groupby(level=[0,1,2]).apply(tuple).unstack()
        
        if use_points_or_percents == 'percents':
            index_list = list(percent_index_dict.keys())
            index_renamer = percent_index_dict
        else:
            index_list = list(index_dict.keys())
            index_renamer = index_dict
        
        index_list.insert(0, 'const')
        
        #index_list = set(index_list).intersection(set(table.index.get_level_values(1)))
        
        index_list = [x for x in index_list if x in table.index.get_level_values(1)]
        
        tables = dict()
        for i in range(1, k):
            tables[i] = tuple_table.loc[i].loc[index_list] # Reorder according to index_dict
        
        tuple_table = pd.concat(tables, names=['Cluster', 'Coef. name'])
        tuple_table = tuple_table.rename(index=index_renamer)
        
        if table_type == 'month':
            tuple_table = tuple_table.reindex(columns=[month_dict[i] for i in list(bs.get_valid_months(CITY, YEAR))])
        elif table_type == 'only_coef':
            tuple_table = tuple_table.reindex(columns=[bs.name_dict[i] for i in city_list])
            
        if k == 3:
            tuple_table = tuple_table.rename(index={1: 'Morning Sink', 2: 'Morning Source'})
        
        
        latex_table = tuple_table.to_latex(column_format='@{}ll'+('r'*len(tuple_table.columns)) + '@{}', multirow=True, formatters = [tuple_formatter]*len(tuple_table.columns), escape=False)
        print(latex_table)
        
        # collist = list(tuple_table.columns)
        # latex_table = tuple_table[collist[:4]].to_latex(column_format='@{}ll'+('r'*len(collist[:4])) + '@{}', multirow=True, formatters = [tuple_formatter]*len(collist[:4]), escape=False)
        # print(latex_table)
        # latex_table = tuple_table[collist[4:]].to_latex(column_format='@{}ll'+('r'*len(collist[4:])) + '@{}', multirow=True, formatters = [tuple_formatter]*len(collist[4:]), escape=False)
        # print(latex_table)
        
        with open('figures/coef_table.tex','a') as file:
            file.write(latex_table + "\n")

    elif print_type == "pvalues":
        
        signif_table = table.xs('pvalues', level=1, axis=1) < 0.05
        
        tabo = table*0
        
        tabo.iloc[:,tabo.columns.get_level_values(1) == 'pvalues'] = signif_table
        # coeftable = table.xs('coef', level=1, axis=1)
        
        #tuple_table = pd.concat([table,tabo]).stack(dropna=False).groupby(level=[0,1,2]).apply(tuple).unstack()
        c = np.stack([table.to_numpy(), tabo.to_numpy()], axis=2)
        tuple_table = pd.DataFrame([[(c[i, j, 0], c[i, j, 1]) for j in range(c.shape[1])] for i in range(c.shape[0])], columns=table.columns, index=table.index)
        
        if use_points_or_percents == 'percents':
            index_list = list(percent_index_dict.keys())
            index_renamer = percent_index_dict
        else:
            index_list = list(index_dict.keys())
            index_renamer = index_dict
        
        index_list.insert(0, 'const')
        
        #index_list = set(index_list).intersection(set(table.index.get_level_values(1)))
        
        index_list = [x for x in index_list if x in table.index.get_level_values(1)]
        
        tables = dict()
        for i in range(1, k):
            tables[i] = tuple_table.loc[i].loc[index_list] # Reorder according to index_dict
        
        
        tuple_table = pd.concat(tables, names=['Cluster', 'Coef. name'])
        tuple_table = tuple_table.rename(index=index_renamer)
        
        if k == 3:
            tuple_table = tuple_table.rename(index={1: 'Morning Sink', 2: 'Morning Source'})
        
        print(tuple_table.to_latex(column_format='@{}ll'+ ('r'*len(table.columns)) + '@{}', multicolumn_format='c', multirow=True, formatters = [tuple_formatter]*len(table.columns), escape=False))
    
    
    else:
        
        if use_points_or_percents == 'percents':
            index_list = list(percent_index_dict.keys())
            index_renamer = percent_index_dict
        else:
            index_list = list(index_dict.keys())
            index_renamer = index_dict
        
        index_list.insert(0, 'const')
        
        #index_list = set(index_list).intersection(set(table.index.get_level_values(1)))
        
        index_list = [x for x in index_list if x in table.index.get_level_values(1)]
        
        tables = dict()
        for i in range(1, k):
            tables[i] = table.loc[i].loc[index_list] # Reorder according to index_dict
        
        
        table = pd.concat(tables, names=['Cluster', 'Coef. name'])
        table = table.rename(index=index_renamer)
        
        print(table.to_latex(multicolumn_format='c', multirow=True, formatters = [formatter]*len(table.columns), escape=False))
        
        
