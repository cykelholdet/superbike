#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:37:32 2021

@author: dbvd
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredText

def plot_trips_pr_hour_year(df, city, year, savefig=True, n_bins=24):
    """
    Plot a month calendar with the data. It looks like a calendar okay.

    Parameters
    ----------
    df
    """
    hours = np.arange(0, 24)
    
    cal = bs.get_cal(city)
    holidays = cal.holidays(year)
    holidays = pd.DataFrame(holidays, columns=['day', 'name'])

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"]
    
    month_abbr = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    precip = bs.get_weather_year(city, 2019)
    
    num_days = datetime.date(year, 12, 31).timetuple().tm_yday
    st_w = datetime.date(2019, 1, 1).weekday()
    
    print('Setting up plot...')
    plt.style.use('seaborn-darkgrid')
    plt.ioff()
    fig, ax = plt.subplots( num_days//7 + 1, 7,
                           sharex=True, sharey=True, figsize=(12, 100))

    fig.subplots_adjust(hspace=0.05, wspace=0.0)
    fig.subplots_adjust(top=0.97)
    
    twinax = dict()
    twinax_list = list()
    for row in range(num_days//7 + 1):
        for column in range(7):
            if twinax_list == []:
                twinax[row, column] = fig.add_axes(ax[row, column].get_position(True),
                                                   sharex=ax[row, column])

                twinax_list.append(twinax[row, column])
            else:
                twinax[row, column] = fig.add_axes(ax[row, column].get_position(True),
                                                   sharex=ax[row, column],
                                                   sharey=twinax_list[0])

            twinax[row, column].yaxis.tick_right()
            twinax[row, column].yaxis.set_label_position('right')
            twinax[row, column].yaxis.set_offset_position('right')
            

            twinax[row, column].set_autoscalex_on(ax[row, column].get_autoscalex_on())
            ax[row, column].yaxis.tick_left()
            twinax[row, column].xaxis.set_visible(False)
            twinax[row, column].patch.set_visible(False)
            twinax[row, column].grid(False)
            if column != 6:
                a = 0

                twinax[row, column].yaxis.set_tick_params(labelright=False)
    
    bins = np.linspace(0, 23, n_bins)
    
    df['doy'] = df.start_dt.dt.day_of_year
    df['hour'] = df.start_dt.dt.hour
    print('Plotting...')
    for day in df['doy'].unique():
        #ln = ax[(day+st_w)//7, (datetime.datetime(year, 1, 1)+datetime.timedelta(days=day-1)).weekday()].bar(
        #    hours, , label='rides', color='C1'
        #    )
        day_dt = datetime.datetime(year, 1, 1)+datetime.timedelta(days=int(day-1))
        d_ax = ax[(day+st_w-1)//7, day_dt.weekday()]
        d_twinax = twinax[(day+st_w-1)//7, day_dt.weekday()]
        df.loc[df['doy'] == day]['hour'].hist(bins=bins, ax=d_ax, width=0.9, label='rides', color='C1')
        ln2 = d_twinax.plot(
                precip[precip['time_dt'].dt.date == day_dt.date()].reset_index()['precipMM'],
                label='precipitation'
                )
        
        
        
        if day_dt.weekday() == 0:
            d_ax.set_ylabel("# rides")
        elif day_dt.weekday() == 6:
            d_twinax.set_ylabel("precipitation (mm)")
        
        if (day_dt.date() == holidays['day']).sum() > 0:
            d_ax.set_facecolor('#d4f0d3')
            text_box = AnchoredText(f"{month_abbr[day_dt.month]}{day_dt.day}\n{holidays[holidays['day'] == day_dt.date()]['name'].iloc[0]}", frameon=False, loc='upper left', pad=0.3)        
            d_ax.add_artist(text_box)
        elif day_dt.weekday() == 6 and day < 8:
            pass
        else:
            text_box = AnchoredText(f"{month_abbr[day_dt.month]}{day_dt.day}", frameon=False, loc='upper left', pad=0.3)        
            d_ax.add_artist(text_box)

    for i in range(7):
        ax[0, i].set_title(weekdays[i])
        ax[-1, i].set_xticks([0, 6, 12, 18])
        ax[-1, i].set_xlabel("hour")
    
    lines = [ax[17, -1].patches[0], ln2[0]]
    labels = [line.get_label() for line in lines]
    ax[0, -1].legend(lines, labels, loc=0)

    ymin, ymax = twinax[0, 0].get_ylim()
    twinax[0, 0].set_ylim(bottom=0)
    # if ymax < 10:
    #     twinax[0, 0].set_ylim(top=11)
    # if ymax > 20:
    #     twinax[0, 0].set_ylim(top=19)
    twinax[0, 0].set_ylim(top=11.9)
    
    fig.suptitle(f'Number of rides by day in {bs.name_dict[city]} {year:d}',  fontsize=20)

    if savefig:
        plt.savefig(f"./figures/trips_pr_year-{city}{year:d}.pdf",
                    bbox_inches='tight', pad_inches=0
                    )
    print(f"{bs.name_dict[city]} saved.")
    plt.clf()


if __name__ == '__main__':
    import bikeshare as bs
    
    cities = bs.name_dict.keys()
    
    cities = ['guadalajara', 'mexico']
    
    for city in cities:
        year = 2019
        df_year = bs.get_data_year(city, year)[0]
        plot_trips_pr_hour_year(df_year, city, year, savefig=True)
