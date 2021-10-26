#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:37:32 2021

@author: dbvd
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText

def plot_trips_pr_hour_year(df, city, year, savefig=False, n_bins=24):
    """
    Plot a month calendar with the data. It looks like a calendar okay.

    Parameters
    ----------
    df
    """
    hours = np.arange(0, 24)

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"]
    
    month_abbr = {1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'}
    
    num_days = datetime.date(year, 12, 31).timetuple().tm_yday
    st_w = datetime.date(2019, 1, 1).weekday()
    
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots( num_days//7 + 1, 7,
                           sharex=True, sharey=True, figsize=(12, 120))

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.subplots_adjust(top=0.93)
    
    df['doy'] = df.start_dt.dt.day_of_year
    df['hour'] = df.start_dt.dt.hour

    for day in df['doy'].unique():
        #ln = ax[(day+st_w)//7, (datetime.datetime(year, 1, 1)+datetime.timedelta(days=day-1)).weekday()].bar(
        #    hours, , label='rides', color='C1'
        #    )
        day_dt = datetime.datetime(year, 1, 1)+datetime.timedelta(days=int(day-1))
        d_ax = ax[(day+st_w-1)//7, day_dt.weekday()]
        df.loc[df['doy'] == day]['hour'].hist(bins=n_bins, ax=d_ax)
        text_box = AnchoredText(f"{month_abbr[day_dt.month]}{day_dt.day}", frameon=False, loc='upper left', pad=0.3)
        d_ax.add_artist(text_box)
    for i in range(7):
        ax[0, i].set_title(weekdays[i])

    #fig.suptitle(f'Number of rides by day in {name_dict[city]} {months[month]} {year:d}',  fontsize=20)

    if savefig:
        plt.savefig(f"./figures/trips_pr_year-{city}{year:d}.pdf",
                    bbox_inches='tight', pad_inches=0
                    )

    plt.show()

if __name__ == '__main__':
    import bikeshare as bs
    
    city = 'madrid'
    year = 2019
    df_year = bs.get_data_year(city, year)[0]
    plot_trips_pr_hour_year(df_year, city, year)