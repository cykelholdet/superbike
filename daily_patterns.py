"""
@author: Mattek Group 3

Plot calendar as well as average weekday and weekend day.
Plot demographics.
"""

import numpy as np
import matplotlib.pyplot as plt
import bikeshare as bs
import pandas as pd
import scipy.stats as stats
from matplotlib.offsetbox import AnchoredText


def get_trips_pr_hour(citydata, condition, n_categories):
    """
    Get the amount of trips by their output to the given condition.

    Parameters
    ----------
    citydata : object of Data class
        Contains all of the data.
    condition : lambda function
        condition to fulfill. Output of this will divide into categories.
    n_categories : TYPE
        Number of categories to split the data in based on the condition.

    Returns
    -------
    trips_pr_hour : np array
        category, day number, hour.

    """
    trips_pr_hour = np.zeros((n_categories, citydata.num_days+1, 24))

    print("Calculating trips per hour. Grab a snack, this can take a while...")
    con = condition(citydata.df)
    day = citydata.df['start_dt'].dt.day
    hour = citydata.df['start_dt'].dt.hour

    for c, d, h in zip(con, day, hour):
        trips_pr_hour[c, d, h] += 1

    return trips_pr_hour


def get_trips_pr_hour_nocondition(citydata):
    """
    Get the amount of trips by their output to the given condition.

    Parameters
    ----------
    citydata : object of Data class
        Contains all of the data.
    condition : lambda function
        condition to fulfill. Output of this will divide into categories.
    n_categories : TYPE
        Number of categories to split the data in based on the condition.

    Returns
    -------
    trips_pr_hour : np array
        category, day number, hour.

    """
    trips_pr_hour = np.zeros((1, citydata.num_days+1, 24))
    print("Calculating trips per hour. Grab a snack, this can take a while...")
    day = citydata.df['start_dt'].dt.day
    hour = citydata.df['start_dt'].dt.hour
    for d, h in zip(day, hour):
        trips_pr_hour[0, d, h] += 1
    return trips_pr_hour


def plot_trips_pr_hour_month(citydata, trips_pr_hour, labels, savefig=False, name=None):
    """
    Plot a month calendar with the data. It looks like a calendar okay.

    Parameters
    ----------
    citydata : object of Data class
        Contains all of the data.
    trips_pr_hour : np array
        category, day number, hour.
    labels : list
        Labels for plot . same length as number of conditions. [''] for blank.
    savefig : bool, optional
        Whether to save the figure. The default is False.
    name : str, optional
        name of the condition we are testing for. The default is None.

    """
    data_shape = trips_pr_hour.shape

    hours = np.arange(0, 24)

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"]

    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots((citydata.num_days+citydata.weekdays[0])//7 + 1, 7,
                           sharex=True, sharey=True, figsize=(12, 10))

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.subplots_adjust(top=0.93)

    width = (1 - 1/(3*data_shape[0]**2))/data_shape[0]

    twinax = dict()
    twinax_list = list()
    for row in range(6):
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

    for day in range(0, citydata.num_days):
        for i in range(data_shape[0]):
            ln1 = ax[(day+citydata.weekdays[0])//7, citydata.weekdays[day]].bar(
                hours - width*((data_shape[0] - 1)/2 - i),
                trips_pr_hour[i, day+1], width, label='rides', color='C1'
                )
            ln2 = twinax[(day+citydata.weekdays[0])//7, citydata.weekdays[day]].plot(
                weather[weather['day'] == day+1].reset_index()['precipMM'],
                label='precipitation'
                )

            text_box = AnchoredText(str(day+1), frameon=False, loc='upper left', pad=0.3)
            ax[(day+citydata.weekdays[0])//7, citydata.weekdays[day]].add_artist(text_box)
            if day+1 in holidays[city]:
                ax[(day+citydata.weekdays[0])//7, citydata.weekdays[day]].set_facecolor('#d4f0d3')
        # ax[(day+citydata.weekdays[0])//7, citydata.weekdays[day]].set_xticks([0,3,6,9,12,15,18,21])
        ax[(day+citydata.weekdays[0])//7, citydata.weekdays[day]].set_xticks([0, 6, 12, 18])
        ax[(day+citydata.weekdays[0])//7, 0].set_ylabel("# rides")
        twinax[(day+citydata.weekdays[0])//7, 6].set_ylabel("precipitation (mm)")
        ax[5, day % 7].set_xlabel("hour")

    for i in range(7):
        ax[0, i].set_title(weekdays[i])

    lines = [ln1, ln2[0]]
    labels = [line.get_label() for line in lines]
    ax[0, 5].legend(lines, labels, loc=0)

    ymin, ymax = twinax[0, 0].get_ylim()
    twinax[0, 0].set_ylim(bottom=0)
    # if ymax < 10:
    #     twinax[0, 0].set_ylim(top=11)
    # if ymax > 20:
    #     twinax[0, 0].set_ylim(top=19)
    twinax[0, 0].set_ylim(top=12)

    fig.suptitle(f'Number of rides by day in {name_dict[city]} {months[month]} {year:d}',  fontsize=20)

    if savefig:
        plt.savefig(f"./figures/trips_pr_hour_month_precipitation_{name}-{city}{year:d}{month:02d}.pdf",
                    bbox_inches='tight', pad_inches=0
                    )

    plt.show()
    return a

def plot_trips_pr_hour_avg(citydata, trips_pr_hour, labels, normalize=False,
                           savefig=False, name=None, cityname='', lineplot=False):
    """
    Plot bar charts of weekdays and weekend next to each other.

    Parameters
    ----------
    citydata : object of Data class
        Contains all of the data.
    trips_pr_hour : np array
        category, day number, hour.
    labels : list
        Labels for plot . same length as number of conditions. [''] for blank.
    normalize : bool, optional
        Whether or not to normalize by traffic over the day. shows percentage
        of daily traffic by hour
        The default is False.
    savefig : bool, optional
        Whether to save the figure. The default is False.
    name : str, optional
        name of the condition we are testing for. The default is None.

    Returns
    -------
    trips_pr_hour_avg : np array
        contaits trips averaged over weekdays and weekend respecitvely.
        condition, weekday/weekend, hour

    """
    data_shape = trips_pr_hour.shape
    trips_pr_hour_avg = np.zeros((data_shape[0], 2, 24))
    wd = np.array(citydata.weekdays)

    weekdays = np.where(wd < 5)[0]
    workdays = weekdays[np.isin(weekdays, holidays[citydata.city], invert=True)]

    for day in workdays:
        trips_pr_hour_avg[:, 0, :] += trips_pr_hour[:, day+1, :]

    trips_pr_hour_avg[:, 0, :] = trips_pr_hour_avg[:, 0, :] / len(workdays)

    for day in np.where(wd >= 5)[0]:  # weekend
        trips_pr_hour_avg[:, 1, :] += trips_pr_hour[:, day+1, :]

    trips_pr_hour_avg[:, 1, :] = trips_pr_hour_avg[:, 1, :] / len(np.where(wd >= 5)[0])

    if normalize:
        trips_pr_hour_avg[:, 0] = (trips_pr_hour_avg[:, 0].T / np.sum(trips_pr_hour_avg[:, 0], axis=1)).T
        trips_pr_hour_avg[:, 1] = (trips_pr_hour_avg[:, 1].T / np.sum(trips_pr_hour_avg[:, 1], axis=1)).T

    hours = np.arange(0, 24)

    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5, 2.5))

    fig.subplots_adjust(hspace=0.1, wspace=0.05)

    width = 0.8/data_shape[0]

    if lineplot is True:
        for day in range(0, 2):
            for i in range(data_shape[0]):
                ax[day].plot(range(24), trips_pr_hour_avg[i, day], '.-', label=labels[i])

            ax[day].set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
            if normalize:
                ax[0].set_ylabel("% of rides")
            else:
                ax[0].set_ylabel("# rides")
            ax[day].set_xlabel("Hour")
    else:
        for day in range(0, 2):
            for i in range(data_shape[0]):
                ax[day].bar(hours - width*((data_shape[0] - 1)/2 - i),
                            trips_pr_hour_avg[i, day], width, label=labels[i])

            ax[day].set_xticks([0, 3, 6, 9, 12, 15, 18, 21])
            if normalize:
                ax[0].set_ylabel("% of rides")
            else:
                ax[0].set_ylabel("# rides")
            ax[day].set_xlabel("Hour")

    ax[0].set_title("Weekdays")
    ax[1].set_title("Weekend")

    ax[1].legend()

    if normalize:
        fig.suptitle(f'Normalised average number of rides in {city.upper()} {months[month]} {year:d}',
                     fontsize=16, y=0.5)
    elif name is not None:
        fig.suptitle(f'{name} in {cityname} {months[month]} {year:d}',
                     fontsize=12, y=1.04)
    else:
        fig.suptitle(f'Average # rides in {cityname} {months[month]} {year:d}',
                     fontsize=12, y=1.04)

    if savefig is True:
        plt.savefig(f"./figures/trips_pr_hour_avg-{name}-{city}{year:d}{month:02d}{'normalized'*int(normalize)}.pdf",
                    bbox_inches='tight', pad_inches=0)
    plt.show()
    
    return trips_pr_hour_avg


def duration_kde(citydata):
    xmax = 50
    N = 300
    
    weekdays_list = np.where(np.array(citydata.weekdays) < 5)[0] + 1
    weekend_list = np.where(np.array(citydata.weekdays) >= 5)[0] + 1
    
    df = citydata.df
    
    wdays = df.loc[df['day'].isin(weekdays_list)]
    wend  = df.loc[df.day.isin(weekend_list)]
    
    ker_wday = stats.gaussian_kde(wdays['duration'], 0.005)
    ker_wend = stats.gaussian_kde(wend['duration'], 0.005)
    x = np.linspace(0, xmax, N)
    
    y_wday = ker_wday(x)
    y_wend = ker_wend(x)
    
    plt.plot(x, y_wday, label="weekdays")
    plt.plot(x, y_wend, label="weekend")
    
    plt.legend()
    plt.ylabel("density")
    plt.xlabel("Trip duration (min)")
    plt.savefig("figures/duration_kde_wday_wend.pdf")
    plt.show()


def duration_demog_kde(citydata, demog, cat_dict, categories, labels, n_cats):
    xmax = 50
    N = 300
    weekdays_list = np.where(np.array(citydata.weekdays) < 5)[0] + 1
    weekend_list = np.where(np.array(citydata.weekdays) >= 5)[0] + 1
    
    df = citydata.df
    df[demog] = df[demog].map(cat_dict)
    wdays = df.loc[df['start_dt'].dt.day.isin(weekdays_list)]
    wends = df.loc[df['start_dt'].dt.day.isin(weekend_list)]
    
    x = np.linspace(0, xmax, N)
    
    wday_dem = dict()
    wday_ker = dict()
    wday_y = dict()
    for category in range(n_cats):
        wday_dem[category] = wdays.loc[wdays[demog] == category]
        wday_ker[category] = stats.gaussian_kde(wday_dem[category]['duration']/60, 0.01)
        wday_y[category] = wday_ker[category](x)
        
    wend_dem = dict()
    wend_ker = dict()
    wend_y = dict()
    for category in range(n_cats):
        wend_dem[category] = wends.loc[wends[demog] == category]
        wend_ker[category] = stats.gaussian_kde(wend_dem[category]['duration']/60, 0.01)
        wend_y[category] = wend_ker[category](x)
    
    
    
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(5, 2.5))

    fig.subplots_adjust(hspace=0.1, wspace=0.05)
    
    for category in range(n_cats):
        ax[0].plot(x, wday_y[category], label=labels[category])
        ax[1].plot(x, wend_y[category], label=labels[category])
    
    
    ax[0].legend()
    ax[0].set_ylabel("density")
    ax[0].set_xlabel("Trip duration (min)")
    #plt.savefig("figures/duration_kde_cus_sub_wday.pdf")
    #plt.show()

    
    ax[1].legend()
    #ax[1].ylabel("density")
    ax[1].set_xlabel("Trip duration (min)")
    
    ax[0].set_title("Weekdays")
    ax[1].set_title("Weekend")
    
    fig.suptitle(f'{dname_dict[demog]} in {name_dict[citydata.city]} {months[month]} {year:d}',
                 fontsize=12, y=1.04)
    
    plt.savefig(f"figures/duration_kde_{citydata.city}_{demog}.pdf",
                    bbox_inches='tight', pad_inches=0)
    plt.show()


# %%
months = ["", "January", "February", "March", "April", "Nay", "June", "July",
              "August", "September", "October", "November", "December"]

cities = ['chic', 'london', 'madrid', 'mexico', 'nyc', 'sfran', 'taipei', 'washDC']
city_names = ['Chicago', 'London', 'Madrid', 'Mexico City', 'New York City',
              'San Francisco', 'Taipei', 'Washington DC']

name_dict = {'chic': 'Chicago',
             'london': 'London',
             'madrid': 'Madrid',
             'mexico': 'Mexico City',
             'nyc': 'New York City',
             'sfran': 'San Francisco',
             'taipei': 'Taipei',
             'washDC': 'Washington DC'}

holidays = dict()
holidays['chic'] = [2]  # Labor Day
holidays['london'] = []
holidays['madrid'] = []
holidays['mexico'] = [16]  # Independence day
holidays['nyc'] = holidays['chic']
holidays['sfran'] = holidays['chic']
holidays['taipei'] = [13]  # Mid autumn festival
holidays['washDC'] = holidays['chic']

year = 2019
month = 9

savefig = True

for city in cities:
    data = bs.Data(city, year, month)
    _, weather = bs.get_weather(city, year, month)

    trips_pr_hour = get_trips_pr_hour_nocondition(data)

    plot_trips_pr_hour_month(data, np.array([trips_pr_hour.sum(axis=0)]),
                             [""], savefig=savefig, name="total")

    if city == 'taipei':
        holidays['taipei'] = [13, 30]  # Typhoon on the 30th removed from average
    num_trips = plot_trips_pr_hour_avg(data, np.array([trips_pr_hour.sum(axis=0)]), [""],
                           savefig=savefig, cityname=name_dict[city])
    print(np.sum(num_trips, axis=2))
    holidays['taipei'] = [13]

# %%
demographic_info = dict()
demographic_info['chic'] = ['user_type', 'birth_year', 'gender']
demographic_info['london'] = []
demographic_info['madrid'] = ['user_type', 'ageRange']
demographic_info['mexico'] = ['age', 'gender']
demographic_info['nyc'] = ['user_type', 'birth_year', 'gender']
demographic_info['sfran'] = ['user_type']
demographic_info['taipei'] = []
demographic_info['washDC'] = ['user_type']

dname_dict = {'user_type': 'User Type', 'age_group': 'Age Group',
              'gender': 'Gender', 'ageRange': 'Age Group'}

table_dict = dict()

for city in cities:
    if demographic_info[city] != []:
        data = bs.Data(city, year, month)
        di = demographic_info[city]
        table_dict[city] = dict()
        for demog in di:
            print(f'{city} {demog}')
            if demog == 'birth_year':
                data.df['age'] = data.df['birth_year']*(-1) + year
                data.df['age_group'] = np.floor(data.df['age']/10)*10
                data.df.loc[data.df['age_group'] > 70, 'age_group'] = 70
                demog = 'age_group'

            if demog == 'age':
                data.df['age_group'] = np.floor(data.df['age']/10)*10
                data.df.loc[data.df['age_group'] > 70, 'age_group'] = 70
                demog = 'age_group'

            condition = lambda df: df[demog]
            categories = data.df[demog].value_counts(dropna=False).sort_index()
            n_cats = len(categories)
            labels = list(categories.index.fillna('Other/NA'))
            cat_dict = dict(zip(categories.index, range(n_cats)))
            #duration_demog_kde(data, demog, cat_dict, categories, labels, n_cats)
            if demog == 'age_group':
                labels = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70+', 'Other/NA']
            if demog == 'ageRange':
                labels = ['0-16', '17-18', '19-26', '27-40', '41-65', '66+', 'N/A']
            if city == 'nyc' and demog == 'age_group':
                data.df.drop(data.df[(data.df['gender'] == 0) & (data.df['birth_year'] == 1969)].index, inplace=True)
            if city == 'nyc' and demog == 'gender':
                labels = ['Other', 'Male', 'Female']
                data.df.drop(data.df[(data.df['gender'] == 0) & (data.df['birth_year'] == 1969)].index, inplace=True)
            if city == 'madrid' and demog == 'user_type':
                labels = ['Annual user', 'Occasional user', 'Company worker']

            
            condition = lambda df: df[demog].map(cat_dict)

            tph_condition = get_trips_pr_hour(data, condition, n_cats)
            if demog == 'ageRange':
                tph_condition = np.concatenate(
                    (tph_condition[1:], np.expand_dims(tph_condition[0], axis=0)))
                labels = ['0-16', '17-18', '19-26', '27-40', '41-65', '66+', 'N/A']
            if city == 'nyc' and demog == 'gender':
                tph_condition = tph_condition[::-1]
                labels = ['Female', 'Male', 'Other']
                #categories = pd.Series([categories[2], categories[1], categories[0]], name='gender')
                cat_dict = {0: 2, 1: 1, 2: 0}
            if city == 'madrid' and demog == 'user_type':
                labels = ['Occasional User', 'Annual user', 'Company worker']
                tph_condition = np.concatenate(
                    (np.expand_dims(tph_condition[1], axis=0),
                     np.expand_dims(tph_condition[0], axis=0),
                     np.expand_dims(tph_condition[2], axis=0),))
                cat_dict = {1: 1, 2: 0, 3: 2}
            if city == 'mexico' and demog == 'gender':
                labels = ['Female', 'Male']
            table_dict[city][demog] = np.sum(np.sum(tph_condition, axis=1), axis=1)
            for i in range(n_cats):
                print(f'{name_dict[city]:15} {table_dict[city][demog][i]:10} {labels[i]:10}')

            plot_trips_pr_hour_avg(data, tph_condition, labels,
                                   cityname=name_dict[city], name=dname_dict[demog],
                                   lineplot=True, savefig=savefig)
            if demog in ['user_type', 'gender']:
                duration_demog_kde(data, demog, cat_dict, categories, labels, n_cats)


# %%
kernel_weights = {'chic': 0.005,
                  'london': 0.01,
                  'madrid': 0.0025,
                  'mexico': 0.005, 
                  'nyc': 0.0025,
                  'sfran': 0.015,
                  'taipei': 0.003,
                  'washDC': 0.01,
                  }
y_wdays = dict()
y_wends = dict()
y_tot = dict()
for city in cities:
    data = bs.Data(city, year, month)
    xmax = 50
    N = 300
    
    weekdays_list = np.where(np.array(data.weekdays) < 5)[0] + 1
    weekend_list = np.where(np.array(data.weekdays) >= 5)[0] + 1
    
    df = data.df
    
    wdays = df.loc[df['start_dt'].dt.day.isin(weekdays_list)]
    wend  = df.loc[df['start_dt'].dt.day.isin(weekend_list)]
    
    ker_wday = stats.gaussian_kde(wdays['duration']/60, kernel_weights[city])
    ker_wend = stats.gaussian_kde(wend['duration']/60, kernel_weights[city])
    x = np.linspace(0, xmax, N)
    
    y_wdays[city] = ker_wday(x)
    y_wends[city] = ker_wend(x)
    
    ker_tot = stats.gaussian_kde(df['duration']/60, kernel_weights[city])
    y_tot[city] = ker_tot(x)

# %%

plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))

fig.subplots_adjust(hspace=0.1, wspace=0.05)

for city in cities:
    ax[0].plot(x, y_wdays[city], label=name_dict[city])
    ax[1].plot(x, y_wends[city], label=name_dict[city])

ax[0].legend()
ax[0].set_ylabel("density")
ax[0].set_xlabel("Trip duration (min)")

ax[1].legend()
#ax[1].ylabel("density")
ax[1].set_xlabel("Trip duration (min)")

ax[0].set_title("Weekdays")
ax[1].set_title("Weekend")

fig.suptitle(f'Trip Duration {months[month]} {year:d}')

plt.savefig("figures/duration_kde_all_wday_wend.pdf",
                bbox_inches='tight', pad_inches=0)
plt.show()

plt.style.use('seaborn-darkgrid')
fig = plt.figure(figsize=(8, 4))

for city in cities:
    plt.plot(x, y_tot[city], label=name_dict[city])

plt.legend()
plt.ylabel("density")
plt.xlabel("Trip duration (min)")


plt.suptitle(f'Trip Duration {months[month]} {year:d}')

plt.savefig("figures/duration_kde_all_tot.pdf",
                bbox_inches='tight', pad_inches=0)
plt.show()

# %%

city = "nyc"
year = 2019
month = 9

citydata = bs.Data(city, year, month)

savefig = True


# %% Age groups
# =============================================================================
# Age groups in NYC bigger figures
# =============================================================================
age_groups = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                       100, 110, 120, 130, 140, 150, 160])

year_groups = -1*age_groups[1:] + year

year_age = dict()

for age in range(0, 150):
    birth_year = year - age
    year_age[birth_year] = np.min(np.where(birth_year > year_groups)[0])

age_group = lambda df: df['birth_year'].map(year_age)

trips_pr_hour_age = get_trips_pr_hour(citydata, age_group, n_categories=16)

labels_age = [f"Age: {age_groups[i-1]} to {age_groups[i]}" for i in range(1, 17)]
labels_age[7] = 'Age: 70+'

plot_trips_pr_hour_month(citydata, trips_pr_hour_age, labels_age)

trips_pr_hour_age_avg = plot_trips_pr_hour_avg(citydata, trips_pr_hour_age, labels_age)

plot_trips_pr_hour_avg(citydata, trips_pr_hour_age, labels_age, normalize=True)

for i in [1, 2, 3, 4, 5, 6, 7]:
    plt.plot(range(24), trips_pr_hour_age_avg[i, 0, :], '.-', label=labels_age[i])
    plt.xticks([0, 3, 6, 9, 12, 15, 18, 21])
    plt.xlabel("Hour")
    plt.ylabel("# rides")
    plt.title(f"Weekday rides per hour by age group in {city.upper()} {months[month]} {year:d}")

plt.legend()

if savefig:
    plt.savefig(f"./figures/weekday_rides_by_age-{city}{year:d}{month:02d}.pdf",
                bbox_inches='tight', pad_inches=0)

plt.show()

print("Average number of daily trips by age group weekday")
print("Age        Avg # trips")
for i in range(16):
    print("{:10} {:10.2f}".format(labels_age[i][5:], trips_pr_hour_age_avg[i, 0].sum()))

for i in [1, 2, 3, 4, 5, 6, 7]:
    plt.plot(range(24), trips_pr_hour_age_avg[i, 1, :], '.-', label=labels_age[i])
    plt.xticks([0, 3, 6, 9, 12, 15, 18, 21])
    plt.xlabel("Hour")
    plt.ylabel("# rides")
    plt.title(f"Weekend rides per hour by age group in {city.upper()} {months[month]} {year:d}")

plt.legend()

if savefig:
    plt.savefig(f"./figures/weekend_rides_by_age-{city}{year:d}{month:02d}.pdf",
                bbox_inches='tight', pad_inches=0)
plt.show()

print("Average number of daily trips by age group weekend")
print("Age        Avg # trips")
for i in range(16):
    print("{:10} {:10.2f}".format(labels_age[i][5:], trips_pr_hour_age_avg[i, 1].sum()))

# =============================================================================
# Gender to year comparison
# =============================================================================
# %%
gender_age = np.zeros((150, 3), int)
print("Calculating trips per hour. Grab a snack, this can take a while...")
age = citydata.df['birth_year'] - 1870
gender = citydata.df['gender']

for a, g in zip(age, gender):
    gender_age[a, g] += 1

year_ga = dict()

for yearr in range(1880, 2019):
    year_ga[yearr] = gender_age[yearr-1870]

print(f"Year Age {'Other':>7} {'Male':>7} {'Female':>7}")
for yearr in reversed(range(1880, 2019)):
    print(f"{yearr} {2019-yearr:3d} {year_ga[yearr][0]:7d} {year_ga[yearr][1]:7d} {year_ga[yearr][2]:7d}")


plt.plot(np.arange(15, 80), np.flip(gender_age)[15:80], '.-')
plt.xticks(np.arange(15, 85, 5))
plt.xlabel("Age")
plt.ylabel("# rides")
plt.legend(["Female", "Male", "Other"])

if savefig:
    plt.savefig(f"./figures/trips_gender_age-{city}{year:d}{month:02d}.pdf")

# %%

subscriber_condition = lambda df: (df['user_type'] == "Subscriber").astype(int)

trips_pr_hour_sub = get_trips_pr_hour(citydata, subscriber_condition, n_categories=2)

labels_sub = ["Customer", "Subscriber"]

tph_sub = plot_trips_pr_hour_avg(citydata, trips_pr_hour_sub, labels_sub,
                                 savefig=savefig, name="sub")

print(f"Average number of weekday Customer trips: {tph_sub[0,0].sum():.0f}")
print(f"Average number of weekday Subscriber trips: {tph_sub[1,0].sum():.0f}")

print(f"Average number of weekend Customer trips: {tph_sub[0,1].sum():.0f}")
print(f"Average number of weekend Subscriber trips: {tph_sub[1,1].sum():.0f}")
