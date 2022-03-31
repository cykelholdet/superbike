# Analysis of Bike-Sharing Schemes

Scripts for the analysis of bike-sharing data. These scipts have been developed for the purpose of a masters project in Mathematical Engineering at Aalborg University:

D. B. van Diepen and N. A. Weinreich, Aalborg University, Denmark, 2022


## Dependencies
This project is created with `Python 3.9`

Dependencies:
```
matplotlib
numpy
pandas
scipy
sklearn
```

### Conda environment
If working with Conda, you could for example make a conda environment as follows. (You can ignore spyder if you wish to use another development environment)

```
conda create -n bike_env numpy matplotlib pandas scipy holoviews hvplot geoviews geopandas spyder

conda activate bike_env
```

Then install the remaining packages
```
pip install scikit-learn-extra scikit-image statsmodels workalendar geopy openpyxl smopy rarfile
pip install --upgrade shapely
```

#### Conda-forge
Alternatively, if you want to get more recent versions of packages, including Python 3.10, you can get packages from conda-forge. As of March 2022, cartopy and scikit-learn-extra are not available from PyPI as wheels or from the regular conda repository for Python 3.10, but we can get them from conda-forge.

```
conda create -n bike_env
conda activate bike_env

conda install -c conda-forge cartopy scikit-learn-extra numba llvmlite numpy matplotlib pandas scipy holoviews hvplot geoviews geopandas scikit-image statsmodels workalendar geopy openpyxl

pip install smopy rarfile
```

There are some issues with the IPython debugger and with Spyder in 3.10. If you wish to use Python 3.9 instead, you can instead do.

```
conda create -n bike_env
conda activate bike_env

conda install -c conda-forge python=3.9 cartopy scikit-learn-extra numba llvmlite numpy matplotlib pandas scipy holoviews hvplot geoviews geopandas scikit-image statsmodels workalendar geopy openpyxl

pip install smopy rarfile
```
If you have `panel=0.12.6`, you may need to downgrade `jinja2` if you wish to use the interactive plot features.
```
conda install -c conda-forge jinja2=3.0
```


## Directory structure

The data should be organised as follows. Please create directories `data`, `python_variables`, `python_variables/big_data`, and `figures` as necessary.

```
./data
├── (Put data .csv/.json files here)
|
├── boston
│   └── 201909-bluebikes-tripdata.csv
|
├── chicago
│   └── Divvy_Trips_2019_Q3.csv
|
├── london
|   ├── 177JourneyDataExtract28Aug2019-03Sep2019.csv
|   ├── 178JourneyDataExtract04Sep2019-10Sep2019.csv
|   ├── 179JourneyDataExtract11Sep2019-17Sep2019.csv
|   ├── 180JourneyDataExtract18Sep2019-24Sep2019.csv
|   ├── 181JourneyDataExtract25Sep2019-01Oct2019.csv
|   └── london_stations.csv
|
├── madrid
|   ├── 201908_movements.json
|   ├── 201909_movements.json
|   └──  201909_stations_madrid.json
|
├── nyc
│   └── 201909-citibike-tripdata.csv
|
└── washdc
    ├── 201909-capitalbikeshare-tripdata.csv
    └── Capital_Bike_Share_Locations.csv


./python_variables
└── (Pickle files will be here)


./figures
└── (Figures will be here)
```

## Scripts:

`bikeshare.py`
	- Module script containing various functions used in the other scripts. Includes importing data, computing adjacency matrices etc.


# Data Sources
Trip data can be accessed at the following locations

| City             | System name         | # stations 2019 | # trips 2019 | link                                                                                 | Comment                                                       |
|------------------|---------------------|-------------------:|----------------:|--------------------------------------------------------------------------------------|---------------------------------------------------------------|
| Bergen           | Bergen Bysykkel     |                 90 |         898.276 | https://bergenbysykkel.no/en/open-data/historical                                    |                                                               |
| Boston           | Bluebikes           |                341 |       2.522.771 | https://www.bluebikes.com/system-data                                                |                                                               |
| Buenos Aires     | EcoBici             |                417 |       5.238.643 | https://data.buenosaires.gob.ar/dataset/bicicletas-publicas                          |                                                               |
| Chicago          | Divvy               |                593 |       3.614.078 | https://www.divvybikes.com/system-data                                               |                                                               |
| Edinburgh        | Just Eat Cycles     |                163 |         123.684 | https://edinburghcyclehire.com/open-data/historical                                  | Discontinued September 2021                                   |
| Guadalajara      | MiBici              |                275 |       4.625.130 | https://www.mibici.net/en/open-data/                                                 |                                                               |
| Helsinki         | Helsinki City Bikes |                348 |       3.784.877 | https://hri.fi/data/en_GB/dataset/helsingin-ja-espoon-kaupunkipyorilla-ajatut-matkat | Encompasses Helsinki & Espoo. Only open from April to October |
| London           | Santander Cycles    |                753 |       8.829.104 | https://cycling.data.tfl.gov.uk/                                                     |                                                               |
| Los Angeles      | Metro Bike Share    |                    |                 | https://bikeshare.metro.net/about/data/                                              |                                                               |
| Madrid           | BiciMad             |                214 |       3.956.099 | https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)                    |                                                               |
| Mexico City      | EcoBici             |                480 |       8.349.075 | https://www.ecobici.cdmx.gob.mx/en/informacion-del-servicio/open-data                |                                                               |
| Minneapolis      | Nice Ride           |                179 |         263.169 | https://www.niceridemn.com/system-data                                               | Only open from April to November                              |
| Montreal         | Bixi                |                619 |       5.442.288 | https://bixi.com/en/open-data                                                        | Only open from April to October                               |
| New York City    | Citi Bike           |                938 |      20.551.396 | https://www.citibikenyc.com/system-data                                              |                                                               |
| Oslo             | Oslo Bysykkel       |                254 |       2.237.092 | https://oslobysykkel.no/en/open-data/historical                                      | 2019 missing data from January to March                       |
| San Francisco    | Bay Wheels          |                351 |       2.296.199 | https://www.lyft.com/bikes/bay-wheels/system-data                                    | Split in three main parts by the San Francisco Bay             |
| Taipei           | YouBike             |                399 |      26.484.903 | https://drive.google.com/drive/folders/1QsROgp8AcER6qkTJDxpuV8Mt1Dy6lGQO             | Transitioning to partly dockless YouBike 2.0 since 2020       |
| Trondheim        | Trondheim Bysykkel  |                 56 |         356.189 | https://trondheimbysykkel.no/en/open-data/historical                                 | Only open from April to November                              |
| Washington, D.C. | Capital Bikeshare   |                429 |       3.281.231 | https://www.capitalbikeshare.com/system-data                                         |                                                               |

