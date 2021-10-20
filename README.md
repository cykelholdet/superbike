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
conda create -n bike_env numpy matplotlib pandas scipy holoviews hvplot geopandas spyder

conda activate bike_env
```

Then install the remaining packages
```
pip install scikit-learn-extra scikit-image
```

## Directory structure

The data should be organised as follows. Please create directories `data`, `python_variables`, `python_variables/big_data`, and `figures` as necessary.

```
./data
├── (Put data .csv/.json files here)
├── Divvy_Trips_2019_Q3.csv
│
├── 177JourneyDataExtract28Aug2019-03Sep2019.csv
├── 178JourneyDataExtract04Sep2019-10Sep2019.csv
├── 179JourneyDataExtract11Sep2019-17Sep2019.csv
├── 180JourneyDataExtract18Sep2019-24Sep2019.csv
├── 181JourneyDataExtract25Sep2019-01Oct2019.csv
├── london_stations.csv
│
├── 201908_movements.json
├── 201909_movements.json
├── 201909_stations_madrid.json
│
├── 2019-09-mexico.csv
├── stations_mexico.json
│
├── 201909-citibike-tripdata.csv
│
├── 201909-baywheels-tripdata.csv
│
├── 201909-taipei.csv
├── stations_taipei.csv
│
├── 201909-capitalbikeshare-tripdata.csv
└── Capital_Bike_Share_Locations.csv


./python_variables
├── big_data
│   └── (Dataframe pickles will be here)
│
└── (Pickle files will be here)


./figures
└── (Figures will be here)
```

## Scripts:

`bikeshare.py`
	- Module script containing various functions used in the other scripts. Includes importing data, computing adjacency matrices etc.


# Data Sources
Trip data can be accessed at the following locations

| City             | Link                                                                                   |
|------------------|----------------------------------------------------------------------------------------|
| Bergen           | https://bergenbysykkel.no/en/open-data/historical                                      |
| Boston           | https://www.bluebikes.com/system-data                                                  |
| Buenos Aires     | https://data.buenosaires.gob.ar/dataset/bicicletas-publicas                            |
| Chicago          | https://www.divvybikes.com/system-data                                                 |
|                  | https://data.cityofchicago.org/Transportation/Divvy-Bicycle-Stations-All-Map/bk89-9dk7 |
| Edinburgh        | https://edinburghcyclehire.com/open-data/historical                                    |
| Helsinki         | https://hri.fi/data/en_GB/dataset/helsingin-ja-espoon-kaupunkipyorilla-ajatut-matkat   |
| London           | https://cycling.data.tfl.gov.uk/                                                       |
| Los Angeles      | https://bikeshare.metro.net/about/data/                                                |
| Madrid           | https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)                      |
| Mexico City      | https://www.ecobici.cdmx.gob.mx/en/informacion-del-servicio/open-data                  |
| Montreal         | https://bixi.com/en/open-data                                                          |
| New York City    | https://www.citibikenyc.com/system-data                                                |
| Oslo             | https://oslobysykkel.no/en/open-data/historical                                        |
| San Francisco    | https://www.lyft.com/bikes/bay-wheels/system-data                                      |
| Taipei           | https://drive.google.com/drive/folders/1QsROgp8AcER6qkTJDxpuV8Mt1Dy6lGQO               |
| Trondheim        | https://trondheimbysykkel.no/en/open-data/historical                                   |
| Washington, D.C. | https://www.capitalbikeshare.com/system-data                                           |

