"""
Created on Thu Feb 11 14:55:16 2021.

@author: Mattek Group 3

Standard key:

    duration
    start_t
    end_t
    start_stat_id
    start_stat_name
    start_stat_lat
    start_stat_long
    end_stat_id
    end_stat_name
    end_stat_lat
    end_stat_long

"""


def get_key(city):
    """
    Create a dictionary used when converting DataFrame coloumn names to the
    standard key.

    Parameters
    ----------
    city : str
        The identification of the city. For a list of supported cities, see
        the documentation for the Data class.

    Returns
    -------
    key : dict
        Dictionary used for renaming coloumn labels.

    """
    if city in ['nyc', 'boston']:
        key = {'tripduration': 'duration',
               'starttime': 'start_t',
               'stoptime': 'end_t',
               'start station id': 'start_stat_id',
               'start station name': 'start_stat_name',
               'start station latitude': 'start_stat_lat',
               'start station longitude': 'start_stat_long',
               'end station id': 'end_stat_id',
               'end station name': 'end_stat_name',
               'end station latitude': 'end_stat_lat',
               'end station longitude': 'end_stat_long',
               'bikeid': 'bike_id',
               'usertype': 'user_type',
               'birth year': 'birth_year',
               'gender': 'gender'}

    elif city == "la":
        key = {'start_time': 'start_t',
               'end_time': 'end_t',
               'start_station': 'start_stat_id',
               'start_lat': 'start_stat_lat',
               'start_lon': 'start_stat_long',
               'end_station': 'end_stat_id',
               'end_lat': 'end_stat_lat',
               'end_lon': 'end_stat_long',
               'passholder_type': 'user_type'}

    elif city == "washdc":
        key = {'Duration': 'duration',
               'Start date': 'start_t',
               'End date': 'end_t',
               'Start station number': 'start_stat_id',
               'Start station': 'start_stat_name',
               'End station number': 'end_stat_id',
               'End station': 'end_stat_name',
               'Bike number': 'bike_id',
               'Member type': 'user_type'}

    elif city == "chicago":
        key = {'tripduration': 'duration',
               'start_time': 'start_t',
               'end_time': 'end_t',
               'from_station_id': 'start_stat_id',
               'from_station_name': 'start_stat_name',
               'to_station_id': 'end_stat_id',
               'to_station_name': 'end_stat_name',
               'bikeid': 'bike_id',
               'usertype': 'user_type',
               'birthyear': 'birth_year',
               'gender': 'gender',
               '01 - Rental Details Rental ID': 'trip_id',
               '01 - Rental Details Local Start Time': 'start_t',
               '01 - Rental Details Local End Time': 'end_t',
               '01 - Rental Details Bike ID': 'bike_id',
               '01 - Rental Details Duration In Seconds Uncapped': 'duration',
               '03 - Rental Start Station ID': 'start_stat_id',
               '03 - Rental Start Station Name': 'start_stat_name',
               '02 - Rental End Station ID': 'end_stat_id',
               '02 - Rental End Station Name': 'end_stat_name',
               'User Type': 'user_type',
               'Member Gender': 'gender',
               '05 - Member Details Member Birthday Year': 'birth_year'}
    
    elif city == "minneapolis":
        key = {'ride_id' : 'ride_id',
               'rideable_type' : 'bike_type',
               'started_at' : 'start_t',
               'ended_at' : 'end_t',
               'start_station_name' : 'start_stat_name',
               'start_station_id' : 'start_stat_id',
               'end_station_name' : 'end_stat_name',
               'end_station_id' : 'end_stat_id',
               'start_lat' : 'start_stat_lat',
               'start_lng' : 'start_stat_long',
               'end_lat' : 'end_stat_lat',
               'end_lng' : 'end_stat_long',
               'member_casual' : 'user_type',
               'tripduration': 'duration',
               'start_time': 'start_t',
               'end_time': 'end_t',
               'start station id': 'start_stat_id',
               'start station name': 'start_stat_name',
               'start station latitude': 'start_stat_lat',
               'start station longitude': 'start_stat_long',
               'end station id': 'end_stat_id',
               'end station name': 'end_stat_name',
               'end station latitude': 'end_stat_lat',
               'end station longitude': 'end_stat_long',
               'bikeid': 'bike_id',
               'usertype': 'user_type',
               'birth year': 'birth_year',
               'gender': 'gender',
               'bike type' : 'bike_type'}
    
    elif city in ['sfran', 'sjose']:
        key = {'duration_sec': 'duration',
               'start_time': 'start_t',
               'end_time': 'end_t',
               'start_station_id': 'start_stat_id',
               'start_station_name': 'start_stat_name',
               'start_station_latitude': 'start_stat_lat',
               'start_station_longitude': 'start_stat_long',
               'end_station_id': 'end_stat_id',
               'end_station_name': 'end_stat_name',
               'end_station_latitude': 'end_stat_lat',
               'end_station_longitude': 'end_stat_long',
               'bike_id': 'bike_id',
               'user_type': 'user_type',
               'bike_share_for_all_trip': 'bike_share_for_all_trip'}

    elif city == "london":
        key = {'Rental Id': 'trip_id',
               'Duration': 'duration',
               'Start Date': 'start_t',
               'End Date': 'end_t',
               'StartStation Id': 'start_stat_id',
               'StartStation Name': 'start_stat_name',
               'EndStation Id': 'end_stat_id',
               'EndStation Name': 'end_stat_name',
               'Bike Id': 'bike_id'}

    elif city in ['oslo', 'bergen', 'trondheim', 'edinburgh']:
        key = {# 2019 and later
               'duration': 'duration',
               'started_at': 'start_t',
               'ended_at': 'end_t',
               'start_station_id': 'start_stat_id',
               'start_station_name': 'start_stat_name',
               'start_station_latitude': 'start_stat_lat',
               'start_station_longitude': 'start_stat_long',
               'end_station_id': 'end_stat_id',
               'end_station_name': 'end_stat_name',
               'end_station_latitude': 'end_stat_lat',
               'end_station_longitude': 'end_stat_long',
               # before 2019
               'Start station': 'start_stat_id',
               'Start time': 'start_t',
               'End station': 'end_stat_id',
               'End time': 'end_t',
               }
    
    elif city == 'helsinki':
        key = {'Duration (sec.)': 'duration',
               'Departure': 'start_t',
               'Return': 'end_t',
               'Departure station id': 'start_stat_id',
               'Departure station name': 'start_stat_name',
               'Return station id': 'end_stat_id',
               'Return station name': 'end_stat_name',
               'Covered distance (m)': 'distance'}

    elif city == "buenos_aires":
        key = {'periodo': 'year',
               'id_usuario': 'user_id',
               'genero_usuario': 'gender',
               'fecha_origen_recorrido': 'start_t',
               'id_estacion_origen': 'start_stat_id',
               'nombre_estacion_origen': 'start_stat_name',
               'long_estacion_origen': 'start_stat_long',
               'lat_estacion_origen': 'start_stat_lat',
               'domicilio_estacion_origen': 'start_stat_desc',
               'duracion_recorrido': 'duration',
               'fecha_destino_recorrido': 'end_t',
               'id_estacion_destino': 'end_stat_id',
               'nombre_estacion_destino': 'end_stat_name',
               'long_estacion_destino': 'end_stat_long',
               'lat_estacion_destino': 'end_stat_lat',
               'domicilio_estacion_destino': 'end_stat_desc'}

    elif city == "madrid":
        key = {'_id': '_id',
               'user_day_code': 'user_day_code',
               'idplug_base': 'end_base_id',
               'idunplug_base': 'start_base_id',
               'user_type': 'user_type',
               'travel_time': 'duration',
               'idplug_station': 'end_stat_id',
               'idunplug_station': 'start_stat_id',
               'age_range': 'age_range',
               'unplug_hourTime': 'start_t',
               'zip_code': 'zip_code'}

    elif city == "mexico":
        key = {'Genero_Usuario': 'gender',
               'Edad_Usuario': 'age',
               'Bici': 'bike_id',
               'Ciclo_Estacion_Retiro': 'start_stat_id',
               'Fecha_Retiro': 'start_date',
               'Hora_Retiro': 'start_time',
               'Ciclo_Estacion_Arribo': 'end_stat_id',
               'Fecha_Arribo': 'end_date',
               'Hora_Arribo': 'end_time'}
        
    elif city == "guadalajara":
        key = {'Genero': 'gender',
               'Año_de_nacimiento': 'birth_year',
               'Viaje_Id': 'bike_id',
               'Usuario_Id': 'user_id',
               'Origen_Id': 'start_stat_id',
               'Inicio_del_viaje': 'start_t',
               'Destino_Id': 'end_stat_id',
               'Fin_del_viaje': 'end_t'}
    
    elif city == "montreal":
        key = {'is_member': 'user_type',
               'start_station_code': 'start_stat_id',
               'start_date': 'start_t',
               'end_station_code': 'end_stat_id',
               'end_date': 'end_t',
               'duration_sec': 'duration'}
    
    elif city == "la":
        key = {"trip_id": "trip_id",
               'duration': 'duration',
               'start_time': 'start_t',
               'end_time': 'end_t',
               'start_station': 'start_stat_id',
               'start_lat': 'start_stat_lat',
               'start_lon': 'start_stat_long',
               'end_station': 'end_stat_id',
               'end_lat': 'end_stat_lat',
               'end_lon': 'end_stat_long',
               'bike_id': 'bike_id',
               'plan_duration': 'plan_duration',
               'passholder_type': 'passholder_type',
               'bike_type': 'bike_type'}
        
    return key


def get_land_use_key(city):
    if city == 'nyc':
        key = {'ZONEDIST': 'zone_type',
               'geometry': 'geometry',}
    elif city == 'chicago':
        key = {'zone_class': 'zone_type',
               'geometry': 'geometry',}
    # elif city == 'boston':
        # key = {'ZONE_': 'zone_type',
        #        'geometry': 'geometry',}
    elif city == 'minneapolis':
        key = {'ZONE_CODE': 'zone_type',
               'geometry': 'geometry'}
    elif city == 'washdc':
        key = {'ZONING_LABEL': 'zone_type',
               'geometry': 'geometry',}
    elif city in ['helsinki', 'madrid', 'oslo', 'london', 'bergen', 'trondheim', 'edinburgh']:
        key = {'code_2018': 'zone_type',
               'geometry': 'geometry',}
    else:
        key = {}
    return key


def get_census_key(city):
        
    if city in ['nyc', 'chicago', 'boston', 'minneapolis', 'washdc']:
        key = {'GEO_ID': 'census_geo_id',
                'P1_001N': 'population',}
    
    elif city in ['helsinki', 'madrid', 'oslo', 'london']:
        key = {'code_2018': 'zone_type',
               'geometry': 'geometry',}
    else:
        key = {}
    return key













