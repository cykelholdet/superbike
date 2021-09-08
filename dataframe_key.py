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
    if city == "nyc":
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

    elif city == "washDC":
        key = {'Duration': 'duration',
               'Start date': 'start_t',
               'End date': 'end_t',
               'Start station number': 'start_stat_id',
               'Start station': 'start_stat_name',
               'End station number': 'end_stat_id',
               'End station': 'end_stat_name',
               'Bike number': 'bike_id',
               'Member type': 'user_type'}

    elif city == "chic":
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
               'gender': 'gender'}

    elif city == "sfran":
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

    elif city == "sjose":
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
        key = {'duration': 'duration',
               'started_at': 'start_t',
               'ended_at': 'end_t',
               'start_station_id': 'start_stat_id',
               'start_station_name': 'start_stat_name',
               'start_station_latitude': 'start_stat_lat',
               'start_station_longitude': 'start_stat_long',
               'end_station_id': 'end_stat_id',
               'end_station_name': 'end_stat_name',
               'end_station_latitude': 'end_stat_lat',
               'end_station_longitude': 'end_stat_long'}

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

    return key
