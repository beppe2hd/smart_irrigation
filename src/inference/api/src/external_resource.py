import mysql.connector
import openmeteo_requests
from retry_requests import retry
import requests_cache
import pandas as pd
import numpy as np


def retrieve_meteo_data(mode, features, lat, lon, start_dt="", end_dt=""):
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    if mode == "historical":
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,  # 52.52,
            "longitude": lon,  # 13.41,
            "start_date": start_dt,
            "end_date": end_dt,
            "hourly": features,
        }

    if mode == "forecast":
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,  # 52.52,
            "longitude": lon,  # 13.41,
            "hourly": features,
            # "forecast_days": 3,
        }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }

    for i, p in enumerate(features):
        hourly_data[p] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data=hourly_data)





def retrive_sensor_data(host, user, password, database, start_dt, end_dt, field_id):
    
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )

    cursor = conn.cursor()

    full_index = pd.date_range(start=start_dt, end=end_dt, freq="h")

    query = f"""
    SELECT ts, lai
    FROM lai
    WHERE ts < CURDATE() AND field_id = {field_id}
    ORDER BY ts ASC;
    """

    cursor.execute(query)
    lai = cursor.fetchall()

    if len(lai)==0:
        return [], []

    df = pd.DataFrame(lai, columns=['ts','lai'])
    full_index_lai = pd.date_range(start=df['ts'][0], end=end_dt, freq="h")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    df = df.reindex(full_index_lai)
    df = df.interpolate()
    filtered_df = df.loc[start_dt: end_dt]
    lai = filtered_df['lai'].to_list()

    zones = {'s_b', 's_w'}

    outSensor = {}
    for zone in zones:
        query = f"""
        SELECT ts, water_content
        FROM soil_moisture
        WHERE sensor_zone = '{zone}' and ts BETWEEN %s AND %s AND field_id = {field_id};
        """

        cursor.execute(query, (start_dt, end_dt))
        rows = cursor.fetchall()
        print(rows)

        if len(rows)==0:
            return [], []

        df = pd.DataFrame(rows, columns=['ts','water_content'])
        df["ts"] = pd.to_datetime(df["ts"])
        df["ts"] = df["ts"].dt.round('h')
        df = df.groupby('ts').mean()
        full_index = pd.date_range(start=start_dt, end=end_dt, freq="h")
        df = df.reindex(full_index)
        df = df.interpolate()
        df.fillna(20.0, inplace=True)

        # df = pd.DataFrame(rows, columns=['ts','water_content'])
        # df["ts"] = pd.to_datetime(df["ts"])
        # df["ts"] = df["ts"].dt.round('h')
        # df = df.groupby('ts').mean()
     
        outSensor[zone] = df['water_content']
    

    query = f"""
    SELECT ts, water_volume
    FROM irrigation
    WHERE ts BETWEEN %s AND %s AND field_id = {field_id};
    """

    cursor.execute(query, (start_dt, end_dt))
    rows = cursor.fetchall()

    if len(rows)==0:
        irr = len(full_index) * [0.0]
    else:
        df = pd.DataFrame(rows, columns=['ts', 'water_volume'])
        df["ts"] = pd.to_datetime(df["ts"])
        full_index = pd.date_range(start=start_dt, end=end_dt, freq="h")
        df = df.set_index('ts')
        df = df.reindex(full_index)
        df.fillna(0.0, inplace=True)
        irr = df['water_volume'].to_list()

    elements = []
    for i in range(len(irr)):
        elements.append({'s_b': np.float32(outSensor['s_b'].iloc[i]), 's_w': np.float32(outSensor['s_w'].iloc[i]), 'irr': irr[i], 'datetime': str(full_index.to_list()[i]), 'LAI': lai[i]})

    return elements, full_index

def write_irrigation(host, user, password, database, date, water_volume, field_id):
    
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    cursor = conn.cursor(dictionary=True)

    query = f"""
    INSERT IGNORE INTO irrigation (ts, water_volume, field_id)
    VALUES (%s, %s, %s);
    """

    cursor.execute(query, (date, water_volume, field_id))
    conn.commit()
    print(f"Inserted 1 row. ID: {cursor.lastrowid}")

    cursor.close()
    conn.close()

def write_lai(host, user, password, database, date, lai, field_id):
    
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    cursor = conn.cursor(dictionary=True)

    query = f"""
    INSERT IGNORE INTO lai (ts, lai, field_id)
    VALUES (%s, %s, %s);
    """

    cursor.execute(query, (date, lai, field_id))
    conn.commit()
    print(f"Inserted 1 row. ID: {cursor.lastrowid}")

    cursor.close()
    conn.close()

def retriev_field_list(host, user, password, database):
    
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )

    cursor = conn.cursor()

    query = f"""
    SELECT id FROM field;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows

def retriev_last_irr(host, user, password, database):
    
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )

    cursor = conn.cursor()

    query = f"""
    SELECT ts, water_volume, field_id
    FROM (
        SELECT
            id,
            ts,
            water_volume,
            field_id,
            ROW_NUMBER() OVER (
                PARTITION BY field_id
                ORDER BY ts DESC
            ) AS rn
        FROM irrigation
    ) t
    WHERE rn <= 2
    ORDER BY field_id, ts DESC;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows

def retriev_last_lai(host, user, password, database):

    
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )

    cursor = conn.cursor()

    query = f"""
    SELECT ts, lai, field_id
    FROM (
        SELECT
            id,
            ts,
            lai,
            field_id,
            ROW_NUMBER() OVER (
                PARTITION BY field_id
                ORDER BY ts DESC
            ) AS rn
        FROM lai
    ) t
    WHERE rn <= 2
    ORDER BY field_id, ts DESC;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows

def write_sensor(host, user, password, database, date, plot_id, sensor_zone, water_content):
    

    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    cursor = conn.cursor()

    query = f"""
    INSERT IGNORE INTO soil_moisture (ts, sensor_zone, water_content, field_id)
    VALUES (%s, %s, %s, %s);
    """

    data = list(zip(date, sensor_zone, water_content, plot_id))

    cursor.executemany(query, data)
    conn.commit()

    print(f"Inserted {len(date)} row. ID: {cursor.lastrowid}")

    cursor.close()
    conn.close()

    


