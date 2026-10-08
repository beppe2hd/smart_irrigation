from fastapi import FastAPI
from contextlib import asynccontextmanager
import sys, os
from dotenv import load_dotenv
import torch
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd

load_dotenv()
sys.path.append(os.getenv("PYTHONPATH"))
print(os.getenv("PYTHONPATH"))
from src.commons.architectures.model_handler import create_model, inference, load_weights_and_scale
from src.inference.api.src.external_resource import (
    retrive_sensor_data,
    retrieve_meteo_data,
    write_irrigation,
    write_lai,
    retriev_field_list,
    retriev_last_irr,
    retriev_last_lai,
    write_sensor
)
from src.commons.utils import (
    get_config_file,
    get_start_end_date,
    reset_df_start_end_hours,
)

class SensorReading(BaseModel):
    id: int
    time: str
    sensor: str
    water_SOIL: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config_file(os.getenv("configFile"))
    model = create_model(config)
    model, scaler, output_scale_index, x_scale_index, x_f_scale_index = load_weights_and_scale(model, config)
    model.eval()

    app.state.model = model
    app.state.config = config
    app.state.scaler = scaler
    app.state.output_scale_index = output_scale_index
    app.state.x_scale_index = x_scale_index
    app.state.x_f_scale_index = x_f_scale_index

    print(f"App running with {config['name']}")


    yield
    del model, scaler, output_scale_index, x_scale_index, x_f_scale_index


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://purecircle.ngrok.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/forecast")
def forecast(field_id: int):
    model = app.state.model
    config = app.state.config
    scaler = app.state.scaler
    output_scale_index = app.state.output_scale_index
    x_scale_index = app.state.x_scale_index
    x_f_scale_index = app.state.x_f_scale_index

    start_dt_historical, end_dt_historical = get_start_end_date(
        config["features"]["input_seq_len"], "past"
    )
    start_dt_forecast, end_dt_forecast = get_start_end_date(
        config["features"]["output_seq_len"], "future"
    )

    fields_feaures = config["features"]["input"]["fiedls"]
    meteo_features_historical = config["features"]["input"]["meteo_historical"]
    meteo_features_forecast = config["features"]["input"]["meteo_forecast"]
    meteo_features_forecast = [item[:-2] for item in meteo_features_forecast]
    lat = config["features"]["geo_coordinate"]["lat"]
    lon = config["features"]["geo_coordinate"]["lon"]

    historical_sensor_data, date_index = retrive_sensor_data(
        host=os.getenv("host"),
        user=os.getenv("user"),
        password=os.getenv("password"),
        database=os.getenv("database"),
        start_dt=start_dt_historical,
        end_dt=end_dt_historical,
        field_id=field_id,
    )
    
    if len(historical_sensor_data) == 0:
        return {}

    meteo_data_historical = retrieve_meteo_data(
        mode="historical",
        features=meteo_features_historical,
        lat=lat,
        lon=lon,
        start_dt=start_dt_historical.split(" ")[
            0
        ],  # meteo service do not need seconds information
        end_dt=end_dt_historical.split(" ")[0],
    )
    meteo_data_forecast = retrieve_meteo_data(
        mode="forecast", features=meteo_features_forecast, lat=lat, lon=lon
    )

    meteo_data_historical = reset_df_start_end_hours(
        meteo_data_historical, start_dt_historical, end_dt_historical
    )
    meteo_data_forecast = reset_df_start_end_hours(
        meteo_data_forecast, start_dt_forecast, end_dt_forecast
    )

    x = []
    x_f = []

    for i in range(0, config["features"]["input_seq_len"]):
        r = historical_sensor_data[i]
        a = [r[item] for item in fields_feaures]
        hmd = meteo_data_historical.iloc[i][meteo_features_historical].to_list()
        a.extend(hmd)
        x.append(a)

    for i in range(0, config["features"]["output_seq_len"]):
        fmd = meteo_data_forecast.iloc[i][meteo_features_forecast].to_list()
        x_f.append(fmd)
    # Convert to NumPy array for convenience
    x = torch.tensor(x)
    x_f = torch.tensor(x_f)

    print("-"*50)
    print("-"*50)
    print(f"Field id = {field_id}")
    print(x)
    print(x_f)
    print("-"*50)
    print("-"*50)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    x = x.to(device)
    x_f = x_f.to(device)
    
    #y = inference(model, x, x_f, output_scale_index, x_scale_index, x_f_scale_index, scaler, config["delta_mode"])
    y = inference(model.to(device), x, x_f, output_scale_index, x_scale_index, x_f_scale_index, scaler, config["delta_mode"], device)


    date_index = pd.date_range(start=start_dt_forecast, end=end_dt_forecast, freq="h")
    date_index = [d.strftime('%m-%d %H:00') for d in date_index.to_list()]
    
    if len(y.shape)>1:
        print(f"{type(y)}, {y.shape})--------*****7&&&&&&&")
        list1 = list(list(zip(*y.tolist()))[0])
        list2 = list(list(zip(*y.tolist()))[0])
    else:
        #y = np.expand_dims(y, axis=1)
        print(f"{type(y)}, {y.shape})--------*****7&&&&&&&")
        list1 = y.tolist()
        list2 = list1

    irrigation = 0
    idx = 0
    suggested_volume = 0
    suggested_time = 0

    dur_a = config["irr_param"]["durartio"]["a"]
    dur_b = config["irr_param"]["durartio"]["b"]
    vol_a = config["irr_param"]["volume"]["a"]
    vol_b = config["irr_param"]["volume"]["b"]

    if np.array(list1).mean()<config["irr_param"]["threshold"]:
        irrigation = 1
        list1 = np.array(list1)
        mask = list1 < config["irr_param"]["threshold"]
        if mask.any():
            idx = np.argmax(mask)
            val = list1[idx]
  
        suggested_volume = val*vol_a+vol_b
        suggested_time = val*dur_a+dur_b
        suggested_volume = float(suggested_volume)
        suggested_time = float(suggested_time)
        idx = int(idx+1)
        list1 = list1.tolist()

    print("[list1- values]", list1)
    print("[list2- values]", list2)
    print("[list1- max]", max(list1))
    print("[list2- max]", max(list2))


    return {
        "list1": list1,
        "list2": list2,
        "data_index": date_index,
        "irrigation": irrigation,
        "volume": suggested_volume,
        "duration": suggested_time,
        "time": idx
    }


@app.post("/add_irr")
def forecast(field_id: int, date: str, water_volume: float):

    write_irrigation(
        host=os.getenv("host"),
        user=os.getenv("user"),
        password=os.getenv("password"),
        database=os.getenv("database"),
        date=date,
        water_volume=water_volume,
        field_id=field_id,
    )

    return {"output": "ok"}


@app.post("/add_lai")
def forecast(field_id: int, date: str, lai: float):

    write_lai(
        host=os.getenv("host"),
        user=os.getenv("user"),
        password=os.getenv("password"),
        database=os.getenv("database"),
        date=date,
        lai=lai,
        field_id=field_id,
    )

    return {"output": "ok"}


@app.get("/field_list")
def get_field():
    field_listret = retriev_field_list(
        host=os.getenv("host"),
        user=os.getenv("user"),
        password=os.getenv("password"),
        database=os.getenv("database"),
    )
    field_listret = [i[0] for i in field_listret]

    return {"fields": field_listret}

@app.get("/last_irr")
def get_last_irr():
    last_irr = retriev_last_irr(
        host=os.getenv("host"),
        user=os.getenv("user"),
        password=os.getenv("password"),
        database=os.getenv("database"),
    )
    #last_irr = [i[0] for i in last_irr]

    return {"last_irr": last_irr}

@app.get("/last_lai")
def get_last_lai():
    last_lai = retriev_last_lai(
        host=os.getenv("host"),
        user=os.getenv("user"),
        password=os.getenv("password"),
        database=os.getenv("database"),
    )
    #last_lai = [i[0] for i in last_lai]

    return {"last_lai": last_lai}

@app.post("/sensor-data")
def receive_bulk_data(data: List[SensorReading]):
    host=os.getenv("host"),
    user=os.getenv("user"),
    password=os.getenv("password"),
    database=os.getenv("database"),

    # Here you can store to DB, log, etc.
    sens_to_field_map = {'sensor_014':['s_w', 5],
                         'sensor_015':['s_b', 5],
                         'sensor_032':['s_w', 11],
                         'sensor_033':['s_b', 11],
                         'sensor_089':['s_w', 30],
                         'sensor_090':['s_b', 30],
                         'sensor_166':['s_w', 32],
                         'sensor_174':['s_b', 32],
                         'sensor_172':['s_w', 52],
                         'sensor_156':['s_b', 52],
                         'sensor_161':['s_w', 54],
                         'sensor_162':['s_b', 54]
                         }
    
    sensor_id = []
    ts = []
    sm = []
    plot = []
    zone = []

    for i in range(len(data)):
        sensor_id.append(int(data[i].sensor.split('_')[1]))
        ts.append(data[i].time)
        sm.append(data[i].water_SOIL)
        plot.append(sens_to_field_map[data[i].sensor][1])
        zone.append(sens_to_field_map[data[i].sensor][0])
    write_sensor(host=os.getenv("host"), user=os.getenv("user"), password = os.getenv("password"), database = os.getenv("database"), date=ts, plot_id=plot, sensor_zone=zone, water_content=sm)
    print(f"Received {len(data)} records")
    return {"status": "ok", "received": len(data)}

# if __name__ == "__main__":

#     uvicorn.run(
#         "src.inference.api.src.app:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#     )

