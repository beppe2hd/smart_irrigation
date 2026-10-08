import yaml
from datetime import datetime, timedelta
import pandas as pd


def get_config_file(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_start_end_date(delta, direction):
    now = datetime.now()
    date_now = pd.to_datetime(now)
    date_now_round = date_now.round("h")
    # print(str(date_now_round))
    if direction == "past":
        end_dt = str(date_now_round)
        start_dt = str(date_now_round - timedelta(hours=delta - 1))
    if direction == "future":
        start_dt = str(date_now_round + timedelta(hours=1))
        end_dt = str(date_now_round + timedelta(hours=delta))

    return (start_dt, end_dt)


def reset_df_start_end_hours(h, start_dt, end_dt):

    index = pd.date_range(start_dt, end_dt, freq="h")

    h["date"] = h["date"].dt.tz_localize(None)
    h.set_index("date", inplace=True)
    h = h.loc[h.index.intersection(index)]
    h = h.fillna(0)

    return h
