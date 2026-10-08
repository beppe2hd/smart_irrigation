import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests


class WeatherRequest:

    def __init__(self, start_date, end_date, lat, lon):

        self.start_date = start_date
        self.end_date = end_date
        self.lat = lat
        self.lon = lon
        # self.features = features #["temperature_2m", "relative_humidity_2m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm", "soil_temperature_28_to_100cm", "wind_speed_10m", "cloud_cover", "rain", "wind_direction_10m", "precipitation", "et0_fao_evapotranspiration"]
        # self.features_forecast = features_forecast #["temperature_2m", "relative_humidity_2m", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm", "rain"]

    def make_request(self, url, params):
        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }

        print("done")

        for i, feature in enumerate(params["hourly"]):
            hourly_data[feature] = hourly.Variables(i).ValuesAsNumpy()

        hourly_dataframe = pd.DataFrame(data=hourly_data)

        return hourly_dataframe

    def meteo_request_histor(self, features):

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.lat,  # 34.97,
            "longitude": self.lon,  # 2.4,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "hourly": features,
        }
        hourly_dataframe = self.make_request(url, params)
        return hourly_dataframe

        # print("\nHourly data\n", hourly_dataframe)

    def meteo_request_forecast(self, features):

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.lat,  # 34.97,
            "longitude": self.lon,  # 2.4,
            "hourly": features,
        }

        hourly_dataframe = self.make_request(url, params)
        return hourly_dataframe

        # print("\nHourly data\n", hourly_dataframe)

    def extractsubset(self, h, index):
        h["date"] = h["date"].dt.tz_localize(None)
        h.set_index("date", inplace=True)
        h = h.loc[h.index.intersection(index)]

        return h
