import numpy as np
import pandas as pd


def generate_synthetic_soil_moisture_csv(
    output_csv="synthetic_soil_moisture_with_lags.csv",
    start="2024-01-01 00:00:00",
    periods=24 * 365,   # 1 year hourly
    freq="h",
    seed=42,
    dropna=True
):
    rng = np.random.default_rng(seed)

    # Time axis
    time_index = pd.date_range(start=start, periods=periods, freq=freq)
    t = np.arange(periods)
    hour = time_index.hour.values
    doy = time_index.dayofyear.values

    # -------------------------
    # Synthetic weather signals
    # -------------------------

    # Temperature [°C]
    temperature_2m = (
        16
        + 9 * np.sin(2 * np.pi * doy / 365.25 - 0.9)
        + 5 * np.sin(2 * np.pi * hour / 24 - np.pi / 2)
        + rng.normal(0, 1.3, periods)
    )

    # Relative humidity [%]
    relative_humidity_2m = (
        70
        - 0.85 * (temperature_2m - np.mean(temperature_2m))
        + 8 * np.sin(2 * np.pi * hour / 24 + 0.8)
        + rng.normal(0, 4.5, periods)
    )
    relative_humidity_2m = np.clip(relative_humidity_2m, 20, 100)

    # Cloud cover [%]
    cloud_cover = (
        42
        + 0.45 * (relative_humidity_2m - 60)
        + 18 * np.sin(2 * np.pi * t / (24 * 6) + 0.4)
        + rng.normal(0, 12.0, periods)
    )
    cloud_cover = np.clip(cloud_cover, 0, 100)

    # Wind speed [m/s]
    wind_speed_10m = (
        3.8
        + 1.1 * np.sin(2 * np.pi * hour / 24 + 1.2)
        + 0.9 * np.sin(2 * np.pi * t / (24 * 4) - 0.7)
        + rng.normal(0, 0.8, periods)
    )
    wind_speed_10m = np.clip(wind_speed_10m, 0.1, 12)

    # -------------------------
    # Precipitation [mm/h]
    # -------------------------
    # Event probability increases with humidity and cloud cover
    rain_prob = (
        0.03
        + 0.20 * (relative_humidity_2m / 100.0 - 0.5)
        + 0.20 * (cloud_cover / 100.0 - 0.4)
    )
    
    rain_prob = np.clip(rain_prob, 0.01, 0.55)

    rain_flag = rng.uniform(0, 1, periods) < rain_prob

    # Rain intensity: skewed distribution with occasional stronger events
    rain_intensity = rng.gamma(shape=1.8, scale=1.2, size=periods)  # mostly light rain
    heavy_event = rng.uniform(0, 1, periods) < 0.06
    rain_intensity[heavy_event] *= 3.5

    precipitation = np.where(rain_flag, rain_intensity, 0.0)
    precipitation = np.clip(precipitation, 0, 25)

    # -------------------------
    # Soil moisture generation
    # -------------------------
    temp_n = (temperature_2m - temperature_2m.min()) / (temperature_2m.max() - temperature_2m.min())
    rh_n = relative_humidity_2m / 100.0
    cloud_n = cloud_cover / 100.0
    wind_n = wind_speed_10m / 12.0
    rain_n = np.clip(precipitation / 10.0, 0, 1)

    # Rolling rainfall memory for delayed infiltration effect
    rain_6h = pd.Series(precipitation).rolling(6, min_periods=1).sum().values
    rain_24h = pd.Series(precipitation).rolling(24, min_periods=1).sum().values

    rain_6h_n = np.clip(rain_6h / 20.0, 0, 1)
    rain_24h_n = np.clip(rain_24h / 60.0, 0, 1)

    soil_moisture = np.zeros(periods, dtype=float)
    soil_moisture[0] = 0.28

    for i in range(1, periods):
        evap_demand = (
            0.40 * temp_n[i]
            + 0.25 * wind_n[i]
            - 0.20 * rh_n[i]
            - 0.10 * cloud_n[i]
        )

        wetting = (
            0.18 * rain_n[i]
            + 0.12 * rain_6h_n[i]
            + 0.10 * rain_24h_n[i]
            + 0.04 * rh_n[i]
        )

        target = (
            0.30
            + wetting
            - 0.16 * evap_demand
            + 0.01 * np.sin(2 * np.pi * i / (24 * 5))
            + rng.normal(0, 0.004)
        )

        # Inertia + faster response during rain
        alpha = 0.94 if precipitation[i] == 0 else 0.88
        soil_moisture[i] = alpha * soil_moisture[i - 1] + (1 - alpha) * target

    soil_moisture = np.clip(soil_moisture, 0.05, 0.60)

    # -------------------------
    # DataFrame
    # -------------------------
    df = pd.DataFrame({
        "soil_moisture": soil_moisture*100,
        "temperature_2m": temperature_2m,
        "relative_humidity_2m": relative_humidity_2m,
        "cloud_cover": cloud_cover,
        "wind_speed_10m": wind_speed_10m,
        "precipitation": precipitation,
        "temperature_2m_f": temperature_2m,
        "relative_humidity_2m_f": relative_humidity_2m,
        "cloud_cover_f": cloud_cover,
        "wind_speed_10m_f": wind_speed_10m,
        "precipitation_f": precipitation,
    }, index=time_index)

    df.index.name = "time"

    # -------------------------
    # Lag features
    # -------------------------
    lag_steps = {
        "soil_moisture": soil_moisture,
        "temperature_2m": temperature_2m,
        "relative_humidity_2m": relative_humidity_2m,
        "cloud_cover": cloud_cover,
        "wind_speed_10m": wind_speed_10m,
        "precipitation": precipitation,
        "temperature_2m_f": temperature_2m,
        "relative_humidity_2m_f": relative_humidity_2m,
        "cloud_cover_f": cloud_cover,
        "wind_speed_10m_f": wind_speed_10m,
        "precipitation_f": precipitation,
    }

    # for col_name, lag in lag_steps.items():
    #     base_col = col_name.replace(f"_lag_{lag}", "")
    #     df[col_name] = df[base_col].shift(lag)

    # # Optional rolling features, very useful for forecasting
    # df["precipitation_roll_6h"] = df["precipitation"].rolling(6, min_periods=1).sum()
    # df["precipitation_roll_24h"] = df["precipitation"].rolling(24, min_periods=1).sum()
    # df["temperature_2m_roll_24h"] = df["temperature_2m"].rolling(24, min_periods=1).mean()
    # df["relative_humidity_2m_roll_24h"] = df["relative_humidity_2m"].rolling(24, min_periods=1).mean()

    if dropna:
        df = df.dropna()

    df.to_csv(output_csv)
    return df


if __name__ == "__main__":
    df = generate_synthetic_soil_moisture_csv(
        output_csv="./synthetic_soil_moisture_with_lags_train.csv",
        start="2024-01-01 00:00:00",
        periods=24 * 365 * 3,
        freq="h",
        seed=42,
        dropna=True
    )

    print(df.head())
    print(f"\nSaved CSV with {len(df)} rows and {len(df.columns)} columns.")

    df = generate_synthetic_soil_moisture_csv(
        output_csv="./synthetic_soil_moisture_with_lags_test.csv",
        start="2027-01-01 00:00:00",
        periods=24 * 365 * 1,
        freq="h",
        seed=45,
        dropna=True
    )
