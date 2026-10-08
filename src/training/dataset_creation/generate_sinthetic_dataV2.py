"""
Logica usata
irr ha effetto dominante sulla soil_moisture_drip_line, perché lungo la linea di irrigazione l’acqua viene applicata direttamente e la risposta del suolo è più rapida e intensa; invece soil_moisture_between_rows riceve un effetto più smorzato e ritardato, mentre la pioggia influisce su entrambe in modo più uniforme. Ho anche introdotto LAI come fattore che tende a ridurre l’evaporazione diretta dal suolo tramite ombreggiamento, pur mantenendo una dinamica sintetica semplice e controllabile.

Unità consigliate
Nel CSV, le due colonne di soil moisture sono espresse come contenuto idrico volumetrico sintetico in scala circa 0–1, quindi interpretabile come 
m3/m3 
m3/m3
(forma comune per rappresentare il soil moisture). irr e rain sono trattate come apporti idrici orari equivalenti in mm/h sintetici, utili per costruire dinamiche plausibili di bagnamento e asciugamento.

Forecast values are identical to the historical one assuming the forscat is 100% reliable
"""


import numpy as np
import pandas as pd


def generate_synthetic_soil_moisture_dual_csv(
    output_csv="synthetic_soil_moisture_dual.csv",
    start="2024-04-01 00:00:00",
    periods=24 * 180,   # 180 days hourly
    freq="h",
    seed=42,
    dropna=False
):
    rng = np.random.default_rng(seed)

    time_index = pd.date_range(start=start, periods=periods, freq=freq)
    t = np.arange(periods)
    hour = time_index.hour.values
    doy = time_index.dayofyear.values

    # -------------------------
    # Synthetic meteorological drivers
    # -------------------------

    # Temperature [°C]
    temperature_2m = (
        20
        + 7 * np.sin(2 * np.pi * doy / 365.25 - 0.7)
        + 5 * np.sin(2 * np.pi * hour / 24 - np.pi / 2)
        + rng.normal(0, 1.2, periods)
    )

    # Relative humidity [%]
    relative_humidity_2m = (
        65
        - 0.8 * (temperature_2m - np.mean(temperature_2m))
        + 7 * np.sin(2 * np.pi * hour / 24 + 0.8)
        + rng.normal(0, 4.0, periods)
    )
    relative_humidity_2m = np.clip(relative_humidity_2m, 20, 100)

    # Dew point [°C] - physically related to temp and RH
    dew_point_2m = temperature_2m - (100 - relative_humidity_2m) / 5.0 + rng.normal(0, 0.6, periods)
    dew_point_2m = np.minimum(dew_point_2m, temperature_2m)

    # Cloud cover [%]
    cloud_cover = (
        40
        + 0.45 * (relative_humidity_2m - 60)
        + 18 * np.sin(2 * np.pi * t / (24 * 6) + 0.4)
        + rng.normal(0, 12.0, periods)
    )
    cloud_cover = np.clip(cloud_cover, 0, 100)

    # Wind speed [m/s]
    wind_speed_10m = (
        3.5
        + 1.0 * np.sin(2 * np.pi * hour / 24 + 1.0)
        + 0.8 * np.sin(2 * np.pi * t / (24 * 4) - 0.7)
        + rng.normal(0, 0.7, periods)
    )
    wind_speed_10m = np.clip(wind_speed_10m, 0.1, 12)

    # Wind direction [degrees]
    wind_direction_10m = (
        180
        + 90 * np.sin(2 * np.pi * t / (24 * 3))
        + 40 * np.sin(2 * np.pi * hour / 24)
        + rng.normal(0, 20, periods)
    ) % 360

    # Soil temperature 0-7 cm [°C]
    soil_temperature_0_to_7cm = (
        18
        + 6 * np.sin(2 * np.pi * doy / 365.25 - 0.5)
        + 2 * np.sin(2 * np.pi * hour / 24 - 1.0)
        + 0.4 * temperature_2m
        + rng.normal(0, 0.8, periods)
    )

    # Leaf Area Index [-]
    # smooth seasonal signal with moderate variability
    LAI = (
        1.2
        + 1.8 * np.maximum(0, np.sin(2 * np.pi * (doy - 80) / 365.25))
        + 0.15 * np.sin(2 * np.pi * t / (24 * 10))
        + rng.normal(0, 0.08, periods)
    )
    LAI = np.clip(LAI, 0.2, 4.5)

    # -------------------------
    # Rain [mm/h]
    # -------------------------
    rain_prob = (
        0.03
        + 0.20 * (relative_humidity_2m / 100.0 - 0.5)
        + 0.22 * (cloud_cover / 100.0 - 0.35)
    )
    rain_prob = np.clip(rain_prob, 0.01, 0.60)

    rain_flag = rng.uniform(0, 1, periods) < rain_prob
    rain_intensity = rng.gamma(shape=1.7, scale=1.1, size=periods)
    heavy_event = rng.uniform(0, 1, periods) < 0.05
    rain_intensity[heavy_event] *= 4.0

    rain = np.where(rain_flag, rain_intensity, 0.0)
    rain = np.clip(rain, 0, 30)

    # -------------------------
    # Irrigation events [mm/h equivalent]
    # -------------------------
    # Irrigation mostly occurs early morning, every few days, stronger in warmer/drier conditions
    dry_signal = (
        0.45 * (temperature_2m - temperature_2m.min()) / (temperature_2m.max() - temperature_2m.min())
        + 0.30 * (1 - relative_humidity_2m / 100.0)
        + 0.25 * (wind_speed_10m / 12.0)
    )
    dry_signal = np.clip(dry_signal, 0, 1)

    morning_window = ((hour >= 4) & (hour <= 7)).astype(float)
    base_irr_prob = 0.015 + 0.14 * dry_signal * morning_window

    irr_flag = rng.uniform(0, 1, periods) < base_irr_prob
    irr_amount = rng.gamma(shape=2.2, scale=1.8, size=periods)
    irr = np.where(irr_flag, irr_amount, 0.0)
    irr = np.clip(irr, 0, 20)

    # Avoid too much overlap rain+irrigation
    irr = np.where(rain > 2.0, irr * 0.25, irr)

    # -------------------------
    # Rolling memories
    # -------------------------
    rain_6h = pd.Series(rain).rolling(6, min_periods=1).sum().values
    rain_24h = pd.Series(rain).rolling(24, min_periods=1).sum().values
    irr_6h = pd.Series(irr).rolling(6, min_periods=1).sum().values
    irr_24h = pd.Series(irr).rolling(24, min_periods=1).sum().values

    # delayed infiltration between rows
    irr_12h_shift = pd.Series(irr).shift(3).fillna(0).rolling(12, min_periods=1).sum().values

    # -------------------------
    # Normalization
    # -------------------------
    def minmax(x):
        x = np.asarray(x, dtype=float)
        den = x.max() - x.min()
        if den == 0:
            return np.zeros_like(x)
        return (x - x.min()) / den

    temp_n = minmax(temperature_2m)
    rh_n = relative_humidity_2m / 100.0
    dew_n = minmax(dew_point_2m)
    rain_n = np.clip(rain / 10.0, 0, 1)
    rain_6h_n = np.clip(rain_6h / 20.0, 0, 1)
    rain_24h_n = np.clip(rain_24h / 60.0, 0, 1)
    cloud_n = cloud_cover / 100.0
    wind_n = wind_speed_10m / 12.0
    soil_temp_n = minmax(soil_temperature_0_to_7cm)
    lai_n = np.clip(LAI / 5.0, 0, 1)
    irr_n = np.clip(irr / 10.0, 0, 1)
    irr_6h_n = np.clip(irr_6h / 25.0, 0, 1)
    irr_24h_n = np.clip(irr_24h / 60.0, 0, 1)
    irr_12h_shift_n = np.clip(irr_12h_shift / 40.0, 0, 1)

    # Optional directional factor: some orientations slightly increase evaporative exposure
    wind_dir_rad = np.deg2rad(wind_direction_10m)
    wind_exposure = 0.5 + 0.5 * np.cos(wind_dir_rad - np.deg2rad(225))
    wind_exposure = np.clip(wind_exposure, 0, 1)

    # -------------------------
    # Dual soil moisture generation
    # -------------------------
    soil_moisture_drip_line = np.zeros(periods, dtype=float)
    soil_moisture_between_rows = np.zeros(periods, dtype=float)

    soil_moisture_drip_line[0] = 0.24
    soil_moisture_between_rows[0] = 0.20

    for i in range(1, periods):
        evap_demand = (
            0.30 * temp_n[i]
            + 0.20 * wind_n[i]
            + 0.18 * soil_temp_n[i]
            + 0.08 * wind_exposure[i]
            - 0.18 * rh_n[i]
            - 0.10 * dew_n[i]
            - 0.08 * cloud_n[i]
        )

        # Higher LAI can reduce direct soil evaporation through shading,
        # but also slightly increases transpiration demand.
        lai_effect_drip = -0.05 * lai_n[i] + 0.015 * np.sqrt(lai_n[i])
        lai_effect_between = -0.07 * lai_n[i] + 0.020 * np.sqrt(lai_n[i])

        wetting_drip = (
            0.24 * irr_n[i]
            + 0.20 * irr_6h_n[i]
            + 0.10 * irr_24h_n[i]
            + 0.10 * rain_n[i]
            + 0.10 * rain_6h_n[i]
            + 0.05 * rain_24h_n[i]
        )

        wetting_between = (
            0.05 * irr_n[i]
            + 0.10 * irr_6h_n[i]
            + 0.16 * irr_12h_shift_n[i]
            + 0.10 * irr_24h_n[i]
            + 0.12 * rain_n[i]
            + 0.14 * rain_6h_n[i]
            + 0.08 * rain_24h_n[i]
        )

        target_drip = (
            0.26
            + wetting_drip
            - 0.14 * evap_demand
            + 0.03 * rh_n[i]
            + 0.02 * dew_n[i]
            + lai_effect_drip
            + rng.normal(0, 0.004)
        )

        target_between = (
            0.22
            + wetting_between
            - 0.12 * evap_demand
            + 0.03 * rh_n[i]
            + 0.02 * dew_n[i]
            + lai_effect_between
            + rng.normal(0, 0.004)
        )

        # Faster response on drip line, slower between rows
        alpha_drip = 0.86 if (irr[i] > 0 or rain[i] > 0) else 0.94
        alpha_between = 0.91 if (rain[i] > 0 or irr_12h_shift[i] > 0) else 0.965

        soil_moisture_drip_line[i] = (
            alpha_drip * soil_moisture_drip_line[i - 1]
            + (1 - alpha_drip) * target_drip
        )

        soil_moisture_between_rows[i] = (
            alpha_between * soil_moisture_between_rows[i - 1]
            + (1 - alpha_between) * target_between
        )

    soil_moisture_drip_line = np.clip(soil_moisture_drip_line, 0.05, 0.65)
    soil_moisture_between_rows = np.clip(soil_moisture_between_rows, 0.04, 0.55)

    # -------------------------
    # Build dataframe
    # -------------------------
    df = pd.DataFrame({
        "s_w": soil_moisture_drip_line,
        "s_b": soil_moisture_between_rows,
        "LAI": LAI,
        "irr": irr,
        "temperature_2m": temperature_2m,
        "relative_humidity_2m": relative_humidity_2m,
        "dew_point_2m": dew_point_2m,
        "rain": rain,
        "cloud_cover": cloud_cover,
        "wind_speed_10m": wind_speed_10m,
        "wind_direction_10m": wind_direction_10m,
        "soil_temperature_0_to_7cm": soil_temperature_0_to_7cm,
        
        "temperature_2m_f": temperature_2m,
        "relative_humidity_2m_f": relative_humidity_2m,
        "dew_point_2m_f": dew_point_2m,
        "rain_f": rain,
        "cloud_cove_f": cloud_cover,
        "wind_speed_10m_f": wind_speed_10m,
        "wind_direction_10m_f": wind_direction_10m,
        "soil_temperature_0_to_7cm_f": soil_temperature_0_to_7cm,
    }, index=time_index)

    df.index.name = "time"

    # Optional lag features
    lag_cols = [
        "soil_moisture_drip_line",
        "soil_moisture_between_rows",
        "irr",
        "rain",
        "temperature_2m",
        "relative_humidity_2m",
        "soil_temperature_0_to_7cm",
    ]

    # for col in lag_cols:
    #     df[f"{col}_lag_1"] = df[col].shift(1)
    #     df[f"{col}_lag_24"] = df[col].shift(24)

    # df["irr_roll_6h"] = df["irr"].rolling(6, min_periods=1).sum()
    # df["irr_roll_24h"] = df["irr"].rolling(24, min_periods=1).sum()
    # df["rain_roll_6h"] = df["rain"].rolling(6, min_periods=1).sum()
    # df["rain_roll_24h"] = df["rain"].rolling(24, min_periods=1).sum()

    if dropna:
        df = df.dropna()

    df.to_csv(output_csv)
    return df


if __name__ == "__main__":
    df = generate_synthetic_soil_moisture_dual_csv(
        output_csv="./synthetic_soil_moisture_with_lags_trainV2.csv",
        start="2024-01-01 00:00:00",
        periods=24 * 365 * 3,
        freq="h",
        seed=42,
        dropna=True
    )

    print(df.head())
    print(f"\nSaved CSV with {len(df)} rows and {len(df.columns)} columns.")

    df = generate_synthetic_soil_moisture_dual_csv(
        output_csv="./synthetic_soil_moisture_with_lags_testV2.csv",
        start="2027-01-01 00:00:00",
        periods=24 * 365 * 1,
        freq="h",
        seed=45,
        dropna=True
    )

    print(df.head())
    print(f"\nSaved CSV with {len(df)} rows and {len(df.columns)} columns.")

    print(df.head())
    print(df[[
        "s_w",
        "s_b",
        "irr",
        "rain"
    ]].describe())





