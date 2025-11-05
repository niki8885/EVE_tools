import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime

DATA_PATH = "plex/data/price.csv"
REGION_ID = 19000001
TYPE_ID = 44992

def fetch_market_data(region_id: int, type_id: int) -> dict:
    url = f"https://evetycoon.com/api/v1/market/stats/{region_id}/{type_id}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "timestamp": timestamp,
        "buyVolume": data["buyVolume"],
        "sellVolume": data["sellVolume"],
        "buyOrders": data["buyOrders"],
        "sellOrders": data["sellOrders"],
        "buyOutliers": data["buyOutliers"],
        "sellOutliers": data["sellOutliers"],
        "buyThreshold": data["buyThreshold"],
        "sellThreshold": data["sellThreshold"],
        "buyAvgFivePercent": data["buyAvgFivePercent"],
        "sellAvgFivePercent": data["sellAvgFivePercent"]
    }


def append_to_csv(data: dict, path: str):
    df_new = pd.DataFrame([data])

    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(path, index=False)


def calc_order_imbalance(df):
    df["OI"] = (df["buyVolume"] - df["sellVolume"]) / (df["buyVolume"] + df["sellVolume"])
    return df

def calc_volume_ratio(df):
    df["VR"] = df["buyVolume"] / df["sellVolume"]
    return df

def calc_liquidity_index(df):
    df["LI"] = (df["buyOrders"] + df["sellOrders"]) / (df["buyVolume"] + df["sellVolume"])
    return df

def calc_volume_per_order(df):
    df["VPO"] = (df["buyVolume"] + df["sellVolume"]) / (df["buyOrders"] + df["sellOrders"])
    return df

def calc_delta_volume(df):
    df["ΔVolume"] = (df["buyVolume"] + df["sellVolume"]).diff()
    return df

def calc_roc(df, column="buyAvgFivePercent", period=1):
    df[f"ROC_{column}_{period}"] = df[column].pct_change(periods=period) * 100
    return df

def calc_cumulative_oi(df):
    df["COI"] = df["OI"].cumsum()
    return df

def calc_rolling_volatility(df, column="OI", window=24):
    df[f"Volatility_{column}_{window}"] = df[column].rolling(window).std()
    return df

def calc_market_sentiment(df):
    oi_min, oi_max = df["OI"].min(), df["OI"].max()
    df["MSI"] = 2 * ((df["OI"] - oi_min) / (oi_max - oi_min)) - 1
    return df

def calc_sma(df, column="buyAvgFivePercent", periods=[10, 24, 48, 120]):
    for p in periods:
        df[f"SMA_{column}_{p}"] = df[column].rolling(p).mean()
    return df

def calc_ema(df, column="buyAvgFivePercent", periods=[10, 24, 48, 120]):
    for p in periods:
        df[f"EMA_{column}_{p}"] = df[column].ewm(span=p, adjust=False).mean()
    return df

def calc_macd(df, column="buyAvgFivePercent"):
    short = df[column].ewm(span=12, adjust=False).mean()
    long = df[column].ewm(span=26, adjust=False).mean()
    df["MACD"] = short - long
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def calc_rsi(df, column="buyAvgFivePercent", period=14):
    delta = df[column].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def calc_rolling_std(df, column="buyAvgFivePercent", window=24):
    df[f"RollingStd_{column}_{window}"] = df[column].rolling(window).std()
    return df

def calc_bollinger_bands(df, column="buyAvgFivePercent", window=24, k=2):
    sma = df[column].rolling(window).mean()
    std = df[column].rolling(window).std()
    df["Boll_Upper"] = sma + k * std
    df["Boll_Lower"] = sma - k * std
    return df

def calc_coefficient_variation(df, column="buyAvgFivePercent", window=24):
    mean = df[column].rolling(window).mean()
    std = df[column].rolling(window).std()
    df[f"CV_{column}_{window}"] = std / mean
    return df

def calc_volume_ma(df, window_list=[10, 24, 48, 120]):
    total_vol = df["buyVolume"] + df["sellVolume"]
    for w in window_list:
        df[f"VMA_{w}"] = total_vol.rolling(w).mean()
    return df

def calc_volume_oscillator(df, short=14, long=28):
    vol = df["buyVolume"] + df["sellVolume"]
    ema_short = vol.ewm(span=short, adjust=False).mean()
    ema_long = vol.ewm(span=long, adjust=False).mean()
    df["VO"] = (ema_short - ema_long) / ema_long * 100
    return df

def calc_stochastic(df, column="buyAvgFivePercent", window=14):
    low_min = df[column].rolling(window).min()
    high_max = df[column].rolling(window).max()
    df["%K"] = (df[column] - low_min) / (high_max - low_min) * 100
    df["%D"] = df["%K"].rolling(3).mean()
    return df

def calc_momentum(df, column="buyAvgFivePercent", period=10):
    df[f"Momentum_{period}"] = df[column] - df[column].shift(period)
    return df

def calc_williams_r(df, column="buyAvgFivePercent", window=14):
    high_max = df[column].rolling(window).max()
    low_min = df[column].rolling(window).min()
    df["WilliamsR"] = (high_max - df[column]) / (high_max - low_min) * -100
    return df

def calc_cci(df, column="buyAvgFivePercent", window=20, c=0.015):
    tp = df[column]
    sma = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df["CCI"] = (tp - sma) / (c * mad)
    return df

def calc_atr(df, high="buyAvgFivePercent", low="sellAvgFivePercent", close="buyAvgFivePercent", window=14):
    high_low = df[high] - df[low]
    high_close = (df[high] - df[close].shift()).abs()
    low_close = (df[low] - df[close].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window).mean()
    return df

def calc_zscore(df, column="buyAvgFivePercent", window=24):
    mean = df[column].rolling(window).mean()
    std = df[column].rolling(window).std()
    df[f"ZScore_{column}_{window}"] = (df[column] - mean) / std
    return df

def calc_mfi(df, column="buyAvgFivePercent", volume_col="buyVolume", window=14):
    typical_price = df[column]
    money_flow = typical_price * df[volume_col]
    positive_flow = np.where(typical_price > typical_price.shift(), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(), money_flow, 0)
    pos_mf = pd.Series(positive_flow).rolling(window).sum()
    neg_mf = pd.Series(negative_flow).rolling(window).sum()
    mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
    df["MFI"] = mfi
    return df

def calc_acceleration(df, column="buyAvgFivePercent"):
    df["Acceleration"] = df[column].diff().diff()
    return df

def calc_composite_strength(df):
    df["CompositeIndex"] = (
        0.4 * df["RSI"].fillna(50) +
        0.3 * (100 - df["WilliamsR"].fillna(50)) +
        0.3 * (df["%K"].fillna(50))
    ) / 100
    return df

def request_plex_info():
    new_data = fetch_market_data(REGION_ID, TYPE_ID)
    append_to_csv(new_data, DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    df = calc_order_imbalance(df)
    df = calc_volume_ratio(df)
    df = calc_liquidity_index(df)
    df = calc_volume_per_order(df)
    df = calc_delta_volume(df)
    df = calc_roc(df)
    df = calc_cumulative_oi(df)
    df = calc_rolling_volatility(df)
    df = calc_market_sentiment(df)
    df = calc_sma(df)
    df = calc_ema(df)
    df = calc_macd(df)
    df = calc_rsi(df)
    df = calc_rolling_std(df)
    df = calc_bollinger_bands(df)
    df = calc_coefficient_variation(df)
    df = calc_volume_ma(df)
    df = calc_volume_oscillator(df)
    df = calc_stochastic(df)
    df = calc_momentum(df)
    df = calc_williams_r(df)
    df = calc_cci(df)
    df = calc_atr(df)
    df = calc_zscore(df)
    df = calc_mfi(df)
    df = calc_acceleration(df)
    df = calc_composite_strength(df)

    df.to_csv(DATA_PATH, index=False)
    print(f"Updated {len(df)} records in {DATA_PATH}")
