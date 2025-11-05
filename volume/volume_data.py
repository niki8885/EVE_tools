import csv
import requests
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

API_BASE = "https://evetycoon.com/api/v1/market/stats"
REGION_ID = 10000002


def fetch_market_stats(type_id: int, region_id: int = REGION_ID):
    url = f"{API_BASE}/{region_id}/{type_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def load_items(path=DATA_DIR / "items.csv"):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def append_to_csv(path, row, fieldnames):
    file_exists = path.exists()
    with open(path, "a", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def collect_total_market_volume():
    items = load_items()
    total_buy_volume = total_sell_volume = 0
    total_buy_orders = total_sell_orders = 0
    total_buy_outliers = total_sell_outliers = 0

    for item in items:
        try:
            stats = fetch_market_stats(int(item["type_id"]))
            total_buy_volume += stats.get("buyVolume", 0)
            total_sell_volume += stats.get("sellVolume", 0)
            total_buy_orders += stats.get("buyOrders", 0)
            total_sell_orders += stats.get("sellOrders", 0)
            total_buy_outliers += stats.get("buyOutliers", 0)
            total_sell_outliers += stats.get("sellOutliers", 0)
        except Exception as e:
            print(f"Error fetching {item['name']}: {e}")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "buyVolume": total_buy_volume,
        "sellVolume": total_sell_volume,
        "buyOrders": total_buy_orders,
        "sellOrders": total_sell_orders,
        "buyOutliers": total_buy_outliers,
        "sellOutliers": total_sell_outliers
    }

    out_path = DATA_DIR / "volume.csv"
    append_to_csv(out_path, row, fieldnames=row.keys())
    print(f"[+] Added aggregated market stats at {row['timestamp']}")


def calculate_volume_indicators(df: pd.DataFrame):
    df["TotalVolume"] = df["buyVolume"] + df["sellVolume"]
    df["NetVolume"] = df["buyVolume"] - df["sellVolume"]
    df["VolumeRatio"] = df["buyVolume"] / df["sellVolume"].replace(0, 1)
    df["BuyVolumeShare"] = df["buyVolume"] / df["TotalVolume"]
    df["SellVolumeShare"] = df["sellVolume"] / df["TotalVolume"]
    df["VolumeImbalance"] = (df["buyVolume"] - df["sellVolume"]) / df["TotalVolume"]

    df["OrderCountRatio"] = df["buyOrders"] / df["sellOrders"].replace(0, 1)
    df["AvgBuyOrderSize"] = df["buyVolume"] / df["buyOrders"].replace(0, 1)
    df["AvgSellOrderSize"] = df["sellVolume"] / df["sellOrders"].replace(0, 1)
    df["OrderSizeRatio"] = df["AvgBuyOrderSize"] / df["AvgSellOrderSize"].replace(0, 1)

    df["LiquidityIndex"] = df["TotalVolume"] / (df["buyOrders"] + df["sellOrders"])
    df["OrderFlowPressure"] = (df["buyOrders"] - df["sellOrders"]) / (df["buyOrders"] + df["sellOrders"])
    df["MarketDepthProxy"] = df["TotalVolume"] / (df["buyOrders"] + df["sellOrders"])

    df["VolumeChange"] = df["TotalVolume"].diff()
    df["VROC"] = df["VolumeChange"] / df["TotalVolume"].shift(1)
    df["VMomentum"] = df["TotalVolume"].diff(3)
    df["VolumeVolatility"] = df["TotalVolume"].rolling(10).std()
    df["CoeffVariation"] = df["VolumeVolatility"] / df["TotalVolume"].rolling(10).mean()

    return df


def aggregate_volume_stats(path=DATA_DIR / "volume.csv", out=DATA_DIR / "avg_volume.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["weekday"] = df["timestamp"].dt.day_name()
    df["month_day"] = df["timestamp"].dt.day
    df["year_day"] = df["timestamp"].dt.dayofyear
    df["hour"] = df["timestamp"].dt.hour

    agg = (
        df.groupby(["weekday", "month_day", "year_day", "hour"])
        .mean(numeric_only=True)
        .reset_index()
    )

    agg.to_csv(out, index=False)
    print(f"[+] Saved aggregated volume averages to {out}")


def request_volume_data():
    collect_total_market_volume()
    df = pd.read_csv(DATA_DIR / "volume.csv")
    df = calculate_volume_indicators(df)
    df.to_csv(DATA_DIR / "volume_indicators.csv", index=False)
    aggregate_volume_stats()
