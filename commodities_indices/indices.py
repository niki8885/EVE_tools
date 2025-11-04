import requests
import pandas as pd
import time
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ESI_SEARCH_URL = "https://esi.evetech.net/latest/search/"
FUZZWORK_TYPEID = "https://www.fuzzwork.co.uk/api/typeid.php"
FUZZWORK_API = "https://market.fuzzwork.co.uk/aggregates/"
JITA_REGION_ID = 10000002
BASE_URL = "https://evetycoon.com/api/v1/market/stats"
REGION_ID = 10000002

def get_type_id(item_name):
    params = {"categories": "inventory_type", "search": item_name}
    r = requests.get(ESI_SEARCH_URL, params=params)
    if r.status_code == 200:
        data = r.json()
        if "inventory_type" in data:
            return data["inventory_type"][0]

    fw = requests.get(FUZZWORK_TYPEID, params={"typename": item_name})
    if fw.status_code == 200:
        d = fw.json()
        if "typeID" in d:
            return d["typeID"]
    return None


def get_jita_volume(type_id):
    url = f"{FUZZWORK_API}?region={JITA_REGION_ID}&types={type_id}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if str(type_id) in data:
            sell_data = data[str(type_id)].get("sell", {})
            return sell_data.get("volume", 0)
    return 0


def build_commodities_index(csv_path, output_path="commodities_index.csv"):
    df = pd.read_csv(csv_path)
    results = []

    for name in df["name"]:
        type_id = get_type_id(name)
        if type_id is None:
            print(f"No type_id {name}")
            continue

        volume = get_jita_volume(type_id)
        print(f"{name}: type_id={type_id}, volume={volume}")

        results.append({"name": name, "type_id": type_id, "volume": volume})
        time.sleep(0.3)

    if not results:
        raise ValueError("Empty data")

    result_df = pd.DataFrame(results)
    result_df["volume"] = pd.to_numeric(result_df["volume"], errors="coerce")
    total_volume = result_df["volume"].sum()
    result_df["weight"] = result_df["volume"] / total_volume

    result_df.to_csv(output_path, index=False)
    print(f"{output_path}")
    return result_df

def get_market_stats(type_id, region_id=REGION_ID):
    url = f"{BASE_URL}/{region_id}/{type_id}"
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        print(f"Error  {type_id}: {r.status_code}")
        return None


def compute_entropy(weights):
    weights = np.array(weights)
    weights = weights[weights > 0]
    return -np.sum(weights * np.log(weights))


def compute_h_index(weights):
    return np.sum(np.array(weights) ** 2)


def compute_top3_share(weights):
    sorted_w = np.sort(weights)[::-1]
    return np.sum(sorted_w[:3])


def update_commodity_index(csv_path, region_id=REGION_ID):
    index_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_dir = f"prices/{index_name}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "index.csv")

    df = pd.read_csv(csv_path)

    prices, volumes = [], []
    for _, row in df.iterrows():
        type_id = row["type_id"]
        stats = get_market_stats(type_id, region_id)
        if not stats:
            prices.append(np.nan)
            volumes.append(np.nan)
            continue
        prices.append(stats.get("sellAvgFivePercent", np.nan))
        volumes.append(stats.get("sellVolume", np.nan))
        time.sleep(0.25)

    df["price"] = prices
    df["market_volume"] = volumes

    df["weighted_price"] = df["weight"] * df["price"]
    df["weighted_volume"] = df["weight"] * df["market_volume"]
    price_index = df["weighted_price"].sum()
    volume_index = df["weighted_volume"].sum()

    weights = df["weight"].values
    top3_share = compute_top3_share(weights)
    h_index = compute_h_index(weights)
    entropy = compute_entropy(weights)

    liquidity_index = np.nanmean(df["market_volume"]) / np.nanstd(df["market_volume"]) if df["market_volume"].std() != 0 else np.nan

    result = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price_index": price_index,
        "volume_index": volume_index,
        "top3_share": top3_share,
        "h_index": h_index,
        "entropy": entropy,
        "liquidity_index": liquidity_index,
    }])

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            hist = pd.read_csv(out_path)
            hist = pd.concat([hist, result], ignore_index=True)
        except pd.errors.EmptyDataError:
            print(f"{out_path} empty.")
            hist = result
    else:
        hist = result

    hist["sma_10"] = hist["price_index"].rolling(10).mean()
    hist["returns"] = hist["price_index"].pct_change()
    hist["volatility"] = hist["returns"].rolling(10).std()

    delta = hist["price_index"].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    roll_up, roll_down = up.rolling(14).mean(), down.rolling(14).mean()
    rs = roll_up / roll_down
    hist["rsi"] = 100 - (100 / (1 + rs))

    ema12 = hist["price_index"].ewm(span=12, adjust=False).mean()
    ema26 = hist["price_index"].ewm(span=26, adjust=False).mean()
    hist["macd"] = ema12 - ema26

    hist.to_csv(out_path, index=False)
    print(f"Index {index_name} updated → {out_path}")
    return hist


def plot_index_grids(index_name: str, index_path: str):
    os.makedirs(f"plots/{index_name}", exist_ok=True)
    df = pd.read_csv(index_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # ========== GRID 1 ==========
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{index_name.capitalize()} Index Dynamics", fontsize=16)
    gs = GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df["timestamp"], df["price_index"], label="Price Index", color="blue")
    ax1.set_title("Price Index")
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df["timestamp"], df["volume_index"], label="Volume Index", color="purple")
    ax2.set_title("Volume Index")
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df["timestamp"], df["volatility"], label="Volatility", color="orange")
    ax3.set_title("Volatility")
    ax3.grid(True)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df["timestamp"], df["rsi"], label="RSI", color="green")
    ax4.axhline(30, color="red", linestyle="--", linewidth=0.8)
    ax4.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax4.set_title("RSI")
    ax4.grid(True)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df["timestamp"], df["macd"], label="MACD", color="brown")
    ax5.set_title("MACD")
    ax5.grid(True)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(df["timestamp"], df["sma_10"], label="SMA 10", color="teal")
    ax6.set_title("SMA 10")
    ax6.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"plots/{index_name}/grid_index_{index_name}.png", dpi=200)
    plt.close(fig)

    # ========== GRID 2 ==========
    fig2 = plt.figure(figsize=(14, 10))
    fig2.suptitle(f"{index_name.capitalize()} Distributions & Risk", fontsize=16)
    gs2 = GridSpec(2, 2, figure=fig2)

    ax1 = fig2.add_subplot(gs2[0, 0])
    ax1.hist(df["price_index"], bins=20, color="skyblue", alpha=0.7)
    current_price = df["price_index"].iloc[-1]
    ax1.axvline(current_price, color="red", linestyle="--", label=f"Current: {current_price:.2f}")
    ax1.set_title("Price Distribution")
    ax1.legend()

    ax2 = fig2.add_subplot(gs2[0, 1])
    ax2.hist(df["volume_index"], bins=20, color="mediumpurple", alpha=0.7)
    current_vol = df["volume_index"].iloc[-1]
    ax2.axvline(current_vol, color="red", linestyle="--", label=f"Current: {current_vol:,.0f}")
    ax2.set_title("Volume Distribution")
    ax2.legend()

    returns = df["price_index"].pct_change().dropna()

    if len(returns) > 0:
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
    else:
        var_95 = np.nan
        cvar_95 = np.nan

    ax3 = fig2.add_subplot(gs2[1, 0])
    if len(returns) > 0:
        ax3.hist(returns, bins=30, color="gray", alpha=0.7)
        ax3.axvline(var_95, color="orange", linestyle="--", label=f"VaR 95% = {var_95:.3%}")
        ax3.axvline(cvar_95, color="red", linestyle="--", label=f"CVaR 95% = {cvar_95:.3%}")
    else:
        ax3.text(0.5, 0.5, "Not enough data", ha="center", va="center", fontsize=12)
    ax3.set_title("VaR & CVaR Distribution")
    ax3.legend()

    ax4 = fig2.add_subplot(gs2[1, 1])
    if len(returns) >= 20:
        rolling_returns = returns.rolling(20).apply(lambda x: np.percentile(x, 5))
        ax4.plot(df["timestamp"].iloc[-len(rolling_returns):], rolling_returns, color="red")
    else:
        ax4.text(0.5, 0.5, "Not enough data for rolling VaR", ha="center", va="center", fontsize=12)
    ax4.set_title("Rolling VaR (20-period)")
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(f"plots/{index_name}/grid_risk_{index_name}.png", dpi=200)
    plt.close(fig2)

    print(f"Saved grids to plots/{index_name}/")


def recive_indicies():
    update_commodity_index("data/minerals.csv")
    update_commodity_index("data/moon.csv")
    update_commodity_index("data/PI.csv")
    update_commodity_index("data/ice.csv")
    update_commodity_index("data/war.csv")
    plot_index_grids(
        index_name="minerals",
        index_path="prices/minerals/index.csv"
    )
    plot_index_grids(
        index_name="moon",
        index_path="prices/moon/index.csv"
    )
    plot_index_grids(
        index_name="PI",
        index_path="prices/PI/index.csv"
    )
    plot_index_grids(
        index_name="ice",
        index_path="prices/ice/index.csv"
    )
    plot_index_grids(
        index_name="war",
        index_path="prices/war/index.csv"
    )
