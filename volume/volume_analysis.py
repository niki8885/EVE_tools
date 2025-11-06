import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.style.use("seaborn-v0_8-darkgrid")


def generate_volume_plots(data_dir: str = "volume/data", plot_dir: str = "volume/plots"):

    data_dir = Path(data_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df_vol = pd.read_csv(data_dir / "volume.csv")
    df_ind = pd.read_csv(data_dir / "volume_indicators.csv")

    for df in (df_vol, df_ind):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)

    min_len = min(len(df_vol), len(df_ind))
    df_vol = df_vol.iloc[:min_len].reset_index(drop=True)
    df_ind = df_ind.iloc[:min_len].reset_index(drop=True)

    df = pd.concat([df_ind.add_suffix("_ind"), df_vol.add_suffix("_raw")], axis=1)
    df["timestamp"] = df["timestamp_raw"]

    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()
    df["month"] = df["timestamp"].dt.month_name()

    exclude_cols = ["timestamp_raw", "timestamp_ind", "timestamp", "hour", "weekday", "month"]
    numeric_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        plt.plot(df["timestamp"], df[col], label=col, color="tab:blue")
        plt.title(f"{col} over time")
        plt.xlabel("Timestamp")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{col}_timeseries.png")
        plt.close()

    buy_col = "buyVolume_raw" if "buyVolume_raw" in df.columns else "buyVolume"
    sell_col = "sellVolume_raw" if "sellVolume_raw" in df.columns else "sellVolume"
    total_col = "TotalVolume_raw" if "TotalVolume_raw" in df.columns else "TotalVolume_ind"

    plt.figure(figsize=(8, 4))
    plt.plot(df["timestamp"], df[buy_col], label="Buy Volume", color="tab:green")
    plt.plot(df["timestamp"], df[sell_col], label="Sell Volume", color="tab:red")
    if total_col in df.columns:
        plt.plot(df["timestamp"], df[total_col], label="Total Volume", color="tab:gray", linestyle="--")
    plt.title("Buy / Sell / Total Volume Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "buy_sell_total_volume.png")
    plt.close()

    df["weekday_num"] = df["timestamp"].dt.weekday
    pivot = df.pivot_table(values=total_col, index="weekday_num", columns="hour", aggfunc="mean")

    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={'label': 'Avg Total Volume'})
    plt.title("Average Total Volume by Day of Week and Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.yticks(ticks=np.arange(7) + 0.5, labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    plt.tight_layout()
    plt.savefig(plot_dir / "heatmap_weekday_hour.png")
    plt.close()

    pivot_month = df.pivot_table(values=total_col, index="month", columns="hour", aggfunc="mean")
    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot_month, cmap="magma", cbar_kws={'label': 'Avg Total Volume'})
    plt.title("Average Total Volume by Month and Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Month")
    plt.tight_layout()
    plt.savefig(plot_dir / "heatmap_month_hour.png")
    plt.close()

    df_daily = (
        df.set_index("timestamp")
        .resample("1D")[[buy_col, sell_col, total_col]]
        .mean()
    )

    plt.figure(figsize=(8, 4))
    plt.plot(df_daily.index, df_daily[buy_col], label="Buy Volume", color="tab:green")
    plt.plot(df_daily.index, df_daily[sell_col], label="Sell Volume", color="tab:red")
    plt.plot(df_daily.index, df_daily[total_col], label="Total Volume", color="tab:gray", linestyle="--")
    plt.title("Daily Average Volumes")
    plt.xlabel("Date")
    plt.ylabel("Average Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "daily_avg_volumes.png")
    plt.close()

    print(f"All volume plots and heatmaps saved to: {plot_dir.resolve()}")
