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

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = ["TotalVolume_ind", "LiquidityIndex_ind", "VolumeVolatility_ind", "OrderFlowPressure_ind"]
    features = [f for f in features if f in df.columns]
    if len(features) < 3:
        print("Not enough indicators for clustering — skipping.")
        return

    X = df[features].dropna().copy()
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = np.nan
    df.loc[X.index, "Cluster"] = kmeans.fit_predict(X_scaled)

    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            X[features[0]],
            X[features[1]],
            X[features[2]],
            c=kmeans.labels_,
            cmap="tab10",
            s=10,
            alpha=0.7
        )
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        plt.title("KMeans Market State Clusters (3D View)")
        plt.colorbar(sc, label="Cluster")
        plt.tight_layout()
        plt.savefig(plot_dir / "clusters_3d.png")
        plt.close()
    except Exception as e:
        print("3D plot skipped:", e)

    try:
        from umap import UMAP
        reducer = UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(X_scaled)
        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=kmeans.labels_,
            palette="tab10",
            s=15
        )
        plt.title("UMAP projection of Market States (KMeans Clusters)")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.legend(title="Cluster", loc="best")
        plt.tight_layout()
        plt.savefig(plot_dir / "clusters_umap.png")
        plt.close()
    except ImportError:
        print("UMAP not installed — skipping dimensionality reduction plot.")

    heatmap_features = {
        "LiquidityIndex_ind": "Liquidity Index",
        "VolumeVolatility_ind": "Volume Volatility",
        "OrderFlowPressure_ind": "Order Flow Pressure",
    }

    df["weekday_num"] = df["timestamp"].dt.weekday
    for col, title in heatmap_features.items():
        if col not in df.columns:
            continue

        pivot = df.pivot_table(values=col, index="weekday_num", columns="hour", aggfunc="mean")

        plt.figure(figsize=(10, 5))
        cmap = "RdBu_r" if "Pressure" in title else "magma" if "Volatility" in title else "YlGnBu"
        sns.heatmap(pivot, cmap=cmap, center=0 if "Pressure" in title else None, cbar_kws={'label': title})
        plt.title(f"Average {title} by Day of Week and Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week")
        plt.yticks(ticks=np.arange(7) + 0.5, labels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], rotation=0)
        plt.tight_layout()
        plt.savefig(plot_dir / f"heatmap_{col}.png")
        plt.close()

    df["Cluster"] = df["Cluster"].fillna(-1).astype(int)

    df = df[df["Cluster"] >= 0]

    cluster_pivot = df.pivot_table(
        values="Cluster",
        index="Item",
        columns="Date",
        aggfunc=lambda x: np.bincount(x.astype(int)).argmax() if len(x) > 0 else -1
    )

    plt.figure(figsize=(10, 5))
    sns.heatmap(cluster_pivot, cmap="tab10", cbar_kws={'label': 'Dominant Cluster'})
    plt.title("Dominant Market Cluster by Day and Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.yticks(ticks=np.arange(7) + 0.5, labels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], rotation=0)
    plt.tight_layout()
    plt.savefig(plot_dir / "heatmap_clusters_weekday_hour.png")
    plt.close()

    print(f"All volume plots and heatmaps saved to: {plot_dir.resolve()}")
