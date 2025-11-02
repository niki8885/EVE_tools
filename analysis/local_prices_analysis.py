import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

region_colors = {
    "C-J6MT": "#8172b3",
    "UALX-3": "#dd8452",
    "jita":   "#4c72b0",
    "amarr":  "#ff7f0e",
    "dodixie": "#55a868"
}

def analyze_market_timeline(input_dir: str, regions: list, save_base_dir: str = "analysis"):
    input_name = os.path.basename(input_dir.rstrip("/\\"))
    save_dir = os.path.join(save_base_dir, input_name)
    plots_dir = os.path.join(save_dir, "plots")
    reports_dir = os.path.join(save_dir, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    all_dfs = []
    for item_folder in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item_folder)
        if not os.path.isdir(item_path):
            continue
        for region in regions:
            csv_path = os.path.join(item_path, f"{region}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df["Region"] = region
                df["Item"] = item_folder
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                all_dfs.append(df)

    if not all_dfs:
        print("No data found in input directory.")
        return None

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.sort_values("Timestamp", inplace=True)

    df_all["Sell_Buy_Spread"] = df_all["Sell_Min"] - df_all["Buy_Max"]
    df_all["Sell_Buy_%"] = (df_all["Sell_Buy_Spread"] / df_all["Buy_Max"]) * 100
    df_all["Daily_Change_%"] = df_all.groupby(["Region", "Item"])["Sell_Min"].pct_change() * 100
    df_all["Volatility_7d"] = df_all.groupby(["Region", "Item"])["Sell_Min"].transform(
        lambda x: x.rolling(7, min_periods=2).std()
    )

    items = sorted(df_all["Item"].unique())
    n_items = len(items)
    n_cols = 4
    n_rows = int(np.ceil(n_items / n_cols))

    def plot_grid(df_all, value_col, title, filename):
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axs = axs.flatten()
        for i, item in enumerate(items):
            df_item = df_all[df_all["Item"] == item]
            for region in regions:
                color = region_colors.get(region, "gray")
                data = df_item[df_item["Region"] == region][value_col].dropna()
                if len(data) == 0:
                    continue
                sns.lineplot(ax=axs[i], x=range(len(data)), y=data.values, color=color, label=region)
            axs[i].set_title(item)
            axs[i].set_ylabel(value_col)
            axs[i].set_xlabel("Time index")
            axs[i].legend()
        for j in range(i + 1, len(axs)):
            axs[j].axis("off")
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(plots_dir, filename))
        plt.close()

    plot_grid(df_all, "Sell_Min", "Average Sell Price Grid", "average_sell_price_grid.png")
    plot_grid(df_all, "Buy_Max", "Average Buy Price Grid", "average_buy_price_grid.png")
    plot_grid(df_all, "Sell_Buy_Spread", "Spread Grid", "spread_grid.png")

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axs = axs.flatten()
    for i, item in enumerate(items):
        df_item = df_all[df_all["Item"] == item]
        for region in regions:
            color = region_colors.get(region, "gray")
            data = df_item[df_item["Region"] == region]["Sell_Min"].dropna()
            if len(data) == 0:
                continue
            if data.nunique() > 1:
                sns.kdeplot(data, ax=axs[i], color=color, fill=True, alpha=0.3, label=region, warn_singular=False)
            else:
                axs[i].axvline(data.mean(), color=color, linestyle="--", label=f"{region}: {data.mean():.0f}")
        axs[i].set_title(item)
        axs[i].set_xlabel("Sell_Min (ISK)")
        axs[i].set_ylabel("Density")
        axs[i].legend()
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")
    plt.suptitle("Distribution Grid", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(plots_dir, "distribution_grid.png"))
    plt.close()

    df_all.to_csv(os.path.join(reports_dir, "full_timeseries.csv"), index=False)

    print(f"Market analysis complete. Results saved to: {save_dir}")

    return df_all
