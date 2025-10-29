import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

DELIVERY_FEE = 1200  # ISK per 1 m³

def trading_analysis(items_path: str, prices_path: str, region1: str = None, region2: str = None):
    """
    Analyze trade opportunities between two EVE Online regions.
    Generates CSV reports and visualizations showing profit potential for items.
    """

    # === 1. Load item list ===
    items_df = pd.read_csv(items_path)

    # === 2. Determine the trade subfolder name (e.g., 'fuel', 'minerals') ===
    trade_name = os.path.basename(os.path.normpath(prices_path))
    base_trade_dir = os.path.join("trade", trade_name)
    reports_dir = os.path.join(base_trade_dir, "reports")
    plots_dir = os.path.join(base_trade_dir, "plots")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # === 3. Detect available regions ===
    first_item = items_df.iloc[0]["name"]
    regions = sorted(os.listdir(os.path.join(prices_path, first_item)))

    if len(regions) < 2:
        raise ValueError("Less than two regions found in price data structure!")

    # === 4. Region selection ===
    if region1 is None or region2 is None:
        print("Available regions:")
        for i, r in enumerate(regions, 1):
            print(f"{i}. {r}")
        if region1 is None:
            region1 = regions[int(input("Enter the number for the first region: ")) - 1]
        if region2 is None:
            region2 = regions[int(input("Enter the number for the second region: ")) - 1]

    if region1 == region2:
        raise ValueError("Regions must be different!")

    print(f"Selected regions: {region1} and {region2}")

    # === 5. Description of scenarios ===
    scenario_labels = {
        1: f"Buy({region1}) → Buy({region2})",
        2: f"Sell({region1}) → Buy({region2})",
        3: f"Buy({region1}) → Sell({region2})",
        4: f"Sell({region1}) → Sell({region2})"
    }

    results = {1: [], 2: [], 3: [], 4: []}

    # === 6. Main item loop ===
    for _, row in items_df.iterrows():
        name, type_id, volume = row["name"], row["type_id"], row["volume_m3"]

        try:
            df1 = pd.read_csv(os.path.join(prices_path, name, f"{region1}.csv"))
            df2 = pd.read_csv(os.path.join(prices_path, name, f"{region2}.csv"))
        except FileNotFoundError:
            print(f"⚠️ Missing data for {name}")
            continue

        # Use latest price data
        p1 = df1.iloc[-1]
        p2 = df2.iloc[-1]

        buy1, sell1 = p1["Buy_Max"], p1["Sell_Min"]
        buy2, sell2 = p2["Buy_Max"], p2["Sell_Min"]

        fee_cost = DELIVERY_FEE * volume

        # === 7. Profit scenarios ===
        res1 = buy2 - (buy1 + fee_cost)
        res2 = buy2 - (sell1 + fee_cost)
        res3 = sell2 - (buy1 + fee_cost)
        res4 = sell2 - (sell1 + fee_cost)

        # === 8. ROI calculations ===
        roi1 = res1 / (buy1 + fee_cost) if buy1 + fee_cost > 0 else 0
        roi2 = res2 / (sell1 + fee_cost) if sell1 + fee_cost > 0 else 0
        roi3 = res3 / (buy1 + fee_cost) if buy1 + fee_cost > 0 else 0
        roi4 = res4 / (sell1 + fee_cost) if sell1 + fee_cost > 0 else 0

        data = {
            "Item": name,
            "TypeID": type_id,
            "Volume_m3": volume,
            "Region_1": region1,
            "Region_2": region2,
            "Buy1": buy1,
            "Sell1": sell1,
            "Buy2": buy2,
            "Sell2": sell2,
            "Fee_per_m3": DELIVERY_FEE,
            "Profit_1": res1,
            "Profit_2": res2,
            "Profit_3": res3,
            "Profit_4": res4,
            "ROI_1": roi1,
            "ROI_2": roi2,
            "ROI_3": roi3,
            "ROI_4": roi4,
            "Timestamp": datetime.now()
        }

        for i in range(1, 5):
            results[i].append(data)

    # === 9. Save reports and visualizations ===
    for i in range(1, 5):
        df = pd.DataFrame(results[i])
        df = df.sort_values(by=f"Profit_{i}", ascending=False)

        # Save CSV report
        report_name = (
            f"trade_report_{i}_{region1}_vs_{region2}_"
            f"{scenario_labels[i].replace(' ', '').replace('→', 'to')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        report_path = os.path.join(reports_dir, report_name)
        df.to_csv(report_path, index=False)

        # --- Top 10 profit bar chart ---
        top10 = df.head(10)
        plt.figure(figsize=(10, 5))
        plt.bar(top10["Item"], top10[f"Profit_{i}"], color="skyblue", edgecolor="black")
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Top 10 Profitable Items\n{scenario_labels[i]} ({region1} → {region2})")
        plt.ylabel("Profit (ISK)")
        plt.xlabel("Item")
        plt.tight_layout()
        plot_name = f"top10_{region1}_vs_{region2}_{scenario_labels[i].replace(' ', '_').replace('→', 'to')}.png"
        plot_path = os.path.join(plots_dir, plot_name)
        plt.savefig(plot_path)
        plt.close()

        # --- ROI vs Profit scatter plot ---
        plt.figure(figsize=(8, 6))
        plt.scatter(df[f"Profit_{i}"], df[f"ROI_{i}"], alpha=0.6, edgecolor='k')
        plt.title(f"Profit vs ROI — {scenario_labels[i]}\n{region1} → {region2}")
        plt.xlabel("Profit (ISK)")
        plt.ylabel("ROI (Return on Investment)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        scatter_name = f"roi_vs_profit_{region1}_vs_{region2}_{scenario_labels[i].replace(' ', '_').replace('→', 'to')}.png"
        scatter_path = os.path.join(plots_dir, scatter_name)
        plt.savefig(scatter_path)
        plt.close()

    print(f"✅ All trade reports and plots saved in '{base_trade_dir}'.")
