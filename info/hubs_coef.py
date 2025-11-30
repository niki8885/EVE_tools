import requests
import pandas as pd
from typing import List, Dict

BASE_URL = "https://evetycoon.com/api/v1"

def get_region_id_by_name(region_name: str) -> int:
    url = f"{BASE_URL}/market/regions"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    regions = resp.json()
    for region in regions:
        if region['name'].lower() == region_name.lower():
            return region['id']
    raise ValueError(f"Region '{region_name}' not found")

def get_type_id_by_name(item_name: str) -> int:
    url = f"https://www.fuzzwork.co.uk/api/typeid.php?typename={item_name}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if 'typeID' in data:
        return data['typeID']
    raise ValueError(f"Item '{item_name}' not found")

def allocate_item_profit_based(regions_names: List[str], item_name: str, total_amount: int, buy_price: float) -> Dict[str, Dict[str, float]]:
    type_id = get_type_id_by_name(item_name)
    region_ids = [get_region_id_by_name(r) for r in regions_names]

    region_scores = {}
    min_sell_prices = {}

    for name, rid in zip(regions_names, region_ids):
        stats_url = f"{BASE_URL}/market/stats/{rid}/{type_id}"
        try:
            resp = requests.get(stats_url, timeout=10)
            resp.raise_for_status()
            stats = resp.json()
            sell_price = stats.get('sellAvgFivePercent', 0)
            min_sell_prices[name] = sell_price
        except Exception as e:
            print(f"Error fetching stats for region {name}: {e}")
            min_sell_prices[name] = 0
            sell_price = 0

        hist_url = f"{BASE_URL}/market/history/{rid}/{type_id}"
        try:
            resp = requests.get(hist_url, timeout=10)
            resp.raise_for_status()
            history = resp.json()
            total_volume = sum(day['volume'] for day in history)
        except Exception as e:
            print(f"Error fetching history for region {name}: {e}")
            total_volume = 0

        expected_profit = sell_price - buy_price
        score = expected_profit * total_volume
        if score > 0:
            region_scores[name] = score

    if not region_scores:
        print("No profitable regions found. Nothing to allocate.")
        return {}

    total_score = sum(region_scores.values())
    allocation = {}
    remaining = total_amount

    for i, (name, score) in enumerate(region_scores.items()):
        if i == len(region_scores) - 1:
            amt = remaining
        else:
            amt = round(total_amount * (score / total_score))
            remaining -= amt
        expected_profit = (min_sell_prices[name] - buy_price) * amt
        allocation[name] = {
            "amount": amt,
            "min_sell_price": min_sell_prices[name],
            "expected_profit": expected_profit
        }

    return allocation


def process_items(
    regions: List[str],
    items: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    items = {
        "Item Name": {"amount": int, "buy_price": float}
    }
    """
    all_rows = []

    for item_name, params in items.items():
        amount = params["amount"]
        buy_price = params["buy_price"]
        print(f"\nProcessing {item_name} ({amount} units)...")
        allocation = allocate_item_profit_based(regions, item_name, amount, buy_price)
        for region, data in allocation.items():
            all_rows.append({
                "item": item_name,
                "region": region,
                "amount": data["amount"],
                "min_sell_price": data["min_sell_price"],
                "expected_profit": data["expected_profit"]
            })

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(by=["item", "expected_profit"], ascending=[True, False])
    return df


def items_to_hubs(
    regions: List[str],
    items: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    df_allocations = process_items(regions, items)
    print("\n--- Allocation Table ---")
    print(df_allocations.to_string(index=False))






regions = ["Molden Heath", "Heimatar", "Metropolis", "Genesis", "Sinq Laison","G-R00031","Placid"]
regions_production = ["The Forge","Molden Heath", "Heimatar", "Metropolis", "Genesis", "Sinq Laison","G-R00031","Placid"]

items = {
    "Strip Miner I": {"amount": 40, "buy_price": 4235000},
    "650mm Artillery Cannon II": {"amount": 75, "buy_price": 1300000},
    "Neural Lace 'Blackglass' Net Intrusion 920-40": {"amount": 3, "buy_price": 98990000},
    "Sisters Core Scanner Probe": {"amount": 224, "buy_price": 754150},
    "Small Ancillary Armor Repairer": {"amount": 20, "buy_price": 25600}

}

items_2 = {
    "Curator II": {"amount": 100, "buy_price": 1520000},
    "Light Armor Maintenance Bot II": {"amount": 360, "buy_price": 305000},
    "Prototype Cloaking Device I": {"amount": 300, "buy_price": 1781000},
    "Capacitor Power Relay II": {"amount": 500, "buy_price": 340000},
    "Gravimetric ECM II": {"amount": 300, "buy_price": 997000},
    "Gyrostabilizer II": {"amount": 280, "buy_price": 758000},
    "Nanofiber Internal Structure II": {"amount": 100, "buy_price": 87000},
    "Tracking Computer II": {"amount": 520, "buy_price": 965000},
    "Tracking Enhancer II": {"amount": 230, "buy_price": 680000}
}

items_to_hubs(regions, items)
items_to_hubs(regions, items_2)
