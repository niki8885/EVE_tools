import requests
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

regions_production = ["The Forge", "Molden Heath", "Heimatar", "Metropolis", "Genesis", "Sinq Laison"]
regions = ["Molden Heath", "Heimatar", "Metropolis", "Genesis", "Sinq Laison","G-R00031"]
item = "Damage Control II"
total_amount = 200
buy_price = 393700.0

allocation = allocate_item_profit_based(regions, item, total_amount, buy_price)
for region, data in allocation.items():
    print(f"{region}: {data['amount']} units, min sell price: {data['min_sell_price']}, expected profit: {data['expected_profit']}")