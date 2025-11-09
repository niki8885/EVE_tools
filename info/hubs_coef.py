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

def allocate_item_with_prices(regions_names: List[str], item_name: str, total_amount: int) -> Dict[str, Dict[str, float]]:
    type_id = get_type_id_by_name(item_name)
    region_ids = [get_region_id_by_name(r) for r in regions_names]

    region_volumes = {}
    min_sell_prices = {}

    for name, rid in zip(regions_names, region_ids):
        hist_url = f"{BASE_URL}/market/history/{rid}/{type_id}"
        try:
            resp = requests.get(hist_url, timeout=10)
            resp.raise_for_status()
            history = resp.json()
            total_volume = sum(day['volume'] for day in history)
            region_volumes[name] = total_volume
        except Exception as e:
            print(f"Error fetching history for region {name}: {e}")
            region_volumes[name] = 0

        stats_url = f"{BASE_URL}/market/stats/{rid}/{type_id}"
        try:
            resp = requests.get(stats_url, timeout=10)
            resp.raise_for_status()
            stats = resp.json()
            min_sell_prices[name] = stats.get('sellAvgFivePercent', 0)
        except Exception as e:
            print(f"Error fetching stats for region {name}: {e}")
            min_sell_prices[name] = 0

    if all(v == 0 for v in region_volumes.values()):
        equal_amount = total_amount // len(regions_names)
        allocation = {name: {"amount": equal_amount, "min_sell_price": min_sell_prices[name]} for name in regions_names}
        return allocation

    total_volume_sum = sum(region_volumes.values())
    allocation = {}
    remaining = total_amount

    for i, name in enumerate(regions_names):
        if i == len(regions_names) - 1:
            amt = remaining
        else:
            amt = round(total_amount * (region_volumes[name] / total_volume_sum))
            remaining -= amt
        allocation[name] = {"amount": amt, "min_sell_price": min_sell_prices[name]}

    return allocation


regions = ["The Forge", "Molden Heath", "Heimatar", "Metropolis", "Genesis", "Sinq Laison","G-R00031"]
item = "Damage Control II"
total_amount = 200

allocation = allocate_item_with_prices(regions, item, total_amount)
for region, data in allocation.items():
    print(f"{region}: {data['amount']} units, min sell price: {data['min_sell_price']}")
