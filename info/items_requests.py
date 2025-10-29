import pandas as pd
import requests
from tqdm import tqdm
import time


def fetch_eve_item_data(input_csv: str, output_csv: str, volume: bool = True):
    """
    Fetch EVE Online item type IDs (via fuzzwork API) and optional volume (via ESI API).
    """
    items_df = pd.read_csv(input_csv)
    if 'name' not in items_df.columns:
        raise ValueError("The input CSV must contain a 'name' column.")

    type_ids = []
    volumes = [] if volume else None

    for name in tqdm(items_df['name'], desc="Fetching type IDs"):
        type_id = None
        vol = None

        try:
            lookup_url = f"https://www.fuzzwork.co.uk/api/typeid.php?typename={name}"
            resp = requests.get(lookup_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if 'typeID' in data:
                type_id = data['typeID']

                if volume:
                    type_url = f"https://esi.evetech.net/latest/universe/types/{type_id}/?datasource=tranquility"
                    type_resp = requests.get(type_url, timeout=10)
                    type_resp.raise_for_status()
                    type_data = type_resp.json()
                    vol = type_data.get('volume', None)
            else:
                print(f"No match found for '{name}'")

        except Exception as e:
            print(f"Error fetching '{name}': {e}")

        type_ids.append(type_id)
        if volume:
            volumes.append(vol)

        time.sleep(0.2)

    items_df['type_id'] = type_ids
    if volume:
        items_df['volume_m3'] = volumes

    items_df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")
