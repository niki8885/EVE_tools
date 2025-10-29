import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
from datetime import datetime

BASE_URL = "https://appraise.gnf.lt/item/{}#{}"


def parse_prices_from_page(type_id, region):
    url = BASE_URL.format(type_id, region)
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        tab = soup.find("div", id=region)
        if not tab:
            print(f"Region tab {region} not found for item {type_id}")
            return None

        tables = tab.find_all("table")
        if not tables or len(tables) < 2:
            print(f"Missing expected tables for {region} item {type_id}")
            return None

        def extract_table_data(table):
            result = {}
            for row in table.find_all("tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    key = th.text.strip()
                    val = td.text.strip().replace(",", "").replace(" ISK", "")
                    try:
                        val = float(val)
                    except:
                        val = None
                    result[key] = val
            return result

        sell_data = extract_table_data(tables[0])
        buy_data = extract_table_data(tables[1])

        combined = {}
        for k, v in sell_data.items():
            combined[f"Sell_{k}"] = v
        for k, v in buy_data.items():
            combined[f"Buy_{k}"] = v

        return combined

    except Exception as e:
        print(f"Error fetching {type_id} {region}: {e}")
        return None


def update_prices_for_item(item_name, type_id, regions, output_dir):
    item_dir = os.path.join(output_dir, item_name.replace("/", "_"))
    os.makedirs(item_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for region in regions:
        print(f"→ {item_name} ({type_id}) in {region}")
        prices = parse_prices_from_page(type_id, region)
        if prices:
            prices["Item"] = item_name
            prices["TypeID"] = type_id
            prices["Region"] = region
            prices["Timestamp"] = timestamp

            output_file = os.path.join(item_dir, f"{region}.csv")

            if os.path.exists(output_file):
                old_df = pd.read_csv(output_file)
                new_df = pd.DataFrame([prices])
                combined_df = pd.concat([old_df, new_df], ignore_index=True)
                combined_df.to_csv(output_file, index=False)
            else:
                pd.DataFrame([prices]).to_csv(output_file, index=False)

        else:
            print(f"No data for {item_name} in {region}")

        time.sleep(0.3)


def get_all_prices(input_items_csv, output_dir, regions):
    df_items = pd.read_csv(input_items_csv)

    for _, row in df_items.iterrows():
        item_name = row["name"]
        type_id = int(row["type_id"])
        update_prices_for_item(item_name, type_id, regions, output_dir)
