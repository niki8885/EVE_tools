from info.items_requests import fetch_eve_item_data
from local_prices.goon_prices import get_all_prices
from analysis.local_prices_analysis import analyze_market_timeline

if __name__ == "__main__":
    # input_path = "fuel.csv"
    # output_path = "local_prices/fuel_items.csv"
    # fetch_eve_item_data(input_path, output_path, volume=True)

    regions = ["C-J6MT", "UALX-3", "jita", "amarr", "dodixie"]

    input_items_csv = "local_prices/fuel_items.csv"
    output_dir = "local_prices/fuel"

    get_all_prices(input_items_csv, output_dir, regions)

    input_dir = "local_prices/fuel"

    analyze_market_timeline(input_dir, regions)
