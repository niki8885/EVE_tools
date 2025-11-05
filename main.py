from info.items_requests import fetch_eve_item_data
from local_prices.goon_prices import get_all_prices
from analysis.local_prices_analysis import analyze_market_timeline
from trade.hub_trading import trading_analysis
from commodities_indices.indices import recive_indicies
from plex.plex_price import request_plex_info

if __name__ == "__main__":
    # input_path = "doctrine.csv"
    # output_path = "local_prices/doctrine_items.csv"
    # fetch_eve_item_data(input_path, output_path, volume=True)

    regions = ["C-J6MT", "UALX-3", "jita", "amarr", "dodixie"]
    regions_short = ["C-J6MT", "jita"]

    input_fuel_items_csv = "local_prices/fuel_items.csv"
    input_minerals_items_csv = "local_prices/minerals_items.csv"
    input_doctrine_items_csv = "local_prices/doctrine_items.csv"
    output_doctrine_dir = "local_prices/doctrine"
    output_fuel_dir = "local_prices/fuel"
    output_minerals_dir = "local_prices/minerals"

    get_all_prices(input_fuel_items_csv, output_fuel_dir, regions_short)
    get_all_prices(input_minerals_items_csv, output_minerals_dir, regions_short)
    get_all_prices(input_doctrine_items_csv, output_doctrine_dir, regions_short)

    input_fuel_dir = "local_prices/fuel"
    input_minerals_dir = "local_prices/minerals"

    analyze_market_timeline(input_fuel_dir, regions_short)
    analyze_market_timeline(input_minerals_dir, regions_short)

    trading_analysis("local_prices/fuel_items.csv", "local_prices/fuel", "jita", "C-J6MT")
    trading_analysis("local_prices/minerals_items.csv", "local_prices/minerals", "jita", "C-J6MT")
    trading_analysis("local_prices/doctrine_items.csv", "local_prices/doctrine", "jita", "C-J6MT")

    recive_indicies()

    request_plex_info()
