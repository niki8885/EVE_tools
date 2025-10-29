# EVE Online Market Price Analyzer

This project fetches and analyzes EVE Online market prices for various items across multiple regions.  
It supports automatic scraping, accumulation of historical data, and generating visual reports.

## Features

- Fetch item prices from multiple regions:
  - C-J6MT
  - UALX-3
  - Jita
  - Amarr
  - Dodixie
- Store historical prices in structured folders.
- Generate analysis with plots:
  - Average Sell Price Grid
  - Average Buy Price Grid
  - Spread Grid
  - Distribution Grid
- Save combined CSV reports for further analysis.

## Project Structure
```
├── main.py # Entry point for scraping and updating prices
├── local_prices/ # Historical price data
│ └── fuel/
│ ├── Oxygen/ # Item folder
│ │ ├── C-J6MT.csv # Prices in C-J6MT region
│ │ ├── jita.csv # Prices in Jita region
│ │ └── ... # Other regions
│ └── ... # Other items
├── analysis/ # Generated plots and reports
│ └── fuel/
│ ├── plots/ # Visualizations: grids, distributions, spread
│ └── reports/ # Combined CSV reports and analysis outputs
├── local_prices/goon_prices.py # Scripts for fetching and updating item prices
├── analysis/local_prices_analysis.py# Scripts for analyzing historical market data
├── requirements.txt # Python dependencies
└── LICENSE.md # GNU GPL v3 License
```

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
````

2. Run price scraping:

```bash
python main.py
```

## Regions

```python
regions = ["C-J6MT", "UALX-3", "jita", "amarr", "dodixie"]
```

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE.md](LICENSE.md).


