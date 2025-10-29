# EVE Online Market Price Analyzer

This project fetches, analyzes, and visualizes EVE Online market prices for various items across multiple regions.  
It supports automatic scraping, accumulation of historical data, and generating detailed trade and market reports.

## Features

- Fetch item prices from multiple regions:
  - C-J6MT
  - UALX-3
  - Jita
  - Amarr
  - Dodixie
- Store historical prices in structured folders for different categories:
  - Fuel
  - Minerals
  - Doctrine items
- Generate analysis with plots and reports:
  - Average Sell Price Grid
  - Average Buy Price Grid
  - Spread Grid
  - Distribution Grid
  - Top Profitable Items per scenario
  - Profit vs ROI scatter plots
- Save combined CSV reports for further analysis.
- Automatically generates `trade/*` folders with scenario-based profit analysis and visualizations.

## Project Structure
```
├── main.py # Entry point for scraping and updating prices
├── local_prices/ # Historical price data
│ ├── fuel/
│ │ ├── Oxygen/
│ │ │ ├── C-J6MT.csv
│ │ │ ├── jita.csv
│ │ │ └── ...
│ │ └── ...
│ ├── minerals/
│ │ └── ...
│ └── doctrine/
│ └── ...
├── analysis/ # Generated plots and reports
│ ├── fuel/
│ │ ├── plots/
│ │ └── reports/
│ ├── minerals/
│ │ ├── plots/
│ │ └── reports/
│ └── doctrine/
│ ├── plots/
│ └── reports/
├── trade/ # Scenario-based trade reports and visualizations
│ ├── fuel/
│ │ ├── plots/
│ │ └── reports/
│ ├── minerals/
│ │ ├── plots/
│ │ └── reports/
│ └── doctrine/
│ ├── plots/
│ └── reports/
├── local_prices/goon_prices.py # Scripts for fetching and updating item prices
├── analysis/local_prices_analysis.py # Scripts for analyzing historical market data
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
