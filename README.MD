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

## **Extended Module: Commodities Indices**

The `commodities_indices` module introduces **automated index computation** and **analytics** for EVE Online commodity groups (e.g., fuel, minerals, ice, etc.).
It integrates directly with the [EVE Tycoon API](https://evetycoon.com/api) and produces historical records with computed metrics and visualizations.

### **Data Flow**

1. Reads input from CSVs (e.g., `/data/minerals.csv`):

   ```
   name,type_id,volume,weight
   Wasp II,2436,3724.0,5.894e-06
   Praetor II,2195,7283.0,1.152e-05
   ```
2. Fetches market stats from:

   ```
   https://evetycoon.com/api/v1/market/stats/{regionId}/{typeId}
   ```
3. Calculates and appends indicator values to:

   ```
   /prices/<category>/index.csv
   ```
4. Generates plots and grid visualizations in:

   ```
   /plots/<category>/
   ```

---

## **Computed Indicators**

| Indicator                          | Description                                            |
| ---------------------------------- | ------------------------------------------------------ |
| **Price Index (PI)**               | Weighted average of `sellAvgFivePercent` prices        |
| **Volume Index (VI)**              | Total `sellVolume` across all items                    |
| **Volatility**                     | Rolling standard deviation of the price index          |
| **Top-3 Share**                    | Share of top 3 items by weight in the index            |
| **Weight Concentration (H-index)** | Herfindahl-Hirschman concentration index               |
| **Entropy of Weights**             | Market diversification entropy                         |
| **SMA-10**                         | 10-period Simple Moving Average of PI                  |
| **RSI**                            | 14-period Relative Strength Index for price momentum   |
| **MACD**                           | 12/26-period MACD signal indicator                     |
| **Liquidity Index**                | Average ratio of total volume to volatility            |
| **VaR / CVaR**                     | Value-at-Risk and Conditional Value-at-Risk of returns |

## **Mathematical Definitions**

* **SMA(n)** = average of last *n* values
* **RSI** = 100 − 100 / (1 + RS), where RS = avg(gains) / avg(losses)
* **MACD** = EMA(12) − EMA(26), Signal = EMA(9) of MACD
* **Volatility** = std(returns over last *n* periods)
* **Entropy** = −∑(wᵢ * log(wᵢ))
* **Herfindahl Index (H)** = ∑(wᵢ²)
* **VaR (α)** = quantile(return, α)
* **CVaR (α)** = mean(return < VaR(α))

---

## **Visualization Grids**

Two main visualization grids are generated for each commodity category (e.g., `minerals`, `ice`):

### **1. Index Dynamics Grid**

Includes:

* Price Index (PI)
* Volume Index (VI)
* Volatility
* RSI
* MACD
* SMA-10
  All shown as line plots on a synchronized time axis.

### **2. Distribution Grid**

Includes:

* **Price Distribution** — with a vertical line marking current price
* **Volume Distribution** — with current volume marked
* **Value-at-Risk (VaR)** and **Conditional VaR (CVaR)** plots

Saved as:

```
plots/<category>/<category>_index_grid.png
plots/<category>/<category>_distribution_grid.png
```
---

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
├── commodities_indices/          # Commodity index analytics module
│   ├── indices.py               # Core index builder and analyzer
│   ├── data/                    # Source item lists
│   ├── prices/                  # Historical index CSVs
│   └── plots/                   # Generated analysis visuals
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
