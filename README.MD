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

## **Commodities Indices**

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

## **PLEX Market Monitor**

The `plex` module introduces **automated market tracking** and **indicator computation** for the PLEX item (`type_id = 44992`).
It collects real-time data directly from the [EVE Tycoon API](https://evetycoon.com/api/v1/market/stats/19000001/44992), stores it as a time series, and computes a broad set of **technical and liquidity indicators**.

### **Data Flow**

1. Fetches latest PLEX market statistics from:

   ```
   https://evetycoon.com/api/v1/market/stats/{regionId}/{typeId}
   ```

   (default region: `19000001`)
2. Appends the fetched data to:

   ```
   plex/data/price.csv
   ```
3. Computes derived indicators and metrics:

   * Order and volume-based indicators
   * Momentum and volatility metrics
   * Moving averages and oscillators
4. Exports enriched historical data for long-term monitoring and visualization.

---

### **Computed Market Indicators**

| Category               | Indicator                         | Description                                            |
| ---------------------- | --------------------------------- | ------------------------------------------------------ |
| **Order Flow**         | **Order Imbalance (OI)**          | Ratio of buy vs sell volume imbalance                  |
|                        | **Volume Ratio (VR)**             | `buyVolume / sellVolume`                               |
|                        | **Liquidity Index (LI)**          | `(buyOrders + sellOrders) / (buyVolume + sellVolume)`  |
|                        | **Volume per Order (VPO)**        | Average traded volume per active order                 |
| **Volume Dynamics**    | **ΔVolume**                       | Change in total traded volume                          |
|                        | **Rate of Change (ROC)**          | Percentage change of PLEX price                        |
|                        | **Cumulative OI (COI)**           | Cumulative order imbalance over time                   |
|                        | **Rolling Volatility**            | Standard deviation of OI over a moving window          |
| **Market Sentiment**   | **Market Sentiment Index (MSI)**  | Normalized OI between −1 and +1                        |
| **Trend Indicators**   | **SMA (10, 24, 48, 120)**         | Simple Moving Averages of PLEX price                   |
|                        | **EMA (10, 24, 48, 120)**         | Exponential Moving Averages                            |
|                        | **MACD / Signal**                 | Moving Average Convergence Divergence                  |
|                        | **RSI (14)**                      | Relative Strength Index for momentum                   |
| **Volatility Metrics** | **Rolling Std**                   | Rolling standard deviation of price                    |
|                        | **Bollinger Bands**               | Price envelopes based on volatility (±2σ)              |
|                        | **Coefficient of Variation (CV)** | Ratio of std to mean (volatility normalization)        |
| **Volume Analytics**   | **VMA (10, 24, 48, 120)**         | Volume Moving Averages                                 |
|                        | **VO**                            | Volume Oscillator (EMA short vs long)                  |
| **Oscillators**        | **Stochastic (%K, %D)**           | Relative position of current price within recent range |

---

### **Data Output Example**

```
timestamp,buyVolume,sellVolume,buyOrders,sellOrders,buyAvgFivePercent,sellAvgFivePercent,OI,VR,LI,VPO,RSI,MACD,MSI,SMA_buyAvgFivePercent_24,...
2025-11-05 10:00:00,215581,831898,203,451,2477927.73,2598726.21,-0.58,0.26,0.0007,1520.9,47.2,-130.3,-0.42,2521000.54,...
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
├── plex/         
│   ├── plex_prices.py     
│   ├── plex_analysis.py       
│   ├── data/                    
│   ├── models/                  
│   └── plots/    
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
