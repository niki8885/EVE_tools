import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150

def _ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _roc(series: pd.Series, period: int = 1) -> pd.Series:
    return series.pct_change(periods=period) * 100

def _zscore(series: pd.Series, window: int = 24) -> pd.Series:
    return (series - series.rolling(window).mean()) / series.rolling(window).std()

def _stochastic_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_window=14, d_window=3) -> Tuple[pd.Series,pd.Series]:
    lowest = low.rolling(k_window).min()
    highest = high.rolling(k_window).max()
    k = 100 * (close - lowest) / (highest - lowest)
    d = k.rolling(d_window).mean()
    return k.fillna(50), d.fillna(50)

def _cci(high: pd.Series, low: pd.Series, close: pd.Series, n=20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - sma) / (0.015 * md)

def _macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def generate_market_plots(
    csv_path: str,
    output_dir: str = "plots",
    item_name: Optional[str] = "plex",
    time_col: str = "timestamp",
    price_col_candidates: List[str] = ["buyAvgFivePercent", "midPrice", "price", "close"],
    max_plots: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Read CSV, filter by item_name if present, compute indicators (when possible),
    and save many plots to output_dir. Returns dict with 'files' (list of saved PNGs)
    and 'summary_csv' (path to CSV describing produced plots).
    """
    csv_path = Path(csv_path)
    out = Path(output_dir)
    _ensure_dir(out)

    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"Timestamp column '{time_col}' not found in CSV.")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).set_index(time_col)

    if "item_name" in df.columns and item_name is not None:
        df = df[df["item_name"].astype(str).str.lower() == item_name.lower()]

    price_col = None
    for c in price_col_candidates:
        if c in df.columns:
            price_col = c
            break

    files = []
    meta_rows = []

    def save_fig(fig, name_suffix):
        filename = f"{item_name or 'item'}_{name_suffix}.png"
        path = out / filename
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        files.append(str(path))
        meta_rows.append({"file": str(path), "desc": name_suffix})

    # 1) Basic volume / orders time series
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    if "buyVolume" in df.columns:
        ax[0].plot(df.index, df["buyVolume"])
        ax[0].set_ylabel("buyVolume")
    if "sellVolume" in df.columns:
        ax[1].plot(df.index, df["sellVolume"])
        ax[1].set_ylabel("sellVolume")
    orders_plot = False
    if "buyOrders" in df.columns or "sellOrders" in df.columns:
        if "buyOrders" in df.columns:
            ax[2].plot(df.index, df["buyOrders"], label="buyOrders")
            orders_plot = True
        if "sellOrders" in df.columns:
            ax[2].plot(df.index, df["sellOrders"], label="sellOrders")
            orders_plot = True
        if orders_plot:
            ax[2].legend()
            ax[2].set_ylabel("orders")
    ax[-1].set_xlabel("time")
    save_fig(fig, "volumes_and_orders")

    # 2) Price series and moving averages (if price available)
    if price_col is not None:
        price = df[price_col].astype(float)
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(df.index, price, label="price")
        for span in [10, 24, 48, 120]:
            ax[0].plot(df.index, price.rolling(span).mean(), label=f"SMA_{span}")
        ax[0].legend()
        ax[0].set_title("Price and SMA")
        # EMA
        for span in [10, 24, 48, 120]:
            df[f"EMA_{span}"] = price.ewm(span=span, adjust=False).mean()
        for span in [10, 24, 48, 120]:
            ax[1].plot(df.index, df[f"EMA_{span}"], label=f"EMA_{span}")
        ax[1].legend()
        save_fig(fig, "price_sma_ema")
        # MACD + signal
        macd, signal_line, hist = _macd(price)
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax0.plot(df.index, price, label="price")
        ax0.set_title("Price")
        ax1.plot(df.index, macd, label="MACD")
        ax1.plot(df.index, signal_line, label="Signal")
        ax1.bar(df.index, hist, label="MACD_hist")
        ax1.legend()
        save_fig(fig, "macd_signal_hist")
        # RSI
        rsi = _rsi(price)
        fig = plt.figure(figsize=(10,3))
        plt.plot(df.index, rsi)
        plt.axhline(70, linestyle="--")
        plt.axhline(30, linestyle="--")
        plt.title("RSI")
        save_fig(fig, "rsi")
        # Bollinger bands
        ma20 = price.rolling(20).mean()
        sd20 = price.rolling(20).std()
        upper = ma20 + 2*sd20
        lower = ma20 - 2*sd20
        fig, ax = plt.subplots(1,1,figsize=(10,4))
        ax.plot(df.index, price, label="price")
        ax.plot(df.index, upper, label="Bollinger_Upper")
        ax.plot(df.index, lower, label="Bollinger_Lower")
        ax.legend()
        save_fig(fig, "bollinger_bands")
        # Momentum and ROC
        fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)
        momentum = price.diff(10)
        ax[0].plot(df.index, momentum)
        ax[0].set_title("Momentum (10)")
        roc1 = _roc(price, 1)
        ax[1].plot(df.index, roc1)
        ax[1].set_title("ROC (1)")
        save_fig(fig, "momentum_roc")
        # Z-score
        z = _zscore(price, window=24)
        fig = plt.figure(figsize=(10,3))
        plt.plot(df.index, z)
        plt.title("Z-score (window=24)")
        save_fig(fig, "zscore_price_24")
        # Volatility - rolling std
        vol = price.rolling(24).std()
        fig = plt.figure(figsize=(10,3))
        plt.plot(df.index, vol)
        plt.title("Rolling Std (volatility, window=24)")
        save_fig(fig, "rolling_std_vol_24")
        # Stochastic if H/L/C present or approximate
        if all(c in df.columns for c in ["high","low","close"]) :
            k,d = _stochastic_kd(df["high"], df["low"], df["close"])
            fig = plt.figure(figsize=(8,3))
            plt.plot(df.index, k, label="%K")
            plt.plot(df.index, d, label="%D")
            plt.legend()
            plt.title("Stochastic %K/%D")
            save_fig(fig, "stochastic_kd")
        else:
            approx_high = price * 1.001
            approx_low = price * 0.999
            approx_close = price
            k,d = _stochastic_kd(approx_high, approx_low, approx_close)
            fig = plt.figure(figsize=(8,3))
            plt.plot(df.index, k, label="%K (approx)")
            plt.plot(df.index, d, label="%D (approx)")
            plt.legend()
            plt.title("Stochastic %K/%D (approx)")
            save_fig(fig, "stochastic_kd_approx")
        # Williams %R (needs high/low/close)
        if all(c in df.columns for c in ["high","low","close"]):
            highest = df["high"].rolling(14).max()
            lowest = df["low"].rolling(14).min()
            willr = -100 * (highest - df["close"]) / (highest - lowest)
            fig = plt.figure(figsize=(8,3))
            plt.plot(df.index, willr)
            plt.title("Williams %R")
            save_fig(fig, "williams_r")
        else:
            highest = approx_high.rolling(14).max()
            lowest = approx_low.rolling(14).min()
            willr = -100 * (highest - approx_close) / (highest - lowest)
            fig = plt.figure(figsize=(8,3))
            plt.plot(df.index, willr)
            plt.title("Williams %R (approx)")
            save_fig(fig, "williams_r_approx")
        # CCI if possible
        if all(c in df.columns for c in ["high","low","close"]):
            cci = _cci(df["high"], df["low"], df["close"], n=20)
            fig = plt.figure(figsize=(8,3))
            plt.plot(df.index, cci)
            plt.title("CCI (20)")
            save_fig(fig, "cci")
    else:
        # No price column found -> make volume/indicator-only plots
        fig = plt.figure(figsize=(10,4))
        if "buyAvgFivePercent" in df.columns:
            plt.plot(df.index, df["buyAvgFivePercent"], label="buyAvgFivePercent (used as price)")
            plt.legend()
            save_fig(fig, "buyAvgFivePercent_as_price")
        else:
            plt.plot(df.index, df.iloc[:,0])
            plt.title("Fallback time series (first column)")
            save_fig(fig, "fallback_series")

    # 3) Volume-derived indicators and anomalies (if present)
    # VR, VPO, OI, LI, CompositeIndex, ΔVolume etc.
    vol_items = ["VR","VPO","LI","OI","CompositeIndex","DeltaVolume","ΔVolume","Δvolume","ΔVolume"]
    present_vol = [c for c in df.columns if c in vol_items or c.lower() in [v.lower() for v in vol_items]]
    if present_vol:
        fig, axes = plt.subplots(len(present_vol), 1, figsize=(8, 2.5*len(present_vol)), sharex=True)
        if len(present_vol) == 1:
            axes = [axes]
        for ax, col in zip(axes, present_vol):
            ax.plot(df.index, df[col])
            ax.set_ylabel(col)
        save_fig(fig, "volume_indicators")
    # 4) Outliers and thresholds
    out_cols = [c for c in df.columns if "Outlier" in c or "Threshold" in c or "Outliers" in c]
    if out_cols:
        fig, ax = plt.subplots(len(out_cols), 1, figsize=(8, 2.5*len(out_cols)), sharex=True)
        if len(out_cols) == 1:
            ax = [ax]
        for a, c in zip(ax, out_cols):
            a.plot(df.index, df[c])
            a.set_ylabel(c)
        save_fig(fig, "outliers_thresholds")

    # 5) Volume rolling means (VMA) if available
    vma_cols = [c for c in df.columns if c.startswith("VMA_") or c.startswith("vma_")]
    if vma_cols:
        fig, ax = plt.subplots(1,1,figsize=(10,4))
        for c in vma_cols:
            ax.plot(df.index, df[c], label=c)
        ax.legend()
        ax.set_title("VMA series")
        save_fig(fig, "vma_series")

    # 6) Heatmap / correlation of numeric columns (compact)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        fig, ax = plt.subplots(figsize=(10,8))
        im = ax.imshow(corr, aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        plt.colorbar(im, ax=ax)
        plt.title("Correlation matrix (numeric columns)")
        save_fig(fig, "correlation_matrix")

    # 7) Combined risk dashboard: volatility (rolling std), RSI, OI, buy/sell imbalance
    comps = []
    if price_col is not None:
        comps.append(("price", price))
        comps.append(("rsi", _rsi(price)))
        comps.append(("vol", price.rolling(24).std()))
    if "OI" in df.columns:
        comps.append(("OI", df["OI"]))
    if "buyVolume" in df.columns and "sellVolume" in df.columns:
        comps.append(("imbalance", (df["buyVolume"] - df["sellVolume"]) / (df["buyVolume"] + df["sellVolume"] + 1)))
    if comps:
        fig, axes = plt.subplots(len(comps), 1, figsize=(10, 2.6*len(comps)), sharex=True)
        for ax, (name, series) in zip(axes, comps):
            ax.plot(df.index, series)
            ax.set_ylabel(name)
        save_fig(fig, "risk_dashboard")

    # 8) Save metadata CSV
    meta_df = pd.DataFrame(meta_rows)
    summary_csv = out / f"{item_name or 'item'}_plots_summary.csv"
    meta_df.to_csv(summary_csv, index=False)

    # 9) VaR / CVaR risk metrics and distributions
    if price_col is not None:
        returns = price.pct_change().dropna()

        def var_cvar(series: pd.Series, alpha=0.05) -> Tuple[float, float]:
            """Compute parametric (normal) VaR and CVaR."""
            mu, sigma = series.mean(), series.std()
            var = mu - sigma * np.abs(np.percentile(np.random.randn(100000), alpha * 100))
            # CVaR for normal
            from scipy.stats import norm
            cvar = mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha
            return var, cvar

        def historical_var_cvar(series: pd.Series, alpha=0.05) -> Tuple[float, float]:
            """Empirical (historical) VaR and CVaR."""
            sorted_ret = np.sort(series)
            idx = int(alpha * len(sorted_ret))
            var = sorted_ret[idx]
            cvar = sorted_ret[:idx].mean() if idx > 0 else var
            return var, cvar

        def monte_carlo_var_cvar(series: pd.Series, alpha=0.05, n=10000) -> Tuple[float, float, np.ndarray]:
            """Monte Carlo simulation VaR/CVaR."""
            mu, sigma = series.mean(), series.std()
            sim = np.random.normal(mu, sigma, n)
            sorted_sim = np.sort(sim)
            idx = int(alpha * n)
            var = sorted_sim[idx]
            cvar = sorted_sim[:idx].mean() if idx > 0 else var
            return var, cvar, sim

        # Compute metrics
        param_var, param_cvar = var_cvar(returns)
        hist_var, hist_cvar = historical_var_cvar(returns)
        mc_var, mc_cvar, sim_returns = monte_carlo_var_cvar(returns)

        # Plot 1: Distribution of returns + VaR lines
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(returns, bins=50, density=True, alpha=0.6, label="Empirical returns")
        ax.axvline(param_var, color="r", linestyle="--", label=f"Param VaR(95%)={param_var:.4f}")
        ax.axvline(hist_var, color="g", linestyle="--", label=f"Hist VaR(95%)={hist_var:.4f}")
        ax.axvline(mc_var, color="b", linestyle="--", label=f"MC VaR(95%)={mc_var:.4f}")
        ax.legend()
        ax.set_title("Distribution of returns with VaR (95%)")
        save_fig(fig, "returns_distribution_var")

        # Plot 2: QQ plot (returns vs normal)
        from scipy import stats
        fig = plt.figure(figsize=(6, 6))
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title("QQ-plot (returns vs normal)")
        save_fig(fig, "qq_plot_returns")

        # Plot 3: Monte Carlo vs empirical distribution
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(returns, bins=50, density=True, alpha=0.6, label="Empirical")
        ax.hist(sim_returns, bins=50, density=True, alpha=0.4, label="Monte Carlo")
        ax.legend()
        ax.set_title("Empirical vs Monte Carlo return distributions")
        save_fig(fig, "empirical_vs_mc_distribution")

        # Plot 4: CVaR visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(returns, bins=50, density=True, alpha=0.6)
        ax.axvline(param_cvar, color="r", linestyle="--", label=f"Param CVaR(95%)={param_cvar:.4f}")
        ax.axvline(hist_cvar, color="g", linestyle="--", label=f"Hist CVaR(95%)={hist_cvar:.4f}")
        ax.axvline(mc_cvar, color="b", linestyle="--", label=f"MC CVaR(95%)={mc_cvar:.4f}")
        ax.legend()
        ax.set_title("Conditional Value at Risk (CVaR) 95%")
        save_fig(fig, "cvar_distribution")

        risk_summary = pd.DataFrame({
            "Metric": ["Parametric", "Historical", "Monte Carlo"],
            "VaR_95": [param_var, hist_var, mc_var],
            "CVaR_95": [param_cvar, hist_cvar, mc_cvar],
        })
        risk_csv = out / f"{item_name or 'item'}_risk_metrics.csv"
        risk_summary.to_csv(risk_csv, index=False)
        meta_rows.append({"file": str(risk_csv), "desc": "risk_metrics_csv"})


    return {"files": files, "summary_csv": str(summary_csv)}
