import argparse
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import joblib  # type: ignore
from ml.train_utils import read_sentiment_csv

FEATURE_ORDER = [
    # core oscillators and MACD family
    "rsi",
    "macd",
    "macd_hist",
    # moving averages and distances
    "sma_20",
    "sma_50",
    "sma_200",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    # trend and momentum proxies
    "trend_strength",
    "adx_strength",
    "signal_strength",
    # stochastic
    "stoch_k",
    "stoch_d",
    # bollinger
    "bb_upper",
    "bb_lower",
    "bb_middle",
    "bb_percent_b",
    # returns and vol
    "daily_return_pct",
    # sentiment
    "integrated_score",
    "sentiment_score",
]


def fetch_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise RuntimeError("No data from yfinance")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    def sma(series: pd.Series, n: int) -> pd.Series:
        return series.rolling(n).mean()

    def ema(series: pd.Series, n: int) -> pd.Series:
        return series.ewm(span=n).mean()

    def rsi(series: pd.Series, n: int = 14) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0).rolling(n).mean()
        down = (-delta.clip(upper=0)).rolling(n).mean().replace(0, 1e-9)
        rs = up / down
        return 100 - 100 / (1 + rs)

    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd_line, 9)
    macd_hist = macd_line - macd_signal

    # Simple trend/adx proxies
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    trend_strength = ((close - sma50) / sma50 * 100).fillna(0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    adx = true_range.rolling(14).mean().fillna(0)

    # Simple rules-based signal strength proxy
    signal_strength = (
        (close > sma20).astype(int) + (sma20 > sma50).astype(int) - (close < sma20).astype(int) - (sma20 < sma50).astype(int)
    )

    # stochastic
    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()
    stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low)).replace([np.inf, -np.inf], np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    # bollinger bands
    bb_mid = sma(close, 20)
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    feats = pd.DataFrame(
        {
            "rsi": rsi(close),
            "macd": macd_line,
            "macd_hist": macd_hist,
            "sma_20": sma20,
            "sma_50": sma50,
            "sma_200": sma200,
            "trend_strength": trend_strength,
            "adx_strength": adx,
            "signal_strength": signal_strength,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_middle": bb_mid,
        },
        index=df.index,
    ).fillna(0)

    return feats


def compute_sentiment_proxy(df: pd.DataFrame) -> pd.Series:
    # Placeholder: neutral score (replace with real sentiment later)
    return pd.Series(50.0, index=df.index)


def build_dataset(ticker: str, period: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = fetch_price_data(ticker, period)
    feats = compute_indicators(df)

    # Add daily_return_pct
    daily_ret = df["Close"].pct_change() * 100
    feats["daily_return_pct"] = daily_ret.fillna(0)
    # distances to MAs
    feats["dist_sma_20"] = ((df["Close"] - feats["sma_20"]) / feats["sma_20"]).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    feats["dist_sma_50"] = ((df["Close"] - feats["sma_50"]) / feats["sma_50"]).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    feats["dist_sma_200"] = ((df["Close"] - feats["sma_200"]) / feats["sma_200"]).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    # bb percent b
    denom = (feats["bb_upper"] - feats["bb_lower"]).replace(0, np.nan)
    feats["bb_percent_b"] = ((df["Close"] - feats["bb_lower"]) / denom).clip(lower=0, upper=1).fillna(0.5)

    # Sentiment features: prefer historical CSV if available, else proxy
    hist_senti = read_sentiment_csv(ticker, df.index)
    if hist_senti is not None:
        feats["integrated_score"] = hist_senti.fillna(50.0)
        feats["sentiment_score"] = hist_senti.fillna(0.0)
    else:
        feats["integrated_score"] = compute_sentiment_proxy(df)
        feats["sentiment_score"] = 0.0

    # Align and drop early NaNs
    feats = feats.fillna(0)

    # Labels: next-day direction (1 if next close > today close, else 0)
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)
    feats = feats.iloc[:-1]
    y = y.iloc[:-1]

    # Reorder columns
    # Ensure any missing columns are added as zeros to match FEATURE_ORDER
    for col in FEATURE_ORDER:
        if col not in feats.columns:
            feats[col] = 0.0
    feats = feats[FEATURE_ORDER]

    return feats, y


def train_and_save(tickers: List[str], period: str, out_dir: str) -> str:
    X_all = []
    y_all = []
    for t in tickers:
        X, y = build_dataset(t, period)
        X_all.append(X)
        y_all.append(y)
    X = pd.concat(X_all).astype(float)
    y = pd.concat(y_all).astype(int)

    # Time series split evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    accs = []
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1, class_weight="balanced")

    for train_idx, test_idx in tscv.split(X):
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = rf.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(y.iloc[test_idx], proba)
        except Exception:
            auc = float("nan")
        acc = accuracy_score(y.iloc[test_idx], pred)
        aucs.append(auc)
        accs.append(acc)

    rf.fit(X, y)

    os.makedirs(out_dir, exist_ok=True)
    artifact = {
        "model": rf,
        "feature_names": FEATURE_ORDER,
        "metadata": {
            "trained_at": datetime.utcnow().isoformat(),
            "tickers": tickers,
            "period": period,
            "cv_auc_mean": float(np.nanmean(aucs)) if len(aucs) else None,
            "cv_acc_mean": float(np.nanmean(accs)) if len(accs) else None,
        },
    }
    out_path = os.path.join(out_dir, f"rf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib")
    joblib.dump(artifact, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"])
    parser.add_argument("--period", default="2y")
    parser.add_argument("--out", default=os.path.join("ml", "models"))
    args = parser.parse_args()

    path = train_and_save(args.tickers, args.period, args.out)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()

