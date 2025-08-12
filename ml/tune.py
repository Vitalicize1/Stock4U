from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

from ml.features import build_features_from_state


def _build_state_for_index(ticker: str, hist: pd.DataFrame, i: int) -> dict:
    price = float(hist["Close"].iloc[i])
    prev = float(hist["Close"].iloc[i - 1]) if i > 0 else price
    return {
        "ticker": ticker,
        "timeframe": "1d",
        "data": {"price_data": {"current_price": price, "previous_close": prev}},
        "technical_analysis": {"technical_score": 50},
        "sentiment_integration": {"integrated_analysis": {"integrated_score": 50}},
    }


def _load_hist(ticker: str, period: str) -> pd.DataFrame:
    hist = yf.Ticker(ticker).history(period=period, interval="1d")
    if hist is None or hist.empty:
        raise RuntimeError(f"No data for {ticker} {period}")
    hist = hist.rename_axis("Date").reset_index().set_index("Date")
    return hist


def _make_dataset(tickers: List[str], period: str) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    rows = []
    feature_names: List[str] = []
    for t in tickers:
        h = _load_hist(t, period)
        for i in range(60, len(h) - 1):  # warmup of 60 bars
            state = _build_state_for_index(t, h, i)
            X, feats = build_features_from_state(state)
            if not feature_names:
                feature_names = list(feats.keys())
            y = 1 if (float(h["Close"].iloc[i + 1]) / float(h["Close"].iloc[i]) - 1.0) > 0 else 0
            rows.append({**feats, "y": y})
    df = pd.DataFrame(rows)
    y = df.pop("y").astype(int)
    X = df[feature_names].astype(float)
    return X, y, feature_names


def tune_rf(X: pd.DataFrame, y: pd.Series, n_trials: int = 30) -> dict:
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        aucs = []
        for tr_idx, te_idx in tscv.split(X):
            clf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", random_state=42, **params)
            clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            proba = clf.predict_proba(X.iloc[te_idx])[:, 1]
            try:
                aucs.append(roc_auc_score(y.iloc[te_idx], proba))
            except Exception:
                aucs.append(float("nan"))
        return float(np.nanmean(aucs))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params | {"best_value": study.best_value}


def save_model(X: pd.DataFrame, y: pd.Series, feature_names: List[str], params: dict, out_dir: str) -> str:
    clf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", random_state=42, **{k: v for k, v in params.items() if k != "best_value"})
    clf.fit(X, y)
    os.makedirs(out_dir, exist_ok=True)
    artifact = {
        "model": clf,
        "feature_names": feature_names,
        "metadata": {
            "trained_at": datetime.utcnow().isoformat(),
            "params": params,
        },
    }
    path = os.path.join(out_dir, f"rf_tuned_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib")
    joblib.dump(artifact, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune ML hyperparameters and save best model")
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA,GOOGL", help="Comma-separated list of tickers for dataset")
    parser.add_argument("--period", default="1y", help="History period, e.g., 1y")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials")
    parser.add_argument("--outdir", default=os.path.join("ml", "models"))
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    X, y, feature_names = _make_dataset(tickers, args.period)
    params = tune_rf(X, y, n_trials=args.trials)
    path = save_model(X, y, feature_names, params, args.outdir)
    print(f"Saved tuned model to {path}")


if __name__ == "__main__":
    main()


