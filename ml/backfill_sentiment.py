from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional

import requests

from ml.sentiment_logger import append_sentiment_sample
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(override=False)
except Exception:
    pass


def backfill_recent_proxy(tickers: List[str], days: int) -> None:
    """Backfill by reusing today's overall sentiment for the past N days (proxy)."""
    from agents.tools.sentiment_analyzer_tools import compute_overall_sentiment

    for t in tickers:
        try:
            res = compute_overall_sentiment.invoke({"ticker": t})
            overall = (res.get("sentiment_analysis", {}) or {}).get("overall_sentiment", {})
            score = float(overall.get("sentiment_score", 0.0) or 0.0)
            for i in range(days):
                ts = (datetime.utcnow() - timedelta(days=i)).isoformat()
                append_sentiment_sample(t, ts, score)
            print(f"Proxy backfilled {t} for {days} days with score={score:.3f}")
        except Exception as e:
            print(f"Failed proxy backfill for {t}: {e}")


def fetch_newsapi_day(ticker: str, company_name: Optional[str], day: datetime, page_size: int = 50) -> List[dict]:
    """Fetch news for a single UTC day using NewsAPI if configured. Returns list of articles."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        since = day.strftime("%Y-%m-%d")
        until = (day + timedelta(days=1)).strftime("%Y-%m-%d")
        terms = [ticker]
        if company_name:
            terms.append(company_name)
        collected: List[dict] = []
        for term in terms:
            params = {
                "q": f"{term} stock",
                "apiKey": api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "from": since,
                "to": until,
            }
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok":
                    for a in data.get("articles", []):
                        collected.append({
                            "title": a.get("title"),
                            "description": a.get("description"),
                            "publishedAt": a.get("publishedAt"),
                            "source": (a.get("source") or {}).get("name"),
                        })
        return collected
    except Exception:
        return []


def score_texts(texts: List[str]) -> float:
    """Use the built-in analyzer tool to compute average sentiment score for a list of texts."""
    try:
        from agents.tools.sentiment_analyzer_tools import analyze_text_sentiment
        res = analyze_text_sentiment.invoke({"texts": texts})
        return float(res.get("average", 0.0) or 0.0)
    except Exception:
        return 0.0


def backfill_historical_news(tickers: List[str], days: int, company_name: Optional[str] = None) -> None:
    """Backfill by fetching NewsAPI articles per day and scoring them with the heuristic analyzer.

    If NEWS_API_KEY is not set, this function is a no-op.
    """
    if not os.getenv("NEWS_API_KEY"):
        print("NEWS_API_KEY not set; skipping historical news backfill.")
        return

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    for t in tickers:
        for i in range(days):
            day = today - timedelta(days=i)
            articles = fetch_newsapi_day(t, company_name, day)
            texts = [f"{a.get('title','')} {a.get('description','')}" for a in articles]
            score = score_texts(texts) if texts else 0.0
            ts = day.isoformat()
            append_sentiment_sample(t, ts, score)
        print(f"Historical news backfilled {t} for {days} days.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"])
    parser.add_argument("--days", type=int, default=90, help="Number of past days to backfill")
    parser.add_argument("--mode", choices=["proxy", "news", "both"], default="both")
    args = parser.parse_args()

    if args.mode in ("news", "both"):
        backfill_historical_news(args.tickers, args.days)
    if args.mode in ("proxy", "both"):
        # Use proxy as a fallback to fill any gaps quickly
        backfill_recent_proxy(args.tickers, args.days)


if __name__ == "__main__":
    main()


