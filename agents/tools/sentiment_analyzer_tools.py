# agents/sentiment_analyzer_tools.py
"""
Deterministic tools for sentiment analysis. These tools avoid LLM usage and can be
composed by the workflow or called directly from the sentiment analyzer node.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
import requests
from langchain_core.tools import tool


POSITIVE_WORDS = {
    "bullish", "buy", "buying", "strong", "growth", "profit", "earnings", "beat",
    "positive", "up", "rise", "gain", "rally", "surge", "jump", "climb", "soar",
    "outperform", "upgrade", "target", "recommend", "favorable", "optimistic",
}

NEGATIVE_WORDS = {
    "bearish", "sell", "selling", "weak", "decline", "loss", "miss", "negative", "down",
    "fall", "drop", "crash", "plunge", "tank", "dump", "downgrade", "underperform",
    "risk", "concern", "worry", "pessimistic",
}


def _score_text_sentiment(text: str) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in lower)
    neg = sum(1 for w in NEGATIVE_WORDS if w in lower)
    tot = pos + neg
    if tot == 0:
        return 0.0
    return (pos - neg) / tot


def _label_from_score(score: float) -> str:
    if score > 0.3:
        return "very_positive"
    if score > 0.1:
        return "positive"
    if score < -0.3:
        return "very_negative"
    if score < -0.1:
        return "negative"
    return "neutral"


@tool
def fetch_news_articles(ticker: str, company_name: Optional[str] = None,
                        window_days: int = 7, page_size: int = 20) -> Dict[str, Any]:
    """
    Fetch recent news via NewsAPI if NEWS_API_KEY is configured. Falls back to empty list.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return {"status": "success", "articles": []}

    try:
        terms = [ticker]
        if company_name:
            terms.append(company_name)
        since = (datetime.now() - timedelta(days=window_days)).strftime("%Y-%m-%d")
        collected: List[Dict[str, Any]] = []
        for term in terms:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": f"{term} stock",
                "apiKey": api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": page_size,
                "from": since,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "ok":
                for a in data.get("articles", []):
                    collected.append({
                        "title": a.get("title"),
                        "description": a.get("description"),
                        "url": a.get("url"),
                        "publishedAt": a.get("publishedAt"),
                        "source": (a.get("source") or {}).get("name"),
                    })
        return {"status": "success", "articles": collected}
    except Exception as e:
        return {"status": "error", "error": f"news fetch failed: {e}", "articles": []}


@tool
def fetch_reddit_posts(ticker: str, company_name: Optional[str] = None,
                       subreddits: Optional[List[str]] = None,
                       limit: int = 10, window: str = "week") -> Dict[str, Any]:
    """
    Fetch recent Reddit posts via OAuth if credentials provided. Falls back to empty list.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "StockPredictor/1.0")
    if not (client_id and client_secret):
        return {"status": "success", "posts": []}

    try:
        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
        token_resp = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": user_agent},
            auth=auth,
            timeout=15,
        )
        token_resp.raise_for_status()
        token = token_resp.json().get("access_token")
        if not token:
            return {"status": "success", "posts": []}

        headers = {"Authorization": f"Bearer {token}", "User-Agent": user_agent}
        terms = [ticker]
        if company_name:
            terms.append(company_name)
        if not subreddits:
            subreddits = ["stocks", "investing", "wallstreetbets", "StockMarket"]

        posts: List[Dict[str, Any]] = []
        for term in terms:
            for sr in subreddits:
                url = f"https://oauth.reddit.com/r/{sr}/search"
                params = {"q": term, "t": window, "limit": limit, "sort": "hot"}
                r = requests.get(url, headers=headers, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    for ch in data.get("data", {}).get("children", []):
                        d = ch.get("data", {})
                        posts.append({
                            "subreddit": sr,
                            "title": d.get("title"),
                            "selftext": d.get("selftext"),
                            "score": d.get("score"),
                            "created_utc": d.get("created_utc"),
                            "permalink": f"https://reddit.com{d.get('permalink')}" if d.get("permalink") else None,
                        })
        return {"status": "success", "posts": posts}
    except Exception as e:
        return {"status": "error", "error": f"reddit fetch failed: {e}", "posts": []}


@tool
def analyze_text_sentiment(texts: List[str]) -> Dict[str, Any]:
    """Compute per-text and aggregate sentiment score/label using keyword heuristic."""
    try:
        scores = [
            _score_text_sentiment(t or "") for t in (texts or [])
        ]
        avg = sum(scores) / len(scores) if scores else 0.0
        label = _label_from_score(avg)
        return {"status": "success", "scores": scores, "average": avg, "label": label}
    except Exception as e:
        return {"status": "error", "error": str(e), "scores": [], "average": 0.0, "label": "neutral"}


@tool
def aggregate_sentiment(news_articles: List[Dict[str, Any]], reddit_posts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate sentiment from news and reddit with simple weighting.
    Returns overall score, label, confidence, and breakdown.
    """
    try:
        news_texts = [f"{a.get('title','')} {a.get('description','')}" for a in news_articles or []]
        reddit_texts = [f"{p.get('title','')} {p.get('selftext','')}" for p in reddit_posts or []]
        news_res = analyze_text_sentiment.invoke({"texts": news_texts})
        reddit_res = analyze_text_sentiment.invoke({"texts": reddit_texts})

        n_avg = news_res.get("average", 0.0)
        r_avg = reddit_res.get("average", 0.0)
        n_count = len(news_texts)
        r_count = len(reddit_texts)
        # weight news higher
        nw = 0.7
        rw = 0.3
        overall = (n_avg * nw) + (r_avg * rw)
        label = _label_from_score(overall)
        # confidence: bounded by volume of items
        confidence = min((n_count / 10.0) * 0.7 + (r_count / 10.0) * 0.3, 1.0)
        return {
            "status": "success",
            "overall": {
                "sentiment_score": overall,
                "sentiment_label": label,
                "confidence": confidence,
            },
            "news": {"average": n_avg, "count": n_count},
            "reddit": {"average": r_avg, "count": r_count},
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@tool
def compute_overall_sentiment(ticker: str, company_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Orchestrate fetching news and reddit, analyze sentiment, and return a UI-compatible payload.
    """
    try:
        news = fetch_news_articles.invoke({"ticker": ticker, "company_name": company_name})
        reddit = fetch_reddit_posts.invoke({"ticker": ticker, "company_name": company_name})
        agg = aggregate_sentiment.invoke({
            "news_articles": news.get("articles", []),
            "reddit_posts": reddit.get("posts", []),
        })

        overall = agg.get("overall", {}) if isinstance(agg, dict) else {}
        result = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "news_sentiment": {
                "sentiment_score": (agg.get("news", {}) or {}).get("average", 0.0),
                "sentiment_label": _label_from_score((agg.get("news", {}) or {}).get("average", 0.0)),
                "article_count": (agg.get("news", {}) or {}).get("count", 0),
                "confidence": min(((agg.get("news", {}) or {}).get("count", 0)) / 10.0, 1.0),
                "recent_articles": (news.get("articles", []) or [])[:5],
            },
            "reddit_sentiment": {
                "sentiment_score": (agg.get("reddit", {}) or {}).get("average", 0.0),
                "sentiment_label": _label_from_score((agg.get("reddit", {}) or {}).get("average", 0.0)),
                "post_count": (agg.get("reddit", {}) or {}).get("count", 0),
                "confidence": min(((agg.get("reddit", {}) or {}).get("count", 0)) / 10.0, 1.0),
            },
            "overall_sentiment": {
                "sentiment_score": overall.get("sentiment_score", 0.0),
                "sentiment_label": overall.get("sentiment_label", "neutral"),
                "confidence": overall.get("confidence", 0.0),
            },
            "sentiment_summary": (
                "Very positive sentiment detected across sources." if overall.get("sentiment_score", 0) > 0.3 else
                "Generally positive sentiment with some mixed signals." if overall.get("sentiment_score", 0) > 0.1 else
                "Neutral sentiment with balanced positive and negative signals." if overall.get("sentiment_score", 0) > -0.1 else
                "Generally negative sentiment with some concerns." if overall.get("sentiment_score", 0) > -0.3 else
                "Very negative sentiment detected across sources."
            ),
            "key_sentiment_factors": [
                "News volume: {} items".format((agg.get("news", {}) or {}).get("count", 0)),
                "Reddit volume: {} posts".format((agg.get("reddit", {}) or {}).get("count", 0)),
            ],
            "sentiment_trend": (
                "bullish" if overall.get("sentiment_score", 0.0) > 0.2 else
                "bearish" if overall.get("sentiment_score", 0.0) < -0.2 else
                "neutral"
            ),
        }

        return {"status": "success", "sentiment_analysis": result, "next_agent": "sentiment_integrator"}
    except Exception as e:
        return {"status": "error", "error": f"compute_overall_sentiment failed: {e}"}


