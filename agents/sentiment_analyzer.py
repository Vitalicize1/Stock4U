# agents/sentiment_analyzer.py
from typing import Dict, Any, List
import requests
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from agents.tools.sentiment_analyzer_tools import compute_overall_sentiment
import time

# Load environment variables
load_dotenv()

class SentimentAnalyzerAgent:
    """
    Agent responsible for sentiment analysis from multiple sources:
    - News sentiment (using NewsAPI)
    - Reddit sentiment (using Reddit API)
    - Social media sentiment analysis
    """
    
    def __init__(self):
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.reddit_client_id = os.getenv("REDDIT_CLIENT_ID")
        self.reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "StockPredictor/1.0")
        
        # Initialize sentiment scores
        self.sentiment_scores = {
            "news_sentiment": 0.0,
            "reddit_sentiment": 0.0,
            "overall_sentiment": 0.0,
            "sentiment_confidence": 0.0
        }
    
    def analyze_sentiment(self, ticker: str, company_name: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis for a stock.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name for better search results
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            print(f"📰 Analyzing sentiment for {ticker}...")
            
            # Get news sentiment
            news_sentiment = self._analyze_news_sentiment(ticker, company_name)
            
            # Get Reddit sentiment
            reddit_sentiment = self._analyze_reddit_sentiment(ticker, company_name)
            
            # Calculate overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(news_sentiment, reddit_sentiment)
            
            # Compile results
            sentiment_results = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "news_sentiment": news_sentiment,
                "reddit_sentiment": reddit_sentiment,
                "overall_sentiment": overall_sentiment,
                "sentiment_summary": self._generate_sentiment_summary(overall_sentiment),
                "key_sentiment_factors": self._extract_key_factors(news_sentiment, reddit_sentiment),
                "sentiment_trend": self._determine_sentiment_trend(overall_sentiment)
            }
            
            return {
                "status": "success",
                "sentiment_analysis": sentiment_results,
                "next_agent": "sentiment_integrator"
            }
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "sentiment_analysis": self._get_default_sentiment(ticker)
            }
    
    def _analyze_news_sentiment(self, ticker: str, company_name: str = None) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        
        if not self.news_api_key:
            return self._get_default_news_sentiment()
        
        try:
            # Search terms for news
            search_terms = [ticker]
            if company_name:
                search_terms.append(company_name)
            
            all_articles = []
            
            for term in search_terms:
                # Get recent news articles
                url = "https://newsapi.org/v2/everything"
                params = {
                    "q": f"{term} stock",
                    "apiKey": self.news_api_key,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 20,
                    "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                if data.get("status") == "ok":
                    all_articles.extend(data.get("articles", []))
            
            # Analyze sentiment of articles
            sentiment_scores = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for article in all_articles[:30]:  # Limit to 30 articles
                title = article.get("title", "")
                description = article.get("description", "")
                content = f"{title} {description}"
                
                # Simple keyword-based sentiment analysis
                sentiment_score = self._analyze_text_sentiment(content)
                sentiment_scores.append(sentiment_score)
                
                if sentiment_score > 0.1:
                    positive_count += 1
                elif sentiment_score < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            return {
                "sentiment_score": avg_sentiment,
                "sentiment_label": self._get_sentiment_label(avg_sentiment),
                "article_count": len(all_articles),
                "positive_articles": positive_count,
                "negative_articles": negative_count,
                "neutral_articles": neutral_count,
                "confidence": min(len(all_articles) / 10, 1.0),  # Higher confidence with more articles
                "recent_articles": all_articles[:5]  # Keep 5 most recent for reference
            }
            
        except Exception as e:
            print(f"News sentiment analysis failed: {str(e)}")
            return self._get_default_news_sentiment()
    
    def _analyze_reddit_sentiment(self, ticker: str, company_name: str = None) -> Dict[str, Any]:
        """Analyze sentiment from Reddit posts and comments."""
        
        if not all([self.reddit_client_id, self.reddit_client_secret]):
            return self._get_default_reddit_sentiment()
        
        try:
            # Get Reddit access token
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                "grant_type": "client_credentials"
            }
            auth_headers = {
                "User-Agent": self.reddit_user_agent
            }
            
            response = requests.post(
                auth_url,
                data=auth_data,
                headers=auth_headers,
                auth=(self.reddit_client_id, self.reddit_client_secret)
            )
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data.get("access_token")
            
            if not access_token:
                return self._get_default_reddit_sentiment()
            
            # Search Reddit for posts about the stock
            search_terms = [ticker]
            if company_name:
                search_terms.append(company_name)
            
            all_posts = []
            
            for term in search_terms:
                # Search in relevant subreddits
                subreddits = ["stocks", "investing", "wallstreetbets", "StockMarket"]
                
                for subreddit in subreddits:
                    url = f"https://oauth.reddit.com/r/{subreddit}/search"
                    headers = {
                        "Authorization": f"Bearer {access_token}",
                        "User-Agent": self.reddit_user_agent
                    }
                    params = {
                        "q": term,
                        "t": "week",  # Last week
                        "limit": 10,
                        "sort": "hot"
                    }
                    
                    response = requests.get(url, headers=headers, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        posts = data.get("data", {}).get("children", [])
                        all_posts.extend(posts)
            
            # Analyze sentiment of posts
            sentiment_scores = []
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for post in all_posts[:20]:  # Limit to 20 posts
                post_data = post.get("data", {})
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                content = f"{title} {selftext}"
                
                # Simple keyword-based sentiment analysis
                sentiment_score = self._analyze_text_sentiment(content)
                sentiment_scores.append(sentiment_score)
                
                if sentiment_score > 0.1:
                    positive_count += 1
                elif sentiment_score < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            return {
                "sentiment_score": avg_sentiment,
                "sentiment_label": self._get_sentiment_label(avg_sentiment),
                "post_count": len(all_posts),
                "positive_posts": positive_count,
                "negative_posts": negative_count,
                "neutral_posts": neutral_count,
                "confidence": min(len(all_posts) / 10, 1.0),
                "recent_posts": all_posts[:5]  # Keep 5 most recent for reference
            }
            
        except Exception as e:
            print(f"Reddit sentiment analysis failed: {str(e)}")
            return self._get_default_reddit_sentiment()
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment analysis."""
        
        # Positive keywords
        positive_words = [
            "bullish", "buy", "buying", "strong", "growth", "profit", "earnings", "beat", "positive",
            "up", "rise", "gain", "rally", "surge", "jump", "climb", "soar", "moon", "rocket",
            "outperform", "upgrade", "target", "recommend", "favorable", "optimistic"
        ]
        
        # Negative keywords
        negative_words = [
            "bearish", "sell", "selling", "weak", "decline", "loss", "miss", "negative", "down",
            "fall", "drop", "crash", "plunge", "tank", "dump", "bear", "short", "downgrade",
            "underperform", "risk", "concern", "worry", "pessimistic", "bearish"
        ]
        
        # Count occurrences
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_words
        return sentiment_score
    
    def _calculate_overall_sentiment(self, news_sentiment: Dict, reddit_sentiment: Dict) -> Dict[str, Any]:
        """Calculate overall sentiment from multiple sources."""
        
        news_score = news_sentiment.get("sentiment_score", 0)
        reddit_score = reddit_sentiment.get("sentiment_score", 0)
        
        # Weight the sources (news is more reliable)
        news_weight = 0.7
        reddit_weight = 0.3
        
        overall_score = (news_score * news_weight) + (reddit_score * reddit_weight)
        
        # Calculate confidence based on data availability
        news_confidence = news_sentiment.get("confidence", 0)
        reddit_confidence = reddit_sentiment.get("confidence", 0)
        
        overall_confidence = (news_confidence * news_weight) + (reddit_confidence * reddit_weight)
        
        return {
            "sentiment_score": overall_score,
            "sentiment_label": self._get_sentiment_label(overall_score),
            "confidence": overall_confidence,
            "news_contribution": news_score * news_weight,
            "reddit_contribution": reddit_score * reddit_weight
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.3:
            return "very_positive"
        elif score > 0.1:
            return "positive"
        elif score < -0.3:
            return "very_negative"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _generate_sentiment_summary(self, overall_sentiment: Dict) -> str:
        """Generate a summary of sentiment analysis."""
        
        score = overall_sentiment.get("sentiment_score", 0)
        label = overall_sentiment.get("sentiment_label", "neutral")
        confidence = overall_sentiment.get("confidence", 0)
        
        if label == "very_positive":
            summary = "Very positive sentiment detected across news and social media sources."
        elif label == "positive":
            summary = "Generally positive sentiment with some mixed signals."
        elif label == "neutral":
            summary = "Neutral sentiment with balanced positive and negative signals."
        elif label == "negative":
            summary = "Generally negative sentiment with some concerns."
        else:  # very_negative
            summary = "Very negative sentiment detected across multiple sources."
        
        if confidence < 0.5:
            summary += " Low confidence due to limited data availability."
        
        return summary
    
    def _extract_key_factors(self, news_sentiment: Dict, reddit_sentiment: Dict) -> List[str]:
        """Extract key factors from sentiment analysis."""
        
        factors = []
        
        # News factors
        news_score = news_sentiment.get("sentiment_score", 0)
        news_count = news_sentiment.get("article_count", 0)
        
        if news_count > 0:
            if news_score > 0.2:
                factors.append("Positive news coverage")
            elif news_score < -0.2:
                factors.append("Negative news coverage")
            else:
                factors.append("Mixed news sentiment")
        
        # Reddit factors
        reddit_score = reddit_sentiment.get("sentiment_score", 0)
        reddit_count = reddit_sentiment.get("post_count", 0)
        
        if reddit_count > 0:
            if reddit_score > 0.2:
                factors.append("Positive social media sentiment")
            elif reddit_score < -0.2:
                factors.append("Negative social media sentiment")
            else:
                factors.append("Mixed social media sentiment")
        
        # Overall factors
        if not factors:
            factors.append("Limited sentiment data available")
        
        return factors
    
    def _determine_sentiment_trend(self, overall_sentiment: Dict) -> str:
        """Determine sentiment trend direction."""
        
        score = overall_sentiment.get("sentiment_score", 0)
        
        if score > 0.2:
            return "bullish"
        elif score < -0.2:
            return "bearish"
        else:
            return "neutral"
    
    def _get_default_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get default sentiment when analysis fails."""
        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "news_sentiment": self._get_default_news_sentiment(),
            "reddit_sentiment": self._get_default_reddit_sentiment(),
            "overall_sentiment": {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0
            },
            "sentiment_summary": "Sentiment analysis unavailable",
            "key_sentiment_factors": ["Limited sentiment data"],
            "sentiment_trend": "neutral"
        }
    
    def _get_default_news_sentiment(self) -> Dict[str, Any]:
        """Get default news sentiment."""
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "article_count": 0,
            "positive_articles": 0,
            "negative_articles": 0,
            "neutral_articles": 0,
            "confidence": 0.0,
            "recent_articles": []
        }
    
    def _get_default_reddit_sentiment(self) -> Dict[str, Any]:
        """Get default Reddit sentiment."""
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "post_count": 0,
            "positive_posts": 0,
            "negative_posts": 0,
            "neutral_posts": 0,
            "confidence": 0.0,
            "recent_posts": []
        }

# Function for LangGraph integration
def analyze_sentiment(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node function for sentiment analysis.
    
    Args:
        state: Current state containing ticker and company info
        
    Returns:
        Updated state with sentiment analysis
    """
    ticker = state.get("ticker")
    company_info = (state.get("data", {}) or {}).get("company_info", {})
    # Try to derive a reasonable company name for search
    company_name = (
        (company_info.get("basic_info", {}) or {}).get("name")
        or (company_info.get("basic_info", {}) or {}).get("short_name")
        or company_info.get("name")
    )

    if not ticker:
        return {"error": "No ticker provided"}

    # Prefer deterministic tool pipeline for consistent results
    try:
        tool_res = compute_overall_sentiment.invoke({
            "ticker": ticker,
            "company_name": company_name,
        })
        if isinstance(tool_res, dict) and tool_res.get("status") == "success":
            state.update(tool_res)
            return state
    except Exception:
        pass

    # Fallback to in-module implementation
    agent = SentimentAnalyzerAgent()
    result = agent.analyze_sentiment(ticker, company_name)
    state.update(result)
    return state