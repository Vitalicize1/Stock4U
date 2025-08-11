import os
from typing import Dict, Any
from dotenv import load_dotenv
import json
import requests

# Load environment variables
load_dotenv()

class GroqClient:
    """
    Client for Groq AI model integration.
    Handles stock prediction analysis using Groq's fast LLM models.
    """
    
    def __init__(self, model: str = "llama3-8b-8192"):
        """
        Initialize Groq client.
        
        Args:
            model: Groq model to use (default: llama3-8b-8192)
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_stock_data(self, analysis_summary: str) -> Dict[str, Any]:
        """
        Analyze stock data using Groq and return structured prediction.
        
        Args:
            analysis_summary: Comprehensive stock analysis summary
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            prompt = f"""
You are an expert financial analyst and AI stock prediction specialist. Analyze the following stock data and provide a structured prediction.

{analysis_summary}

Please provide your analysis in the following JSON format:
{{
    "direction": "UP/DOWN/NEUTRAL",
    "confidence": 0-100,
    "price_target": null or specific price,
    "price_range": {{
        "low": price,
        "high": price
    }},
    "reasoning": "Detailed explanation of your prediction",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_factors": ["risk1", "risk2", "risk3"]
}}

Focus on:
1. Technical indicators and their significance
2. Market context and trends
3. Support/resistance levels
4. Volume analysis
5. Risk assessment

Be conservative in your predictions and always consider market volatility.
"""

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial analyst expert. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            
            # Check for quota/rate limit errors
            if response.status_code == 429:
                # Rate limit hit - wait and retry once
                import time
                retry_after = response.headers.get('retry-after', 2)
                print(f"⚠️ Rate limit hit, waiting {retry_after} seconds...")
                time.sleep(float(retry_after))
                
                # Retry once
                response = requests.post(self.base_url, headers=self.headers, json=payload)
                if response.status_code == 429:
                    raise ValueError("Rate limit still exceeded after retry. Please wait a moment and try again.")
            elif response.status_code == 403:
                raise ValueError("Groq API key invalid or quota exceeded.")
            
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            # Parse the response
            try:
                # Try to extract JSON from the response
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                else:
                    json_str = content.strip()
                
                prediction = json.loads(json_str)
                
                # Validate and clean the prediction
                return self._validate_prediction(prediction)
                
            except json.JSONDecodeError as e:
                # Fallback: parse the response manually
                return self._parse_fallback_response(content)
                
        except ValueError as e:
            # Re-raise quota errors
            raise e
        except Exception as e:
            print(f"Error in Groq analysis: {str(e)}")
            return self._get_default_prediction()
    
    def _validate_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean the prediction data."""
        
        # Ensure required fields exist
        required_fields = ["direction", "confidence", "reasoning", "key_factors", "risk_factors"]
        for field in required_fields:
            if field not in prediction:
                prediction[field] = self._get_default_value(field)
        
        # Validate direction
        if prediction["direction"] not in ["UP", "DOWN", "NEUTRAL"]:
            prediction["direction"] = "NEUTRAL"
        
        # Validate confidence
        try:
            confidence = float(prediction["confidence"])
            prediction["confidence"] = max(0, min(100, confidence))
        except (ValueError, TypeError):
            prediction["confidence"] = 50.0
        
        # Ensure lists are actually lists
        if not isinstance(prediction["key_factors"], list):
            prediction["key_factors"] = ["Technical analysis indicates mixed signals"]
        
        if not isinstance(prediction["risk_factors"], list):
            prediction["risk_factors"] = ["Market volatility may impact predictions"]
        
        # Add price range if missing
        if "price_range" not in prediction:
            prediction["price_range"] = {"low": None, "high": None}
        
        return prediction
    
    def _parse_fallback_response(self, response_text: str) -> Dict[str, Any]:
        """Parse response when JSON parsing fails."""
        
        # Simple keyword-based parsing
        direction = "NEUTRAL"
        if "UP" in response_text.upper() or "BULLISH" in response_text.upper():
            direction = "UP"
        elif "DOWN" in response_text.upper() or "BEARISH" in response_text.upper():
            direction = "DOWN"
        
        # Extract confidence (look for numbers followed by %)
        import re
        confidence_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response_text)
        confidence = float(confidence_match.group(1)) if confidence_match else 50.0
        
        return {
            "direction": direction,
            "confidence": confidence,
            "price_target": None,
            "price_range": {"low": None, "high": None},
            "reasoning": response_text[:500] + "..." if len(response_text) > 500 else response_text,
            "key_factors": ["Analysis based on technical indicators and market conditions"],
            "risk_factors": ["Market volatility and external factors may affect predictions"]
        }
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """Return a default prediction when analysis fails."""
        return {
            "direction": "NEUTRAL",
            "confidence": 50.0,
            "price_target": None,
            "price_range": {"low": None, "high": None},
            "reasoning": "Unable to generate prediction due to technical issues. Consider manual analysis.",
            "key_factors": ["Technical analysis unavailable"],
            "risk_factors": ["Prediction confidence is low due to analysis failure"]
        }
    
    def _get_default_value(self, field: str) -> Any:
        """Get default values for missing fields."""
        defaults = {
            "direction": "NEUTRAL",
            "confidence": 50.0,
            "price_target": None,
            "reasoning": "Analysis incomplete",
            "key_factors": ["Insufficient data for analysis"],
            "risk_factors": ["Analysis quality may be compromised"]
        }
        return defaults.get(field, None)

# Function for easy integration
def get_groq_prediction(analysis_summary: str, model: str = "llama3-8b-8192") -> Dict[str, Any]:
    """
    Get prediction from Groq for easy integration.
    
    Args:
        analysis_summary: Stock analysis summary
        model: Groq model to use
        
    Returns:
        Prediction dictionary
    """
    try:
        client = GroqClient(model)
        return client.analyze_stock_data(analysis_summary)
    except Exception as e:
        print(f"Groq prediction failed: {str(e)}")
        return {
            "direction": "NEUTRAL",
            "confidence": 50.0,
            "price_target": None,
            "price_range": {"low": None, "high": None},
            "reasoning": f"Prediction failed: {str(e)}",
            "key_factors": ["Analysis failed"],
            "risk_factors": ["Unable to assess risks"]
        }

def get_groq_client():
    """
    Get a LangChain-compatible Groq client for use with create_react_agent.
    
    Returns:
        LangChain ChatGroq client
    """
    try:
        from langchain_groq import ChatGroq
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Create LangChain client without making a test call
        client = ChatGroq(
            model_name="llama3-8b-8192",
            groq_api_key=api_key,
            temperature=0.1,
            max_tokens=4000
        )
        
        # Don't make a test call - this was causing rate limit issues
        # The client will be tested when actually used
        return client
        
    except Exception as e:
        print(f"Failed to create Groq client: {str(e)}")
        raise
