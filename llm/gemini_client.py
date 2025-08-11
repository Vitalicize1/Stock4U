import google.generativeai as genai
import os
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class GeminiClient:
    """
    Client for Google's Gemini AI model integration.
    Handles stock prediction analysis using Gemini Pro.
    """
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_stock_data(self, analysis_summary: str) -> Dict[str, Any]:
        """
        Analyze stock data using Gemini and return structured prediction.
        
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

            response = self.model.generate_content(prompt)
            
            # Parse the response
            try:
                # Try to extract JSON from the response
                response_text = response.text
                
                # Find JSON in the response (sometimes Gemini wraps it in markdown)
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                else:
                    json_str = response_text.strip()
                
                prediction = json.loads(json_str)
                
                # Validate and clean the prediction
                return self._validate_prediction(prediction)
                
            except json.JSONDecodeError as e:
                # Fallback: parse the response manually
                return self._parse_fallback_response(response.text)
                
        except Exception as e:
            print(f"Error in Gemini analysis: {str(e)}")
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
def get_gemini_prediction(analysis_summary: str) -> Dict[str, Any]:
    """
    Get prediction from Gemini for easy integration.
    
    Args:
        analysis_summary: Stock analysis summary
        
    Returns:
        Prediction dictionary
    """
    try:
        client = GeminiClient()
        return client.analyze_stock_data(analysis_summary)
    except Exception as e:
        print(f"Gemini prediction failed: {str(e)}")
        return {
            "direction": "NEUTRAL",
            "confidence": 50.0,
            "price_target": None,
            "price_range": {"low": None, "high": None},
            "reasoning": f"Prediction failed: {str(e)}",
            "key_factors": ["Analysis failed"],
            "risk_factors": ["Unable to assess risks"]
        }

def get_gemini_client():
    """
    Get a LangChain-compatible Gemini client for use with create_react_agent.
    
    Returns:
        LangChain ChatGoogleGenerativeAI client
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Create LangChain client with no retries and short timeout to avoid long backoffs
        client = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            max_tokens=4000,
            max_retries=0,  # Prevent internal retry loops
            timeout=5  # Short timeout to fail fast
        )

        # Do NOT perform a test API call here to avoid triggering provider-side
        # retry delays when quota is exhausted. Simply return the client; callers
        # can decide whether to use it based on availability/preferences.
        return client
        
    except ValueError as e:
        # Re-raise quota errors
        raise e
    except Exception as e:
        print(f"Failed to create Gemini client: {str(e)}")
        raise
