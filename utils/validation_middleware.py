"""
Validation Middleware for FastAPI

This module provides middleware for additional input validation and security
checks at the request level.
"""

import json
import logging
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from utils.validation import InputValidator, sanitize_input

logger = logging.getLogger(__name__)


class ValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request validation and sanitization.
    
    This middleware performs additional validation on incoming requests
    before they reach the endpoint handlers.
    """
    
    def __init__(self, app, enable_validation: bool = True):
        super().__init__(app)
        self.enable_validation = enable_validation
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process the request through validation middleware.
        
        Args:
            request: The incoming request
            call_next: The next middleware/endpoint in the chain
            
        Returns:
            The response from the next middleware/endpoint
        """
        if not self.enable_validation:
            return await call_next(request)
        
        try:
            # Validate and sanitize query parameters
            await self._validate_query_params(request)
            
            # Validate and sanitize path parameters
            await self._validate_path_params(request)
            
            # Validate request body for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                await self._validate_request_body(request)
            
            # Process the request
            response = await call_next(request)
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log unexpected errors and return 500
            logger.error(f"Validation middleware error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal validation error"}
            )
    
    async def _validate_query_params(self, request: Request) -> None:
        """Validate and sanitize query parameters."""
        query_params = dict(request.query_params)
        
        for key, value in query_params.items():
            # Sanitize the value
            sanitized_value = sanitize_input(value)
            
            # Validate based on parameter type
            if key in ["ticker", "tickers"]:
                if key == "ticker":
                    result = InputValidator.validate_ticker_symbol(sanitized_value)
                else:
                    result = InputValidator.validate_ticker_list(sanitized_value)
                
                if not result.is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid {key}: {result.error_message}"
                    )
            
            elif key in ["timeframe", "period"]:
                if key == "timeframe":
                    result = InputValidator.validate_timeframe(sanitized_value)
                else:
                    result = InputValidator.validate_period(sanitized_value)
                
                if not result.is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid {key}: {result.error_message}"
                    )
            
            elif key in ["cash", "fee_bps", "slip_bps"]:
                try:
                    num_value = float(sanitized_value)
                    if num_value < 0:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid {key}: must be non-negative"
                        )
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid {key}: must be a valid number"
                    )
    
    async def _validate_path_params(self, request: Request) -> None:
        """Validate and sanitize path parameters."""
        path_params = dict(request.path_params)
        
        for key, value in path_params.items():
            # Sanitize the value
            sanitized_value = sanitize_input(str(value))
            
            # Validate ticker in path params
            if key in ["ticker", "symbol"]:
                result = InputValidator.validate_ticker_symbol(sanitized_value)
                if not result.is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid {key}: {result.error_message}"
                    )
    
    async def _validate_request_body(self, request: Request) -> None:
        """Validate request body content."""
        try:
            # Get the request body
            body = await request.body()
            
            if not body:
                return  # Empty body is fine
            
            # Try to parse as JSON
            try:
                body_data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError:
                # Not JSON, might be form data or other format
                return
            
            # Validate the body data
            if isinstance(body_data, dict):
                await self._validate_body_dict(body_data)
            
        except Exception as e:
            logger.warning(f"Request body validation error: {str(e)}")
            # Don't fail the request for body validation errors
            # Let the endpoint handle validation
    
    async def _validate_body_dict(self, data: Dict[str, Any]) -> None:
        """Validate dictionary data in request body."""
        for key, value in data.items():
            # Sanitize string values
            if isinstance(value, str):
                data[key] = sanitize_input(value)
            
            # Validate specific fields
            if key == "ticker" and isinstance(value, str):
                result = InputValidator.validate_ticker_symbol(value)
                if not result.is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid ticker: {result.error_message}"
                    )
            
            elif key == "timeframe" and isinstance(value, str):
                result = InputValidator.validate_timeframe(value)
                if not result.is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid timeframe: {result.error_message}"
                    )
            
            elif key == "tickers":
                result = InputValidator.validate_ticker_list(value)
                if not result.is_valid:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid tickers: {result.error_message}"
                    )


def create_validation_middleware(app, enable_validation: bool = True):
    """
    Create and add validation middleware to the FastAPI app.
    
    Args:
        app: The FastAPI application
        enable_validation: Whether to enable validation middleware
        
    Returns:
        The app with middleware added
    """
    app.add_middleware(ValidationMiddleware, enable_validation=enable_validation)
    return app


# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.
    
    This is a basic implementation. For production, consider using
    a more robust solution like slowapi or fastapi-limiter.
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}  # In production, use Redis or similar
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request with rate limiting."""
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Simple rate limiting (in production, use proper rate limiting)
        # This is just a basic example
        
        response = await call_next(request)
        return response


def create_rate_limit_middleware(app, requests_per_minute: int = 60):
    """
    Create and add rate limiting middleware to the FastAPI app.
    
    Args:
        app: The FastAPI application
        requests_per_minute: Maximum requests per minute per client
        
    Returns:
        The app with middleware added
    """
    app.add_middleware(RateLimitMiddleware, requests_per_minute=requests_per_minute)
    return app
