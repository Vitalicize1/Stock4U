"""
Database-Backed Logger for Stock4U

This module provides a database-backed logging system that stores predictions,
daily picks, and system metrics in PostgreSQL for better performance and analytics.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from decimal import Decimal

from models.database_models import Prediction, DailyPick, SystemMetric, UserSession
from utils.database import get_postgres_session
from peewee import fn

logger = logging.getLogger(__name__)


class DatabaseLogger:
    """Database-backed logger for predictions and system metrics."""
    
    def __init__(self):
        self._postgres_available = False
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize PostgreSQL connection."""
        try:
            with get_postgres_session() as session:
                session.execute("SELECT 1")
            self._postgres_available = True
            logger.info("✅ Database logger initialized")
        except Exception as e:
            logger.warning(f"⚠️ Database logger not available: {e}")
            self._postgres_available = False
    
    def log_prediction(self, ticker: str, result: Dict[str, Any], timeframe: str = "1d") -> Optional[str]:
        """
        Log a prediction result to the database.
        
        Args:
            ticker: Stock ticker symbol
            result: Complete prediction result from workflow
            timeframe: Prediction timeframe
            
        Returns:
            Prediction ID if successful, None otherwise
        """
        if not self._postgres_available:
            logger.warning("Database not available, skipping prediction log")
            return None
        
        try:
            prediction = Prediction.create_from_result(ticker, result, timeframe)
            
            # Record metrics
            self._record_prediction_metrics(prediction, result)
            
            logger.info(f"✅ Prediction logged for {ticker}: {prediction.id}")
            return str(prediction.id)
            
        except Exception as e:
            logger.error(f"❌ Failed to log prediction for {ticker}: {e}")
            return None
    
    def log_daily_picks(self, picks: List[Dict[str, Any]], date: Optional[datetime] = None) -> List[str]:
        """
        Log daily picks to the database.
        
        Args:
            picks: List of daily pick dictionaries
            date: Date for the picks (defaults to today)
            
        Returns:
            List of pick IDs
        """
        if not self._postgres_available:
            logger.warning("Database not available, skipping daily picks log")
            return []
        
        if date is None:
            date = datetime.utcnow()
        
        pick_ids = []
        
        try:
            for pick_data in picks:
                pick = DailyPick.create_pick(
                    date=date,
                    ticker=pick_data.get("ticker"),
                    recommendation=pick_data.get("recommendation", "HOLD"),
                    confidence=pick_data.get("confidence"),
                    reasoning=pick_data.get("reasoning"),
                    target_price=pick_data.get("target_price"),
                    stop_loss=pick_data.get("stop_loss"),
                    source=pick_data.get("source", "system"),
                    tags=pick_data.get("tags")
                )
                pick_ids.append(str(pick.id))
            
            logger.info(f"✅ {len(pick_ids)} daily picks logged for {date.date()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log daily picks: {e}")
        
        return pick_ids
    
    def update_prediction_result(self, prediction_id: str, actual_direction: str, 
                               actual_price: float) -> bool:
        """
        Update a prediction with actual market results.
        
        Args:
            prediction_id: UUID of the prediction
            actual_direction: Actual market direction (UP/DOWN)
            actual_price: Actual closing price
            
        Returns:
            True if successful, False otherwise
        """
        if not self._postgres_available:
            return False
        
        try:
            prediction = Prediction.get(Prediction.id == prediction_id)
            prediction.update_actual_result(actual_direction, actual_price)
            
            # Record accuracy metric
            if prediction.is_correct is not None:
                self._record_accuracy_metric(prediction.is_correct)
            
            logger.info(f"✅ Updated prediction {prediction_id} with actual results")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update prediction {prediction_id}: {e}")
            return False
    
    def get_prediction_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """
        Get prediction accuracy statistics for the last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with accuracy statistics
        """
        if not self._postgres_available:
            return self._get_fallback_accuracy_stats()
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get predictions with actual results
            predictions = Prediction.select().where(
                (Prediction.timestamp >= cutoff_date) & 
                (Prediction.is_correct.is_null(False))
            )
            
            total_predictions = predictions.count()
            
            if total_predictions == 0:
                return {
                    "total_predictions": 0,
                    "predictions_with_results": 0,
                    "accuracy": 0.0,
                    "correct_predictions": 0,
                    "incorrect_predictions": 0,
                    "days_analyzed": days,
                    "avg_confidence": 0.0,
                    "engine_breakdown": {}
                }
            
            correct_predictions = predictions.where(Prediction.is_correct == True).count()
            accuracy = (correct_predictions / total_predictions) * 100
            
            # Calculate average confidence
            avg_confidence = predictions.where(Prediction.confidence.is_null(False)).aggregate(
                fn.AVG(Prediction.confidence)
            ) or 0.0
            
            # Engine breakdown
            engine_breakdown = {}
            for engine in ["ensemble", "ml", "llm", "rule"]:
                engine_predictions = predictions.where(Prediction.prediction_engine == engine)
                engine_count = engine_predictions.count()
                if engine_count > 0:
                    engine_correct = engine_predictions.where(Prediction.is_correct == True).count()
                    engine_accuracy = (engine_correct / engine_count) * 100
                    engine_breakdown[engine] = {
                        "count": engine_count,
                        "accuracy": round(engine_accuracy, 2)
                    }
            
            return {
                "total_predictions": total_predictions,
                "predictions_with_results": total_predictions,
                "accuracy": round(accuracy, 2),
                "correct_predictions": correct_predictions,
                "incorrect_predictions": total_predictions - correct_predictions,
                "days_analyzed": days,
                "avg_confidence": round(float(avg_confidence), 2),
                "engine_breakdown": engine_breakdown
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get prediction accuracy: {e}")
            return self._get_fallback_accuracy_stats()
    
    def get_daily_picks_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """
        Get daily picks accuracy statistics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with daily picks accuracy statistics
        """
        if not self._postgres_available:
            return self._get_fallback_daily_picks_stats()
        
        try:
            cutoff_date = datetime.utcnow().date() - timedelta(days=days)
            
            # Get daily picks
            picks = DailyPick.select().where(DailyPick.date >= cutoff_date)
            total_picks = picks.count()
            
            if total_picks == 0:
                return {
                    "total_daily_picks_days": 0,
                    "total_picks": 0,
                    "picks_with_results": 0,
                    "accuracy": 0.0,
                    "correct_picks": 0,
                    "incorrect_picks": 0,
                    "avg_return": 0.0
                }
            
            # Get picks with performance data
            picks_with_results = picks.where(DailyPick.actual_return.is_null(False))
            picks_with_results_count = picks_with_results.count()
            
            if picks_with_results_count == 0:
                return {
                    "total_daily_picks_days": picks.select(fn.DISTINCT(DailyPick.date)).count(),
                    "total_picks": total_picks,
                    "picks_with_results": 0,
                    "accuracy": 0.0,
                    "correct_picks": 0,
                    "incorrect_picks": 0,
                    "avg_return": 0.0
                }
            
            # Calculate accuracy based on target hits
            correct_picks = picks_with_results.where(DailyPick.hit_target == True).count()
            accuracy = (correct_picks / picks_with_results_count) * 100
            
            # Calculate average return
            avg_return = picks_with_results.aggregate(fn.AVG(DailyPick.actual_return)) or 0.0
            
            return {
                "total_daily_picks_days": picks.select(fn.DISTINCT(DailyPick.date)).count(),
                "total_picks": total_picks,
                "picks_with_results": picks_with_results_count,
                "accuracy": round(accuracy, 2),
                "correct_picks": correct_picks,
                "incorrect_picks": picks_with_results_count - correct_picks,
                "avg_return": round(float(avg_return), 4)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get daily picks accuracy: {e}")
            return self._get_fallback_daily_picks_stats()
    
    def get_recent_predictions(self, ticker: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent predictions from the database.
        
        Args:
            ticker: Optional ticker filter
            limit: Maximum number of predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        if not self._postgres_available:
            return []
        
        try:
            query = Prediction.select().order_by(Prediction.timestamp.desc()).limit(limit)
            
            if ticker:
                query = query.where(Prediction.ticker == ticker)
            
            predictions = []
            for pred in query:
                pred_dict = pred.to_dict()
                pred_dict["analysis_data"] = pred.get_analysis_data()
                predictions.append(pred_dict)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent predictions: {e}")
            return []
    
    def get_prediction_by_id(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific prediction by ID.
        
        Args:
            prediction_id: UUID of the prediction
            
        Returns:
            Prediction dictionary or None if not found
        """
        if not self._postgres_available:
            return None
        
        try:
            prediction = Prediction.get(Prediction.id == prediction_id)
            pred_dict = prediction.to_dict()
            pred_dict["analysis_data"] = prediction.get_analysis_data()
            return pred_dict
            
        except Prediction.DoesNotExist:
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get prediction {prediction_id}: {e}")
            return None
    
    def export_predictions_to_csv(self, output_file: str, days: int = 30) -> bool:
        """
        Export predictions to CSV file.
        
        Args:
            output_file: Output file path
            days: Number of days to export
            
        Returns:
            True if successful, False otherwise
        """
        if not self._postgres_available:
            return False
        
        try:
            import pandas as pd
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            predictions = Prediction.select().where(Prediction.timestamp >= cutoff_date)
            
            # Convert to DataFrame
            data = []
            for pred in predictions:
                row = {
                    "timestamp": pred.timestamp,
                    "ticker": pred.ticker,
                    "timeframe": pred.timeframe,
                    "direction": pred.direction,
                    "confidence": pred.confidence,
                    "predicted_price": pred.predicted_price,
                    "current_price": pred.current_price,
                    "actual_direction": pred.actual_direction,
                    "actual_price": pred.actual_price,
                    "is_correct": pred.is_correct,
                    "prediction_engine": pred.prediction_engine,
                    "processing_time_ms": pred.processing_time_ms,
                    "cache_hit": pred.cache_hit
                }
                data.append(row)
            
            if data:
                df = pd.DataFrame(data)
                df.to_csv(output_file, index=False)
                logger.info(f"✅ Exported {len(data)} predictions to {output_file}")
                return True
            else:
                logger.warning("No prediction data to export")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to export predictions: {e}")
            return False
    
    def cleanup_old_data(self, days: int = 90) -> Dict[str, int]:
        """
        Clean up old data from the database.
        
        Args:
            days: Keep data newer than this many days
            
        Returns:
            Dictionary with cleanup statistics
        """
        if not self._postgres_available:
            return {"predictions": 0, "daily_picks": 0, "system_metrics": 0}
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Clean up old predictions
            old_predictions = Prediction.select().where(Prediction.timestamp < cutoff_date)
            predictions_deleted = old_predictions.count()
            old_predictions.delete_instance()
            
            # Clean up old daily picks
            old_picks = DailyPick.select().where(DailyPick.date < cutoff_date.date())
            picks_deleted = old_picks.count()
            old_picks.delete_instance()
            
            # Clean up old system metrics
            old_metrics = SystemMetric.select().where(SystemMetric.timestamp < cutoff_date)
            metrics_deleted = old_metrics.count()
            old_metrics.delete_instance()
            
            logger.info(f"✅ Cleaned up {predictions_deleted} predictions, {picks_deleted} picks, {metrics_deleted} metrics")
            
            return {
                "predictions": predictions_deleted,
                "daily_picks": picks_deleted,
                "system_metrics": metrics_deleted
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data: {e}")
            return {"predictions": 0, "daily_picks": 0, "system_metrics": 0}
    
    def _record_prediction_metrics(self, prediction: Prediction, result: Dict[str, Any]):
        """Record metrics for a prediction."""
        try:
            # Record processing time
            if prediction.processing_time_ms:
                SystemMetric.record_metric(
                    "prediction_processing_time_ms",
                    prediction.processing_time_ms,
                    context={"ticker": prediction.ticker, "engine": prediction.prediction_engine}
                )
            
            # Record confidence distribution
            if prediction.confidence:
                SystemMetric.record_metric(
                    "prediction_confidence",
                    prediction.confidence,
                    context={"ticker": prediction.ticker, "engine": prediction.prediction_engine}
                )
            
            # Record cache hit rate
            SystemMetric.record_metric(
                "prediction_cache_hit",
                1.0 if prediction.cache_hit else 0.0,
                context={"ticker": prediction.ticker}
            )
            
        except Exception as e:
            logger.warning(f"Failed to record prediction metrics: {e}")
    
    def _record_accuracy_metric(self, is_correct: bool):
        """Record accuracy metric."""
        try:
            SystemMetric.record_metric(
                "prediction_accuracy",
                1.0 if is_correct else 0.0,
                tags={"accuracy_type": "binary"}
            )
        except Exception as e:
            logger.warning(f"Failed to record accuracy metric: {e}")
    
    def _get_fallback_accuracy_stats(self) -> Dict[str, Any]:
        """Get fallback accuracy statistics when database is unavailable."""
        return {
            "total_predictions": 0,
            "predictions_with_results": 0,
            "accuracy": 0.0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "days_analyzed": 30,
            "avg_confidence": 0.0,
            "engine_breakdown": {}
        }
    
    def _get_fallback_daily_picks_stats(self) -> Dict[str, Any]:
        """Get fallback daily picks statistics when database is unavailable."""
        return {
            "total_daily_picks_days": 0,
            "total_picks": 0,
            "picks_with_results": 0,
            "accuracy": 0.0,
            "correct_picks": 0,
            "incorrect_picks": 0,
            "avg_return": 0.0
        }


# Global logger instance
db_logger = DatabaseLogger()


# Convenience functions for backward compatibility
def log_prediction(ticker: str, prediction: Dict[str, Any], actual_result: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Log a prediction (backward compatibility)."""
    # Convert old format to new format
    result = {
        "prediction_result": prediction,
        "ticker": ticker,
        "actual_result": actual_result
    }
    return db_logger.log_prediction(ticker, result)


def log_daily_picks(picks: List[Dict[str, Any]]) -> List[str]:
    """Log daily picks (backward compatibility)."""
    return db_logger.log_daily_picks(picks)


def get_accuracy_summary(days: int = 30) -> Dict[str, Any]:
    """Get accuracy summary (backward compatibility)."""
    return db_logger.get_prediction_accuracy(days)


def get_daily_picks_accuracy(days: int = 30) -> Dict[str, Any]:
    """Get daily picks accuracy (backward compatibility)."""
    return db_logger.get_daily_picks_accuracy(days)


def export_predictions_to_csv(output_file: str = "cache/prediction_logs/predictions_export.csv") -> bool:
    """Export predictions to CSV (backward compatibility)."""
    return db_logger.export_predictions_to_csv(output_file)


# Additional utility functions
def get_recent_predictions(ticker: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent predictions."""
    return db_logger.get_recent_predictions(ticker, limit)


def get_prediction_by_id(prediction_id: str) -> Optional[Dict[str, Any]]:
    """Get prediction by ID."""
    return db_logger.get_prediction_by_id(prediction_id)


def cleanup_old_data(days: int = 90) -> Dict[str, int]:
    """Clean up old data."""
    return db_logger.cleanup_old_data(days)


def initialize_database_logger() -> Dict[str, bool]:
    """Initialize database logger."""
    return {
        "postgres": db_logger._postgres_available
    }
