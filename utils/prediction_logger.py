from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


class PredictionLogger:
    """Logger for tracking daily prediction accuracy and performance."""
    
    def __init__(self, log_dir: str = "cache/prediction_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.daily_log_file = self.log_dir / "daily_predictions.json"
        self.accuracy_log_file = self.log_dir / "accuracy_summary.json"
        
    def log_prediction(self, ticker: str, prediction: Dict[str, Any], actual_result: Optional[Dict[str, Any]] = None) -> None:
        """Log a single prediction with optional actual result."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "ticker": ticker,
            "prediction": {
                "direction": prediction.get("direction"),
                "confidence": prediction.get("confidence"),
                "timeframe": prediction.get("timeframe", "1d"),
                "predicted_price": prediction.get("predicted_price"),
                "current_price": prediction.get("current_price"),
            },
            "actual_result": actual_result,
            "correct": None,  # Will be updated when actual result is available
        }
        
        # Load existing logs
        logs = self._load_daily_logs()
        logs.append(log_entry)
        
        # Save updated logs
        self._save_daily_logs(logs)
        
    def log_daily_picks(self, picks: List[Dict[str, Any]]) -> None:
        """Log daily picks for tracking."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "daily_picks",
            "picks": picks,
        }
        
        logs = self._load_daily_logs()
        logs.append(log_entry)
        self._save_daily_logs(logs)
        
    def update_actual_results(self, ticker: str, date: str, actual_result: Dict[str, Any]) -> None:
        """Update predictions with actual market results."""
        logs = self._load_daily_logs()
        
        # Find predictions for this ticker on the given date
        target_date = datetime.fromisoformat(date.replace("Z", "+00:00")).date()
        
        for log in logs:
            if log.get("ticker") == ticker and "timestamp" in log:
                log_date = datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00")).date()
                if log_date == target_date and log.get("actual_result") is None:
                    log["actual_result"] = actual_result
                    log["correct"] = self._check_prediction_accuracy(log["prediction"], actual_result)
                    break
        
        self._save_daily_logs(logs)
        
    def get_accuracy_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get accuracy summary for the last N days."""
        logs = self._load_daily_logs()
        
        # Filter logs from last N days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_logs = [
            log for log in logs 
            if "timestamp" in log and datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None) >= cutoff_date
        ]
        
        # Count predictions with actual results
        predictions_with_results = [log for log in recent_logs if log.get("actual_result") is not None]
        
        if not predictions_with_results:
            return {
                "total_predictions": len(recent_logs),
                "predictions_with_results": 0,
                "accuracy": 0.0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "days_analyzed": days,
            }
        
        correct_predictions = sum(1 for log in predictions_with_results if log.get("correct") is True)
        accuracy = (correct_predictions / len(predictions_with_results)) * 100
        
        return {
            "total_predictions": len(recent_logs),
            "predictions_with_results": len(predictions_with_results),
            "accuracy": round(accuracy, 2),
            "correct_predictions": correct_predictions,
            "incorrect_predictions": len(predictions_with_results) - correct_predictions,
            "days_analyzed": days,
        }
        
    def get_daily_picks_accuracy(self, days: int = 30) -> Dict[str, Any]:
        """Get accuracy summary specifically for daily picks."""
        logs = self._load_daily_logs()
        
        # Filter daily picks logs
        daily_picks_logs = [log for log in logs if log.get("type") == "daily_picks"]
        
        if not daily_picks_logs:
            return {
                "total_daily_picks_days": 0,
                "total_picks": 0,
                "picks_with_results": 0,
                "accuracy": 0.0,
                "correct_picks": 0,
                "incorrect_picks": 0,
            }
        
        # Count total picks
        total_picks = sum(len(log.get("picks", [])) for log in daily_picks_logs)
        
        # For now, we'll need to manually track actual results for daily picks
        # This could be enhanced with automatic price checking
        return {
            "total_daily_picks_days": len(daily_picks_logs),
            "total_picks": total_picks,
            "picks_with_results": 0,  # Will be updated when we implement actual result tracking
            "accuracy": 0.0,
            "correct_picks": 0,
            "incorrect_picks": 0,
        }
        
    def export_to_csv(self, output_file: str = "cache/prediction_logs/predictions_export.csv") -> None:
        """Export prediction logs to CSV for analysis."""
        logs = self._load_daily_logs()
        
        # Convert to DataFrame
        df_data = []
        for log in logs:
            if "ticker" in log:  # Skip daily picks logs
                row = {
                    "timestamp": log.get("timestamp"),
                    "ticker": log.get("ticker"),
                    "predicted_direction": log.get("prediction", {}).get("direction"),
                    "predicted_confidence": log.get("prediction", {}).get("confidence"),
                    "timeframe": log.get("prediction", {}).get("timeframe"),
                    "predicted_price": log.get("prediction", {}).get("predicted_price"),
                    "current_price": log.get("prediction", {}).get("current_price"),
                    "actual_direction": log.get("actual_result", {}).get("direction"),
                    "actual_price": log.get("actual_result", {}).get("price"),
                    "correct": log.get("correct"),
                }
                df_data.append(row)
        
        if df_data:
            df = pd.DataFrame(df_data)
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Prediction logs exported to: {output_path}")
        else:
            print("ℹ️ No prediction data to export")
            
    def _load_daily_logs(self) -> List[Dict[str, Any]]:
        """Load daily prediction logs from file."""
        if not self.daily_log_file.exists():
            return []
        try:
            with open(self.daily_log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
            
    def _save_daily_logs(self, logs: List[Dict[str, Any]]) -> None:
        """Save daily prediction logs to file."""
        try:
            with open(self.daily_log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error saving prediction logs: {e}")
            
    def _check_prediction_accuracy(self, prediction: Dict[str, Any], actual_result: Dict[str, Any]) -> Optional[bool]:
        """Check if a prediction was correct based on actual result."""
        pred_direction = prediction.get("direction", "").upper()
        actual_direction = actual_result.get("direction", "").upper()
        
        if pred_direction and actual_direction:
            return pred_direction == actual_direction
        return None


# Global logger instance
prediction_logger = PredictionLogger()


def log_prediction(ticker: str, prediction: Dict[str, Any], actual_result: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to log a prediction."""
    prediction_logger.log_prediction(ticker, prediction, actual_result)


def log_daily_picks(picks: List[Dict[str, Any]]) -> None:
    """Convenience function to log daily picks."""
    prediction_logger.log_daily_picks(picks)


def get_accuracy_summary(days: int = 30) -> Dict[str, Any]:
    """Convenience function to get accuracy summary."""
    return prediction_logger.get_accuracy_summary(days)


def get_daily_picks_accuracy(days: int = 30) -> Dict[str, Any]:
    """Convenience function to get daily picks accuracy summary."""
    return prediction_logger.get_daily_picks_accuracy(days)


def export_predictions_to_csv(output_file: str = "cache/prediction_logs/predictions_export.csv") -> None:
    """Convenience function to export predictions to CSV."""
    prediction_logger.export_to_csv(output_file)
