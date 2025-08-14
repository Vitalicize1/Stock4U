#!/usr/bin/env python3
"""CLI tool for managing prediction logs and viewing accuracy statistics."""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from utils.prediction_logger import (
    get_accuracy_summary,
    get_daily_picks_accuracy,
    export_predictions_to_csv,
    prediction_logger
)


def display_accuracy_summary(days: int = 30):
    """Display accuracy summary for the specified number of days."""
    print(f"\n📊 Prediction Accuracy Summary (Last {days} days)")
    print("=" * 50)
    
    accuracy = get_accuracy_summary(days)
    
    print(f"Total Predictions: {accuracy['total_predictions']}")
    print(f"Predictions with Results: {accuracy['predictions_with_results']}")
    print(f"Correct Predictions: {accuracy['correct_predictions']}")
    print(f"Incorrect Predictions: {accuracy['incorrect_predictions']}")
    print(f"Accuracy Rate: {accuracy['accuracy']}%")
    
    if accuracy['predictions_with_results'] > 0:
        success_rate = (accuracy['correct_predictions'] / accuracy['predictions_with_results']) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    else:
        print("No predictions with results yet.")


def display_daily_picks_summary(days: int = 30):
    """Display daily picks summary."""
    print(f"\n📈 Daily Picks Summary (Last {days} days)")
    print("=" * 50)
    
    picks_accuracy = get_daily_picks_accuracy(days)
    
    print(f"Total Daily Picks Days: {picks_accuracy['total_daily_picks_days']}")
    print(f"Total Picks: {picks_accuracy['total_picks']}")
    print(f"Picks with Results: {picks_accuracy['picks_with_results']}")
    print(f"Correct Picks: {picks_accuracy['correct_picks']}")
    print(f"Incorrect Picks: {picks_accuracy['incorrect_picks']}")
    print(f"Accuracy Rate: {picks_accuracy['accuracy']}%")


def export_logs(output_file: str = None):
    """Export prediction logs to CSV."""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"cache/prediction_logs/predictions_export_{timestamp}.csv"
    
    print(f"\n📤 Exporting prediction logs to: {output_file}")
    print("=" * 50)
    
    try:
        export_predictions_to_csv(output_file)
        print(f"✅ Export completed successfully!")
        
        # Check if file was created
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size
            print(f"📁 File size: {file_size} bytes")
        else:
            print("⚠️ File was not created")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")


def show_recent_predictions(limit: int = 10):
    """Show recent predictions."""
    print(f"\n🕒 Recent Predictions (Last {limit})")
    print("=" * 50)
    
    logs = prediction_logger._load_daily_logs()
    
    # Filter for individual predictions (not daily picks)
    predictions = [log for log in logs if "ticker" in log and "type" not in log]
    
    # Sort by timestamp (newest first)
    predictions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    for i, pred in enumerate(predictions[:limit]):
        timestamp = pred.get("timestamp", "")
        ticker = pred.get("ticker", "")
        direction = pred.get("prediction", {}).get("direction", "")
        confidence = pred.get("prediction", {}).get("confidence", "")
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        print(f"{i+1:2d}. {formatted_time} | {ticker:6s} | {direction:4s} | {confidence:5.1f}%")


def show_recent_daily_picks(limit: int = 5):
    """Show recent daily picks."""
    print(f"\n🎯 Recent Daily Picks (Last {limit})")
    print("=" * 50)
    
    logs = prediction_logger._load_daily_logs()
    
    # Filter for daily picks
    daily_picks_logs = [log for log in logs if log.get("type") == "daily_picks"]
    
    # Sort by timestamp (newest first)
    daily_picks_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    for i, picks_log in enumerate(daily_picks_logs[:limit]):
        timestamp = picks_log.get("timestamp", "")
        picks = picks_log.get("picks", [])
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        print(f"\n{i+1}. {formatted_time}")
        for j, pick in enumerate(picks):
            ticker = pick.get("ticker", "")
            direction = pick.get("direction", "")
            confidence = pick.get("confidence", "")
            print(f"   {j+1}. {ticker:6s} | {direction:4s} | {confidence:5.1f}%")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Manage prediction logs and view accuracy statistics")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze (default: 30)")
    parser.add_argument("--export", type=str, help="Export logs to CSV file")
    parser.add_argument("--recent", type=int, default=10, help="Show recent predictions (default: 10)")
    parser.add_argument("--picks", type=int, default=5, help="Show recent daily picks (default: 5)")
    parser.add_argument("--all", action="store_true", help="Show all statistics")
    
    args = parser.parse_args()
    
    if args.all:
        display_accuracy_summary(args.days)
        display_daily_picks_summary(args.days)
        show_recent_predictions(args.recent)
        show_recent_daily_picks(args.picks)
    elif args.export:
        export_logs(args.export)
    else:
        # Default: show accuracy summary
        display_accuracy_summary(args.days)
        display_daily_picks_summary(args.days)


if __name__ == "__main__":
    main()
