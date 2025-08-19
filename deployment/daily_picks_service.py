#!/usr/bin/env python3
"""
Standalone Daily Picks Service
Generates and serves daily stock picks for all Stock4U instances.
Can be deployed independently or as part of the main API.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the daily picks logic
from utils.daily_picks import compute_top_picks, run_daily_picks_job

app = FastAPI(
    title="Stock4U Daily Picks Service",
    description="Centralized daily stock picks for all Stock4U instances",
    version="1.0.0"
)

# CORS middleware to allow all origins for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Configuration
PICKS_FILE = Path(os.getenv("DAILY_PICKS_PATH", "cache/daily_picks.json"))
PICKS_MAX_AGE_HOURS = int(os.getenv("DAILY_PICKS_MAX_AGE_HOURS", "24"))
AUTO_GENERATE = os.getenv("DAILY_PICKS_AUTO_GENERATE", "true").lower() == "true"

def load_picks_file() -> Dict:
    """Load picks from the JSON file."""
    if not PICKS_FILE.exists():
        return {"generated_at": None, "picks": []}
    
    try:
        with open(PICKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) and "picks" in data else {"generated_at": None, "picks": []}
    except Exception:
        return {"generated_at": None, "picks": []}

def is_picks_stale(data: Dict) -> bool:
    """Check if picks data is stale."""
    if not data.get("generated_at"):
        return True
    
    try:
        ts = data["generated_at"]
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return datetime.utcnow() >= (dt.replace(tzinfo=None) + timedelta(hours=PICKS_MAX_AGE_HOURS))
    except Exception:
        return True

@app.get("/")
def root():
    """Service status and info."""
    return {
        "service": "Stock4U Daily Picks",
        "status": "active",
        "endpoints": {
            "picks": "/daily_picks",
            "health": "/health",
            "generate": "/generate" 
        },
        "config": {
            "picks_file": str(PICKS_FILE),
            "max_age_hours": PICKS_MAX_AGE_HOURS,
            "auto_generate": AUTO_GENERATE
        }
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    data = load_picks_file()
    is_stale = is_picks_stale(data)
    
    return {
        "status": "ok",
        "picks_available": len(data.get("picks", [])) > 0,
        "picks_fresh": not is_stale,
        "last_generated": data.get("generated_at"),
        "picks_file_exists": PICKS_FILE.exists()
    }

@app.get("/daily_picks")
def get_daily_picks():
    """Get the latest daily picks. Auto-generates if stale and enabled."""
    data = load_picks_file()
    
    # Auto-generate if stale and enabled
    if AUTO_GENERATE and is_picks_stale(data):
        try:
            print("🔄 Auto-generating fresh daily picks...")
            fresh_data = run_daily_picks_job()
            return fresh_data
        except Exception as e:
            print(f"⚠️ Auto-generation failed: {e}")
            # Return stale data as fallback
    
    return data

@app.post("/generate")
def generate_picks():
    """Manually trigger picks generation."""
    try:
        print("🔄 Manually generating daily picks...")
        data = run_daily_picks_job()
        return {
            "status": "success",
            "message": "Daily picks generated successfully",
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate picks: {str(e)}")

@app.get("/picks/status")
def picks_status():
    """Detailed status of picks data."""
    data = load_picks_file()
    is_stale = is_picks_stale(data)
    
    status = {
        "picks_count": len(data.get("picks", [])),
        "generated_at": data.get("generated_at"),
        "is_stale": is_stale,
        "file_path": str(PICKS_FILE),
        "file_exists": PICKS_FILE.exists(),
        "auto_generate": AUTO_GENERATE
    }
    
    if data.get("picks"):
        status["sample_pick"] = data["picks"][0]
    
    return status

if __name__ == "__main__":
    # Ensure cache directory exists
    PICKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate initial picks if none exist
    if AUTO_GENERATE and not PICKS_FILE.exists():
        print("🚀 Generating initial daily picks...")
        try:
            run_daily_picks_job()
        except Exception as e:
            print(f"⚠️ Initial generation failed: {e}")
    
    # Start the service
    port = int(os.getenv("DAILY_PICKS_PORT", "8001"))
    host = os.getenv("DAILY_PICKS_HOST", "0.0.0.0")
    
    print(f"🚀 Starting Daily Picks Service on {host}:{port}")
    print(f"📊 Daily picks will be served at http://{host}:{port}/daily_picks")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
