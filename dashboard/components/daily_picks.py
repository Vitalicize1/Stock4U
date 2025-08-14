from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import streamlit as st

from utils.daily_picks import compute_top_picks


def _load_cache(path: Path) -> dict:
    """Load cached daily picks."""
    if not path.exists():
        return {}
    try:
        txt = path.read_text(encoding="utf-8").strip()
        return json.loads(txt) if txt else {}
    except Exception:
        return {}


def _is_stale(payload: dict, max_age_hours: int = 24) -> bool:
    """Check if cached data is stale."""
    try:
        ts = payload.get("generated_at")
        if not ts:
            return True
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return datetime.utcnow() >= (dt.replace(tzinfo=None) + timedelta(hours=max_age_hours))
    except Exception:
        return True


def display_daily_picks(
    cache_path: str = "cache/daily_picks.json",
    auto_refresh_once: bool = True,
    top_n: int = 3,
) -> dict:
    """Display daily top picks on the dashboard."""
    
    st.subheader("📈 Daily Top Picks")
    st.markdown("Today's best stock recommendations based on our analysis.")
    
    # Load cached data
    data = _load_cache(Path(cache_path))
    
    # Auto-refresh if stale (background only - no blocking)
    if auto_refresh_once and _is_stale(data) and not st.session_state.get("_daily_picks_refreshed", False):
        # Mark as refreshed to prevent multiple background calls
        st.session_state["_daily_picks_refreshed"] = True
        # Note: Background refresh happens via scheduled job, not blocking UI
    
    # Display picks
    if data and data.get("picks"):
        picks = data["picks"]
        
        # Show generation time
        if "generated_at" in data:
            st.caption(f"Generated: {data['generated_at']}")
        
        # Display each pick
        for i, pick in enumerate(picks[:top_n]):
            ticker = pick.get("ticker", "")
            direction = pick.get("direction", "Unknown")
            confidence = pick.get("confidence", 0)
            
            # Create columns for layout
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric(f"#{i+1} {ticker}", direction)
            
            with col2:
                # Color code the direction
                if direction.upper() == "UP":
                    st.markdown("🟢 **UP**")
                elif direction.upper() == "DOWN":
                    st.markdown("🔴 **DOWN**")
                else:
                    st.markdown("🟡 **NEUTRAL**")
            
            with col3:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Add a small gap between picks
            st.markdown("---")
    
    else:
        st.info("No daily picks available. Click 'Refresh Picks' to generate new recommendations.")
    
    # Refresh button (non-blocking)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Refresh Picks"):
            st.info("Daily picks are refreshed automatically via scheduled job. Check back in a few minutes!")
    with col2:
        st.caption("💡 Daily picks are computed automatically at 2:00 PM UTC on weekdays")
    
    return data
