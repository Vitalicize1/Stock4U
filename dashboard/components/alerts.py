from __future__ import annotations

import json
from pathlib import Path
import streamlit as st


def display_alerts(alerts_file: str = "cache/metrics/alerts.log", max_rows: int = 200) -> None:
    st.subheader("⚠️ Alerts (latest)")
    p = Path(alerts_file)
    if not p.exists():
        st.info("No alerts yet.")
        return
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        lines = lines[-max_rows:]
        rows = [json.loads(x) for x in lines if x.strip()]
        if not rows:
            st.info("No alerts to display.")
            return
        # Show newest first
        rows = list(reversed(rows))
        for rec in rows:
            level = (rec.get("level") or "info").lower()
            title = rec.get("title", "")
            body = rec.get("body", "")
            kv = rec.get("kv", {})
            box = st.warning if level in ("warning", "warn") else st.error if level == "error" else st.info
            with box:
                st.markdown(f"**{title}**")
                st.write(body)
                if kv:
                    st.json(kv)
    except Exception as e:
        st.error(f"Failed to load alerts: {e}")


