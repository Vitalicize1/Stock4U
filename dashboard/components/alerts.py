from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, List, Dict, Tuple
import streamlit as st
import pandas as pd


STATE_PATH_DEFAULT = "cache/metrics/alerts_state.json"


def _normalize_ts(rec: Dict[str, Any]) -> str:
    for k in ("timestamp", "time", "ts", "dt"):
        v = rec.get(k)
        if v:
            try:
                # Try to parse common formats; otherwise return as-is
                if isinstance(v, (int, float)):
                    return datetime.fromtimestamp(float(v)).isoformat(sep=" ", timespec="seconds")
                return str(v)
            except Exception:
                return str(v)
    return ""


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _parse_dt(value: Any) -> datetime | None:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value))
        s = str(value).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        content = state_path.read_text(encoding="utf-8").strip()
        return json.loads(content) if content else {}
    except Exception:
        return {}


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    try:
        _ensure_parent(state_path)
        state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def _status_for(alert_id: str, state: Dict[str, Any]) -> Tuple[str, datetime | None]:
    entry = state.get(alert_id) or {}
    status = entry.get("status", "new")
    snooze_until = _parse_dt(entry.get("snooze_until"))
    return status, snooze_until


def _is_pending(alert_id: str, state: Dict[str, Any]) -> bool:
    status, snooze_until = _status_for(alert_id, state)
    if status == "resolved":
        return False
    if snooze_until is None:
        return True
    return datetime.now(tz=snooze_until.tzinfo) >= snooze_until


def display_alerts(alerts_file: str = "cache/metrics/alerts.log", max_rows: int = 200, simple: bool = False) -> None:
    st.subheader("Alerts (latest)")
    p = Path(alerts_file)
    if not p.exists():
        st.info("No alerts yet.")
        return
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        lines = lines[-max_rows:]
        rows: List[Dict[str, Any]] = []
        for idx, x in enumerate(lines):
            if not x.strip():
                continue
            try:
                rec = json.loads(x)
                # Best-effort ID if not present
                rec.setdefault("id", rec.get("alert_id") or f"{rec.get('timestamp', idx)}:{rec.get('ticker', '')}:{rec.get('rule', rec.get('title','rule'))}")
                rows.append(rec)
            except Exception:
                continue
        if not rows:
            st.info("No alerts to display.")
            return

        # Load persistent alert state
        state_path = Path(STATE_PATH_DEFAULT)
        state = _load_state(state_path)

        # Prepare dataframe for filtering and summary
        df = pd.DataFrame(rows)
        if "level" not in df:
            df["level"] = "info"
        df["level"] = df["level"].astype(str).str.lower()
        if "title" not in df:
            df["title"] = ""
        if "body" not in df:
            df["body"] = ""
        if "id" not in df:
            df["id"] = [f"row-{i}" for i in range(len(df))]
        if "timestamp" not in df:
            df["timestamp"] = [ _normalize_ts(r) for r in rows ]
        else:
            # Normalize textual timestamp for display
            df["timestamp"] = [ _normalize_ts(r) for r in rows ]

        # Derive status/pending from state
        statuses: List[str] = []
        pendings: List[bool] = []
        for alert_id in df["id"].tolist():
            stt, snz = _status_for(alert_id, state)
            statuses.append(stt)
            pendings.append(_is_pending(alert_id, state))
        df["status"] = statuses
        df["pending"] = pendings

        # Header metrics (simplified when simple mode is on)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("New", int((df["status"] == "new").sum()))
        with c2:
            st.metric("Pending", int(df["pending"].sum()))
        if not simple:
            with c3:
                st.metric("Snoozed", int((df["status"] == "snoozed").sum()))
            with c4:
                st.metric("Resolved Today", int(((df["status"] == "resolved")).sum()))

        if simple:
            # Simple view: just show recent pending cards with minimal filters
            levels = sorted(df["level"].unique().tolist())
            lvl = st.selectbox("Level", options=["all"] + levels, index=0)
            query = st.text_input("Search", value="")
            mask = df["pending"] == True
            if lvl != "all":
                mask &= df["level"] == lvl
            if query:
                ql = query.lower()
                mask &= df["title"].astype(str).str.lower().str.contains(ql) | df["body"].astype(str).str.lower().str.contains(ql)
            dfv = df[mask].copy().iloc[::-1].head(max_rows)
            for _, rec in dfv.iterrows():
                level = str(rec.get("level", "info")).lower()
                title = str(rec.get("title", "")).strip() or level.title()
                body = str(rec.get("body", "")).strip()
                ts = rec.get("timestamp") or _normalize_ts(rec.to_dict())
                rule = rec.get("rule") or rec.get("category") or ""
                ticker = rec.get("ticker") or ""
                meta_parts = [p for p in [ticker, rule, ts] if p]
                with st.container():
                    st.markdown(f"**{title}**")
                    if body:
                        st.write(body)
                    if meta_parts:
                        st.caption(" · ".join(meta_parts))
            return

        tab_pending, tab_history = st.tabs(["Pending", "History"])

        def render_view(df_in: pd.DataFrame, pending_only: bool) -> None:
            # Filters row
            colf1, colf2, colf3 = st.columns([1, 2, 1])
            with colf1:
                levels = sorted(df_in["level"].unique().tolist())
                level_filter = st.multiselect("Level", options=levels, default=levels, key=f"level_{'p' if pending_only else 'h'}")
            with colf2:
                query = st.text_input("Search (title/body)", value="", key=f"q_{'p' if pending_only else 'h'}")
            with colf3:
                view_mode = st.selectbox("View", ["Cards", "Table"], index=0, key=f"view_{'p' if pending_only else 'h'}")

            # Apply filters
            mask = df_in["level"].isin(level_filter)
            if query:
                ql = query.lower()
                mask &= df_in["title"].astype(str).str.lower().str.contains(ql) | df_in["body"].astype(str).str.lower().str.contains(ql)
            dfv = df_in[mask].copy()

            # Selection controls
            sel_key = f"sel_{'p' if pending_only else 'h'}"
            selected_ids = set(st.session_state.get(sel_key, set()))
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            with sc1:
                if st.button("Select all", key=f"selall_{sel_key}"):
                    st.session_state[sel_key] = set(dfv["id"].tolist())
                    st.experimental_rerun()
            with sc2:
                if st.button("Clear", key=f"clear_{sel_key}"):
                    st.session_state[sel_key] = set()
                    st.experimental_rerun()
            with sc3:
                if st.button("Acknowledge", key=f"ack_{sel_key}"):
                    for aid in selected_ids:
                        stt, _ = _status_for(aid, state)
                        state[aid] = {**state.get(aid, {}), "status": "acknowledged", "ack_at": datetime.now().isoformat()}
                    _save_state(state_path, state)
                    st.experimental_rerun()
            with sc4:
                if st.button("Snooze 1h", key=f"sn1_{sel_key}"):
                    until = datetime.now() + timedelta(hours=1)
                    for aid in selected_ids:
                        state[aid] = {**state.get(aid, {}), "status": "snoozed", "snooze_until": until.isoformat()}
                    _save_state(state_path, state)
                    st.experimental_rerun()
            with sc5:
                if st.button("Snooze 1d", key=f"sn2_{sel_key}"):
                    until = datetime.now() + timedelta(days=1)
                    for aid in selected_ids:
                        state[aid] = {**state.get(aid, {}), "status": "snoozed", "snooze_until": until.isoformat()}
                    _save_state(state_path, state)
                    st.experimental_rerun()
            with sc6:
                if st.button("Resolve", key=f"res_{sel_key}"):
                    for aid in selected_ids:
                        state[aid] = {**state.get(aid, {}), "status": "resolved", "resolved_at": datetime.now().isoformat()}
                    _save_state(state_path, state)
                    st.experimental_rerun()

            # Show newest first
            dfv = dfv.iloc[::-1]

            if view_mode == "Table":
                table_cols = [c for c in ("id", "timestamp", "level", "status", "ticker", "rule", "title", "body") if c in dfv.columns]
                if table_cols:
                    # Add selection checkboxes in a separate column
                    def _row_checkbox(row) -> str:
                        cid = f"chk_{row['id']}"
                        checked = row['id'] in selected_ids
                        if st.checkbox("", value=checked, key=cid):
                            selected_ids.add(row['id'])
                        else:
                            selected_ids.discard(row['id'])
                        return ""

                    st.write("Select rows using the checkboxes on the left.")
                    # Render checkboxes + table
                    for _, row in dfv.iterrows():
                        cols = st.columns([0.1, 0.9])
                        with cols[0]:
                            cid = f"chk_{row['id']}"
                            checked = row['id'] in selected_ids
                            if st.checkbox("", value=checked, key=cid):
                                selected_ids.add(row['id'])
                            else:
                                selected_ids.discard(row['id'])
                        with cols[1]:
                            st.write({k: row.get(k) for k in table_cols})
                    st.session_state[sel_key] = selected_ids
                else:
                    st.info("No columns to display.")
            else:
                for _, rec in dfv.iterrows():
                    level = str(rec.get("level", "info")).lower()
                    title = str(rec.get("title", "")).strip() or level.title()
                    body = str(rec.get("body", "")).strip()
                    kv = rec.get("kv") if isinstance(rec.get("kv"), dict) else None
                    ts = rec.get("timestamp") or _normalize_ts(rec.to_dict())
                    rule = rec.get("rule") or rec.get("category") or ""
                    ticker = rec.get("ticker") or ""
                    status = rec.get("status", "new")
                    row_id = rec.get("id")

                    cols = st.columns([0.05, 0.95])
                    with cols[0]:
                        cid = f"chk_{row_id}"
                        checked = row_id in selected_ids
                        if st.checkbox("", value=checked, key=cid):
                            selected_ids.add(row_id)
                        else:
                            selected_ids.discard(row_id)
                    with cols[1]:
                        meta_parts = [p for p in [ticker, rule, f"status={status}", ts] if p]
                        st.markdown(f"**{title}**")
                        if body:
                            st.write(body)
                        if meta_parts:
                            st.caption(" · ".join(meta_parts))

                        # Per-alert actions
                        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
                        with ac1:
                            if st.button("Acknowledge", key=f"p_ack_{row_id}"):
                                state[row_id] = {**state.get(row_id, {}), "status": "acknowledged", "ack_at": datetime.now().isoformat()}
                                _save_state(state_path, state)
                                st.experimental_rerun()
                        with ac2:
                            if st.button("Snooze 1h", key=f"p_sn1_{row_id}"):
                                until = datetime.now() + timedelta(hours=1)
                                state[row_id] = {**state.get(row_id, {}), "status": "snoozed", "snooze_until": until.isoformat()}
                                _save_state(state_path, state)
                                st.experimental_rerun()
                        with ac3:
                            if st.button("Snooze 1d", key=f"p_sn2_{row_id}"):
                                until = datetime.now() + timedelta(days=1)
                                state[row_id] = {**state.get(row_id, {}), "status": "snoozed", "snooze_until": until.isoformat()}
                                _save_state(state_path, state)
                                st.experimental_rerun()
                        with ac4:
                            if st.button("Resolve", key=f"p_res_{row_id}"):
                                state[row_id] = {**state.get(row_id, {}), "status": "resolved", "resolved_at": datetime.now().isoformat()}
                                _save_state(state_path, state)
                                st.experimental_rerun()
                        with ac5:
                            if kv is not None:
                                with st.expander("Details", expanded=False):
                                    st.json(kv)

                    st.session_state[sel_key] = selected_ids

        with tab_pending:
            render_view(df[df["pending"] == True], pending_only=True)
        with tab_history:
            render_view(df[df["pending"] == False], pending_only=False)
    except Exception as e:
        st.error(f"Failed to load alerts: {e}")


