import streamlit as st
import json
import re
import pandas as pd
from dashboard.utils import humanize_label


def _render_reasoning(reasoning: object) -> None:
    """Render reasoning with better formatting.

    - If dict/list or JSON string: pretty-print as JSON
    - If long text: show preview and expandable full text
    - Otherwise: render as markdown to keep bullets/line breaks
    """
    # Dict/list provided directly → prefer table render if dict
    if isinstance(reasoning, dict):
        _render_parsed_json(reasoning)
        return
    if isinstance(reasoning, list):
        st.json(reasoning)
        return

    # String input
    if isinstance(reasoning, str):
        text = reasoning.strip()
        # Show any leading preamble above the JSON, if present
        preamble = None
        # Handle fenced blocks first
        fenced_match = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced_match:
            preamble = text[:fenced_match.start()].strip() or None
            json_candidate = fenced_match.group(1).strip()
            json_candidate = _normalize_json_like(json_candidate)
            try:
                obj = json.loads(json_candidate)
                # Hide noisy preamble like "Here is the analysis..."
                _render_parsed_json(obj)
                # Optionally show preamble in an expander for transparency
                if preamble:
                    with st.expander("Show LLM preamble"):
                        st.markdown(preamble)
                return
            except Exception:
                # fall through to other strategies
                pass

        # Try to parse JSON if the string looks like JSON
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                obj = json.loads(_normalize_json_like(text))
                _render_parsed_json(obj)
                return
            except Exception:
                pass

        # Otherwise, attempt to extract the first JSON object/array inside the text
        extracted = _extract_first_json_block(text)
        if extracted is not None:
            obj, before = extracted
            # Hide leading preamble in main view; make it available in an expander
            _render_parsed_json(obj)
            if before:
                with st.expander("Show LLM preamble"):
                    st.markdown(before)
            return

        # Try to extract structured fields from free text first
        parsed = _best_effort_extract_fields(text)
        if parsed:
            _render_parsed_json(parsed)
            # Keep full text accessible but not front-and-center
            with st.expander("Show full reasoning"):
                st.markdown(text)
        else:
            # Long content → preview + expander fallback
            if len(text) > 500 or text.count("\n") > 6:
                preview = text[:400].rstrip()
                if len(text) > 400:
                    preview += "..."
                st.write(preview)
                with st.expander("Show full reasoning"):
                    st.markdown(text)
            else:
                st.markdown(text)
        return

    # Fallback for unexpected types
    st.write(str(reasoning))


def _normalize_json_like(s: str) -> str:
    """Normalize common non-JSON characters to improve json parsing.

    - Replace smart quotes with straight quotes
    - Normalize dashes
    """
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
    }
    out = s
    for k, v in replacements.items():
        out = out.replace(k, v)
    # Also replace literal characters if already decoded
    out = out.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-").replace("—", "-")
    return out


def _extract_first_json_block(text: str):
    """Extract the first JSON object/array found inside arbitrary text.

    Returns tuple (parsed_obj, leading_text) or None.
    """
    # Prefer balanced object first
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        end = _find_balanced_end(text, start, opener, closer)
        if end is None:
            continue
        segment = text[start:end + 1].strip()
        candidate = _normalize_json_like(segment)
        try:
            obj = json.loads(candidate)
            leading = text[:start].strip()
            return obj, (leading or None)
        except Exception:
            pass
    return None


def _find_balanced_end(text: str, start: int, opener: str, closer: str):
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return i
    return None


def _render_parsed_json(obj: dict) -> None:
    """Render a human-friendly table for a parsed reasoning JSON object."""
    try:
        direction = obj.get("direction")
        confidence = obj.get("confidence")
        price_target = obj.get("price_target")
        pr = obj.get("price_range") or {}
        low = pr.get("low")
        high = pr.get("high")

        table = pd.DataFrame(
            {
                "Value": [
                    humanize_label(direction) if direction is not None else None,
                    confidence,
                    price_target,
                    low,
                    high,
                ]
            },
            index=[
                "Direction",
                "Confidence (%)",
                "Price Target",
                "Range Low",
                "Range High",
            ],
        )
        st.table(table)

        # Show natural-language reasoning if present
        rn = obj.get("reasoning")
        if rn:
            with st.expander("Detailed reasoning", expanded=False):
                st.markdown(str(rn))

        # Optional lists
        kf = obj.get("key_factors") or []
        rf = obj.get("risk_factors") or []
        if kf:
            st.write("Key Factors:")
            for it in kf:
                st.write(f"- {it}")
        if rf:
            st.write("Risk Factors:")
            for it in rf:
                st.write(f"- {it}")
    except Exception:
        # Fallback to pretty JSON if table rendering fails
        st.json(obj)


def _best_effort_extract_fields(text: str) -> dict | None:
    """Extract key fields from a noisy reasoning string to build a table.

    This is used when strict JSON parsing fails.
    """
    try:
        def _m(pat):
            m = re.search(pat, text, flags=re.IGNORECASE)
            return m.group(1).strip() if m else None

        direction = _m(r'"direction"\s*:\s*"([A-Z]+)"')
        conf = _m(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        pt = _m(r'"price_target"\s*:\s*(null|[0-9]+(?:\.[0-9]+)?)')
        low = _m(r'"low"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        high = _m(r'"high"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
        reasoning = _m(r'"reasoning"\s*:\s*"([\s\S]*?)"\s*(?:,\s*"|}$)')

        if direction or conf or low or high:
            parsed = {
                "direction": direction,
                "confidence": float(conf) if conf else None,
                "price_target": None if (pt == 'null') else (float(pt) if pt else None),
                "price_range": {"low": float(low) if low else None, "high": float(high) if high else None},
            }
            if reasoning:
                parsed["reasoning"] = reasoning
            return parsed
    except Exception:
        return None
    return None


def display_prediction_details(prediction_summary: dict, evaluation_summary: dict) -> None:
    """Display detailed prediction information."""

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction Analysis")

        st.write("**Direction:**", humanize_label(prediction_summary.get("direction", "Unknown")))
        st.write("**Confidence:**", f"{prediction_summary.get('confidence', 0):.1f}%")

        price_target = prediction_summary.get("price_target")
        if price_target is not None:
            st.write("**Price Target:**", f"${price_target:.2f}")
        else:
            st.write("**Price Target:**", "N/A")

        st.subheader("Reasoning")
        reasoning = prediction_summary.get("reasoning", "No reasoning provided")
        _render_reasoning(reasoning)

        st.subheader("Key Factors")
        key_factors = prediction_summary.get("key_factors", [])
        if key_factors:
            for factor in key_factors:
                st.write(f"• {factor}")
        else:
            st.write("No key factors identified")

    with col2:
        st.subheader("Evaluation Results")

        overall_score = evaluation_summary.get("overall_score", 0)
        st.metric("Overall Score", f"{overall_score:.1f}/100")

        prediction_quality = evaluation_summary.get("prediction_quality", {})
        if prediction_quality:
            st.write("**Prediction Quality:**")
            st.write(f"- Score: {prediction_quality.get('score', 0):.1f}/100")
            st.write(f"- Confidence: {prediction_quality.get('confidence_adequacy', 'Unknown')}")
            st.write(f"- Reasoning: {prediction_quality.get('reasoning_quality', 'Unknown')}")

        # Display sentiment integration if available
        if "sentiment_integration" in evaluation_summary:
            sentiment_integration = evaluation_summary.get("sentiment_integration", {})
            st.write("**Sentiment Integration:**")
            integrated_analysis = sentiment_integration.get("integrated_analysis", {})
            if integrated_analysis:
                st.write(f"- Integrated Score: {integrated_analysis.get('integrated_score', 0):.1f}/100")
                st.write(f"- Technical Contribution: {integrated_analysis.get('technical_contribution', 0):.1f}")
                st.write(f"- Sentiment Contribution: {integrated_analysis.get('sentiment_contribution', 0):.1f}")


