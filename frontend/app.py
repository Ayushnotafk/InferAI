"""
InferAI — Streamlit client (end-user layout).

API URL is read from ``INFERAI_API_URL``, or ``NYAYAX_API_URL`` for backward compatibility,
then ``http://127.0.0.1:8000``.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os

import httpx
import pandas as pd
import streamlit as st

from frontend import components as nx
from frontend.theme import inject_dashboard_theme

DEFAULT_API = (
    os.environ.get("INFERAI_API_URL")
    or os.environ.get("NYAYAX_API_URL")
    or "http://127.0.0.1:8000"
)

ARG_KEY = "nyx_argument_text"

DEMO_EXAMPLES = {
    "observation": (
        "On the telemetry screen, the pressure trace flatlined for six seconds after the valve command. "
        "The operator log records the same gap on two consecutive shifts."
    ),
    "inference": (
        "Latency spikes only during failover in this service, which suggests a race in the controller path. "
        "Therefore the next debugging step should focus on shared mutable state."
    ),
    "analogy": (
        "Teaching students about attention in transformers is similar to explaining a spotlight operator: "
        "the model learns where to look next based on what it already took in."
    ),
    "authority": (
        "According to the published infection-control guideline, hand hygiene remains a cornerstone of prevention. "
        "Experts in the field treat that recommendation as a minimum bar for compliance audits."
    ),
}


def analyze(text: str, base_url: str, include_shap: bool) -> dict:
    url = base_url.rstrip("/") + "/analyze"
    payload = {"text": text, "include_shap": include_shap}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def _render_shap_block(sh: dict) -> None:
    st.markdown(nx.section_title("Contribution analysis", "bar-chart"), unsafe_allow_html=True)

    if isinstance(sh, dict) and "error" in sh:
        st.error(sh["error"])
        return

    note = sh.get("note") or ""
    if note:
        st.markdown(nx.shap_note_block(note), unsafe_allow_html=True)

    rows = sh.get("top_embedding_contributions") or []
    if not rows:
        st.caption("No contribution rows returned.")
        return

    df = pd.DataFrame(rows)
    df = df.rename(columns={"embedding_dim": "Dimension", "shap_value": "SHAP"})
    df["Dimension"] = df["Dimension"].astype(str)
    df["|SHAP|"] = df["SHAP"].abs()
    df = df.sort_values("|SHAP|", ascending=False)

    chart_df = df.set_index("Dimension")[["SHAP"]].head(16)

    c1, c2 = st.columns((1.15, 1.0), gap="large")
    with c1:
        st.caption("By dimension")
        cdf = chart_df.reset_index()
        st.bar_chart(
            cdf,
            x="SHAP",
            y="Dimension",
            horizontal=True,
            height=320,
            sort=False,
        )
    with c2:
        st.caption("Top dimensions")
        show = df[["Dimension", "SHAP"]].head(16).copy()
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            height=320,
        )


def _render_results(data: dict, include_shap: bool) -> None:
    hybrid_label = data.get("hybrid_predicted_pramana") or data.get("predicted_pramana", "—")
    ml_label = data.get("predicted_pramana", "—")
    ml_conf = nx.normalize_confidence(data.get("confidence"))
    adj_conf = nx.normalize_confidence(data.get("adjusted_confidence", data.get("confidence")))
    strength = data.get("reasoning_strength", "—")
    claim = data.get("claim", "") or "—"
    premises = data.get("premises") or data.get("evidence", "") or "—"
    explanation = data.get("explanation", "") or "—"
    indicators = data.get("reasoning_indicators") or []
    highlighted = data.get("highlighted_html", "")
    adaptive_alpha = data.get("adaptive_alpha")
    routing_reason = data.get("routing_reason", "")
    fallacy_detected = data.get("fallacy_detected", False)
    fallacy_type = data.get("fallacy_type")
    fallacy_explanation = data.get("fallacy_explanation")

    st.markdown(nx.section_title("Results", "sparkles"), unsafe_allow_html=True)
    st.markdown(
        nx.metrics_row(
            str(hybrid_label),
            str(ml_label),
            adj_conf,
            str(strength),
            adaptive_alpha=float(adaptive_alpha) if adaptive_alpha is not None else None,
            routing_reason=str(routing_reason) if routing_reason else "",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(nx.spacer("md"), unsafe_allow_html=True)
    st.markdown(nx.confidence_bars(adj_conf, ml_conf), unsafe_allow_html=True)

    if fallacy_detected and fallacy_type:
        st.markdown(
            nx.fallacy_warning_card(str(fallacy_type), str(fallacy_explanation or "")),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(nx.fallacy_ok_card(), unsafe_allow_html=True)

    st.markdown(nx.section_title("Explanation", "file-text"), unsafe_allow_html=True)
    st.markdown(
        nx.content_card("Summary", str(explanation), large=True, icon_name="file-text"),
        unsafe_allow_html=True,
    )

    st.markdown(nx.section_title("Extracted structure", "layers"), unsafe_allow_html=True)
    c_claim, c_evi = st.columns(2, gap="large")
    with c_claim:
        st.markdown(
            nx.content_card("Claim", str(claim), icon_name="message"),
            unsafe_allow_html=True,
        )
    with c_evi:
        st.markdown(
            nx.content_card("Premises & evidence", str(premises), icon_name="layers"),
            unsafe_allow_html=True,
        )

    if indicators:
        st.markdown(nx.tags_block(indicators), unsafe_allow_html=True)

    if highlighted:
        st.markdown(nx.highlight_block(highlighted), unsafe_allow_html=True)

    if include_shap and "shap" in data:
        st.markdown(nx.spacer("md"), unsafe_allow_html=True)
        _render_shap_block(data["shap"])


def main() -> None:
    st.set_page_config(
        page_title="InferAI",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_dashboard_theme()

    if ARG_KEY not in st.session_state:
        st.session_state[ARG_KEY] = ""

    _, main_col, _ = st.columns([0.05, 1.0, 0.05], gap="small")
    with main_col:
        st.markdown(nx.header_block(), unsafe_allow_html=True)

        st.markdown(nx.section_title("Input", "message"), unsafe_allow_html=True)
        st.markdown(nx.examples_label(), unsafe_allow_html=True)

        d1, d2, d3, d4 = st.columns(4, gap="small")
        with d1:
            if st.button("Observation", use_container_width=True):
                st.session_state[ARG_KEY] = DEMO_EXAMPLES["observation"]
                st.rerun()
        with d2:
            if st.button("Inference", use_container_width=True):
                st.session_state[ARG_KEY] = DEMO_EXAMPLES["inference"]
                st.rerun()
        with d3:
            if st.button("Analogy", use_container_width=True):
                st.session_state[ARG_KEY] = DEMO_EXAMPLES["analogy"]
                st.rerun()
        with d4:
            if st.button("Authority", use_container_width=True):
                st.session_state[ARG_KEY] = DEMO_EXAMPLES["authority"]
                st.rerun()

        st.markdown(nx.spacer("md"), unsafe_allow_html=True)

        text = st.text_area(
            "Argument text",
            height=200,
            placeholder="Paste or type a short argument to analyze…",
            label_visibility="collapsed",
            key=ARG_KEY,
        )

        include_shap = st.checkbox(
            "Include contribution chart (SHAP)",
            value=False,
            help="Optional embedding-level attribution chart. First run may take longer.",
        )

        st.markdown(nx.spacer("md"), unsafe_allow_html=True)

        _, btn_col, _ = st.columns([0.9, 1.4, 0.9])
        with btn_col:
            run = st.button("Analyze", type="primary", use_container_width=True)

        if run:
            if not text.strip():
                st.warning("Please enter text to analyze.")
            else:
                with st.spinner("Analyzing…"):
                    try:
                        data = analyze(text.strip(), DEFAULT_API, include_shap)
                        _render_results(data, include_shap)

                    except httpx.HTTPStatusError as e:
                        st.error(f"Backend returned HTTP {e.response.status_code}")
                        st.code(e.response.text)

                    except httpx.RequestError as e:
                        st.error(
                            f"Could not connect to backend at `{DEFAULT_API}`.\n\n"
                            f"Start the API first:\n\n"
                            f"`python -m uvicorn api.app:app --reload`\n\n{e}"
                        )

                    except Exception:
                        import traceback

                        st.error("Unexpected frontend error")
                        st.code(traceback.format_exc())

        st.markdown(nx.footer_block(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
