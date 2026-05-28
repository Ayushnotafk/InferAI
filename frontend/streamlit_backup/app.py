"""
InferAI — Streamlit client (end-user layout).

API URL is read from ``INFERAI_API_URL``, then ``http://127.0.0.1:8000``.
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
    or "http://127.0.0.1:8000"
)

ARG_KEY = "infer_argument_text"

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


def analyze(text: str, base_url: str, include_shap: bool, alpha: float) -> dict:
    url = base_url.rstrip("/") + "/analyze"
    payload = {"text": text, "include_shap": include_shap, "alpha": alpha}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def _render_shap_block(sh: dict) -> None:
    st.markdown(nx.section_title("Contribution analysis"), unsafe_allow_html=True)
    if isinstance(sh, dict) and "error" in sh:
        st.error(sh["error"])
        return

    note = sh.get("note") or ""
    st.markdown(nx.shap_note_block(note), unsafe_allow_html=True)

    rows = sh.get("top_embedding_contributions") or []
    if not rows:
        st.info("No contribution rows returned.")
        return

    df = pd.DataFrame(rows)
    df = df.rename(columns={"embedding_dim": "Dimension", "shap_value": "SHAP"})
    df["Dimension"] = df["Dimension"].astype(str)
    df["|SHAP|"] = df["SHAP"].abs()
    df = df.sort_values("|SHAP|", ascending=False)

    chart_df = df.set_index("Dimension")[["SHAP"]].head(16)

    c1, c2 = st.columns((1.15, 1.0), gap="medium")
    with c1:
        st.markdown(
            '<p class="nyx-glass-head" style="margin-bottom:0.45rem;">By dimension</p>',
            unsafe_allow_html=True,
        )
        cdf = chart_df.reset_index()
        st.bar_chart(
            cdf,
            x="SHAP",
            y="Dimension",
            horizontal=True,
            height=300,
            sort=False,
        )
    with c2:
        st.markdown(
            '<p class="nyx-glass-head" style="margin-bottom:0.45rem;">Top dimensions</p>',
            unsafe_allow_html=True,
        )
        show = df[["Dimension", "SHAP"]].head(16).copy()
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            height=300,
        )


def _render_diagnostic_comparison(data: dict) -> None:
    hybrid_data = data.get("hybrid")
    if not hybrid_data:
        return

    st.markdown(nx.section_title("Neuro-Symbolic Diagnostic Analysis"), unsafe_allow_html=True)

    # 1. Weights display
    weights = hybrid_data.get("weights", {})
    ml_w = weights.get("ml", 0.8)
    rule_w = weights.get("rules", 0.2)

    st.markdown(
        f"""
        <div class="nyx-glass" style="margin-bottom: 1.15rem; padding: 0.85rem 1.15rem;">
            <div class="nyx-glass-head">⚙️ Real-time Weight Allocation</div>
            <div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 0.4rem; color: #cbd5e1;">
                <span><strong>Statistical Machine Learning (α):</strong> {ml_w * 100:.0f}%</span>
                <span><strong>Symbolic Rule Heuristics (1 - α):</strong> {rule_w * 100:.0f}%</span>
            </div>
            <div style="background: rgba(30, 41, 59, 0.9); height: 10px; border-radius: 999px; overflow: hidden; display: flex; border: 1px solid rgba(16, 185, 129, 0.15);">
                <div style="background: linear-gradient(90deg, #10b981, #34d399); width: {ml_w * 100}%; height: 100%;"></div>
                <div style="background: linear-gradient(90deg, #f59e0b, #fbbf24); width: {rule_w * 100}%; height: 100%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Grouped Probability Chart
    classes = hybrid_data.get("class_order", ["Anumana", "Pratyaksha", "Shabda", "Upamana"])
    ml_probs = [p * 100 for p in hybrid_data.get("ml_probs", [0.0] * len(classes))]
    rule_probs = [p * 100 for p in hybrid_data.get("rule_probs", [0.0] * len(classes))]
    fused_probs = [p * 100 for p in hybrid_data.get("fused_probs", [0.0] * len(classes))]

    df_probs = pd.DataFrame(
        {
            "Pramāṇa": classes,
            "Statistical ML (%)": ml_probs,
            "Symbolic Rules (%)": rule_probs,
            "Combined Hybrid (%)": fused_probs,
        }
    ).set_index("Pramāṇa")

    c1, c2 = st.columns((1.3, 1.0), gap="medium")
    with c1:
        st.markdown(
            '<p class="nyx-glass-head" style="margin-bottom:0.45rem;">Probability Blending Comparison</p>',
            unsafe_allow_html=True,
        )
        st.bar_chart(df_probs, height=280)

    with c2:
        st.markdown(
            '<p class="nyx-glass-head" style="margin-bottom:0.45rem;">Diagnostic Data Table</p>',
            unsafe_allow_html=True,
        )
        # Format dataframe values to show percentage
        df_display = df_probs.copy()
        for col in df_display.columns:
            df_display[col] = df_display[col].map("{:.1f}%".format)
        st.dataframe(df_display, use_container_width=True, height=240)


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

    st.markdown(nx.section_title("Result"), unsafe_allow_html=True)

    col_pred, col_metrics = st.columns((1.05, 1.0), gap="medium")

    with col_pred:
        sub = f"Reference label: {ml_label}"
        st.markdown(
            nx.prediction_spotlight(str(hybrid_label), subtitle=sub),
            unsafe_allow_html=True,
        )

    with col_metrics:
        st.markdown(
            '<p class="nyx-glass-head" style="margin-bottom:0.45rem;">Confidence</p>',
            unsafe_allow_html=True,
        )
        st.progress(adj_conf / 100.0, text=f"{adj_conf:.0f}% combined")
        st.progress(ml_conf / 100.0, text=f"{ml_conf:.0f}% model")
        st.markdown(
            '<p class="nyx-glass-head" style="margin-top:0.85rem;margin-bottom:0.45rem;">Reasoning strength</p>',
            unsafe_allow_html=True,
        )
        st.markdown(nx.strength_badge_html(str(strength)), unsafe_allow_html=True)

    # Diagnostic views
    _render_diagnostic_comparison(data)

    st.markdown(nx.section_title("Extracted structure"), unsafe_allow_html=True)
    c_claim, c_evi = st.columns(2, gap="medium")
    with c_claim:
        st.markdown(nx.glass_card("Claim", str(claim), icon="◈"), unsafe_allow_html=True)
    with c_evi:
        st.markdown(
            nx.glass_card("Premises & evidence", str(premises), icon="◇"),
            unsafe_allow_html=True,
        )

    if indicators:
        st.markdown(
            '<p class="nyx-glass-head" style="margin:0.35rem 0 0.35rem;">Signals</p>',
            unsafe_allow_html=True,
        )
        st.markdown(" · ".join(f"`{i}`" for i in indicators))

    if highlighted:
        st.markdown(nx.section_title("Highlighted cues"), unsafe_allow_html=True)
        st.markdown(
            f'<div class="nyx-html nyx-glass" style="padding:0.85rem 1rem;">{highlighted}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(nx.section_title("Explanation"), unsafe_allow_html=True)
    st.markdown(nx.glass_card("Summary", str(explanation), icon="✦"), unsafe_allow_html=True)

    if include_shap and "shap" in data:
        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
        _render_shap_block(data["shap"])


def main() -> None:
    st.set_page_config(
        page_title="InferAI Diagnostic Tool",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_dashboard_theme()

    # Initialize session state for caching responses and parameters
    if ARG_KEY not in st.session_state:
        st.session_state[ARG_KEY] = ""
    if "last_analyzed_text" not in st.session_state:
        st.session_state["last_analyzed_text"] = ""
    if "last_alpha" not in st.session_state:
        st.session_state["last_alpha"] = 0.8
    if "last_include_shap" not in st.session_state:
        st.session_state["last_include_shap"] = False
    if "last_response_data" not in st.session_state:
        st.session_state["last_response_data"] = None

    # Sidebar: Diagnostic Settings Panel
    st.sidebar.markdown(
        '<p class="nyx-section-title" style="margin-top: 0.5rem; margin-bottom: 0.8rem; font-size: 0.82rem;">🛠️ Diagnostic Settings</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.4; margin-bottom: 1.25rem;">'
        "Tune the Neuro-Symbolic weight balance (α) in real-time. This shifts the classification blending "
        "between statistical ML and heuristic rule cues.</div>",
        unsafe_allow_html=True,
    )

    alpha = st.sidebar.slider(
        "Neuro-Symbolic Weight (alpha)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["last_alpha"],
        step=0.05,
        help="Weight of the statistical ML model (alpha) vs. symbolic heuristic rules (1 - alpha).",
    )

    st.sidebar.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)

    include_shap = st.sidebar.checkbox(
        "Include SHAP contributions",
        value=st.session_state["last_include_shap"],
        help="Calculate and draw top embedding dimension contributions (SHAP values).",
    )

    st.sidebar.markdown(
        f"""
        <div style="margin-top: 2rem; border-top: 1px solid rgba(16, 185, 129, 0.15); padding-top: 1rem; font-size: 0.78rem; color: #64748b;">
            <strong>System Status:</strong> De-coupled REST API Mode<br/>
            <strong>Endpoint:</strong> <code>{DEFAULT_API}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, main_col, _ = st.columns([0.05, 1.0, 0.05], gap="small")
    with main_col:
        st.markdown(nx.hero_block(), unsafe_allow_html=True)

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

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

        text = st.text_area(
            "Your argument",
            height=156,
            placeholder="Paste or type a short argument to analyze…",
            label_visibility="collapsed",
            key=ARG_KEY,
        )

        run = st.button("Analyze", type="primary", use_container_width=True)

        # Interaction / execution logic
        should_analyze = False
        target_text = ""

        # 1. User triggered manual analysis button
        if run:
            if not text.strip():
                st.warning("Please enter text to analyze.")
            else:
                should_analyze = True
                target_text = text.strip()

        # 2. Interactive updates: user changed parameters on a previously analyzed text
        elif (
            st.session_state["last_analyzed_text"]
            and (
                alpha != st.session_state["last_alpha"]
                or include_shap != st.session_state["last_include_shap"]
            )
        ):
            should_analyze = True
            target_text = st.session_state["last_analyzed_text"]

        if should_analyze:
            with st.spinner("Analyzing and blending weights…"):
                try:
                    data = analyze(target_text, DEFAULT_API, include_shap, alpha)
                except httpx.HTTPError as e:
                    st.error("Could not reach the analysis service. Try again in a moment.")
                else:
                    # Update cache
                    st.session_state["last_analyzed_text"] = target_text
                    st.session_state["last_alpha"] = alpha
                    st.session_state["last_include_shap"] = include_shap
                    st.session_state["last_response_data"] = data
                    st.success("Analysis complete.")

        # Show cached results if we have them
        if st.session_state["last_response_data"] is not None:
            _render_results(st.session_state["last_response_data"], st.session_state["last_include_shap"])

        st.markdown(nx.footer_block(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
