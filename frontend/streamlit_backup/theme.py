"""Global dashboard CSS for InferAI Streamlit UI."""

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
  font-family: 'DM Sans', system-ui, -apple-system, sans-serif;
}

.stApp {
  background: radial-gradient(ellipse 120% 80% at 50% -20%, rgba(16, 185, 129, 0.18), transparent 50%),
              linear-gradient(168deg, #020617 0%, #0f172a 42%, #022c22 88%, #030712 100%) !important;
  color: #e2e8f0;
}

[data-testid="stAppViewContainer"] > .main {
  background: transparent;
}

.block-container {
  padding-top: 0.75rem !important;
  padding-bottom: 2rem !important;
  max-width: 900px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* Hide sidebar — single-column product layout (Disabled to show diagnostics)
section[data-testid="stSidebar"] {
  display: none !important;
}
div[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
*/

/* ----- Sidebar glass ----- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(2, 6, 23, 0.98) 100%) !important;
  border-right: 1px solid rgba(16, 185, 129, 0.2);
  box-shadow: 4px 0 40px rgba(0, 0, 0, 0.35);
}
section[data-testid="stSidebar"] .block-container {
  padding-top: 1.5rem !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] p {
  color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] .stTextInput input {
  background: rgba(15, 23, 42, 0.8) !important;
  border: 1px solid rgba(16, 185, 129, 0.25) !important;
  border-radius: 10px !important;
  color: #f1f5f9 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] .stCheckbox label span {
  color: #e2e8f0 !important;
}

/* ----- Section chrome ----- */
.nyx-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.45), transparent);
  margin: 1.75rem 0 1.25rem;
  border: 0;
}
.nyx-section-title {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: #6ee7b7;
  margin: 0 0 0.85rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.nyx-section-title--compact {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #5eead4;
  margin: 1.1rem 0 0.5rem 0;
  padding: 0;
  display: block !important;
  align-items: unset !important;
}
.nyx-section-title--compact::after {
  display: none !important;
}

/* ----- Hero ----- */
.nyx-hero {
  text-align: center;
  padding: 1.5rem 1.25rem 1.6rem;
  margin-bottom: 1.1rem;
  border-radius: 20px;
  background: linear-gradient(145deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.55) 45%, rgba(6, 95, 70, 0.2) 100%);
  border: 1px solid rgba(52, 211, 153, 0.35);
  box-shadow: 0 0 0 1px rgba(16, 185, 129, 0.08) inset,
              0 0 80px rgba(16, 185, 129, 0.1),
              0 28px 56px -18px rgba(0, 0, 0, 0.55);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
}
.nyx-hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #a7f3d0;
  background: rgba(16, 185, 129, 0.15);
  border: 1px solid rgba(52, 211, 153, 0.35);
  padding: 0.3rem 0.75rem;
  border-radius: 999px;
  margin-bottom: 0.75rem;
}
.nyx-hero h1 {
  font-size: clamp(2rem, 5vw, 2.65rem);
  font-weight: 700;
  letter-spacing: -0.03em;
  margin: 0;
  background: linear-gradient(120deg, #ecfdf5 0%, #6ee7b7 45%, #34d399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.nyx-hero p {
  margin: 0.65rem auto 0;
  max-width: 32rem;
  font-size: 0.95rem;
  line-height: 1.5;
  color: #94a3b8;
  font-weight: 400;
}

/* ----- Glass cards (HTML blocks) ----- */
.nyx-glass {
  background: rgba(15, 23, 42, 0.55);
  border: 1px solid rgba(16, 185, 129, 0.22);
  border-radius: 16px;
  padding: 1rem 1.15rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255,255,255,0.03) inset;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
.nyx-glass:hover {
  border-color: rgba(52, 211, 153, 0.4);
  box-shadow: 0 4px 28px rgba(0, 0, 0, 0.3), 0 0 40px rgba(16, 185, 129, 0.08);
}
.nyx-glass-head {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #6ee7b7;
  margin-bottom: 0.65rem;
  display: flex;
  align-items: center;
  gap: 0.45rem;
}
.nyx-glass-body {
  color: #e2e8f0;
  font-size: 0.98rem;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}

/* Prediction spotlight */
.nyx-prediction {
  text-align: center;
  padding: 1.75rem 1.5rem;
  border-radius: 18px;
  background: linear-gradient(160deg, rgba(6, 78, 59, 0.45) 0%, rgba(15, 23, 42, 0.75) 100%);
  border: 1px solid rgba(52, 211, 153, 0.45);
  box-shadow: 0 0 60px rgba(16, 185, 129, 0.15), 0 20px 40px rgba(0,0,0,0.35);
}
.nyx-prediction-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: #6ee7b7;
  margin-bottom: 0.5rem;
}
.nyx-prediction-value {
  font-size: clamp(1.65rem, 4vw, 2.15rem);
  font-weight: 700;
  color: #ecfdf5;
  letter-spacing: -0.02em;
  text-shadow: 0 0 40px rgba(52, 211, 153, 0.35);
}

/* Strength badges */
.nyx-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 0.45rem 0.95rem;
  border-radius: 999px;
  border: 1px solid;
}
.nyx-badge-strong {
  color: #a7f3d0;
  background: rgba(16, 185, 129, 0.2);
  border-color: rgba(52, 211, 153, 0.55);
  box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
}
.nyx-badge-moderate {
  color: #fde68a;
  background: rgba(245, 158, 11, 0.15);
  border-color: rgba(251, 191, 36, 0.45);
}
.nyx-badge-weak {
  color: #fecaca;
  background: rgba(248, 113, 113, 0.12);
  border-color: rgba(248, 113, 113, 0.4);
}

/* SHAP note */
.nyx-shap-note {
  font-size: 0.88rem;
  line-height: 1.5;
  color: #94a3b8;
  border-left: 3px solid #10b981;
  padding: 0.75rem 1rem;
  margin-top: 0.75rem;
  background: rgba(16, 185, 129, 0.06);
  border-radius: 0 10px 10px 0;
}

/* Footer */
.nyx-footer {
  text-align: center;
  margin-top: 1.75rem;
  padding-top: 1.15rem;
  border-top: 1px solid rgba(16, 185, 129, 0.12);
  color: #64748b;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
}

/* ----- Main controls: textarea ----- */
div[data-testid="stVerticalBlock"] > div:has(> div > .stTextArea) label,
.stTextArea label {
  color: #cbd5e1 !important;
  font-weight: 600 !important;
  font-size: 0.92rem !important;
}
.stTextArea textarea {
  background: rgba(15, 23, 42, 0.65) !important;
  border: 1px solid rgba(16, 185, 129, 0.28) !important;
  border-radius: 14px !important;
  color: #f1f5f9 !important;
  font-size: 0.95rem !important;
  line-height: 1.5 !important;
  min-height: 140px !important;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) inset !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextArea textarea:focus {
  border-color: rgba(52, 211, 153, 0.65) !important;
  box-shadow: 0 0 0 1px rgba(16, 185, 129, 0.25), 0 0 28px rgba(16, 185, 129, 0.12) !important;
  outline: none !important;
}
.stTextArea textarea::placeholder {
  color: #64748b !important;
}

/* ----- Primary button (Analyze) ----- */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%) !important;
  color: #020617 !important;
  font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.65rem 1.25rem !important;
  box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35), 0 0 0 1px rgba(255,255,255,0.1) inset !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(16, 185, 129, 0.45), 0 0 48px rgba(52, 211, 153, 0.2) !important;
  filter: brightness(1.05);
}
.stButton > button[kind="primary"]:active {
  transform: translateY(0);
}

/* Progress bar emerald */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #047857, #10b981, #6ee7b7) !important;
  border-radius: 999px !important;
  box-shadow: 0 0 12px rgba(52, 211, 153, 0.5);
}
.stProgress > div > div > div {
  background: rgba(30, 41, 59, 0.9) !important;
  border-radius: 999px !important;
}

/* Alerts */
div[data-testid="stAlert"] {
  border-radius: 12px !important;
  border: 1px solid rgba(16, 185, 129, 0.25) !important;
}

/* Dataframe / charts in glass context */
[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"] {
  border-radius: 12px !important;
  overflow: hidden !important;
  border: 1px solid rgba(16, 185, 129, 0.2) !important;
}

/* Spinner */
div[data-testid="stSpinner"] {
  color: #34d399 !important;
}

/* Hide Streamlit menu clutter optional */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* Phrase highlights returned by the API */
.nyx-html mark {
  border-radius: 6px;
  padding: 0.05rem 0.25rem;
}
.nyx-html mark.cue-authority {
  background: rgba(16, 185, 129, 0.35);
  color: #ecfdf5;
}
.nyx-html mark.cue-inference {
  background: rgba(52, 211, 153, 0.22);
  color: #ecfdf5;
}
.nyx-html mark.cue-analogy {
  background: rgba(110, 231, 183, 0.18);
  color: #022c22;
}
.nyx-html mark.cue-observation {
  background: rgba(45, 212, 191, 0.18);
  color: #042f2e;
}
.nyx-prediction-sub {
  margin-top: 0.65rem;
  font-size: 0.88rem;
  color: #94a3b8;
}
</style>
"""


def inject_dashboard_theme() -> None:
    """Inject global CSS. Call once at app start."""
    import streamlit as st

    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
