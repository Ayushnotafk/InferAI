"""Global dashboard CSS for InferAI Streamlit UI."""

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap');

:root {
  --bg: #0F172A;
  --surface: #111827;
  --surface-2: #152033;
  --accent: #14B8A6;
  --success: #22C55E;
  --border: #3F4B5C;
  --border-soft: #334155;
  --text: #F8FAFC;
  --muted: #94A3B8;
  --radius: 14px;
  --shadow: 0 1px 2px rgba(0,0,0,0.2), 0 10px 28px rgba(0,0,0,0.22);
}

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  font-size: 16px;
}

.stApp {
  background: var(--bg) !important;
  color: var(--text);
}

[data-testid="stAppViewContainer"] > .main {
  background: transparent;
}

.block-container {
  padding-top: 2rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 980px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

<<<<<<< HEAD
section[data-testid="stSidebar"],
=======
/* Hide sidebar — single-column product layout (Disabled to show diagnostics)
section[data-testid="stSidebar"] {
  display: none !important;
}
>>>>>>> 7fa6014e8b1ce70575677496d1136adac2916b14
div[data-testid="stSidebarCollapsedControl"] {
  display: none !important;
}
*/

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ----- Icons ----- */
.ia-icon {
  width: 18px;
  height: 18px;
  flex-shrink: 0;
  display: inline-block;
  vertical-align: -3px;
}
.ia-label-icon,
.ia-section-icon {
  display: inline-flex;
  align-items: center;
  color: var(--muted);
  margin-right: 0.45rem;
}
.ia-section-icon .ia-icon {
  width: 20px;
  height: 20px;
}

/* ----- Header ----- */
.ia-header {
  margin-bottom: 2.5rem;
  padding-bottom: 1.75rem;
  border-bottom: 1px solid var(--border-soft);
  animation: ia-fade-in 0.4s ease;
}
.ia-brand {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
  font-size: 48px;
  font-weight: 700;
  letter-spacing: -0.04em;
  color: var(--text);
  margin: 0 0 0.55rem 0;
  line-height: 1.1;
}
.ia-tagline {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
  font-size: 18px;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 0.55rem 0;
  line-height: 1.4;
}
.ia-sub {
  font-size: 14px;
  color: var(--muted);
  margin: 0;
  line-height: 1.6;
  max-width: 36rem;
}

/* ----- Section labels ----- */
.ia-section {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
  font-size: 26px;
  font-weight: 650;
  letter-spacing: -0.02em;
  color: var(--text);
  margin: 2.35rem 0 1rem 0;
  display: flex;
  align-items: center;
  line-height: 1.25;
}
.ia-block {
  margin-top: 0.25rem;
}

/* ----- Cards ----- */
.ia-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.35rem 1.45rem;
  box-shadow: var(--shadow);
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
  animation: ia-fade-in 0.45s ease;
  height: 100%;
  box-sizing: border-box;
}
.ia-card:hover {
  border-color: #526175;
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0,0,0,0.22), 0 14px 32px rgba(0,0,0,0.24);
}
.ia-card--focus {
  min-height: 168px;
  padding: 1.55rem 1.55rem 1.45rem;
  display: flex;
  flex-direction: column;
}
.ia-card--prediction {
  border-color: rgba(20, 184, 166, 0.35);
  background: linear-gradient(180deg, #152033 0%, var(--surface) 70%);
}
.ia-card--explain {
  padding: 1.75rem 1.85rem;
}
.ia-card--breakdown {
  padding: 1.5rem 1.55rem 1.65rem;
}

.ia-card-label {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 0.85rem 0;
  display: flex;
  align-items: center;
}
.ia-card-value {
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
  margin: 0 0 0.45rem 0;
  line-height: 1.15;
}
.ia-card-value--xl {
  font-size: 34px;
}
.ia-card-value.accent {
  color: var(--accent);
}
.ia-card-sub {
  font-size: 14px;
  color: var(--muted);
  margin: 0;
  line-height: 1.45;
}
.ia-card-body {
  font-size: 16px;
  color: #E2E8F0;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}
.ia-card-body.lg {
  font-size: 17px;
  line-height: 1.8;
  color: #F1F5F9;
  margin-top: 0.15rem;
}

.ia-card--routing {
  margin-top: 0.75rem;
  margin-bottom: 0.5rem;
}

.ia-fallacy-warning {
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.35);
  border-radius: var(--radius);
  padding: 1.1rem 1.25rem;
  margin: 1rem 0 0.5rem;
}
.ia-fallacy-title {
  font-size: 15px;
  font-weight: 650;
  color: #FCA5A5;
  margin: 0 0 0.4rem 0;
}
.ia-fallacy-body {
  font-size: 14px;
  color: #E2E8F0;
  margin: 0;
  line-height: 1.55;
}
.ia-fallacy-ok {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  margin: 1rem 0 0.5rem;
}

.ia-strength-wrap {
  margin: 0.15rem 0 0.85rem 0;
}

/* ----- Metric row ----- */
.ia-metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin: 0.15rem 0 0.75rem;
  align-items: stretch;
}
.ia-metric-grid--4 {
  grid-template-columns: repeat(2, 1fr);
}
@media (min-width: 900px) {
  .ia-metric-grid--4 {
    grid-template-columns: repeat(4, 1fr);
  }
}
@media (max-width: 820px) {
  .ia-metric-grid {
    grid-template-columns: 1fr;
  }
  .ia-brand {
    font-size: 40px;
  }
  .ia-section {
    font-size: 22px;
  }
  .ia-card-value--xl {
    font-size: 30px;
  }
}

/* ----- Confidence bars ----- */
.ia-conf-row {
  margin-top: 1.15rem;
}
.ia-conf-row:first-of-type {
  margin-top: 0.35rem;
}
.ia-conf-meta {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.5rem;
}
.ia-conf-label {
  font-size: 14px;
  color: var(--muted);
  font-weight: 500;
}
.ia-conf-pct {
  font-size: 14px;
  font-weight: 650;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.ia-bar {
  height: 8px;
  width: 100%;
  background: #1E293B;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 0.85rem;
}
.ia-bar--in-card {
  margin-top: auto;
  padding-top: 0.85rem;
}
.ia-bar--lg {
  height: 10px;
  margin-top: 0;
}
.ia-bar > span {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: var(--accent);
  transition: width 0.55s cubic-bezier(0.22, 1, 0.36, 1);
}
.ia-bar.secondary > span {
  background: #64748B;
}

/* ----- Tags / chips ----- */
.ia-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 0.15rem;
}
.ia-tag {
  display: inline-flex;
  align-items: center;
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.38rem 0.9rem;
  transition: border-color 0.15s ease, background 0.15s ease;
}
.ia-tag:hover {
  border-color: var(--accent);
  background: rgba(20, 184, 166, 0.08);
}

/* ----- Highlighted text ----- */
.ia-highlight {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem 1.6rem;
  line-height: 1.85;
  font-size: 16px;
  color: var(--text);
  box-shadow: var(--shadow);
}
.ia-highlight mark,
.ia-highlight mark.cue-authority,
.ia-highlight mark.cue-inference,
.ia-highlight mark.cue-analogy,
.ia-highlight mark.cue-observation {
  border-radius: 4px;
  padding: 0.08rem 0.25rem;
  background: rgba(20, 184, 166, 0.2);
  color: var(--text);
}

/* ----- Strength pill ----- */
.ia-pill {
  display: inline-flex;
  align-items: center;
  font-size: 15px;
  font-weight: 650;
  padding: 0.55rem 1rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: #0F172A;
  color: var(--text);
  letter-spacing: 0.01em;
}
.ia-pill.strong {
  border-color: rgba(34, 197, 94, 0.4);
  color: #4ADE80;
  background: rgba(34, 197, 94, 0.1);
}
.ia-pill.moderate {
  border-color: rgba(20, 184, 166, 0.4);
  color: #2DD4BF;
  background: rgba(20, 184, 166, 0.1);
}
.ia-pill.weak {
  border-color: rgba(148, 163, 184, 0.35);
  color: #CBD5E1;
  background: rgba(148, 163, 184, 0.08);
}

/* ----- SHAP note ----- */
.ia-note {
  font-size: 14px;
  line-height: 1.6;
  color: var(--muted);
  border-left: 2px solid var(--accent);
  padding: 0.8rem 1rem;
  margin: 0.35rem 0 1.15rem;
  background: rgba(20, 184, 166, 0.05);
  border-radius: 0 10px 10px 0;
}

.ia-examples-label {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 0.65rem 0;
}

/* ----- Footer ----- */
.ia-footer {
  text-align: center;
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border-soft);
  color: var(--muted);
  font-size: 13px;
}

@keyframes ia-fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

.ia-spacer-sm { height: 0.75rem; }
.ia-spacer-md { height: 1.35rem; }
.ia-spacer-lg { height: 2rem; }

/* ----- Streamlit controls ----- */
.stTextArea textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border-soft) !important;
  border-radius: 14px !important;
  color: var(--text) !important;
  font-size: 16px !important;
  line-height: 1.65 !important;
  min-height: 190px !important;
  padding: 1.1rem 1.2rem !important;
  box-shadow: var(--shadow) !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextArea textarea:focus {
  border-color: rgba(20, 184, 166, 0.55) !important;
  box-shadow: 0 0 0 1px rgba(20, 184, 166, 0.25), var(--shadow) !important;
  outline: none !important;
}
.stTextArea textarea::placeholder {
  color: #64748B !important;
  font-size: 15px !important;
  font-weight: 400 !important;
}

.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #042F2E !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important;
  font-size: 16px !important;
  letter-spacing: 0.01em !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.9rem 1.6rem !important;
  min-height: 3.15rem !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.18), 0 8px 20px rgba(20, 184, 166, 0.18) !important;
  transition: background 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
  background: #0D9488 !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 14px rgba(20, 184, 166, 0.28) !important;
}
.stButton > button[kind="primary"]:active {
  transform: translateY(0);
}

.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
  background: var(--surface) !important;
  color: var(--muted) !important;
  border: 1px solid var(--border-soft) !important;
  border-radius: 999px !important;
  font-weight: 550 !important;
  font-size: 13px !important;
  padding: 0.45rem 0.95rem !important;
  min-height: 2.25rem !important;
  transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease, transform 0.15s ease !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
  border-color: var(--accent) !important;
  color: var(--text) !important;
  background: rgba(20, 184, 166, 0.06) !important;
  transform: translateY(-1px);
}

.stCheckbox label span {
  color: var(--muted) !important;
  font-size: 14px !important;
}

.stProgress > div > div > div > div {
  background: var(--accent) !important;
  border-radius: 999px !important;
  box-shadow: none !important;
}
.stProgress > div > div > div {
  background: #1E293B !important;
  border-radius: 999px !important;
}

div[data-testid="stAlert"] {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
}

[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"] {
  border-radius: 12px !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
}

div[data-testid="stSpinner"] {
  color: var(--accent) !important;
}

[data-testid="stCaption"] {
  color: var(--muted) !important;
  font-size: 13px !important;
}
</style>
"""


def inject_dashboard_theme() -> None:
    """Inject global CSS. Call once at app start."""
    import streamlit as st

    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
