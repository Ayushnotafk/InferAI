"""Global dashboard CSS for InferAI Streamlit UI."""

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg: #090C15;
  --surface: rgba(17, 22, 39, 0.75);
  --surface-hover: rgba(23, 30, 54, 0.9);
  --surface-2: rgba(30, 41, 59, 0.4);
  --accent: #6366F1;
  --accent-gradient: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
  --mint: #14B8A6;
  --mint-gradient: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%);
  --accent-glow: rgba(99, 102, 241, 0.12);
  --success: #10B981;
  --border: rgba(255, 255, 255, 0.06);
  --border-hover: rgba(99, 102, 241, 0.3);
  --border-soft: rgba(255, 255, 255, 0.03);
  --text: #F8FAFC;
  --text-muted: #64748B;
  --text-secondary: #94A3B8;
  --radius: 16px;
  --radius-sm: 8px;
  --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
  --font-brand: 'Outfit', 'Plus Jakarta Sans', sans-serif;
  --font-body: 'Plus Jakarta Sans', system-ui, sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  font-size: 15.5px;
}

.stApp {
  background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #090c15 60%, #06080e 100%) !important;
  color: var(--text);
}

[data-testid="stAppViewContainer"] > .main {
  background: transparent;
}

.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3.5rem !important;
  max-width: 1040px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: rgba(8, 10, 18, 0.7) !important;
  backdrop-filter: blur(20px);
  border-right: 1px solid var(--border);
}

div[data-testid="stSidebarCollapsedControl"] {
  background: rgba(15, 23, 42, 0.7);
  border-radius: 0 10px 10px 0;
  border: 1px solid var(--border);
  border-left: none;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ----- Icons ----- */
.ia-icon {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  display: inline-block;
  vertical-align: -2.5px;
}
.ia-label-icon,
.ia-section-icon {
  display: inline-flex;
  align-items: center;
  color: var(--text-muted);
  margin-right: 0.4rem;
}
.ia-section-icon .ia-icon {
  width: 20px;
  height: 20px;
  color: var(--mint);
}

/* ----- Header ----- */
.ia-header {
  margin-bottom: 2.5rem;
  padding-bottom: 1.75rem;
  border-bottom: 1px solid var(--border);
  animation: ia-fade-in 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  text-align: center;
}
.ia-brand {
  font-family: var(--font-brand);
  font-size: 50px;
  font-weight: 800;
  letter-spacing: -0.04em;
  background: linear-gradient(135deg, #FFFFFF 20%, #C7D2FE 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 0.5rem 0;
  line-height: 1.1;
  text-shadow: 0 0 40px rgba(99, 102, 241, 0.1);
}
.ia-tagline {
  font-family: var(--font-brand);
  font-size: 18px;
  font-weight: 600;
  color: var(--mint);
  margin: 0 0 0.6rem 0;
  letter-spacing: -0.01em;
}
.ia-sub {
  font-size: 14.5px;
  color: var(--text-secondary);
  margin: 0 auto;
  line-height: 1.6;
  max-width: 38rem;
}

/* ----- Section labels ----- */
.ia-section-container {
  margin-top: 2.5rem;
  margin-bottom: 2.5rem;
}
.ia-section {
  font-family: var(--font-brand);
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
  margin: 2.5rem 0 1rem 0;
  display: flex;
  align-items: center;
  line-height: 1.25;
}
.ia-section-container .ia-section {
  margin-top: 0 !important;
}

/* ----- Cards ----- */
.ia-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.4rem;
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  animation: ia-fade-in 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  height: 100%;
  box-sizing: border-box;
}
.ia-card:hover {
  border-color: var(--border-hover);
  transform: translateY(-1.5px);
  box-shadow: 0 10px 30px 0 rgba(99, 102, 241, 0.06), var(--shadow);
  background: var(--surface-hover);
}
.ia-card--focus {
  min-height: 156px;
  padding: 1.25rem 1.4rem;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.ia-card--prediction {
  border-color: rgba(99, 102, 241, 0.15);
  background: linear-gradient(180deg, rgba(30, 41, 59, 0.25) 0%, var(--surface) 100%);
}
.ia-card--explain {
  padding: 1.75rem 1.85rem;
}
.ia-card--breakdown {
  padding: 1.35rem 1.45rem;
}

.ia-card-label {
  font-family: var(--font-brand);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 0 0 0.75rem 0;
  display: flex;
  align-items: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ia-card-value {
  font-family: var(--font-brand);
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
  margin: 0 0 0.35rem 0;
  line-height: 1.15;
}
.ia-card-value--xl {
  font-size: 32px;
}
.ia-card-value.accent {
  background: var(--mint-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.ia-card-sub {
  font-size: 13.5px;
  color: var(--text-secondary);
  margin: 0;
  line-height: 1.4;
}
.ia-card-body {
  font-size: 15.5px;
  color: #E2E8F0;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}
.ia-card-body.lg {
  font-size: 16.5px;
  line-height: 1.8;
  color: #F1F5F9;
  margin-top: 0.15rem;
}

.ia-card--routing {
  margin-top: 0.75rem;
  margin-bottom: 0.5rem;
}

/* Fallacy banner */
.ia-fallacy-warning {
  background: rgba(239, 68, 68, 0.04);
  border: 1px solid rgba(239, 68, 68, 0.2);
  border-radius: var(--radius);
  padding: 1.1rem 1.35rem;
  margin: 1rem 0 0.5rem;
  animation: ia-fade-in 0.4s ease;
}
.ia-fallacy-title {
  font-family: var(--font-brand);
  font-size: 15px;
  font-weight: 700;
  color: #FCA5A5;
  margin: 0 0 0.35rem 0;
}
.ia-fallacy-body {
  font-size: 14px;
  color: #CBD5E1;
  margin: 0;
  line-height: 1.55;
}
.ia-fallacy-ok {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.05rem 1.25rem;
  margin: 1rem 0 0.5rem;
  backdrop-filter: blur(16px);
}

.ia-strength-wrap {
  margin: 0.15rem 0 0.75rem 0;
}

/* ----- Metric row ----- */
.ia-metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin: 0.15rem 0 0.85rem;
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
    font-size: 42px;
  }
  .ia-section {
    font-size: 22px;
  }
  .ia-card-value--xl {
    font-size: 28px;
  }
}

/* ----- Confidence bars ----- */
.ia-conf-row {
  margin-top: 1.15rem;
}
.ia-conf-row:first-of-type {
  margin-top: 0.25rem;
}
.ia-conf-meta {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.45rem;
}
.ia-conf-label {
  font-size: 14px;
  color: var(--text-secondary);
  font-weight: 500;
}
.ia-conf-pct {
  font-family: var(--font-brand);
  font-size: 14.5px;
  font-weight: 700;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.ia-bar {
  height: 8px;
  width: 100%;
  background: rgba(15, 23, 42, 0.8);
  border-radius: 999px;
  overflow: hidden;
  margin-top: 0.75rem;
  border: 1px solid rgba(255, 255, 255, 0.03);
}
.ia-bar--in-card {
  margin-top: auto;
  padding-top: 0.75rem;
}
.ia-bar--lg {
  height: 9px;
  margin-top: 0;
}
.ia-bar > span {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: var(--accent-gradient);
  transition: width 0.75s cubic-bezier(0.16, 1, 0.3, 1);
}
.ia-bar.secondary > span {
  background: linear-gradient(90deg, #64748B 0%, #475569 100%);
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
  font-size: 13.5px;
  font-weight: 500;
  color: var(--text);
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.4rem 0.9rem;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}
.ia-tag:hover {
  border-color: var(--border-hover);
  background: var(--accent-glow);
  transform: translateY(-1px);
}

/* ----- Highlighted text ----- */
.ia-highlight {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem 1.75rem;
  line-height: 1.85;
  font-size: 16px;
  color: var(--text);
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
}
.ia-highlight mark,
.ia-highlight mark.cue-authority,
.ia-highlight mark.cue-inference,
.ia-highlight mark.cue-analogy,
.ia-highlight mark.cue-observation {
  border-radius: 5px;
  padding: 0.08rem 0.3rem;
  background: rgba(99, 102, 241, 0.18);
  border-bottom: 2px solid var(--accent);
  color: var(--text);
  font-weight: 500;
}

/* ----- Strength pill ----- */
.ia-pill {
  display: inline-flex;
  align-items: center;
  font-family: var(--font-brand);
  font-size: 13.5px;
  font-weight: 700;
  padding: 0.45rem 0.95rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(8, 10, 18, 0.5);
  color: var(--text);
  letter-spacing: 0.01em;
}
.ia-pill.strong {
  border-color: rgba(16, 185, 129, 0.25);
  color: #34D399;
  background: rgba(16, 185, 129, 0.06);
}
.ia-pill.moderate {
  border-color: rgba(99, 102, 241, 0.25);
  color: #818CF8;
  background: rgba(99, 102, 241, 0.06);
}
.ia-pill.weak {
  border-color: rgba(148, 163, 184, 0.25);
  color: #94A3B8;
  background: rgba(148, 163, 184, 0.04);
}

/* ----- SHAP note ----- */
.ia-note {
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-secondary);
  border-left: 3px solid var(--accent);
  padding: 0.8rem 1rem;
  margin: 0.35rem 0 1.15rem;
  background: var(--accent-glow);
  border-radius: 0 10px 10px 0;
}

.ia-examples-label {
  font-family: var(--font-brand);
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 0 0 0.75rem 0;
}

/* ----- Footer ----- */
.ia-footer {
  text-align: center;
  margin-top: 3.5rem;
  padding-top: 1.75rem;
  border-top: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 13px;
}

@keyframes ia-fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

.ia-spacer-sm { height: 0.75rem; }
.ia-spacer-md { height: 1.25rem; }
.ia-spacer-lg { height: 2rem; }

/* ----- Streamlit controls customization ----- */
.stTextArea textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  color: var(--text) !important;
  font-size: 16px !important;
  line-height: 1.65 !important;
  min-height: 170px !important;
  padding: 1.1rem 1.3rem !important;
  box-shadow: var(--shadow) !important;
  backdrop-filter: blur(16px);
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stTextArea textarea:focus {
  border-color: var(--border-hover) !important;
  box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.12), var(--shadow) !important;
  outline: none !important;
}
.stTextArea textarea::placeholder {
  color: #475569 !important;
}

.stButton > button[kind="primary"] {
  background: var(--accent-gradient) !important;
  color: #FFFFFF !important;
  font-family: var(--font-brand) !important;
  font-weight: 700 !important;
  font-size: 16px !important;
  letter-spacing: 0.01em !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.9rem 1.65rem !important;
  min-height: 3.15rem !important;
  box-shadow: 0 4px 15px 0 rgba(99, 102, 241, 0.2) !important;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-1.5px);
  box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.3) !important;
  filter: brightness(1.1);
}
.stButton > button[kind="primary"]:active {
  transform: translateY(0);
}

.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
  background: var(--surface) !important;
  color: var(--text-secondary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 999px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 0.45rem 1.1rem !important;
  min-height: 2.25rem !important;
  backdrop-filter: blur(16px);
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button:not([kind="primary"]):hover {
  border-color: var(--border-hover) !important;
  color: var(--text) !important;
  background: var(--accent-glow) !important;
  transform: translateY(-1px);
}

.stCheckbox label span {
  color: var(--text-secondary) !important;
  font-size: 13.5px !important;
}

.stProgress > div > div > div > div {
  background: var(--accent-gradient) !important;
  border-radius: 999px !important;
}
.stProgress > div > div > div {
  background: rgba(30, 41, 59, 0.5) !important;
  border-radius: 999px !important;
}

div[data-testid="stAlert"] {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  backdrop-filter: blur(16px);
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
  color: var(--text-muted) !important;
  font-size: 13px !important;
}

.stSlider {
  margin-bottom: 1.25rem;
}
</style>
"""


def inject_dashboard_theme() -> None:
    """Inject global CSS. Call once at app start."""
    import streamlit as st

    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
