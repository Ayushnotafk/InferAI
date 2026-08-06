"""Global dashboard CSS for InferAI Streamlit UI."""

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg: #070814;
  --surface: rgba(13, 17, 30, 0.7);
  --surface-hover: rgba(20, 26, 46, 0.85);
  --surface-2: rgba(30, 41, 59, 0.3);
  --accent: #00F2FE;
  --accent-2: #4FACFE;
  --accent-glow: rgba(0, 242, 254, 0.15);
  --success: #10B981;
  --success-glow: rgba(16, 185, 129, 0.1);
  --border: rgba(255, 255, 255, 0.08);
  --border-hover: rgba(0, 242, 254, 0.3);
  --border-soft: rgba(255, 255, 255, 0.04);
  --text: #F8FAFC;
  --text-muted: #94A3B8;
  --radius: 18px;
  --radius-sm: 10px;
  --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
  --font-brand: 'Outfit', 'Plus Jakarta Sans', sans-serif;
  --font-body: 'Plus Jakarta Sans', system-ui, sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--font-body) !important;
  font-size: 16px;
}

.stApp {
  background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #09090e 65%, #050508 100%) !important;
  color: var(--text);
}

[data-testid="stAppViewContainer"] > .main {
  background: transparent;
}

.block-container {
  padding-top: 2rem !important;
  padding-bottom: 4rem !important;
  max-width: 1040px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: rgba(10, 10, 18, 0.6) !important;
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
  color: var(--text-muted);
  margin-right: 0.5rem;
}
.ia-section-icon .ia-icon {
  width: 22px;
  height: 22px;
  color: var(--accent);
}

/* ----- Header ----- */
.ia-header {
  margin-bottom: 3rem;
  padding-bottom: 2rem;
  border-bottom: 1px solid var(--border);
  animation: ia-fade-in 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  text-align: center;
}
.ia-brand {
  font-family: var(--font-brand);
  font-size: 56px;
  font-weight: 800;
  letter-spacing: -0.05em;
  background: linear-gradient(135deg, #FFF 20%, #A5F3FC 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 0.6rem 0;
  line-height: 1.1;
  text-shadow: 0 0 40px rgba(0, 242, 254, 0.15);
}
.ia-tagline {
  font-family: var(--font-brand);
  font-size: 20px;
  font-weight: 600;
  color: var(--accent);
  margin: 0 0 0.65rem 0;
  letter-spacing: -0.01em;
}
.ia-sub {
  font-size: 15px;
  color: var(--text-muted);
  margin: 0 auto;
  line-height: 1.6;
  max-width: 38rem;
}

/* ----- Section labels ----- */
.ia-section {
  font-family: var(--font-brand);
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
  margin: 2.75rem 0 1.25rem 0;
  display: flex;
  align-items: center;
  line-height: 1.25;
}
.ia-block {
  margin-top: 0.25rem;
}

/* ----- Cards (Glassmorphism) ----- */
.ia-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem 1.65rem;
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  animation: ia-fade-in 0.6s cubic-bezier(0.16, 1, 0.3, 1);
  height: 100%;
  box-sizing: border-box;
}
.ia-card:hover {
  border-color: var(--border-hover);
  transform: translateY(-2px);
  box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.08), var(--shadow);
  background: var(--surface-hover);
}
.ia-card--focus {
  min-height: 180px;
  padding: 1.65rem 1.65rem 1.55rem;
  display: flex;
  flex-direction: column;
}
.ia-card--prediction {
  border-color: rgba(0, 242, 254, 0.18);
  background: linear-gradient(180deg, rgba(30, 41, 59, 0.4) 0%, var(--surface) 100%);
}
.ia-card--explain {
  padding: 2rem;
}
.ia-card--breakdown {
  padding: 1.65rem 1.75rem;
}

.ia-card-label {
  font-family: var(--font-brand);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 0 0 0.95rem 0;
  display: flex;
  align-items: center;
}
.ia-card-value {
  font-family: var(--font-brand);
  font-size: 32px;
  font-weight: 700;
  letter-spacing: -0.04em;
  color: var(--text);
  margin: 0 0 0.5rem 0;
  line-height: 1.15;
}
.ia-card-value--xl {
  font-size: 38px;
}
.ia-card-value.accent {
  background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.ia-card-sub {
  font-size: 14px;
  color: var(--text-muted);
  margin: 0;
  line-height: 1.45;
}
.ia-card-body {
  font-size: 16px;
  color: #E2E8F0;
  line-height: 1.75;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0;
}
.ia-card-body.lg {
  font-size: 17.5px;
  line-height: 1.85;
  color: #F1F5F9;
  margin-top: 0.15rem;
}

.ia-card--routing {
  margin-top: 1rem;
  margin-bottom: 0.5rem;
}

/* Fallacy banner */
.ia-fallacy-warning {
  background: rgba(239, 68, 68, 0.05);
  border: 1px solid rgba(239, 68, 68, 0.25);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  margin: 1.25rem 0 0.75rem;
  animation: ia-fade-in 0.5s ease;
}
.ia-fallacy-title {
  font-family: var(--font-brand);
  font-size: 16px;
  font-weight: 700;
  color: #FCA5A5;
  margin: 0 0 0.45rem 0;
}
.ia-fallacy-body {
  font-size: 14.5px;
  color: #E2E8F0;
  margin: 0;
  line-height: 1.6;
}
.ia-fallacy-ok {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.15rem 1.40rem;
  margin: 1.25rem 0 0.75rem;
  backdrop-filter: blur(16px);
}

.ia-strength-wrap {
  margin: 0.2rem 0 0.9rem 0;
}

/* ----- Metric row ----- */
.ia-metric-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.25rem;
  margin: 0.2rem 0 1rem;
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
    font-size: 44px;
  }
  .ia-section {
    font-size: 24px;
  }
  .ia-card-value--xl {
    font-size: 32px;
  }
}

/* ----- Confidence bars ----- */
.ia-conf-row {
  margin-top: 1.25rem;
}
.ia-conf-row:first-of-type {
  margin-top: 0.35rem;
}
.ia-conf-meta {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 0.55rem;
}
.ia-conf-label {
  font-size: 14.5px;
  color: var(--text-muted);
  font-weight: 500;
}
.ia-conf-pct {
  font-family: var(--font-brand);
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.ia-bar {
  height: 8px;
  width: 100%;
  background: rgba(30, 41, 59, 0.7);
  border-radius: 999px;
  overflow: hidden;
  margin-top: 0.85rem;
  border: 1px solid rgba(255, 255, 255, 0.04);
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
  background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
  transition: width 0.75s cubic-bezier(0.16, 1, 0.3, 1);
}
.ia-bar.secondary > span {
  background: linear-gradient(90deg, #64748B 0%, #475569 100%);
}

/* ----- Tags / chips ----- */
.ia-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.65rem;
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
  padding: 0.45rem 1rem;
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
  padding: 1.75rem 2rem;
  line-height: 1.9;
  font-size: 16.5px;
  color: var(--text);
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
}
.ia-highlight mark,
.ia-highlight mark.cue-authority,
.ia-highlight mark.cue-inference,
.ia-highlight mark.cue-analogy,
.ia-highlight mark.cue-observation {
  border-radius: 6px;
  padding: 0.1rem 0.35rem;
  background: rgba(0, 242, 254, 0.18);
  border-bottom: 2px solid var(--accent);
  color: var(--text);
  font-weight: 500;
}

/* ----- Strength pill ----- */
.ia-pill {
  display: inline-flex;
  align-items: center;
  font-family: var(--font-brand);
  font-size: 14.5px;
  font-weight: 700;
  padding: 0.5rem 1.1rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(10, 10, 18, 0.4);
  color: var(--text);
  letter-spacing: 0.02em;
}
.ia-pill.strong {
  border-color: rgba(16, 185, 129, 0.3);
  color: #34D399;
  background: rgba(16, 185, 129, 0.08);
}
.ia-pill.moderate {
  border-color: rgba(0, 242, 254, 0.3);
  color: #22D3EE;
  background: rgba(0, 242, 254, 0.08);
}
.ia-pill.weak {
  border-color: rgba(148, 163, 184, 0.3);
  color: #CBD5E1;
  background: rgba(148, 163, 184, 0.06);
}

/* ----- SHAP note ----- */
.ia-note {
  font-size: 14.5px;
  line-height: 1.65;
  color: var(--text-muted);
  border-left: 3px solid var(--accent);
  padding: 0.9rem 1.15rem;
  margin: 0.35rem 0 1.25rem;
  background: var(--accent-glow);
  border-radius: 0 12px 12px 0;
}

.ia-examples-label {
  font-family: var(--font-brand);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 0 0 0.8rem 0;
}

/* ----- Footer ----- */
.ia-footer {
  text-align: center;
  margin-top: 4rem;
  padding-top: 2rem;
  border-top: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 13.5px;
}

@keyframes ia-fade-in {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.ia-spacer-sm { height: 0.85rem; }
.ia-spacer-md { height: 1.5rem; }
.ia-spacer-lg { height: 2.25rem; }

/* ----- Streamlit controls customization ----- */
.stTextArea textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  color: var(--text) !important;
  font-size: 16.5px !important;
  line-height: 1.7 !important;
  min-height: 180px !important;
  padding: 1.25rem 1.45rem !important;
  box-shadow: var(--shadow) !important;
  backdrop-filter: blur(16px);
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stTextArea textarea:focus {
  border-color: var(--border-hover) !important;
  box-shadow: 0 0 0 1px rgba(0, 242, 254, 0.15), var(--shadow) !important;
  outline: none !important;
}
.stTextArea textarea::placeholder {
  color: #475569 !important;
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
  color: #042F2E !important;
  font-family: var(--font-brand) !important;
  font-weight: 700 !important;
  font-size: 16.5px !important;
  letter-spacing: 0.02em !important;
  border: none !important;
  border-radius: 14px !important;
  padding: 0.95rem 1.75rem !important;
  min-height: 3.25rem !important;
  box-shadow: 0 4px 20px 0 rgba(0, 242, 254, 0.22) !important;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px 0 rgba(0, 242, 254, 0.35) !important;
  filter: brightness(1.1);
}
.stButton > button[kind="primary"]:active {
  transform: translateY(0);
}

.stButton > button[kind="secondary"],
.stButton > button:not([kind="primary"]) {
  background: var(--surface) !important;
  color: var(--text-muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: 999px !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
  padding: 0.5rem 1.15rem !important;
  min-height: 2.35rem !important;
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
  color: var(--text-muted) !important;
  font-size: 14px !important;
}

.stProgress > div > div > div > div {
  background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%) !important;
  border-radius: 999px !important;
}
.stProgress > div > div > div {
  background: rgba(30, 41, 59, 0.6) !important;
  border-radius: 999px !important;
}

div[data-testid="stAlert"] {
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  backdrop-filter: blur(16px);
}

[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"] {
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
}

div[data-testid="stSpinner"] {
  color: var(--accent) !important;
}

[data-testid="stCaption"] {
  color: var(--text-muted) !important;
  font-size: 13.5px !important;
}

/* Glassmorphic settings cards in sidebar */
.stSlider {
  margin-bottom: 1.5rem;
}
</style>
DASHBOARD_CSS_END
"""
DASHBOARD_CSS = DASHBOARD_CSS.replace("DASHBOARD_CSS_END\\n", "")


def inject_dashboard_theme() -> None:
    """Inject global CSS. Call once at app start."""
    import streamlit as st

    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
