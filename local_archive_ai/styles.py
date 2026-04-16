CSS = """
<style>
/* ============================================================
   THE DIGITAL VAULT - Design System
   Neo-Brutalist editorial UI for Local-Archive AI
   Sharp corners (0px radius), tonal layering, navy-tinted shadows
   Typography: Space Grotesk (headlines) + Inter (body) + Monospace (data)
   ============================================================ */

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --primary: #022448;
  --primary-container: #1e3a5f;
  --on-primary: #ffffff;
  --on-primary-container: #8aa4cf;
  --secondary: #006d3d;
  --secondary-container: #97f3b5;
  --on-secondary: #ffffff;
  --on-secondary-container: #047240;
  --tertiary: #471100;
  --tertiary-container: #6c1e00;
  --on-tertiary-container: #ff7d50;
  --error: #ba1a1a;
  --error-container: #ffdad6;
  --surface: #f7f9fc;
  --surface-dim: #d8dadd;
  --surface-container-lowest: #ffffff;
  --surface-container-low: #f2f4f7;
  --surface-container: #eceef1;
  --surface-container-high: #e6e8eb;
  --surface-container-highest: #e0e3e6;
  --on-surface: #191c1e;
  --on-surface-variant: #43474e;
  --outline: #74777f;
  --outline-variant: #c4c6cf;
  --shadow-sm: 0px 4px 12px rgba(2, 36, 72, 0.04);
  --shadow-md: 0px 8px 24px rgba(2, 36, 72, 0.06);
  --shadow-lg: 0px 12px 32px rgba(2, 36, 72, 0.08);
  --shadow-xl: 0px 16px 40px rgba(2, 36, 72, 0.12);
}

.stApp { background: var(--surface) !important; color: var(--on-surface); }
*, *::before, *::after { border-radius: 0px !important; }

.stApp, .stApp * {
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

h1, h2, h3, h4, h5, h6, .headline,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
  color: var(--primary) !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  font-size: 0.875rem; line-height: 1.6; color: var(--on-surface) !important;
}

[data-testid="stMarkdownContainer"] strong { font-weight: 700 !important; }

.mono { font-family: 'SFMono-Regular', 'Roboto Mono', ui-monospace, Menlo, Monaco, Consolas, monospace !important; }

a, a:visited { color: var(--secondary) !important; font-weight: 600; text-decoration: none; }
a:hover { color: var(--on-tertiary-container) !important; text-decoration: none; }

section[data-testid="stSidebar"] {
  background: var(--surface-container-low) !important;
  border-right: 0px !important;
  box-shadow: none !important;
}
section[data-testid="stSidebar"] * { color: var(--primary) !important; }
section[data-testid="stSidebar"] .stMarkdown h2 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.1rem !important; font-weight: 700 !important;
  letter-spacing: 0.04em; text-transform: uppercase;
}

.sidebar-brand { padding: 1.25rem 0 0.75rem 0; display: flex; align-items: center; gap: 0.75rem; }
.sidebar-brand-icon {
  width: 2rem; height: 2rem; background: var(--primary);
  display: flex; align-items: center; justify-content: center;
  color: var(--on-primary) !important; font-size: 0.875rem; font-weight: 800;
}
.sidebar-brand-text {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1rem; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--primary) !important;
}
.sidebar-version { font-family: monospace !important; font-size: 0.625rem; opacity: 0.5; letter-spacing: 0.05em; }

.sidebar-section-label {
  font-family: monospace !important; font-size: 0.625rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--primary) !important; opacity: 0.5;
  margin-bottom: 0.75rem; margin-top: 1.5rem;
}

.sidebar-footer { padding: 1rem; background: var(--surface-container-high); margin-top: 1rem; }
.sidebar-status { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; }
.sidebar-status-dot { width: 0.5rem; height: 0.5rem; background: var(--secondary); display: inline-block; }
.sidebar-status-dot.offline { background: var(--error); }
.sidebar-status-label { font-family: monospace !important; font-size: 0.6875rem; font-weight: 700; color: var(--secondary) !important; }
.sidebar-stat-row {
  display: flex; justify-content: space-between;
  font-family: monospace !important; font-size: 0.6875rem;
  color: var(--outline) !important; margin-bottom: 0.25rem;
}
.sidebar-stat-row span:last-child { color: var(--on-surface) !important; }

div[data-testid="stVerticalBlock"] > div:has(> .app-shell) { margin-top: 0.25rem; }

.app-shell { animation: fadeIn 200ms ease-out; }

/* ── Top App Bar ── */
.top-app-bar {
  background: var(--surface-container-lowest);
  padding: 0.75rem 1.5rem;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: var(--shadow-sm);
}
.top-app-bar-title {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1rem; font-weight: 700; color: var(--primary) !important;
  letter-spacing: 0.04em; text-transform: uppercase;
}

/* ── Archive Banner ── */
.archive-banner {
  padding: 1.25rem 1.5rem;
  color: var(--on-primary) !important;
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-container) 100%);
  box-shadow: var(--shadow-lg);
  margin-bottom: 1rem;
  transition: box-shadow 160ms ease;
}
.archive-banner:hover { box-shadow: var(--shadow-xl); }
.archive-banner.warning {
  background: linear-gradient(135deg, var(--error), #d32f2f);
}
.archive-banner h1, .archive-banner h2, .archive-banner h3,
.archive-banner p, .archive-banner span, .archive-banner div {
  color: var(--on-primary) !important;
}

/* ── System Pulse Chips ── */
.system-pill {
  display: inline-flex; align-items: center; gap: 0.375rem;
  background: var(--secondary-container);
  color: var(--on-secondary-container) !important;
  padding: 0.25rem 0.625rem;
  font-family: monospace !important;
  font-size: 0.6875rem; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase;
}
.system-pill.offline {
  background: var(--error-container);
  color: var(--error) !important;
}

/* ── Panel / Card ── */
.panel {
  background: var(--surface-container-lowest);
  padding: 1rem 1.125rem;
  box-shadow: var(--shadow-sm);
  margin-bottom: 0.75rem;
}

.layout-stack-sm { margin-bottom: 0.5rem; }
.layout-stack-md { margin-bottom: 0.875rem; }
.layout-stack-lg { margin-bottom: 1.25rem; }

.surface-card {
  background: var(--surface-container-lowest);
  padding: 1rem 1.125rem;
  box-shadow: var(--shadow-sm);
}

.navy-panel {
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-container) 100%);
  color: var(--on-primary) !important;
  padding: 1.125rem;
  box-shadow: var(--shadow-lg);
}
.navy-panel * { color: var(--on-primary) !important; }

/* ── Mini Stats ── */
.mini-stat {
  display: inline-block;
  background: var(--surface-container-low);
  color: var(--primary) !important;
  padding: 0.25rem 0.5rem;
  margin-right: 0.5rem; margin-bottom: 0.375rem;
  font-family: monospace !important;
  font-size: 0.75rem; font-weight: 700;
  transition: background 140ms ease;
}
.mini-stat:hover { background: var(--surface-container-high); }

/* ── Chat Bubbles ── */
.assistant-bubble {
  background: linear-gradient(135deg, var(--primary), var(--primary-container));
  color: var(--on-primary) !important;
  padding: 1rem 1.125rem;
  margin: 0.5rem 0 0.875rem 0;
  line-height: 1.5;
  animation: slideUp 200ms ease-out;
}
.assistant-bubble, .assistant-bubble * { color: var(--on-primary) !important; }

.user-bubble {
  background: var(--surface-container-lowest);
  border-bottom: 2px solid var(--primary);
  padding: 0.75rem 1rem;
  margin: 0.5rem 0;
  text-align: right;
  animation: slideUp 160ms ease-out;
}

/* ── Retrieval Cards ── */
.retrieval-card {
  background: var(--surface-container-lowest);
  box-shadow: var(--shadow-md);
  padding: 0.875rem;
  margin-bottom: 0.75rem;
  transition: box-shadow 150ms ease;
}
.retrieval-card:hover { box-shadow: var(--shadow-lg); }
.retrieval-card .score {
  float: right;
  background: var(--secondary-container);
  color: var(--on-secondary-container) !important;
  padding: 0.125rem 0.375rem;
  font-family: monospace !important;
  font-size: 0.6875rem; font-weight: 700;
}

.metric-head {
  color: var(--on-surface-variant) !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-family: monospace !important;
  font-size: 0.6875rem; font-weight: 700;
}

/* ── Tabs (Neo-Brutalist) ── */
div[data-baseweb="tab-list"] {
  background: var(--surface-container-low);
  padding: 0.25rem;
}
div[data-baseweb="tab"] {
  font-family: monospace !important;
  letter-spacing: 0.06em; font-weight: 700; font-size: 0.75rem;
  text-transform: uppercase;
  transition: background 120ms ease, color 120ms ease;
}
div[data-baseweb="tab"]:hover { background: var(--surface-container-high); }
div[data-baseweb="tab-highlight"] { background: var(--primary) !important; }
div[data-baseweb="tab"][aria-selected="true"] { color: var(--on-primary) !important; }

/* ── Radio group ── */
div[role="radiogroup"] { gap: 0.5rem; margin: 0.375rem 0 1rem 0; }
div[role="radiogroup"] label {
  background: var(--surface-container-lowest);
  border-bottom: 2px solid var(--outline-variant);
  padding: 0.25rem 0.75rem;
  transition: border-color 120ms ease;
}
div[role="radiogroup"] label:hover { border-color: var(--primary); }

/* ── Metrics ── */
div[data-testid="stMetric"] {
  background: var(--surface-container-lowest);
  padding: 0.5rem 0.625rem;
  box-shadow: var(--shadow-sm);
}
div[data-testid="stMetricLabel"] {
  color: var(--on-surface-variant) !important;
  text-transform: uppercase; letter-spacing: 0.06em;
  font-family: monospace !important; font-size: 0.6875rem;
}

/* ── Inputs (bottom-bar style) ── */
div[data-testid="stTextArea"] textarea {
  background: var(--surface-container-lowest) !important;
  border: none !important;
  border-bottom: 2px solid var(--primary) !important;
  color: var(--on-surface) !important;
  caret-color: var(--primary) !important;
  font-weight: 500;
}
div[data-testid="stTextArea"] textarea:focus {
  border-bottom-color: var(--on-tertiary-container) !important;
  box-shadow: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
  color: var(--outline) !important; opacity: 1 !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  background: var(--surface-container-lowest) !important;
  border: none !important;
  border-bottom: 2px solid var(--primary) !important;
  color: var(--on-surface) !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
  border-bottom-color: var(--on-tertiary-container) !important;
}
div[data-baseweb="select"] div { color: var(--on-surface) !important; }

div[data-testid="stDataFrame"] * { color: var(--on-surface) !important; }
div[data-testid="stDataFrame"] [role="columnheader"] {
  font-weight: 700 !important;
  font-family: 'Space Grotesk', sans-serif !important;
}

div[data-testid="stCodeBlock"] pre, code {
  color: var(--on-primary-container);
  font-family: 'SFMono-Regular', 'Roboto Mono', monospace !important;
}

/* ── Buttons (sharp, navy) ── */
div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button {
  background: var(--primary) !important;
  color: var(--on-primary) !important;
  border: none !important;
  font-weight: 700 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: 0.04em;
  text-transform: uppercase; font-size: 0.75rem;
  transition: background 150ms ease, box-shadow 150ms ease !important;
}
div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover {
  background: var(--secondary) !important;
  box-shadow: var(--shadow-md);
}

div[data-testid="stAlert"] { box-shadow: var(--shadow-sm); }

/* ── Debug Screen ── */
.debug-title-row {
  display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 0.875rem;
}
.debug-actions { display: flex; gap: 0.625rem; }

.ghost-btn, .solid-btn {
  display: inline-block;
  padding: 0.5rem 0.875rem;
  font-family: monospace !important;
  font-size: 0.75rem; letter-spacing: 0.04em; font-weight: 700;
  text-transform: uppercase; cursor: pointer;
  transition: background 120ms ease;
}
.ghost-btn {
  background: var(--surface-container-low);
  color: var(--primary) !important;
  border-bottom: 2px solid var(--primary);
}
.ghost-btn:hover { background: var(--surface-container-high); }
.solid-btn {
  background: var(--primary);
  color: var(--on-primary) !important;
}
.solid-btn:hover { background: var(--secondary); }

.debug-embed-box {
  background: var(--surface-container-low);
  border-left: 4px solid var(--tertiary-container);
  padding: 0.875rem 1rem;
  margin-bottom: 1rem;
}
.embed-inline {
  margin-top: 0.5rem;
  padding: 0.5rem 0.625rem;
  background: var(--primary);
  color: var(--on-primary-container) !important;
  font-family: monospace !important;
}

.retrieval-head {
  margin: 0.625rem 0;
  font-family: monospace !important;
  font-weight: 700; letter-spacing: 0.06em; font-size: 0.75rem;
  text-transform: uppercase;
}
.retrieval-head span { float: right; color: var(--secondary) !important; }

.debug-chunk-card {
  background: var(--surface-container-lowest);
  border-top: 4px solid var(--secondary);
  padding: 0.875rem;
  margin-bottom: 0.75rem;
  min-height: 170px;
  box-shadow: var(--shadow-sm);
}

.debug-score {
  float: right;
  background: var(--secondary-container);
  color: var(--on-secondary-container) !important;
  padding: 0.125rem 0.375rem;
  font-family: monospace !important;
  font-size: 0.6875rem; font-weight: 700;
}
.chunk-title {
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700; font-size: 0.875rem; margin-bottom: 0.125rem;
}
.chunk-meta {
  color: var(--outline) !important;
  font-family: monospace !important; font-size: 0.6875rem;
}
.chunk-body {
  margin-top: 0.625rem; line-height: 1.5;
  color: var(--on-surface) !important; font-size: 0.8125rem;
}

.prompt-head {
  margin-top: 0.375rem;
  background: var(--primary);
  color: var(--on-primary) !important;
  padding: 0.5rem 0.75rem;
  font-family: monospace !important;
  font-size: 0.75rem; letter-spacing: 0.06em; font-weight: 700;
  text-transform: uppercase;
}
.prompt-panel {
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-container) 100%);
  color: var(--on-primary-container) !important;
  padding: 0.875rem;
  min-height: 180px;
  white-space: pre-wrap;
  margin-bottom: 0.5rem;
  font-family: monospace !important; font-size: 0.8125rem;
}

/* ── Landing + Pipeline ── */
.landing-hero {
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-container) 100%);
  color: var(--on-primary) !important;
  padding: 1.5rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: var(--shadow-lg);
}
.landing-hero h1 {
  margin: 0.25rem 0 0.5rem 0;
  color: var(--on-primary) !important;
  font-size: 2rem; line-height: 1.15;
}
.landing-hero p {
  margin: 0;
  color: var(--on-primary-container) !important;
  max-width: 880px; font-size: 0.9375rem;
}

.feature-card {
  background: var(--surface-container-lowest);
  padding: 1rem 1.125rem;
  margin-bottom: 0.75rem;
  box-shadow: var(--shadow-sm);
  border-left: 4px solid var(--primary);
  transition: box-shadow 150ms ease;
}
.feature-card:hover { box-shadow: var(--shadow-md); }
.feature-card h4 { margin: 0 0 0.375rem 0; }
.feature-card p { margin: 0; color: var(--on-surface-variant) !important; line-height: 1.5; }

.pipeline-wrap {
  background: var(--surface-container-lowest);
  padding: 1rem 1.125rem;
  margin-bottom: 0.75rem;
  box-shadow: var(--shadow-sm);
}

.pipeline-stage-active {
  background: var(--primary);
  color: var(--on-primary) !important;
  padding: 0.5rem 0.75rem;
}
.pipeline-stage-inactive {
  background: var(--surface-container-low);
  color: var(--primary) !important;
  padding: 0.5rem 0.75rem;
}

.pipeline-connector {
  height: 2px;
  margin: 0.625rem 0 0.875rem 0;
  background: linear-gradient(90deg, var(--primary), var(--primary-container), var(--primary));
}

.pipeline-detail {
  background: var(--surface-container-low);
  border-left: 4px solid var(--primary);
  padding: 1rem 1.125rem;
  min-height: 120px;
}

.pipeline-kpis {
  background: var(--surface-container-lowest);
  padding: 0.875rem 1rem;
  box-shadow: var(--shadow-sm);
}
.pipeline-kpis div { margin-bottom: 0.375rem; }
.pipeline-kpis div:last-child { margin-bottom: 0; }

/* ── Glassmorphism overlay ── */
.glass-overlay {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow: var(--shadow-md);
}

/* ── Streamlit overrides ── */
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.stDeployButton { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface-container-low); }
::-webkit-scrollbar-thumb { background: var(--outline-variant); }
::-webkit-scrollbar-thumb:hover { background: var(--outline); }

/* ── Animations ── */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes slideUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
"""
