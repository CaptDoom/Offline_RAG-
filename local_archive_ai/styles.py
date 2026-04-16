CSS = """
<style>
:root {
  /* Palette: deep slate + teal accents (high contrast) */
  --surface: #f4f6fb;
  --surface-low: #ffffff;
  --surface-high: #eef2fa;
  --panel: #ffffff;
  --ink: #0b1220;
  --muted: #51627a;
  --line: #d6deeb;
  --navy: #0b1f3b;
  --navy-alt: #103056;
  --accent: #0ea5a4;
  --accent-2: #2563eb;
  --accent-soft: #d7fbfb;
  --green: #0f7a44;
  --green-soft: #dbf3e4;
  --orange: #c24a1a;
  --danger: #b3261e;
}

.stApp {
  background: var(--surface);
  color: var(--ink);
}

/* Global text visibility */
.stApp,
.stApp p,
.stApp li,
.stApp label,
.stApp span,
.stMarkdown,
.stCaption,
[data-testid="stMarkdownContainer"] {
  color: var(--ink) !important;
}

/* Typography (clear + bold) */
.stApp, .stApp * {
  font-family: ui-sans-serif, "Inter", "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
  font-weight: 800 !important;
  letter-spacing: -0.01em;
  color: var(--ink) !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  font-size: 0.98rem;
  line-height: 1.55;
}

[data-testid="stMarkdownContainer"] strong {
  font-weight: 800 !important;
}

a, a:visited {
  color: var(--accent-2) !important;
  font-weight: 700;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

section[data-testid="stSidebar"] {
  background: var(--surface-high);
  border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] * {
  color: #132846 !important;
}

div[data-testid="stVerticalBlock"] > div:has(> .app-shell) {
  margin-top: 0.25rem;
}

.app-shell {
  animation: fadeIn 280ms ease-out;
}

.archive-banner {
  padding: 1.1rem 1.2rem;
  color: #fff;
  background: linear-gradient(135deg, var(--navy), var(--navy-alt));
  border: 1px solid #173a63;
  box-shadow: 0 12px 30px rgba(11, 31, 59, 0.14);
  border-radius: 10px;
  margin-bottom: 1rem;
  transition: transform 160ms ease, box-shadow 160ms ease;
}

.archive-banner:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 30px rgba(2, 36, 72, 0.13);
}

.archive-banner.warning {
  background: linear-gradient(135deg, #7f2d2d, #b3261e);
  border-color: #9f2f2a;
}

.system-pill {
  display: inline-block;
  background: var(--accent-soft);
  border: 1px solid #9de8e8;
  color: #0a5a5a;
  padding: 0.2rem 0.52rem;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border-radius: 4px;
  font-weight: 800;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.95rem 1rem;
  box-shadow: 0 6px 20px rgba(5, 33, 66, 0.04);
  margin-bottom: 0.75rem;
}

/* Layout primitives + spacing system */
.layout-stack-sm { margin-bottom: 0.5rem; }
.layout-stack-md { margin-bottom: 0.85rem; }
.layout-stack-lg { margin-bottom: 1.2rem; }

.surface-card {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.9rem 1rem;
}

.navy-panel {
  background: linear-gradient(165deg, var(--navy), var(--navy-alt));
  border: 1px solid #204977;
  border-radius: 10px;
  color: #ffffff;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(2, 36, 72, 0.15);
}

.mini-stat {
  display: inline-block;
  background: var(--surface-low);
  border: 1px solid var(--line);
  border-radius: 6px;
  color: #1b2b44;
  padding: 0.26rem 0.5rem;
  margin-right: 0.45rem;
  margin-bottom: 0.35rem;
  font-size: 0.74rem;
  transition: all 140ms ease;
  font-weight: 800;
}

.mini-stat:hover {
  background: #ffffff;
  transform: translateY(-1px);
}

.assistant-bubble {
  background: linear-gradient(145deg, #0b1f3b, #123a66);
  border: 1px solid #234f86;
  color: white;
  padding: 1rem 1rem;
  border-radius: 10px;
  margin: 0.5rem 0 0.9rem 0;
  line-height: 1.45;
  animation: slideUp 220ms ease-out;
}

.user-bubble {
  background: var(--panel);
  border: 1px solid var(--line);
  padding: 0.75rem 1rem;
  border-radius: 10px;
  margin: 0.5rem 0;
  text-align: right;
  animation: slideUp 180ms ease-out;
}

.retrieval-card {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 10px;
  box-shadow: 0 8px 24px rgba(2, 36, 72, 0.06);
  padding: 0.85rem;
  margin-bottom: 0.8rem;
  transition: box-shadow 150ms ease, transform 150ms ease;
}

.retrieval-card:hover {
  box-shadow: 0 10px 26px rgba(2, 36, 72, 0.09);
  transform: translateY(-1px);
}

.retrieval-card .score {
  float: right;
  background: #e6fffb;
  color: #0a5a5a;
  border: 1px solid #a6f0ee;
  border-radius: 4px;
  padding: 0.1rem 0.35rem;
  font-size: 0.72rem;
  font-weight: 800;
}

.metric-head {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  font-size: 0.7rem;
  font-weight: 700;
}

.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

div[data-baseweb="tab-list"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0.2rem;
}

div[data-baseweb="tab"] {
  letter-spacing: 0.05em;
  font-weight: 600;
  border-radius: 6px;
  transition: background 120ms ease, color 120ms ease;
}

div[data-baseweb="tab"]:hover {
  background: #f3f7fd;
}

div[data-baseweb="tab-highlight"] {
  background: #e6eef8 !important;
  border-radius: 6px;
}

div[role="radiogroup"] {
  gap: 0.45rem;
  margin: 0.3rem 0 1rem 0;
}

div[role="radiogroup"] label {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.25rem 0.8rem;
  transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
}

div[role="radiogroup"] label:hover {
  transform: translateY(-1px);
  border-color: #9bb8db;
  box-shadow: 0 6px 18px rgba(5, 33, 66, 0.06);
}

div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.4rem 0.55rem;
  box-shadow: 0 6px 18px rgba(5, 33, 66, 0.04);
}

div[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

div[data-testid="stTextArea"] textarea {
  border-radius: 10px !important;
  border: 1px solid var(--line) !important;
  background: #ffffff !important;
  color: #102544 !important;
  caret-color: #102544 !important;
  font-weight: 600;
}

div[data-testid="stTextArea"] textarea::placeholder {
  color: #6f7f95 !important;
  opacity: 1 !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] div {
  color: #102544 !important;
}

div[data-testid="stDataFrame"] * {
  color: #0f2240 !important;
}

div[data-testid="stDataFrame"] [role="columnheader"] {
  font-weight: 700 !important;
}

div[data-testid="stCodeBlock"] pre,
code {
  color: #d7e4ff;
}

div[data-testid="stButton"] button,
div[data-testid="stDownloadButton"] button {
  border-radius: 8px !important;
  border: 1px solid #1b4d7b !important;
  background: linear-gradient(180deg, #0f3a64, #0b2e52) !important;
  color: #f4fbff !important;
  font-weight: 800 !important;
  transition: transform 120ms ease, box-shadow 120ms ease, background-color 120ms ease !important;
}

div[data-testid="stButton"] button:hover,
div[data-testid="stDownloadButton"] button:hover {
  transform: translateY(-1px);
  background: linear-gradient(180deg, #145087, #0f3a64) !important;
  box-shadow: 0 8px 20px rgba(2, 36, 72, 0.12);
}

div[data-testid="stAlert"] {
  border-radius: 10px !important;
}

/* Screenshot-matched debug screen */
.debug-title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.8rem;
}

.debug-actions {
  display: flex;
  gap: 0.6rem;
}

.ghost-btn,
.solid-btn {
  display: inline-block;
  border: 1px solid #0d2646;
  padding: 0.5rem 0.85rem;
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  font-weight: 700;
}

.ghost-btn {
  background: #f4f6fb;
  color: #0d2646;
}

.solid-btn {
  background: #032651;
  color: #f4f9ff;
}

.debug-embed-box {
  background: #f1f3f7;
  border: 1px solid #d7deea;
  border-left: 4px solid #7c200f;
  padding: 0.9rem 0.95rem;
  margin-bottom: 1rem;
}

.embed-inline {
  margin-top: 0.55rem;
  padding: 0.5rem 0.65rem;
  background: #032651;
  color: #d8e7ff !important;
  border: 1px solid #1e4a7b;
}

.retrieval-head {
  margin: 0.6rem 0 0.6rem 0;
  font-weight: 700;
  letter-spacing: 0.06em;
  font-size: 0.78rem;
}

.retrieval-head span {
  float: right;
  color: #127448 !important;
}

.debug-chunk-card {
  background: #ffffff;
  border: 1px solid #d7deea;
  border-top: 4px solid #15874f;
  padding: 0.9rem;
  margin-bottom: 0.8rem;
  min-height: 170px;
}

.debug-score {
  float: right;
  background: #d5f3df;
  border: 1px solid #9ad3b1;
  color: #0f6c40 !important;
  padding: 0.08rem 0.36rem;
  font-size: 0.72rem;
  font-weight: 700;
}

.chunk-title {
  font-weight: 800;
  font-size: 0.85rem;
  margin-bottom: 0.1rem;
}

.chunk-meta {
  color: #6a7d96 !important;
  font-size: 0.7rem;
}

.chunk-body {
  margin-top: 0.7rem;
  line-height: 1.45;
  color: #2a3f59 !important;
}

.prompt-head {
  margin-top: 0.35rem;
  background: #032651;
  color: #f4f9ff !important;
  border: 1px solid #1a4779;
  padding: 0.45rem 0.75rem;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
  font-weight: 700;
}

.prompt-panel {
  background: #041a36;
  border: 1px solid #174274;
  color: #cbe0ff !important;
  padding: 0.8rem;
  min-height: 180px;
  white-space: pre-wrap;
  margin-bottom: 0.4rem;
}

/* Landing + Pipeline */
.landing-hero {
  background: radial-gradient(1200px 520px at 20% 20%, rgba(14,165,164,0.25), transparent 60%),
              linear-gradient(150deg, var(--navy) 0%, var(--navy-alt) 70%);
  border: 1px solid #1b4d7b;
  border-radius: 12px;
  color: #e6f1ff !important;
  padding: 1.3rem 1.2rem;
  margin-bottom: 0.9rem;
}

.landing-hero h1 {
  margin: 0.25rem 0 0.45rem 0;
  color: #ffffff !important;
  font-size: 1.9rem;
  line-height: 1.2;
}

.landing-hero p {
  margin: 0;
  color: #d5e6ff !important;
  max-width: 880px;
}

.feature-card {
  background: #ffffff;
  border: 1px solid #d7deea;
  border-radius: 10px;
  padding: 0.85rem 0.95rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 6px 18px rgba(8, 36, 68, 0.05);
}

.feature-card h4 {
  margin: 0 0 0.35rem 0;
}

.feature-card p {
  margin: 0;
  color: #435d7e !important;
  line-height: 1.45;
}

.pipeline-wrap {
  background: #ffffff;
  border: 1px solid #d7deea;
  border-radius: 10px;
  padding: 0.95rem 1rem;
  margin-bottom: 0.8rem;
}

.pipeline-stage-active {
  border: 1px solid #275a8e;
  background: #0f2f56;
  color: #ecf4ff !important;
}

.pipeline-stage-inactive {
  border: 1px solid #d2dcea;
  background: #ffffff;
  color: #143153 !important;
}

.pipeline-connector {
  height: 2px;
  margin: 0.65rem 0 0.95rem 0;
  background: linear-gradient(90deg, #2a5c8d, #3e82c5, #2a5c8d);
}

.pipeline-detail {
  background: #f8fbff;
  border: 1px solid #d6e3f4;
  border-left: 4px solid #2f6cab;
  border-radius: 10px;
  padding: 0.95rem 1rem;
  min-height: 120px;
}

.pipeline-kpis {
  background: #ffffff;
  border: 1px solid #d7deea;
  border-radius: 10px;
  padding: 0.8rem 0.9rem;
}

.pipeline-kpis div {
  margin-bottom: 0.4rem;
}

.pipeline-kpis div:last-child {
  margin-bottom: 0;
}

@media (max-width: 980px) {
  .archive-banner,
  .panel,
  .assistant-bubble,
  .user-bubble,
  .retrieval-card,
  .navy-panel {
    border-radius: 8px;
  }
  .landing-hero h1 {
    font-size: 1.4rem;
  }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes slideUp {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
"""
