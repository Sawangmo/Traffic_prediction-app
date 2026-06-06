
Claude finished the response

import streamlit as st import pandas as pd import numpy as np import plotly.express as px import plotly.figure_factory as ff import warnings warnings.filterwarnings('ignore') # Machine Learning from sklearn.model_selection import train_test_split from sklearn.preprocessing import LabelEnc

pasted

can you edit this code  and rewrite it again since when developing the model all the model is giving perfect r square and it is difficult to identify which model is best for this project and i also want you to make interactive dashboard

5:10 AM

Reading frontend design skill
Reading frontend design skill
Now let me write the improved Streamlit app with fixes for the perfect R² issue and an interactive dashboard:


Ran a command, created a file, read a file
Ran a command, created a file, read a file
Here's what was fixed and what's new:

Why models gave perfect R² — fixed with:

Data leakage detector (Features tab) — scans every feature for suspiciously high correlation with your target and flags anything above |r| > 0.98
Auto ID-column removal — detects and drops columns with "id", "index", etc. that accidentally act as perfect predictors
5-fold cross-validation — every model now shows both Test R² and CV R², so you can see the gap. A big gap means overfitting or leakage, not real performance
Constrained tree depth — Decision Trees and Random Forests now have max_depth limits so they can't perfectly memorise training data
Ridge regression replaces plain Linear Regression (more honest generalisation)
Overfitting flag — each row in the leaderboard explicitly shows ✅ OK or ⚠️ Likely based on test vs CV gap
New interactive dashboard features:

7-tab layout — Dashboard, EDA, Features, ML Models, Deep Learning, Time Series, Model Comparison
Dark themed UI with custom fonts, metric cards, and animated leaderboard rows
EDA tab with scatter explorer, distribution plots, correlation heatmap, and categorical breakdowns
Model Comparison tab — recommends the best model with a generalisation score = 60% CV R² + 40% stability
Training settings panel — adjustable test split, CV toggle, ID-drop toggle
Feature importance charts for tree-based models
LSTM training history plot (loss curves) so you can see if it converged or overfit
Residual histogram for SARIMA forecasts
Run with: streamlit run traffic_accident_prediction.py


Traffic accident prediction
PY 

Want to be notified when Claude responds?




Claude is AI and can make mistakes. Please double-check responses.
Traffic accident prediction · PY
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
 
# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, classification_report
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.inspection import permutation_importance
 
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
 
# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
 
# Time Series
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
 
# --------------------------------------------------
# PAGE CONFIG & CUSTOM THEME
# --------------------------------------------------
st.set_page_config(
    page_title="Traffic Accident Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');
 
    :root {
        --bg: #0d0f14;
        --surface: #161920;
        --surface2: #1e2330;
        --accent: #f97316;
        --accent2: #3b82f6;
        --accent3: #22c55e;
        --warn: #eab308;
        --danger: #ef4444;
        --text: #e2e8f0;
        --muted: #64748b;
        --border: #2a3044;
    }
 
    .stApp {
        background: var(--bg);
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }
 
    h1, h2, h3, h4 {
        font-family: 'Space Mono', monospace;
    }
 
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0d0f14 0%, #1a1f2e 50%, #0d0f14 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 30% 50%, rgba(249,115,22,0.08) 0%, transparent 60%),
                    radial-gradient(ellipse at 70% 50%, rgba(59,130,246,0.06) 0%, transparent 60%);
        pointer-events: none;
    }
    .main-header h1 {
        font-size: 2.2rem;
        margin: 0;
        background: linear-gradient(90deg, #f97316, #fb923c, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .main-header p {
        color: var(--muted);
        margin: 0.5rem 0 0;
        font-size: 0.95rem;
        letter-spacing: 0.05em;
    }
 
    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }
    .metric-card.orange::before { background: var(--accent); }
    .metric-card.blue::before   { background: var(--accent2); }
    .metric-card.green::before  { background: var(--accent3); }
    .metric-card.yellow::before { background: var(--warn); }
    .metric-card.red::before    { background: var(--danger); }
 
    .metric-label {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.5rem;
        font-family: 'Space Mono', monospace;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        color: var(--text);
        line-height: 1;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 0.4rem;
    }
 
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    .section-icon {
        width: 32px; height: 32px;
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
    }
    .section-badge {
        margin-left: auto;
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 99px;
        padding: 0.2rem 0.75rem;
        font-size: 0.7rem;
        color: var(--muted);
        font-family: 'Space Mono', monospace;
    }
 
    /* Alert boxes */
    .alert-warning {
        background: rgba(234,179,8,0.1);
        border: 1px solid rgba(234,179,8,0.3);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #fde68a;
        font-size: 0.9rem;
    }
    .alert-info {
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #93c5fd;
        font-size: 0.9rem;
    }
    .alert-success {
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.3);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #86efac;
        font-size: 0.9rem;
    }
    .alert-danger {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #fca5a5;
        font-size: 0.9rem;
    }
 
    /* Best model badge */
    .best-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        padding: 0.2rem 0.6rem;
        border-radius: 99px;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
 
    /* Leaderboard table */
    .leaderboard-row {
        display: flex;
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 0.5rem;
        gap: 1rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        transition: border-color 0.2s;
    }
    .leaderboard-row:first-child {
        border-color: var(--accent);
        background: rgba(249,115,22,0.05);
    }
    .rank { width: 28px; color: var(--muted); }
    .rank.gold { color: #fbbf24; font-size: 1.1rem; }
    .model-name { flex: 1; color: var(--text); }
    .score-pill {
        background: var(--surface2);
        border-radius: 99px;
        padding: 0.15rem 0.6rem;
        font-size: 0.75rem;
        color: var(--muted);
    }
    .score-pill.good { color: var(--accent3); border: 1px solid rgba(34,197,94,0.3); }
    .score-pill.warn { color: var(--warn); border: 1px solid rgba(234,179,8,0.3); }
    .score-pill.bad  { color: var(--danger); border: 1px solid rgba(239,68,68,0.3); }
 
    /* Sidebar */
    .stSidebar {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
 
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border-radius: 10px;
        padding: 0.25rem;
        gap: 0.25rem;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--muted);
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface2) !important;
        color: var(--text) !important;
    }
 
    /* Plotly charts background */
    .js-plotly-plot .plotly { background: transparent !important; }
 
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        padding: 0.6rem 1.5rem;
        transition: opacity 0.2s, transform 0.1s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
 
    /* Selectbox & inputs */
    .stSelectbox > div > div, .stSlider {
        background: var(--surface2) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
 
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
    }
    .streamlit-expanderContent {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
    }
 
    div[data-testid="stDataFrame"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)
 
# --------------------------------------------------
# PLOTLY THEME
# --------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color='#94a3b8', size=12),
    xaxis=dict(gridcolor='#1e2330', linecolor='#2a3044', zeroline=False),
    yaxis=dict(gridcolor='#1e2330', linecolor='#2a3044', zeroline=False),
    margin=dict(l=20, r=20, t=50, b=20),
    colorway=['#f97316','#3b82f6','#22c55e','#eab308','#a855f7','#ec4899','#14b8a6'],
)
 
COLOR_SEQ = ['#f97316','#3b82f6','#22c55e','#eab308','#a855f7','#ec4899','#14b8a6']
 
# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_numeric_cols(df):
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])]
 
def get_categorical_cols(df):
    return [c for c in df.columns if df[c].dtype == 'object']
 
def drop_datetime_cols(df):
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    return df.drop(columns=dt_cols)
 
def r2_label(r2):
    if r2 >= 0.85:
        return 'good'
    elif r2 >= 0.5:
        return 'warn'
    return 'bad'
 
def cv_scores(model, X, y, task='regression', cv=5):
    """Return cross-validated scores to prevent data leakage inflating metrics."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    if task == 'regression':
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    else:
        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    return scores
 
def make_chart(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig
 
def detect_leakage(X, y, threshold=0.98):
    """Warn if any single feature explains almost all variance."""
    leaky = []
    for col in X.columns:
        try:
            r = np.corrcoef(X[col].astype(float), y.astype(float))[0, 1]
            if abs(r) > threshold:
                leaky.append((col, round(r, 4)))
        except Exception:
            pass
    return leaky
 
# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="main-header">
  <h1>🚦 Traffic Accident Intelligence</h1>
  <p>MACHINE LEARNING · DEEP LEARNING · TIME SERIES FORECASTING · INTERACTIVE DASHBOARD</p>
</div>
""", unsafe_allow_html=True)
 
# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Traffic Accident Dataset (.csv)", type=["csv"])
 
# --------------------------------------------------
# MAIN
# --------------------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
 
    if 'crash_date' in df.columns:
        try:
            df['crash_date'] = pd.to_datetime(df['crash_date'])
        except Exception:
            pass
 
    # ──────────────────────────────────────────────
    # TABS
    # ──────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Dashboard",
        "🔍 EDA",
        "⚙️ Features",
        "🤖 ML Models",
        "🧠 Deep Learning",
        "📅 Time Series",
        "🏆 Model Comparison"
    ])
 
    # ══════════════════════════════════════════════
    # TAB 1: DASHBOARD
    # ══════════════════════════════════════════════
    with tabs[0]:
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">📊</div>
          <p class="section-title">DATASET OVERVIEW</p>
          <span class="section-badge">LIVE</span>
        </div>
        """, unsafe_allow_html=True)
 
        num_cols = get_numeric_cols(df)
        cat_cols = get_categorical_cols(df)
        missing_pct = round(df.isnull().sum().sum() / df.size * 100, 2)
 
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card orange">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-sub">rows in dataset</div>
          </div>
          <div class="metric-card blue">
            <div class="metric-label">Features</div>
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-sub">total columns</div>
          </div>
          <div class="metric-card green">
            <div class="metric-label">Numeric Cols</div>
            <div class="metric-value">{len(num_cols)}</div>
            <div class="metric-sub">continuous features</div>
          </div>
          <div class="metric-card yellow">
            <div class="metric-label">Missing %</div>
            <div class="metric-value">{missing_pct}%</div>
            <div class="metric-sub">{int(df.isnull().sum().sum())} missing cells</div>
          </div>
          <div class="metric-card red">
            <div class="metric-label">Categorical</div>
            <div class="metric-value">{len(cat_cols)}</div>
            <div class="metric-sub">object columns</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
        # Quick visual overview
        c1, c2 = st.columns(2)
        with c1:
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Missing']
            missing_data = missing_data[missing_data['Missing'] > 0].sort_values('Missing', ascending=True)
            if not missing_data.empty:
                fig = px.bar(missing_data, x='Missing', y='Column', orientation='h',
                             title='Missing Values by Column', color='Missing',
                             color_continuous_scale=[[0,'#3b82f6'],[1,'#ef4444']])
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown('<div class="alert-success">✅ No missing values detected in the dataset.</div>', unsafe_allow_html=True)
 
        with c2:
            dtype_counts = df.dtypes.value_counts().reset_index()
            dtype_counts.columns = ['Type', 'Count']
            dtype_counts['Type'] = dtype_counts['Type'].astype(str)
            fig2 = px.pie(dtype_counts, names='Type', values='Count',
                          title='Column Data Types', hole=0.55,
                          color_discrete_sequence=COLOR_SEQ)
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)
 
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">👁️</div>
          <p class="section-title">DATA PREVIEW</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df.head(50), use_container_width=True)
 
    # ══════════════════════════════════════════════
    # TAB 2: EDA
    # ══════════════════════════════════════════════
    with tabs[1]:
        num_cols = get_numeric_cols(df)
        cat_cols = get_categorical_cols(df)
 
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">📈</div>
          <p class="section-title">NUMERICAL ANALYSIS</p>
        </div>
        """, unsafe_allow_html=True)
 
        if num_cols:
            sel_num = st.selectbox("Select numeric column", num_cols, key='eda_num')
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=sel_num, nbins=40,
                                   title=f'Distribution — {sel_num}',
                                   color_discrete_sequence=['#f97316'])
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.box(df, y=sel_num, title=f'Boxplot — {sel_num}',
                              color_discrete_sequence=['#3b82f6'])
                fig2.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)
 
            # Stats summary
            stats = df[sel_num].describe().round(3)
            cols = st.columns(len(stats))
            for i, (k, v) in enumerate(stats.items()):
                cols[i].metric(k, v)
 
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">📊</div>
          <p class="section-title">CATEGORICAL ANALYSIS</p>
        </div>
        """, unsafe_allow_html=True)
 
        if cat_cols:
            sel_cat = st.selectbox("Select categorical column", cat_cols, key='eda_cat')
            vc = df[sel_cat].value_counts().reset_index()
            vc.columns = [sel_cat, 'Count']
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(vc.head(20), x='Count', y=sel_cat, orientation='h',
                             title=f'{sel_cat} Distribution',
                             color='Count', color_continuous_scale=[[0,'#1e2330'],[1,'#f97316']])
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.pie(vc.head(10), names=sel_cat, values='Count',
                              title=f'{sel_cat} Share', hole=0.4,
                              color_discrete_sequence=COLOR_SEQ)
                fig2.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)
 
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">🔥</div>
          <p class="section-title">CORRELATION MATRIX</p>
        </div>
        """, unsafe_allow_html=True)
 
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                            title='Feature Correlation Heatmap',
                            color_continuous_scale='RdYlGn',
                            zmin=-1, zmax=1)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
 
            # Scatter: pick two columns
            st.markdown("**Scatter Explorer**")
            c1, c2, c3 = st.columns(3)
            x_col = c1.selectbox("X axis", num_cols, key='sc_x')
            y_col = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key='sc_y')
            color_col = c3.selectbox("Color by (optional)", ['None'] + cat_cols, key='sc_c')
            color_arg = None if color_col == 'None' else color_col
            fig = px.scatter(df, x=x_col, y=y_col, color=color_arg,
                             opacity=0.6, title=f'{x_col} vs {y_col}',
                             color_discrete_sequence=COLOR_SEQ)
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
 
    # ══════════════════════════════════════════════
    # TAB 3: FEATURE ENGINEERING
    # ══════════════════════════════════════════════
    with tabs[2]:
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">⚙️</div>
          <p class="section-title">FEATURE ENGINEERING</p>
        </div>
        """, unsafe_allow_html=True)
 
        fe_applied = []
        if 'crash_hour' in df.columns:
            if 'crash_day_of_week' in df.columns:
                df['is_weekend'] = df['crash_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
                fe_applied.append("✅ **is_weekend** — weekend flag from crash_day_of_week")
            df['is_night'] = df['crash_hour'].apply(lambda x: 1 if x >= 20 or x <= 5 else 0)
            df['is_rush_hour'] = df['crash_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
            df['time_of_day'] = pd.cut(df['crash_hour'],
                                       bins=[-1, 5, 11, 17, 21, 24],
                                       labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late Night'])
            fe_applied.append("✅ **is_night** — night-time crash flag")
            fe_applied.append("✅ **is_rush_hour** — rush hour flag (7-9am, 4-7pm)")
            fe_applied.append("✅ **time_of_day** — categorical time bucket")
 
        if 'crash_date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['crash_date']):
            df['crash_month'] = df['crash_date'].dt.month
            df['crash_year'] = df['crash_date'].dt.year
            df['crash_quarter'] = df['crash_date'].dt.quarter
            fe_applied.append("✅ **crash_month, crash_year, crash_quarter** — from crash_date")
 
        if fe_applied:
            for feat in fe_applied:
                st.markdown(feat)
        else:
            st.markdown('<div class="alert-info">ℹ️ No crash_hour or crash_date columns found. Feature engineering skipped.</div>', unsafe_allow_html=True)
 
        # Visualize engineered features
        num_cols = get_numeric_cols(df)
        if 'is_rush_hour' in df.columns:
            c1, c2 = st.columns(2)
            with c1:
                rh = df['is_rush_hour'].value_counts().rename({0: 'Non-Rush', 1: 'Rush Hour'})
                fig = px.pie(values=rh.values, names=rh.index,
                             title='Rush Hour vs Normal',
                             color_discrete_sequence=['#f97316','#1e2330'])
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                if 'crash_hour' in df.columns:
                    fig2 = px.histogram(df, x='crash_hour', nbins=24,
                                        title='Crash Distribution by Hour',
                                        color_discrete_sequence=['#3b82f6'])
                    fig2.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig2, use_container_width=True)
 
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">🔍</div>
          <p class="section-title">DATA LEAKAGE DETECTOR</p>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("""
        <div class="alert-warning">
        ⚠️ <b>Why models give perfect R²?</b><br>
        Common causes: (1) Target leakage — a feature that directly encodes the target, (2) ID columns included in features,
        (3) Post-event features that wouldn't exist at prediction time, (4) Overfitting on tiny datasets, (5) Using training set metrics instead of test set.
        <br><br>Use this detector to find suspicious features before training.
        </div>
        """, unsafe_allow_html=True)
 
        data_check = drop_datetime_cols(df.copy())
        for col in data_check.columns:
            if data_check[col].dtype == 'object':
                data_check[col] = LabelEncoder().fit_transform(data_check[col].fillna('missing').astype(str))
            else:
                data_check[col] = data_check[col].fillna(data_check[col].median())
 
        num_cols_check = get_numeric_cols(data_check)
        if num_cols_check:
            leak_target = st.selectbox("Select target to check for leakage", num_cols_check, key='leak_t')
            X_check = data_check.drop(columns=[leak_target])
            y_check = data_check[leak_target]
            leaky = detect_leakage(X_check, y_check)
            if leaky:
                st.markdown(f'<div class="alert-danger">🚨 Potential leakage detected in {len(leaky)} feature(s):</div>', unsafe_allow_html=True)
                for col, corr_val in leaky:
                    st.markdown(f"- **{col}** (correlation = {corr_val}) — consider removing or investigating this feature")
            else:
                st.markdown('<div class="alert-success">✅ No obvious leakage detected (threshold: |r| > 0.98)</div>', unsafe_allow_html=True)
 
            # Correlation with target bar
            corr_with_target = X_check.corrwith(y_check).abs().sort_values(ascending=False).head(20)
            fig = go.Figure(go.Bar(
                x=corr_with_target.values,
                y=corr_with_target.index,
                orientation='h',
                marker=dict(
                    color=corr_with_target.values,
                    colorscale=[[0,'#22c55e'],[0.7,'#eab308'],[1,'#ef4444']],
                    showscale=True
                )
            ))
            fig.update_layout(title='Feature Correlation with Target (absolute)', **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
 
    # ══════════════════════════════════════════════
    # SHARED ENCODING (for ML tabs)
    # ══════════════════════════════════════════════
    data_enc = drop_datetime_cols(df.copy())
    for col in data_enc.columns:
        if data_enc[col].dtype == 'object':
            data_enc[col] = data_enc[col].fillna(data_enc[col].mode()[0] if not data_enc[col].mode().empty else 'missing')
        else:
            data_enc[col] = data_enc[col].fillna(data_enc[col].median())
 
    for col in data_enc.select_dtypes(include='object').columns:
        data_enc[col] = LabelEncoder().fit_transform(data_enc[col].astype(str))
 
    model_num_cols = get_numeric_cols(data_enc)
 
    # ══════════════════════════════════════════════
    # TAB 4: ML MODELS
    # ══════════════════════════════════════════════
    with tabs[3]:
        ml_task = st.radio("Task Type", ['Regression', 'Classification'], horizontal=True)
 
        st.markdown("""
        <div class="alert-info">
        💡 <b>Honest Evaluation</b>: Models are evaluated using both held-out test set AND
        5-fold cross-validation to prevent inflated metrics from data leakage or lucky splits.
        If CV score differs greatly from test score, the model may be overfitting.
        </div>
        """, unsafe_allow_html=True)
 
        # ── Sidebar controls ──
        with st.expander("⚙️ Training Settings", expanded=True):
            test_size = st.slider("Test set size (%)", 10, 40, 25, 5) / 100
            use_cv = st.checkbox("Enable 5-fold Cross Validation", value=True)
            drop_id_cols = st.checkbox("Auto-drop ID-like columns", value=True)
 
        # Optionally drop ID columns
        data_ml = data_enc.copy()
        if drop_id_cols:
            id_candidates = [c for c in data_ml.columns
                             if ('id' in c.lower() or 'index' in c.lower() or 'no' == c.lower())]
            if id_candidates:
                data_ml = data_ml.drop(columns=id_candidates)
                st.markdown(f'<div class="alert-warning">🗑️ Auto-dropped columns: {", ".join(id_candidates)}</div>', unsafe_allow_html=True)
 
        ml_num_cols = get_numeric_cols(data_ml)
 
        if ml_task == 'Regression':
            st.markdown("""
            <div class="section-header">
              <div class="section-icon">📈</div>
              <p class="section-title">REGRESSION MODELS</p>
            </div>
            """, unsafe_allow_html=True)
 
            target = st.selectbox("Select target variable", ml_num_cols, key='reg_target')
 
            if st.button("🚀 Train Regression Models"):
                X = data_ml.drop(columns=[target])
                y = data_ml[target]
 
                # Leakage pre-check
                leaky = detect_leakage(X, y)
                if leaky:
                    st.markdown(f'<div class="alert-danger">🚨 WARNING: Potential leakage in {[c for c,_ in leaky]}. Results may be unreliable.</div>', unsafe_allow_html=True)
 
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
 
                models = {
                    'Linear Regression (Ridge)': Ridge(alpha=1.0),
                    'Decision Tree': DecisionTreeRegressor(max_depth=6, random_state=42),
                    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
                }
                if XGBOOST_AVAILABLE:
                    models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
 
                results = []
                trained_models = {}
 
                progress = st.progress(0, text="Training models...")
                for i, (name, model) in enumerate(models.items()):
                    progress.progress((i+1)/len(models), text=f"Training {name}...")
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
 
                    mae  = mean_absolute_error(y_test, preds)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    r2   = r2_score(y_test, preds)
 
                    cv_r2 = None
                    if use_cv:
                        cv_r2_scores = cv_scores(model, X, y, task='regression')
                        cv_r2 = round(cv_r2_scores.mean(), 4)
 
                    results.append({
                        'Model': name,
                        'Test MAE': round(mae, 4),
                        'Test RMSE': round(rmse, 4),
                        'Test R²': round(r2, 4),
                        'CV R² (mean)': cv_r2 if cv_r2 else 'N/A',
                        'Overfitting?': '⚠️ Likely' if (cv_r2 and abs(r2 - cv_r2) > 0.1) else '✅ OK'
                    })
                    trained_models[name] = (model, preds)
 
                progress.empty()
 
                results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
                st.session_state['reg_results'] = results_df
                st.session_state['reg_models'] = trained_models
                st.session_state['reg_X_test'] = X_test
                st.session_state['reg_y_test'] = y_test
                st.session_state['reg_feature_names'] = list(X.columns)
 
                # Leaderboard
                st.markdown("### 🏆 Model Leaderboard")
                for rank, (_, row) in enumerate(results_df.iterrows()):
                    rank_icon = "🥇" if rank == 0 else f"#{rank+1}"
                    r2_cls = r2_label(row['Test R²'])
                    st.markdown(f"""
                    <div class="leaderboard-row">
                      <span class="rank {'gold' if rank==0 else ''}">{rank_icon}</span>
                      <span class="model-name">{row['Model']}</span>
                      <span class="score-pill {r2_cls}">R² {row['Test R²']}</span>
                      <span class="score-pill">MAE {row['Test MAE']}</span>
                      <span class="score-pill">RMSE {row['Test RMSE']}</span>
                      <span class="score-pill">CV {row['CV R² (mean)']}</span>
                      <span class="score-pill">{row['Overfitting?']}</span>
                    </div>
                    """, unsafe_allow_html=True)
 
                # Charts
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(results_df, x='Model', y='Test R²',
                                 color='Model', title='Test R² Score by Model',
                                 color_discrete_sequence=COLOR_SEQ)
                    fig.add_hline(y=0.8, line_dash='dash', line_color='#22c55e',
                                  annotation_text='Good threshold (0.8)')
                    fig.add_hline(y=0.5, line_dash='dash', line_color='#eab308',
                                  annotation_text='Moderate (0.5)')
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    if use_cv:
                        cv_df = results_df[results_df['CV R² (mean)'] != 'N/A'].copy()
                        cv_df['Test R²'] = cv_df['Test R²'].astype(float)
                        cv_df['CV R² (mean)'] = cv_df['CV R² (mean)'].astype(float)
                        fig2 = go.Figure()
                        fig2.add_bar(name='Test R²', x=cv_df['Model'], y=cv_df['Test R²'],
                                     marker_color='#f97316')
                        fig2.add_bar(name='CV R²', x=cv_df['Model'], y=cv_df['CV R² (mean)'],
                                     marker_color='#3b82f6')
                        fig2.update_layout(barmode='group',
                                           title='Test R² vs Cross-Val R² (gap = overfitting)',
                                           **PLOTLY_LAYOUT)
                        st.plotly_chart(fig2, use_container_width=True)
 
                # Actual vs Predicted for best model
                best_name = results_df.iloc[0]['Model']
                best_preds = trained_models[best_name][1]
                pred_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': best_preds})
                fig_ap = px.scatter(pred_df, x='Actual', y='Predicted',
                                    title=f'Actual vs Predicted — {best_name}',
                                    opacity=0.5, color_discrete_sequence=['#f97316'])
                mn, mx = pred_df.min().min(), pred_df.max().max()
                fig_ap.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                                 line=dict(color='#3b82f6', dash='dash'))
                fig_ap.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_ap, use_container_width=True)
 
                # Feature importance (best model if available)
                best_model_obj = trained_models[best_name][0]
                if hasattr(best_model_obj, 'feature_importances_'):
                    feat_imp = pd.DataFrame({
                        'Feature': st.session_state['reg_feature_names'],
                        'Importance': best_model_obj.feature_importances_
                    }).sort_values('Importance', ascending=False).head(15)
                    fig_fi = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                                    title=f'Feature Importance — {best_name}',
                                    color='Importance',
                                    color_continuous_scale=[[0,'#1e2330'],[1,'#f97316']])
                    fig_fi.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig_fi, use_container_width=True)
 
        else:  # Classification
            st.markdown("""
            <div class="section-header">
              <div class="section-icon">🚦</div>
              <p class="section-title">CLASSIFICATION MODELS</p>
            </div>
            """, unsafe_allow_html=True)
 
            target = st.selectbox("Select target variable", data_ml.columns.tolist(), key='clf_target')
 
            if st.button("🚀 Train Classification Models"):
                X = data_ml.drop(columns=[target])
                y = data_ml[target]
 
                if y.nunique() > 20:
                    st.markdown('<div class="alert-warning">⚠️ Target has many unique values. Consider using Regression.</div>', unsafe_allow_html=True)
 
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
 
                models = {
                    'Logistic Regression': LogisticRegression(max_iter=5000, C=0.1, solver='liblinear'),
                    'Decision Tree (depth=6)': DecisionTreeClassifier(max_depth=6, random_state=42),
                    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
                }
                if XGBOOST_AVAILABLE:
                    models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, verbosity=0, eval_metric='logloss')
 
                results = []
                trained_clf = {}
 
                progress = st.progress(0)
                for i, (name, model) in enumerate(models.items()):
                    progress.progress((i+1)/len(models), text=f"Training {name}...")
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc  = accuracy_score(y_test, preds)
                    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
                    rec  = recall_score(y_test, preds, average='weighted', zero_division=0)
                    f1   = f1_score(y_test, preds, average='weighted', zero_division=0)
 
                    cv_acc = None
                    if use_cv:
                        cv_acc_scores = cv_scores(model, X, y, task='classification')
                        cv_acc = round(cv_acc_scores.mean(), 4)
 
                    results.append({
                        'Model': name,
                        'Test Acc': round(acc, 4),
                        'Precision': round(prec, 4),
                        'Recall': round(rec, 4),
                        'F1': round(f1, 4),
                        'CV Acc': cv_acc if cv_acc else 'N/A',
                        'Overfitting?': '⚠️ Likely' if (cv_acc and abs(acc - cv_acc) > 0.05) else '✅ OK'
                    })
                    trained_clf[name] = (model, preds)
 
                progress.empty()
                results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
 
                # Leaderboard
                for rank, (_, row) in enumerate(results_df.iterrows()):
                    rank_icon = "🥇" if rank == 0 else f"#{rank+1}"
                    acc_cls = 'good' if row['Test Acc'] >= 0.8 else ('warn' if row['Test Acc'] >= 0.6 else 'bad')
                    st.markdown(f"""
                    <div class="leaderboard-row">
                      <span class="rank {'gold' if rank==0 else ''}">{rank_icon}</span>
                      <span class="model-name">{row['Model']}</span>
                      <span class="score-pill {acc_cls}">Acc {row['Test Acc']}</span>
                      <span class="score-pill">F1 {row['F1']}</span>
                      <span class="score-pill">Prec {row['Precision']}</span>
                      <span class="score-pill">Rec {row['Recall']}</span>
                      <span class="score-pill">CV {row['CV Acc']}</span>
                      <span class="score-pill">{row['Overfitting?']}</span>
                    </div>
                    """, unsafe_allow_html=True)
 
                # Metric comparison chart
                plot_df = results_df[['Model','Test Acc','Precision','Recall','F1']].melt(id_vars='Model')
                fig = px.bar(plot_df, x='Model', y='value', color='variable', barmode='group',
                             title='Classification Metrics Comparison',
                             color_discrete_sequence=COLOR_SEQ)
                fig.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
 
                # Confusion matrix for best
                best_name = results_df.iloc[0]['Model']
                best_preds = trained_clf[best_name][1]
                cm = confusion_matrix(y_test, best_preds)
                labels = [str(l) for l in sorted(y_test.unique())]
                fig_cm = ff.create_annotated_heatmap(
                    z=cm, x=labels, y=labels, colorscale='Oranges'
                )
                fig_cm.update_layout(
                    title=f'Confusion Matrix — {best_name}',
                    xaxis_title='Predicted', yaxis_title='Actual',
                    **PLOTLY_LAYOUT
                )
                st.plotly_chart(fig_cm, use_container_width=True)
 
    # ══════════════════════════════════════════════
    # TAB 5: DEEP LEARNING
    # ══════════════════════════════════════════════
    with tabs[4]:
        if not TF_AVAILABLE:
            st.markdown('<div class="alert-danger">❌ TensorFlow not installed. Install with: pip install tensorflow</div>', unsafe_allow_html=True)
        else:
            dl_task = st.radio("Task", ['LSTM Regression', 'LSTM Classification'], horizontal=True)
 
            st.markdown("""
            <div class="alert-info">
            💡 LSTM models are trained with early stopping and validation split to prevent overfitting.
            </div>
            """, unsafe_allow_html=True)
 
            if dl_task == 'LSTM Regression':
                target = st.selectbox("Target", model_num_cols, key='dl_reg_t')
                epochs = st.slider("Max Epochs", 10, 100, 30, 10)
 
                if st.button("Train LSTM Regressor"):
                    with st.spinner("Training LSTM..."):
                        X = data_enc.drop(columns=[target]).values
                        y = data_enc[target].values
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.25, random_state=42
                        )
                        scX = MinMaxScaler(); scy = MinMaxScaler()
                        X_tr = scX.fit_transform(X_train); X_te = scX.transform(X_test)
                        y_tr = scy.fit_transform(y_train.reshape(-1,1))
 
                        X_tr_l = X_tr.reshape(X_tr.shape[0], 1, X_tr.shape[1])
                        X_te_l = X_te.reshape(X_te.shape[0], 1, X_te.shape[1])
 
                        model = Sequential([
                            LSTM(64, return_sequences=True, input_shape=(1, X_tr.shape[1])),
                            Dropout(0.3), BatchNormalization(),
                            LSTM(32), Dropout(0.3),
                            Dense(16, activation='relu'), Dense(1)
                        ])
                        model.compile(optimizer='adam', loss='mse')
                        history = model.fit(
                            X_tr_l, y_tr, epochs=epochs, batch_size=32,
                            validation_split=0.2, callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
                            verbose=0
                        )
                        preds_sc = model.predict(X_te_l)
                        preds = scy.inverse_transform(preds_sc).flatten()
 
                    st.markdown('<div class="alert-success">✅ LSTM Training Complete</div>', unsafe_allow_html=True)
                    c1,c2,c3 = st.columns(3)
                    c1.metric("MAE", f"{mean_absolute_error(y_test, preds):.4f}")
                    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, preds)):.4f}")
                    c3.metric("R²", f"{r2_score(y_test, preds):.4f}")
 
                    hist_df = pd.DataFrame({'Epoch': range(1, len(history.history['loss'])+1),
                                            'Train Loss': history.history['loss'],
                                            'Val Loss': history.history['val_loss']})
                    fig = px.line(hist_df, x='Epoch', y=['Train Loss','Val Loss'],
                                  title='Training History', color_discrete_sequence=['#f97316','#3b82f6'])
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
 
                    pred_df = pd.DataFrame({'Actual': y_test[:200], 'Predicted': preds[:200]})
                    fig2 = px.line(pred_df, title='Actual vs Predicted (first 200)',
                                   color_discrete_sequence=['#22c55e','#f97316'])
                    fig2.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig2, use_container_width=True)
 
            else:  # LSTM Classification
                target = st.selectbox("Target", data_enc.columns.tolist(), key='dl_clf_t')
                epochs = st.slider("Max Epochs", 10, 100, 30, 10)
 
                if st.button("Train LSTM Classifier"):
                    with st.spinner("Training LSTM..."):
                        X = data_enc.drop(columns=[target]).values
                        y_raw = data_enc[target].values
                        ul = np.unique(y_raw)
                        lmap = {v:i for i,v in enumerate(ul)}
                        y = np.array([lmap[v] for v in y_raw])
                        nc = len(ul)
 
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                        scX = MinMaxScaler()
                        X_tr = scX.fit_transform(X_train); X_te = scX.transform(X_test)
                        X_tr_l = X_tr.reshape(X_tr.shape[0],1,X_tr.shape[1])
                        X_te_l = X_te.reshape(X_te.shape[0],1,X_te.shape[1])
 
                        out_u = 1 if nc==2 else nc
                        loss_fn = 'binary_crossentropy' if nc==2 else 'sparse_categorical_crossentropy'
                        out_act = 'sigmoid' if nc==2 else 'softmax'
 
                        model = Sequential([
                            LSTM(64, return_sequences=True, input_shape=(1, X_tr.shape[1])),
                            Dropout(0.3), BatchNormalization(),
                            LSTM(32), Dropout(0.3),
                            Dense(16, activation='relu'),
                            Dense(out_u, activation=out_act)
                        ])
                        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
                        history = model.fit(
                            X_tr_l, y_train, epochs=epochs, batch_size=32,
                            validation_split=0.2, callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
                            verbose=0
                        )
                        prob = model.predict(X_te_l)
                        preds = (prob > 0.5).astype(int).flatten() if nc==2 else np.argmax(prob, axis=1)
 
                    st.markdown('<div class="alert-success">✅ LSTM Classification Complete</div>', unsafe_allow_html=True)
                    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.4f}")
 
                    hist_df = pd.DataFrame({'Epoch': range(1, len(history.history['loss'])+1),
                                            'Train Loss': history.history['loss'],
                                            'Val Loss': history.history['val_loss']})
                    fig = px.line(hist_df, x='Epoch', y=['Train Loss','Val Loss'], title='Training History',
                                  color_discrete_sequence=['#f97316','#3b82f6'])
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
 
                    cm = confusion_matrix(y_test, preds)
                    labels = [str(i) for i in range(cm.shape[0])]
                    fig_cm = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, colorscale='Oranges')
                    fig_cm.update_layout(title='Confusion Matrix', **PLOTLY_LAYOUT)
                    st.plotly_chart(fig_cm, use_container_width=True)
 
    # ══════════════════════════════════════════════
    # TAB 6: TIME SERIES
    # ══════════════════════════════════════════════
    with tabs[5]:
        if not STATSMODELS_AVAILABLE:
            st.markdown('<div class="alert-danger">❌ statsmodels not installed.</div>', unsafe_allow_html=True)
        elif 'crash_date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['crash_date']):
            st.markdown('<div class="alert-danger">❌ Dataset needs a parseable crash_date column.</div>', unsafe_allow_html=True)
        else:
            num_cols_ts = get_numeric_cols(df)
            ts_target = st.selectbox("Time series target", num_cols_ts, key='ts_t')
            resample_freq = st.selectbox("Aggregation frequency", ['ME', 'W', 'D'], key='ts_freq')
 
            if st.button("Run SARIMA Forecast"):
                with st.spinner("Fitting SARIMA..."):
                    ts_df = df[['crash_date', ts_target]].copy().set_index('crash_date')
                    ts_data = ts_df[ts_target].resample(resample_freq).sum().dropna()
 
                    if len(ts_data) < 12:
                        st.markdown('<div class="alert-danger">❌ Need at least 12 observations for SARIMA.</div>', unsafe_allow_html=True)
                    else:
                        split = int(len(ts_data) * 0.7)
                        train_ts = ts_data.iloc[:split]
                        test_ts = ts_data.iloc[split:]
 
                        try:
                            sarima = SARIMAX(train_ts, order=(1,1,1),
                                             seasonal_order=(1,1,1,12) if len(train_ts)>=24 else (0,0,0,0),
                                             enforce_stationarity=False, enforce_invertibility=False)
                            fitted = sarima.fit(disp=False)
                            forecast = fitted.forecast(steps=len(test_ts))
 
                            mae  = mean_absolute_error(test_ts, forecast)
                            rmse = np.sqrt(mean_squared_error(test_ts, forecast))
                            r2   = r2_score(test_ts, forecast)
 
                            st.markdown('<div class="alert-success">✅ Forecasting Completed</div>', unsafe_allow_html=True)
                            c1,c2,c3 = st.columns(3)
                            c1.metric("MAE", f"{mae:.4f}")
                            c2.metric("RMSE", f"{rmse:.4f}")
                            c3.metric("R²", f"{r2:.4f}")
 
                            # Full series plot
                            full_df = pd.DataFrame({
                                'Date': list(train_ts.index) + list(test_ts.index),
                                'Value': list(train_ts.values) + list(test_ts.values),
                                'Type': ['Train']*len(train_ts) + ['Actual Test']*len(test_ts)
                            })
                            fore_df = pd.DataFrame({'Date': test_ts.index, 'Forecast': forecast.values})
 
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=train_ts.index, y=train_ts.values, name='Train',
                                                     line=dict(color='#3b82f6')))
                            fig.add_trace(go.Scatter(x=test_ts.index, y=test_ts.values, name='Actual',
                                                     line=dict(color='#22c55e')))
                            fig.add_trace(go.Scatter(x=test_ts.index, y=forecast.values, name='Forecast',
                                                     line=dict(color='#f97316', dash='dash')))
                            fig.update_layout(title='SARIMA Forecast vs Actual', **PLOTLY_LAYOUT)
                            st.plotly_chart(fig, use_container_width=True)
 
                            # Residuals
                            residuals = test_ts.values - forecast.values
                            fig_res = px.histogram(x=residuals, nbins=30, title='Forecast Residuals',
                                                   color_discrete_sequence=['#a855f7'])
                            fig_res.update_layout(**PLOTLY_LAYOUT)
                            st.plotly_chart(fig_res, use_container_width=True)
 
                        except Exception as e:
                            st.markdown(f'<div class="alert-danger">❌ SARIMA failed: {e}</div>', unsafe_allow_html=True)
 
    # ══════════════════════════════════════════════
    # TAB 7: MODEL COMPARISON
    # ══════════════════════════════════════════════
    with tabs[6]:
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">🏆</div>
          <p class="section-title">MODEL COMPARISON & RECOMMENDATION</p>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("""
        <div class="alert-info">
        This tab helps you pick the <b>best model</b> for your project by comparing metrics that actually matter:
        Cross-validation scores (generalisability), train vs test gap (overfitting), and interpretability.
        </div>
        """, unsafe_allow_html=True)
 
        if 'reg_results' in st.session_state:
            st.markdown("### Regression Results")
            st.dataframe(st.session_state['reg_results'], use_container_width=True)
 
            # Radar chart
            reg_df = st.session_state['reg_results'].copy()
            reg_df = reg_df[reg_df['CV R² (mean)'] != 'N/A'].copy()
            if not reg_df.empty:
                reg_df['Test R²'] = pd.to_numeric(reg_df['Test R²'], errors='coerce')
                reg_df['CV R² (mean)'] = pd.to_numeric(reg_df['CV R² (mean)'], errors='coerce')
                reg_df['Gap'] = (reg_df['Test R²'] - reg_df['CV R² (mean)']).abs()
                reg_df['Score'] = reg_df['CV R² (mean)'] * 0.6 + (1 - reg_df['Gap']) * 0.4
                reg_df = reg_df.sort_values('Score', ascending=False)
 
                fig = go.Figure()
                for i, (_, row) in enumerate(reg_df.iterrows()):
                    fig.add_trace(go.Bar(
                        name=row['Model'],
                        x=['CV R²','Test R²','Generalisation\nScore'],
                        y=[row['CV R² (mean)'], row['Test R²'], row['Score']],
                        marker_color=COLOR_SEQ[i % len(COLOR_SEQ)]
                    ))
                fig.update_layout(barmode='group', title='Multi-metric Model Comparison',
                                  **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
 
                best = reg_df.iloc[0]
                st.markdown(f"""
                <div class="alert-success">
                🏆 <b>Recommended Model: {best['Model']}</b><br>
                Generalisation Score: {best['Score']:.4f} | CV R²: {best['CV R² (mean)']} | Test R²: {best['Test R²']}
                <br><br>This model has the best balance of test performance and cross-validation stability.
                </div>
                """, unsafe_allow_html=True)
 
        st.markdown("""
        ### 📚 How to Choose the Best Model for Traffic Accident Prediction
 
        | Criterion | Best Choice |
        |---|---|
        | **Interpretability needed** | Linear Regression, Decision Tree |
        | **Best accuracy, tabular data** | Gradient Boosting / XGBoost |
        | **Balanced accuracy + speed** | Random Forest |
        | **Temporal/sequential patterns** | LSTM (Deep Learning) |
        | **Forecasting future accident counts** | SARIMA (Time Series) |
 
        ### 🚨 Why Was R² Perfect?
        Perfect R² (≈1.0) is almost always a sign of a problem, not a success:
        - **Target leakage**: A column in features directly encodes the target (e.g. injury_count in features when predicting injury_count)
        - **ID columns**: Row index or crash ID accidentally included as a feature
        - **Post-event features**: Information recorded after the accident (e.g. "report filed = yes")
        - **Data too small**: Only ~100 rows, model memorises training data
        - **Same-set evaluation**: Metrics computed on training data, not a hold-out set
 
        Use the **Leakage Detector** in the Features tab and always compare **Test R²** vs **CV R²**.
        """)
 
else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
      <div style="font-size:4rem; margin-bottom:1rem;">🚦</div>
      <h2 style="font-family:'Space Mono',monospace; color:#64748b;">No Dataset Loaded</h2>
      <p style="color:#475569; max-width:400px; margin:0 auto 2rem;">
        Upload a CSV file above to begin your traffic accident analysis.
        The system will automatically detect columns and guide you through EDA, modelling, and forecasting.
      </p>
      <div style="background:#161920; border:1px solid #2a3044; border-radius:12px; padding:1.5rem; max-width:400px; margin:0 auto; text-align:left;">
        <p style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#64748b; margin:0 0 0.5rem;">EXPECTED COLUMNS (optional)</p>
        <code style="color:#f97316; font-size:0.8rem;">crash_date, crash_hour, crash_day_of_week</code><br>
        <code style="color:#3b82f6; font-size:0.8rem;">injuries_total, crash_type, ...</code>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
