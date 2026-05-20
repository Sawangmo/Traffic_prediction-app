import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Accident Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.main { background: #0d0f1a; }
[data-testid="stSidebar"] { background: #111320; border-right: 1px solid #1e2235; }
.kpi-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #0f1220 100%);
    border: 1px solid #2a2e45; border-radius: 12px;
    padding: 20px 24px; text-align: center;
}
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 600; color: #4f6ef7; }
.kpi-label { font-size: 0.78rem; color: #8891b0; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 4px; }
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #e0e4f5;
    border-left: 3px solid #4f6ef7; padding-left: 12px; margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        return generate_sample_data()

    if 'crash_date' in df.columns:
        df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')
        df['crash_month'] = df['crash_date'].dt.month_name()

    if 'crash_day_of_week' in df.columns:
        df['is_weekend'] = df['crash_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    if 'crash_hour' in df.columns:
        df['is_night']     = df['crash_hour'].apply(lambda x: 1 if x >= 20 or x <= 5 else 0)
        df['is_rush_hour'] = df['crash_hour'].apply(lambda x: 1 if (7<=x<=9) or (16<=x<=19) else 0)

    if 'weather_condition' in df.columns:
        df['weather_risk'] = df['weather_condition'].apply(
            lambda x: 1 if x in ['RAIN','SNOW','FOG','STORM'] else 0)

    if 'roadway_surface_cond' in df.columns:
        df['road_risk'] = df['roadway_surface_cond'].apply(
            lambda x: 1 if x in ['WET','SNOW OR SLUSH','ICE'] else 0)

    present = {c: w for c, w in {'injuries_fatal':5,'injuries_incapacitating':3,
                                   'injuries_non_incapacitating':1}.items() if c in df.columns}
    if present:
        df['severity_score'] = sum(df[c]*w for c,w in present.items())

    return df


def generate_sample_data():
    np.random.seed(42)
    n = 5000
    hours = np.random.choice(range(24), n, p=[
        0.01,0.01,0.01,0.01,0.01,0.02,0.04,0.07,0.06,0.05,
        0.05,0.05,0.05,0.05,0.05,0.05,0.06,0.07,0.06,0.05,
        0.04,0.03,0.02,0.01])
    df = pd.DataFrame({
        'crash_hour': hours,
        'crash_day_of_week': np.random.randint(0, 7, n),
        'crash_month': np.random.choice(['January','February','March','April','May','June',
                                         'July','August','September','October','November','December'], n),
        'weather_condition': np.random.choice(['CLEAR','RAIN','SNOW','CLOUDY','FOG','STORM'], n,
                                               p=[0.55,0.18,0.10,0.10,0.05,0.02]),
        'roadway_surface_cond': np.random.choice(['DRY','WET','SNOW OR SLUSH','ICE','SAND MUD DIRT'], n,
                                                  p=[0.60,0.22,0.10,0.05,0.03]),
        'prim_contributory_cause': np.random.choice([
            'FAILING TO YIELD RIGHT-OF-WAY','FOLLOWING TOO CLOSELY','IMPROPER OVERTAKING',
            'FAILING TO REDUCE SPEED','DISREGARDING TRAFFIC SIGNALS',
            'DISTRACTION - FROM INSIDE VEHICLE','IMPROPER TURNING/NO SIGNAL',
            'WEATHER','ROAD ENGINEERING/SURFACE/MARKING DEFECTS','UNABLE TO DETERMINE'], n),
        'crash_type': np.random.choice(
            ['NO INJURY / DRIVE AWAY','INJURY AND / OR TOW DUE TO CRASH'], n, p=[0.52,0.48]),
        'injuries_fatal':               np.random.poisson(0.02, n),
        'injuries_incapacitating':      np.random.poisson(0.10, n),
        'injuries_non_incapacitating':  np.random.poisson(0.30, n),
        'injuries_total':               np.random.poisson(0.42, n),
    })
    df['is_weekend']     = df['crash_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_night']       = df['crash_hour'].apply(lambda x: 1 if x >= 20 or x <= 5 else 0)
    df['is_rush_hour']   = df['crash_hour'].apply(lambda x: 1 if (7<=x<=9) or (16<=x<=19) else 0)
    df['weather_risk']   = df['weather_condition'].apply(lambda x: 1 if x in ['RAIN','SNOW','FOG','STORM'] else 0)
    df['road_risk']      = df['roadway_surface_cond'].apply(lambda x: 1 if x in ['WET','SNOW OR SLUSH','ICE'] else 0)
    df['severity_score'] = df['injuries_fatal']*5 + df['injuries_incapacitating']*3 + df['injuries_non_incapacitating']*1
    return df


THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Grotesk', color='#8891b0'),
    xaxis=dict(gridcolor='#1e2235', linecolor='#2a2e45'),
    yaxis=dict(gridcolor='#1e2235', linecolor='#2a2e45'),
    margin=dict(l=20, r=20, t=40, b=20),
)
PALETTE = ['#4f6ef7','#34d399','#f59e0b','#ef4444','#a78bfa','#06b6d4','#fb923c','#e879f9']

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚦 Traffic Accident\n### Analytics Dashboard")
    st.markdown("---")
    uploaded = st.file_uploader("Upload traffic_accidents.csv", type=["csv"])
    df = load_data(uploaded)
    st.markdown("---")
    st.markdown("**FILTERS**")
    hour_range = st.slider("Hour of Day", 0, 23, (0, 23))

    if 'weather_condition' in df.columns:
        weather_opts = sorted(df['weather_condition'].dropna().unique().tolist())
        sel_weather  = st.multiselect("Weather Condition", weather_opts, default=weather_opts[:5])
    else:
        sel_weather = []

    if 'crash_type' in df.columns:
        crash_types    = sorted(df['crash_type'].dropna().unique().tolist())
        sel_crash_type = st.multiselect("Crash Type", crash_types, default=crash_types)
    else:
        sel_crash_type = []

    st.markdown("---")
    if uploaded:
        st.success(f"✅ Loaded {len(df):,} records")
    else:
        st.info("📊 Using sample data\nUpload your CSV above to use real data.")

# ── Apply filters ─────────────────────────────────────────────────────────────
fdf = df.copy()
if 'crash_hour' in fdf.columns:
    fdf = fdf[fdf['crash_hour'].between(*hour_range)]
if sel_weather and 'weather_condition' in fdf.columns:
    fdf = fdf[fdf['weather_condition'].isin(sel_weather)]
if sel_crash_type and 'crash_type' in fdf.columns:
    fdf = fdf[fdf['crash_type'].isin(sel_crash_type)]

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER + KPIs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🚦 Traffic Accident Analytics")
st.markdown(f"Showing **{len(fdf):,}** of **{len(df):,}** records after filters")
st.markdown("---")

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (f"{len(fdf):,}", "Total Accidents"),
    (f"{int(fdf['injuries_fatal'].sum()) if 'injuries_fatal' in fdf.columns else 0:,}", "Fatal Injuries"),
    (f"{int(fdf['injuries_total'].sum()) if 'injuries_total' in fdf.columns else 0:,}", "Total Injuries"),
    (f"{round(fdf['is_night'].mean()*100,1) if 'is_night' in fdf.columns else 0}%", "Night-time Crashes"),
    (f"{round(fdf['is_weekend'].mean()*100,1) if 'is_weekend' in fdf.columns else 0}%", "Weekend Crashes"),
]
for col, (val, label) in zip([k1,k2,k3,k4,k5], kpis):
    col.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div>'
                 f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📊 Temporal Analysis","☁️ Conditions","⚠️ Causes & Severity","🔬 Model Insights"])

# ─── TAB 1 ────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Accidents by Hour of Day</div>', unsafe_allow_html=True)
        if 'crash_hour' in fdf.columns:
            hc = fdf['crash_hour'].value_counts().sort_index().reset_index()
            hc.columns = ['Hour','Count']
            fig = px.area(hc, x='Hour', y='Count', color_discrete_sequence=[PALETTE[0]])
            fig.update_traces(fill='tozeroy', fillcolor='rgba(79,110,247,0.15)', line_color=PALETTE[0])
            fig.update_layout(**THEME)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Accidents by Day of Week</div>', unsafe_allow_html=True)
        if 'crash_day_of_week' in fdf.columns:
            day_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
            dc = fdf['crash_day_of_week'].map(day_map).value_counts().reindex(
                ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']).reset_index()
            dc.columns = ['Day','Count']
            colors = [PALETTE[3] if d in ['Sat','Sun'] else PALETTE[0] for d in dc['Day']]
            fig = px.bar(dc, x='Day', y='Count', color='Day', color_discrete_sequence=colors)
            fig.update_layout(**THEME, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Accidents by Month</div>', unsafe_allow_html=True)
    if 'crash_month' in fdf.columns:
        month_order = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December']
        mc = fdf['crash_month'].value_counts().reindex(month_order).dropna().reset_index()
        mc.columns = ['Month','Count']
        fig = px.bar(mc, x='Month', y='Count', color='Count', color_continuous_scale='Blues')
        fig.update_layout(**THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    if 'is_rush_hour' in fdf.columns:
        st.markdown('<div class="section-title">Rush Hour vs Off-Peak vs Night</div>', unsafe_allow_html=True)
        rush = fdf['is_rush_hour'].sum()
        night = fdf['is_night'].sum()
        normal = len(fdf) - rush - night
        fig = px.pie(values=[rush, night, normal],
                     names=['Rush Hour','Night-time','Off-Peak'],
                     color_discrete_sequence=PALETTE[:3], hole=0.55)
        fig.update_layout(**THEME)
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 2 ────────────────────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Weather Conditions</div>', unsafe_allow_html=True)
        if 'weather_condition' in fdf.columns:
            wc = fdf['weather_condition'].value_counts().head(10).reset_index()
            wc.columns = ['Weather','Count']
            fig = px.bar(wc, x='Count', y='Weather', orientation='h',
                         color='Count', color_continuous_scale='Viridis')
            fig.update_layout(**THEME, coloraxis_showscale=False)
            fig.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Road Surface Conditions</div>', unsafe_allow_html=True)
        if 'roadway_surface_cond' in fdf.columns:
            rc = fdf['roadway_surface_cond'].value_counts().head(8).reset_index()
            rc.columns = ['Surface','Count']
            fig = px.bar(rc, x='Count', y='Surface', orientation='h',
                         color='Count', color_continuous_scale='RdYlGn_r')
            fig.update_layout(**THEME, coloraxis_showscale=False)
            fig.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig, use_container_width=True)

    if 'crash_type' in fdf.columns:
        st.markdown('<div class="section-title">Crash Type Distribution</div>', unsafe_allow_html=True)
        ct = fdf['crash_type'].value_counts().reset_index()
        ct.columns = ['Type','Count']
        fig = px.pie(ct, values='Count', names='Type',
                     color_discrete_sequence=[PALETTE[0],PALETTE[2]], hole=0.5)
        fig.update_layout(**THEME)
        st.plotly_chart(fig, use_container_width=True)

    if 'weather_risk' in fdf.columns and 'road_risk' in fdf.columns:
        st.markdown('<div class="section-title">Combined Risk Factor</div>', unsafe_allow_html=True)
        risk_df = fdf.groupby(['weather_risk','road_risk']).size().reset_index(name='Count')
        risk_df['Weather Risk'] = risk_df['weather_risk'].map({0:'Low',1:'High'})
        risk_df['Road Risk']    = risk_df['road_risk'].map({0:'Low',1:'High'})
        fig = px.bar(risk_df, x='Weather Risk', y='Count', color='Road Risk',
                     barmode='group', color_discrete_sequence=[PALETTE[1],PALETTE[3]])
        fig.update_layout(**THEME)
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Top 10 Primary Causes of Accidents</div>', unsafe_allow_html=True)
    if 'prim_contributory_cause' in fdf.columns:
        causes = fdf['prim_contributory_cause'].value_counts().head(10).reset_index()
        causes.columns = ['Cause','Count']
        fig = px.bar(causes, x='Count', y='Cause', orientation='h',
                     color='Count', color_continuous_scale='Reds')
        fig.update_layout(**THEME, coloraxis_showscale=False, height=420)
        fig.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Injury Breakdown</div>', unsafe_allow_html=True)
        inj_cols = [c for c in ['injuries_fatal','injuries_incapacitating',
                                  'injuries_non_incapacitating','injuries_total'] if c in fdf.columns]
        if inj_cols:
            inj_sums = fdf[inj_cols].sum().reset_index()
            inj_sums.columns = ['Type','Count']
            inj_sums['Type'] = inj_sums['Type'].str.replace('injuries_','').str.replace('_',' ').str.title()
            fig = px.bar(inj_sums, x='Type', y='Count', color='Type', color_discrete_sequence=PALETTE)
            fig.update_layout(**THEME, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Severity Score Distribution</div>', unsafe_allow_html=True)
        if 'severity_score' in fdf.columns:
            fig = px.histogram(fdf[fdf['severity_score']>0], x='severity_score',
                               nbins=30, color_discrete_sequence=[PALETTE[2]])
            fig.update_layout(**THEME, xaxis_title="Severity Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    if 'crash_hour' in fdf.columns and 'severity_score' in fdf.columns:
        st.markdown('<div class="section-title">Average Severity Score by Hour</div>', unsafe_allow_html=True)
        sev_hour = fdf.groupby('crash_hour')['severity_score'].mean().reset_index()
        fig = px.line(sev_hour, x='crash_hour', y='severity_score',
                      markers=True, color_discrete_sequence=[PALETTE[2]])
        fig.update_layout(**THEME, xaxis_title="Hour", yaxis_title="Avg Severity Score")
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 4 ────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🤖 ML Model Performance Summary")

    model_results = pd.DataFrame({
        'Model':  ['Linear Regression','Decision Tree','Random Forest',
                   'Gradient Boosting','XGBoost','LSTM'],
        'MAE':    [0.312, 0.198, 0.145, 0.138, 0.132, 0.128],
        'RMSE':   [0.481, 0.312, 0.241, 0.228, 0.219, 0.211],
        'R²':     [0.312, 0.578, 0.712, 0.731, 0.748, 0.762],
    })
    clf_results = pd.DataFrame({
        'Model':    ['Logistic Regression','Decision Tree','Random Forest',
                     'Gradient Boosting','LSTM Classifier'],
        'Accuracy': [0.71, 0.78, 0.84, 0.86, 0.88],
        'F1-Score': [0.71, 0.78, 0.84, 0.86, 0.88],
    })

    st.markdown("#### Regression — predicting `injuries_total`")
    st.dataframe(model_results, use_container_width=True, hide_index=True)
    fig = go.Figure([
        go.Bar(name='R²',  x=model_results['Model'], y=model_results['R²'],  marker_color=PALETTE[0]),
        go.Bar(name='MAE', x=model_results['Model'], y=model_results['MAE'], marker_color=PALETTE[2]),
    ])
    fig.update_layout(**THEME, barmode='group', title="Regression Model Comparison")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Classification — predicting `crash_type`")
    st.dataframe(clf_results, use_container_width=True, hide_index=True)
    fig2 = px.bar(clf_results, x='Model', y='Accuracy',
                  color='Accuracy', color_continuous_scale='Blues')
    fig2.update_layout(**THEME, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.info("💡 Update the model_results and clf_results DataFrames above with your notebook's actual scores.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#4a5270;font-size:0.8rem;'>"
            "Traffic Accident Analytics Dashboard • Streamlit + Plotly</p>", unsafe_allow_html=True)
