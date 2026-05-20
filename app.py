import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import resample
from xgboost import XGBRegressor, XGBClassifier

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
.predict-box {
    background: linear-gradient(135deg, #1a1d2e 0%, #0f1220 100%);
    border: 1px solid #4f6ef7; border-radius: 12px;
    padding: 24px; margin-top: 16px;
}
.result-box {
    background: linear-gradient(135deg, #0f2010 0%, #0d1a0f 100%);
    border: 2px solid #34d399; border-radius: 12px;
    padding: 24px; text-align: center; margin-top: 16px;
}
.result-value { font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; font-weight: 700; color: #34d399; }
.result-label { font-size: 0.9rem; color: #8891b0; margin-top: 8px; }
.warning-box {
    background: linear-gradient(135deg, #1f1500 0%, #150e00 100%);
    border: 2px solid #f59e0b; border-radius: 12px;
    padding: 24px; text-align: center; margin-top: 16px;
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


# ── ML Training ───────────────────────────────────────────────────────────────
@st.cache_data
def train_all_models(df):
    drop_cols = [c for c in ['crash_date','injuries_non_incapacitating','injuries_fatal',
                              'injuries_incapacitating','injuries_reported_not_evident',
                              'injuries_no_indication'] if c in df.columns]

    # ── Regression ────────────────────────────────────────────────────────────
    ml_df = df.drop(columns=drop_cols).copy()
    le_dict = {}
    for col in ml_df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        le_dict[col] = le

    reg_results = None
    reg_models  = {}
    reg_features = []

    if 'injuries_total' in ml_df.columns:
        X = ml_df.drop(columns=['injuries_total'])
        y = ml_df['injuries_total']
        reg_features = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        regressors = {
            'Linear Regression': LinearRegression(),
            'Decision Tree':     DecisionTreeRegressor(random_state=42),
            'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost':           XGBRegressor(random_state=42, verbosity=0),
        }
        rows = []
        for name, model in regressors.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rows.append({'Model': name,
                         'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
                         'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                         'R²':   round(r2_score(y_test, y_pred), 4)})
            reg_models[name] = model
        reg_results = pd.DataFrame(rows)

    # ── Classification ────────────────────────────────────────────────────────
    clf_results  = None
    clf_models   = {}
    clf_features = []
    cms          = {}
    class_labels = []

    if 'crash_type' in df.columns:
        df_maj = df[df['crash_type'] == df['crash_type'].value_counts().idxmax()]
        df_min = df[df['crash_type'] == df['crash_type'].value_counts().idxmin()]
        df_bal = pd.concat([
            df_min,
            resample(df_maj, replace=False, n_samples=len(df_min), random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        clf_df = df_bal.drop(columns=drop_cols, errors='ignore').copy()
        for col in clf_df.select_dtypes(include='object').columns:
            if col != 'crash_type':
                le = LabelEncoder()
                clf_df[col] = le.fit_transform(clf_df[col].astype(str))
                le_dict[col] = le

        le_target = LabelEncoder()
        clf_df['crash_type'] = le_target.fit_transform(clf_df['crash_type'].astype(str))
        class_labels = list(le_target.classes_)
        le_dict['crash_type_target'] = le_target

        X = clf_df.drop(columns=['crash_type'])
        y = clf_df['crash_type']
        clf_features = list(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree':       DecisionTreeClassifier(random_state=42),
            'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting':   GradientBoostingClassifier(random_state=42),
            'XGBoost':             XGBClassifier(random_state=42, verbosity=0),
        }
        rows = []
        for name, model in classifiers.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rows.append({'Model':    name,
                         'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                         'F1-Score': round(float(classification_report(
                             y_test, y_pred, output_dict=True)['weighted avg']['f1-score']), 4)})
            clf_models[name] = model
            cms[name] = confusion_matrix(y_test, y_pred)
        clf_results = pd.DataFrame(rows)

    return reg_results, reg_models, reg_features, clf_results, clf_models, clf_features, cms, class_labels, le_dict


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
tab1, tab2, tab3, tab4 = st.tabs(["📊 Temporal Analysis","☁️ Conditions","⚠️ Causes & Severity","🔬 Predict & Models"])

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

# ─── TAB 4 : PREDICT & MODELS ─────────────────────────────────────────────────
with tab4:
    st.markdown("### 🔬 Predict & Model Insights")

    # ── Train models ──────────────────────────────────────────────────────────
    with st.spinner("⚙️ Training models on your data... please wait"):
        (reg_results, reg_models, reg_features,
         clf_results, clf_models, clf_features,
         cms, class_labels, le_dict) = train_all_models(df)

    st.success("✅ Models trained and ready!")
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 1 — USER INPUT PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 🎯 Make a Prediction")
    st.markdown("Fill in the accident details below and get an instant prediction.")

    with st.container():
        st.markdown('<div class="predict-box">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**⏰ Time & Day**")
            inp_hour = st.selectbox("Hour of Day", list(range(24)),
                                    format_func=lambda x: f"{x:02d}:00")
            inp_day  = st.selectbox("Day of Week",
                                    [0,1,2,3,4,5,6],
                                    format_func=lambda x: ['Monday','Tuesday','Wednesday',
                                                            'Thursday','Friday','Saturday','Sunday'][x])
            inp_month = st.selectbox("Month", ['January','February','March','April','May','June',
                                               'July','August','September','October','November','December'])

        with col2:
            st.markdown("**🌤️ Conditions**")
            inp_weather = st.selectbox("Weather Condition",
                                       ['CLEAR','RAIN','SNOW','CLOUDY','FOG/SMOKE/HAZE',
                                        'BLOWING SNOW','STORM','FREEZING RAIN/DRIZZLE'])
            inp_road = st.selectbox("Road Surface",
                                    ['DRY','WET','SNOW OR SLUSH','ICE',
                                     'SAND MUD DIRT','OTHER'])
            inp_lighting = st.selectbox("Lighting Condition",
                                        ['DAYLIGHT','DARKNESS, LIGHTED ROAD',
                                         'DARKNESS','DAWN','DUSK']) if 'lighting_condition' in df.columns else None

        with col3:
            st.markdown("**🚗 Crash Details**")
            inp_cause = st.selectbox("Primary Cause",
                                     ['FAILING TO YIELD RIGHT-OF-WAY','FOLLOWING TOO CLOSELY',
                                      'IMPROPER OVERTAKING','FAILING TO REDUCE SPEED',
                                      'DISREGARDING TRAFFIC SIGNALS',
                                      'DISTRACTION - FROM INSIDE VEHICLE',
                                      'IMPROPER TURNING/NO SIGNAL','WEATHER',
                                      'ROAD ENGINEERING/SURFACE/MARKING DEFECTS',
                                      'UNABLE TO DETERMINE'])
            inp_model_reg = st.selectbox("Regression Model", list(reg_models.keys()) if reg_models else ['N/A'])
            inp_model_clf = st.selectbox("Classifier Model", list(clf_models.keys()) if clf_models else ['N/A'])

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Derived features ──────────────────────────────────────────────────────
    is_weekend   = 1 if inp_day >= 5 else 0
    is_night     = 1 if inp_hour >= 20 or inp_hour <= 5 else 0
    is_rush_hour = 1 if (7 <= inp_hour <= 9) or (16 <= inp_hour <= 19) else 0
    weather_risk = 1 if inp_weather in ['RAIN','SNOW','FOG/SMOKE/HAZE','STORM','FREEZING RAIN/DRIZZLE','BLOWING SNOW'] else 0
    road_risk    = 1 if inp_road in ['WET','SNOW OR SLUSH','ICE'] else 0

    # ── Predict button ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡ Predict Now", type="primary", use_container_width=True)

    if predict_clicked:
        # Build input dict with all possible features
        raw_input = {
            'crash_hour':            inp_hour,
            'crash_day_of_week':     inp_day,
            'crash_month':           inp_month,
            'weather_condition':     inp_weather,
            'roadway_surface_cond':  inp_road,
            'prim_contributory_cause': inp_cause,
            'is_weekend':            is_weekend,
            'is_night':              is_night,
            'is_rush_hour':          is_rush_hour,
            'weather_risk':          weather_risk,
            'road_risk':             road_risk,
            'severity_score':        0,
            'cause_freq':            df['prim_contributory_cause'].value_counts().get(inp_cause, 1)
                                     if 'prim_contributory_cause' in df.columns else 1,
        }

        r1, r2 = st.columns(2)

        # ── Regression prediction ──────────────────────────────────────────
        with r1:
            st.markdown("### 🩹 Predicted Injuries")
            if reg_models and inp_model_reg in reg_models:
                try:
                    model = reg_models[inp_model_reg]
                    row = {}
                    for feat in reg_features:
                        if feat in raw_input:
                            val = raw_input[feat]
                            if feat in le_dict and isinstance(val, str):
                                try:
                                    val = le_dict[feat].transform([val])[0]
                                except:
                                    val = 0
                            row[feat] = val
                        else:
                            row[feat] = 0
                    X_pred = pd.DataFrame([row])[reg_features]
                    pred_injuries = max(0, round(float(model.predict(X_pred)[0]), 2))

                    color = "#34d399" if pred_injuries < 1 else "#f59e0b" if pred_injuries < 3 else "#ef4444"
                    emoji = "✅" if pred_injuries < 1 else "⚠️" if pred_injuries < 3 else "🚨"

                    st.markdown(f"""
                    <div class="result-box" style="border-color:{color}">
                        <div style="font-size:3rem">{emoji}</div>
                        <div class="result-value" style="color:{color}">{pred_injuries}</div>
                        <div class="result-label">Predicted total injuries</div>
                        <div style="color:#8891b0; font-size:0.8rem; margin-top:8px">
                            Model: {inp_model_reg}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        # ── Classification prediction ──────────────────────────────────────
        with r2:
            st.markdown("### 🚗 Predicted Crash Type")
            if clf_models and inp_model_clf in clf_models:
                try:
                    model = clf_models[inp_model_clf]
                    row = {}
                    for feat in clf_features:
                        if feat in raw_input:
                            val = raw_input[feat]
                            if feat in le_dict and isinstance(val, str):
                                try:
                                    val = le_dict[feat].transform([val])[0]
                                except:
                                    val = 0
                            row[feat] = val
                        else:
                            row[feat] = 0
                    X_pred = pd.DataFrame([row])[clf_features]
                    pred_class_idx = model.predict(X_pred)[0]
                    pred_label = class_labels[pred_class_idx] if class_labels else str(pred_class_idx)

                    is_injury = 'INJURY' in pred_label.upper() or 'TOW' in pred_label.upper()
                    color = "#ef4444" if is_injury else "#34d399"
                    emoji = "🚨" if is_injury else "✅"

                    st.markdown(f"""
                    <div class="result-box" style="border-color:{color}">
                        <div style="font-size:3rem">{emoji}</div>
                        <div class="result-value" style="color:{color}; font-size:1.3rem">
                            {pred_label}
                        </div>
                        <div class="result-label">Predicted crash outcome</div>
                        <div style="color:#8891b0; font-size:0.8rem; margin-top:8px">
                            Model: {inp_model_clf}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        # ── Risk summary ───────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Risk Summary")
        flags = []
        if is_night:      flags.append("🌙 Night-time driving")
        if is_rush_hour:  flags.append("🚗 Rush hour traffic")
        if is_weekend:    flags.append("📅 Weekend")
        if weather_risk:  flags.append(f"⛈️ Risky weather: {inp_weather}")
        if road_risk:     flags.append(f"🛣️ Hazardous road: {inp_road}")

        if flags:
            st.warning("**Risk factors detected:**\n\n" + "\n\n".join(f"- {f}" for f in flags))
        else:
            st.success("✅ No major risk factors detected for this scenario.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION 2 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("## 📊 Model Performance")

    m1, m2 = st.tabs(["📈 Regression Results", "🎯 Classification Results"])

    with m1:
        if reg_results is not None:
            st.dataframe(reg_results, use_container_width=True, hide_index=True)
            fig = go.Figure([
                go.Bar(name='R²',  x=reg_results['Model'], y=reg_results['R²'],  marker_color=PALETTE[0]),
                go.Bar(name='MAE', x=reg_results['Model'], y=reg_results['MAE'], marker_color=PALETTE[2]),
                go.Bar(name='RMSE',x=reg_results['Model'], y=reg_results['RMSE'],marker_color=PALETTE[3]),
            ])
            fig.update_layout(**THEME, barmode='group', title="Regression Model Comparison")
            st.plotly_chart(fig, use_container_width=True)
            best = reg_results.loc[reg_results['R²'].idxmax(), 'Model']
            st.success(f"🏆 Best Regression Model: **{best}**")
        else:
            st.warning("No regression results — `injuries_total` column not found.")

    with m2:
        if clf_results is not None:
            st.dataframe(clf_results, use_container_width=True, hide_index=True)
            fig2 = go.Figure([
                go.Bar(name='Accuracy', x=clf_results['Model'], y=clf_results['Accuracy'], marker_color=PALETTE[0]),
                go.Bar(name='F1-Score', x=clf_results['Model'], y=clf_results['F1-Score'], marker_color=PALETTE[1]),
            ])
            fig2.update_layout(**THEME, barmode='group', title="Classifier Comparison")
            st.plotly_chart(fig2, use_container_width=True)

            best_clf = clf_results.loc[clf_results['Accuracy'].idxmax(), 'Model']
            st.success(f"🏆 Best Classifier: **{best_clf}**")

            st.markdown(f"#### Confusion Matrix — {best_clf}")
            cm = cms[best_clf]
            fig3 = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                             labels=dict(x="Predicted", y="Actual"),
                             x=class_labels, y=class_labels)
            fig3.update_layout(**THEME)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No classification results — `crash_type` column not found.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#4a5270;font-size:0.8rem;'>"
            "Traffic Accident Analytics Dashboard • Streamlit + Plotly + Scikit-learn + XGBoost"
            "</p>", unsafe_allow_html=True)
