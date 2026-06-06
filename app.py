# Traffic Accident Intelligence Dashboard (Part 1)

# Dashboard + EDA + Professional UI

```python
# ============================================================
# TRAFFIC ACCIDENT INTELLIGENCE DASHBOARD
# PART 1 - PROFESSIONAL UI + DASHBOARD + EDA
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Accident Intelligence",
    page_icon="🚦",
    layout="wide"
)

# ------------------------------------------------------------
# PROFESSIONAL LIGHT THEME
# ------------------------------------------------------------
st.markdown("""
<style>

.main {
    background-color:#f8fafc;
}

.stApp{
    background-color:#f8fafc;
}

h1,h2,h3,h4{
    color:#0f172a;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0 2px 10px rgba(0,0,0,0.08);
    text-align:center;
    margin-bottom:10px;
}

.metric-title{
    font-size:16px;
    color:#64748b;
    font-weight:600;
}

.metric-value{
    font-size:34px;
    color:#0f172a;
    font-weight:700;
}

.section-header{
    background:#2563eb;
    color:white;
    padding:12px;
    border-radius:10px;
    font-size:22px;
    font-weight:bold;
    margin-top:20px;
    margin-bottom:20px;
}

.info-box{
    background:white;
    border-left:5px solid #2563eb;
    padding:15px;
    border-radius:10px;
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# PLOTLY THEME
# ------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(size=15),
    title_font=dict(size=22),
    legend=dict(font=dict(size=13)),
    margin=dict(l=40, r=40, t=60, b=40),
    height=550
)

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def get_numeric_cols(df):
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

def get_categorical_cols(df):
    return [
        c for c in df.columns
        if df[c].dtype == "object"
    ]

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("🚦 Traffic Accident Intelligence Dashboard")

st.markdown("""
### Machine Learning • Deep Learning • Forecasting

Interactive platform for traffic accident analysis,
prediction, visualization and decision support.
""")

# ------------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Traffic Accident Dataset",
    type=["csv"]
)

# ------------------------------------------------------------
# NO FILE STATE
# ------------------------------------------------------------
if uploaded_file is None:

    st.info("""
    Upload a CSV dataset to begin.

    Recommended columns:
    - crash_date
    - crash_hour
    - crash_day_of_week
    - injuries_total
    - crash_type
    """)

    st.stop()

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv(uploaded_file)

if "crash_date" in df.columns:
    try:
        df["crash_date"] = pd.to_datetime(df["crash_date"])
    except:
        pass

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2 = st.tabs([
    "📊 Dashboard",
    "🔍 Exploratory Data Analysis"
])

# ============================================================
# DASHBOARD TAB
# ============================================================
with tab1:

    st.markdown(
        '<div class="section-header">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    num_cols = get_numeric_cols(df)
    cat_cols = get_categorical_cols(df)

    missing_pct = round(
        (df.isnull().sum().sum() / df.size) * 100,
        2
    )

    c1,c2,c3,c4,c5 = st.columns(5)

    with c1:
        st.metric(
            "Total Records",
            f"{df.shape[0]:,}"
        )

    with c2:
        st.metric(
            "Total Features",
            df.shape[1]
        )

    with c3:
        st.metric(
            "Numeric Columns",
            len(num_cols)
        )

    with c4:
        st.metric(
            "Categorical Columns",
            len(cat_cols)
        )

    with c5:
        st.metric(
            "Missing %",
            f"{missing_pct}%"
        )

    st.divider()

    # --------------------------------------------------------
    # Missing Values
    # --------------------------------------------------------
    col1,col2 = st.columns(2)

    with col1:

        miss = (
            df.isnull()
            .sum()
            .reset_index()
        )

        miss.columns = [
            "Column",
            "Missing"
        ]

        miss = miss[
            miss["Missing"] > 0
        ]

        if len(miss) > 0:

            fig = px.bar(
                miss.sort_values("Missing"),
                x="Missing",
                y="Column",
                orientation="h",
                title="Missing Values by Column",
                color="Missing"
            )

            fig.update_layout(**PLOTLY_LAYOUT)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.success(
                "No missing values detected."
            )

    # --------------------------------------------------------
    # Data Types
    # --------------------------------------------------------
    with col2:

        dtypes = (
            df.dtypes
            .astype(str)
            .value_counts()
            .reset_index()
        )

        dtypes.columns = [
            "Type",
            "Count"
        ]

        fig = px.pie(
            dtypes,
            names="Type",
            values="Count",
            hole=0.55,
            title="Data Type Distribution"
        )

        fig.update_layout(**PLOTLY_LAYOUT)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-header">Dataset Preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        height=450
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-header">Statistical Summary</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.describe(include="all"),
        use_container_width=True,
        height=450
    )

# ============================================================
# EDA TAB
# ============================================================
with tab2:

    st.markdown(
        '<div class="section-header">Numerical Analysis</div>',
        unsafe_allow_html=True
    )

    numeric_cols = get_numeric_cols(df)

    if len(numeric_cols) > 0:

        selected_num = st.selectbox(
            "Select Numeric Column",
            numeric_cols
        )

        c1,c2 = st.columns(2)

        with c1:

            fig = px.histogram(
                df,
                x=selected_num,
                nbins=40,
                title=f"Distribution of {selected_num}"
            )

            fig.update_layout(**PLOTLY_LAYOUT)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:

            fig = px.box(
                df,
                y=selected_num,
                title=f"Boxplot of {selected_num}"
            )

            fig.update_layout(**PLOTLY_LAYOUT)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.dataframe(
            df[selected_num]
            .describe()
            .to_frame(),
            use_container_width=True
        )

    # --------------------------------------------------------
    # CATEGORICAL ANALYSIS
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-header">Categorical Analysis</div>',
        unsafe_allow_html=True
    )

    cat_cols = get_categorical_cols(df)

    if len(cat_cols) > 0:

        selected_cat = st.selectbox(
            "Select Categorical Column",
            cat_cols
        )

        counts = (
            df[selected_cat]
            .value_counts()
            .reset_index()
        )

        counts.columns = [
            selected_cat,
            "Count"
        ]

        c1,c2 = st.columns(2)

        with c1:

            fig = px.bar(
                counts.head(20),
                x="Count",
                y=selected_cat,
                orientation="h",
                title=f"{selected_cat} Distribution"
            )

            fig.update_layout(**PLOTLY_LAYOUT)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with c2:

            fig = px.pie(
                counts.head(10),
                names=selected_cat,
                values="Count",
                hole=0.5,
                title=f"{selected_cat} Share"
            )

            fig.update_layout(**PLOTLY_LAYOUT)

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # --------------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-header">Correlation Analysis</div>',
        unsafe_allow_html=True
    )

    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )

        fig.update_layout(
            height=700,
            **PLOTLY_LAYOUT
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SCATTER EXPLORER
    # --------------------------------------------------------
    st.markdown(
        '<div class="section-header">Scatter Explorer</div>',
        unsafe_allow_html=True
    )

    if len(numeric_cols) >= 2:

        col1,col2,col3 = st.columns(3)

        x_axis = col1.selectbox(
            "X Axis",
            numeric_cols,
            key="x"
        )

        y_axis = col2.selectbox(
            "Y Axis",
            numeric_cols,
            index=1,
            key="y"
        )

        color_col = col3.selectbox(
            "Color By",
            ["None"] + cat_cols
        )

        color_arg = None if color_col == "None" else color_col

        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_arg,
            title=f"{x_axis} vs {y_axis}",
            opacity=0.7
        )

        fig.update_layout(**PLOTLY_LAYOUT)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ============================================================

# ML HELPER FUNCTIONS

# ============================================================

def prepare_data(data):

```
data = data.copy()

datetime_cols = data.select_dtypes(
    include=["datetime64"]
).columns

if len(datetime_cols) > 0:
    data = data.drop(columns=datetime_cols)

for col in data.columns:

    if data[col].dtype == "object":

        data[col] = data[col].fillna(
            data[col].mode()[0]
        )

        le = LabelEncoder()

        data[col] = le.fit_transform(
            data[col].astype(str)
        )

    else:

        data[col] = data[col].fillna(
            data[col].median()
        )

return data
```

def detect_leakage(X, y):

```
suspicious = []

for col in X.columns:

    try:

        corr = np.corrcoef(
            X[col],
            y
        )[0,1]

        if abs(corr) > 0.98:

            suspicious.append(
                (col, round(corr,4))
            )

    except:
        pass

return suspicious
```
```python
# ============================================================
# MACHINE LEARNING TAB
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-header">Machine Learning Models</div>',
        unsafe_allow_html=True
    )

    if "engineered_df" in st.session_state:
        ml_df = st.session_state["engineered_df"].copy()
    else:
        ml_df = df.copy()

    ml_df = prepare_data(ml_df)

    numeric_cols = get_numeric_cols(ml_df)

    target = st.selectbox(
        "Select Target Variable",
        numeric_cols,
        index=numeric_cols.index("injuries_total")
        if "injuries_total" in numeric_cols
        else 0
    )

    test_size = st.slider(
        "Test Size %",
        10,
        40,
        20
    ) / 100

    run_model = st.button(
        "🚀 Train Models"
    )

    if run_model:

        X = ml_df.drop(columns=[target])
        y = ml_df[target]

        # ----------------------------------------------------
        # LEAKAGE DETECTION
        # ----------------------------------------------------

        suspicious = detect_leakage(X, y)

        if len(suspicious) > 0:

            st.error(
                f"Potential leakage detected: {suspicious}"
            )

        # ----------------------------------------------------
        # TRAIN TEST SPLIT
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42
            )
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        models = {

            "Linear Regression":
                LinearRegression(),

            "Ridge Regression":
                Ridge(alpha=1.0),

            "Decision Tree":
                DecisionTreeRegressor(
                    max_depth=6,
                    random_state=42
                ),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                ),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    n_estimators=200,
                    random_state=42
                )
        }

        if XGB_AVAILABLE:

            models["XGBoost"] = (
                XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42
                )
            )

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        results = []

        trained_models = {}

        progress = st.progress(0)

        for i, (name, model) in enumerate(
            models.items()
        ):

            progress.progress(
                (i + 1) / len(models)
            )

            if name in [
                "Linear Regression",
                "Ridge Regression"
            ]:

                model.fit(
                    X_train_scaled,
                    y_train
                )

                preds = model.predict(
                    X_test_scaled
                )

                cv_score = (
                    cross_val_score(
                        model,
                        scaler.fit_transform(X),
                        y,
                        cv=5,
                        scoring="r2"
                    )
                    .mean()
                )

            else:

                model.fit(
                    X_train,
                    y_train
                )

                preds = model.predict(
                    X_test
                )

                cv_score = (
                    cross_val_score(
                        model,
                        X,
                        y,
                        cv=5,
                        scoring="r2"
                    )
                    .mean()
                )

            mae = mean_absolute_error(
                y_test,
                preds
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    preds
                )
            )

            r2 = r2_score(
                y_test,
                preds
            )

            results.append({

                "Model": name,

                "MAE":
                    round(mae,4),

                "RMSE":
                    round(rmse,4),

                "R2":
                    round(r2,4),

                "CV_R2":
                    round(cv_score,4)
            })

            trained_models[name] = (
                model,
                preds
            )

        progress.empty()

        # ----------------------------------------------------
        # RESULTS TABLE
        # ----------------------------------------------------

        results_df = (
            pd.DataFrame(results)
            .sort_values(
                "R2",
                ascending=False
            )
        )

        st.session_state[
            "results_df"
        ] = results_df

        st.dataframe(
            results_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        best_model_name = (
            results_df.iloc[0]["Model"]
        )

        st.success(
            f"🏆 Best Model: {best_model_name}"
        )

        # ----------------------------------------------------
        # LEADERBOARD CHART
        # ----------------------------------------------------

        fig = px.bar(

            results_df,

            x="Model",

            y="R2",

            color="Model",

            title="Model Performance (R²)"
        )

        fig.update_layout(
            **PLOTLY_LAYOUT
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # ACTUAL VS PREDICTED
        # ----------------------------------------------------

        best_model = (
            trained_models[
                best_model_name
            ][0]
        )

        if best_model_name in [
            "Linear Regression",
            "Ridge Regression"
        ]:

            best_preds = (
                best_model.predict(
                    X_test_scaled
                )
            )

        else:

            best_preds = (
                best_model.predict(
                    X_test
                )
            )

        pred_df = pd.DataFrame({

            "Actual":
                y_test.values,

            "Predicted":
                best_preds
        })

        fig = px.scatter(

            pred_df,

            x="Actual",

            y="Predicted",

            trendline="ols",

            title=
            f"Actual vs Predicted ({best_model_name})"
        )

        fig.update_layout(
            **PLOTLY_LAYOUT
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        if hasattr(
            best_model,
            "feature_importances_"
        ):

            imp = pd.DataFrame({

                "Feature":
                    X.columns,

                "Importance":
                    best_model.feature_importances_
            })

            imp = (
                imp.sort_values(
                    "Importance",
                    ascending=False
                )
                .head(15)
            )

            fig = px.bar(

                imp,

                x="Importance",

                y="Feature",

                orientation="h",

                title=
                "Top 15 Important Features"
            )

            fig.update_layout(
                **PLOTLY_LAYOUT
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # HYPERPARAMETER TUNING
        # ----------------------------------------------------

        st.subheader(
            "⚡ Hyperparameter Tuning"
        )

        if st.button(
            "Tune Random Forest"
        ):

            params = {

                "n_estimators":
                    [100,200,300],

                "max_depth":
                    [4,6,8,None]
            }

            grid = GridSearchCV(

                RandomForestRegressor(),

                params,

                cv=3,

                scoring="r2",

                n_jobs=-1
            )

            grid.fit(
                X_train,
                y_train
            )

            st.success(
                f"Best Score: {grid.best_score_:.4f}"
            )

            st.write(
                grid.best_params_
            )

        # ----------------------------------------------------
        # EXPORT MODEL
        # ----------------------------------------------------

        st.subheader(
            "💾 Export Best Model"
        )

        model_file = (
            f"{best_model_name}.pkl"
        )

        joblib.dump(
            best_model,
            model_file
        )

        with open(
            model_file,
            "rb"
        ) as f:

            st.download_button(

                label=
                "Download Model",

                data=f,

                file_name=model_file,

                mime=
                "application/octet-stream"
            )
```
```python
# ============================================================
# ADVANCED MODEL EXPLAINABILITY
# PART 2B
# ============================================================

from sklearn.inspection import permutation_importance

st.markdown(
    '<div class="section-header">Advanced Model Explainability</div>',
    unsafe_allow_html=True
)

if "results_df" in st.session_state:

    st.subheader("🏆 Model Ranking")

    ranking_df = (
        st.session_state["results_df"]
        .copy()
        .reset_index(drop=True)
    )

    ranking_df.index = ranking_df.index + 1

    st.dataframe(
        ranking_df,
        use_container_width=True
    )

    best_model_name = (
        ranking_df.iloc[0]["Model"]
    )

    st.success(
        f"Recommended Model: {best_model_name}"
    )

# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

st.subheader(
    "🔍 Permutation Feature Importance"
)

try:

    if "engineered_df" in st.session_state:

        explain_df = (
            st.session_state["engineered_df"]
            .copy()
        )

        explain_df = prepare_data(
            explain_df
        )

        target_col = target

        X_perm = explain_df.drop(
            columns=[target_col]
        )

        y_perm = explain_df[target_col]

        if best_model_name in [
            "Linear Regression",
            "Ridge Regression"
        ]:

            model_used = best_model

            perm = permutation_importance(
                model_used,
                X_test_scaled,
                y_test,
                n_repeats=10,
                random_state=42
            )

            imp_df = pd.DataFrame({

                "Feature":
                    X.columns,

                "Importance":
                    perm.importances_mean
            })

        else:

            model_used = best_model

            perm = permutation_importance(
                model_used,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42
            )

            imp_df = pd.DataFrame({

                "Feature":
                    X.columns,

                "Importance":
                    perm.importances_mean
            })

        imp_df = (
            imp_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)
        )

        fig = px.bar(

            imp_df,

            x="Importance",

            y="Feature",

            orientation="h",

            title="Top 15 Features (Permutation Importance)"
        )

        fig.update_layout(
            **PLOTLY_LAYOUT
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

except:
    st.info(
        "Train a model first to view feature importance."
    )

# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-header">Residual Analysis</div>',
    unsafe_allow_html=True
)

try:

    residuals = (
        y_test -
        best_preds
    )

    col1,col2 = st.columns(2)

    with col1:

        fig = px.histogram(

            residuals,

            nbins=40,

            title=
            "Residual Distribution"
        )

        fig.update_layout(
            **PLOTLY_LAYOUT
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.scatter(

            x=best_preds,

            y=residuals,

            labels={
                "x":"Predicted",
                "y":"Residual"
            },

            title=
            "Residual vs Predicted"
        )

        fig.add_hline(
            y=0,
            line_dash="dash"
        )

        fig.update_layout(
            **PLOTLY_LAYOUT
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

except:
    pass

# ============================================================
# ERROR ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-header">Prediction Error Analysis</div>',
    unsafe_allow_html=True
)

try:

    error_df = pd.DataFrame({

        "Actual":
            y_test,

        "Predicted":
            best_preds
    })

    error_df["Absolute_Error"] = (

        abs(
            error_df["Actual"]
            -
            error_df["Predicted"]
        )
    )

    st.dataframe(
        error_df.head(50),
        use_container_width=True
    )

    fig = px.box(

        error_df,

        y="Absolute_Error",

        title=
        "Prediction Error Spread"
    )

    fig.update_layout(
        **PLOTLY_LAYOUT
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except:
    pass

# ============================================================
# PREDICTION INTERFACE
# ============================================================

st.markdown(
    '<div class="section-header">Make New Prediction</div>',
    unsafe_allow_html=True
)

try:

    user_input = {}

    cols = st.columns(3)

    feature_list = list(
        X.columns
    )

    for i, feature in enumerate(
        feature_list
    ):

        user_input[feature] = (
            cols[i % 3]
            .number_input(
                feature,
                value=0.0
            )
        )

    if st.button(
        "Predict Injuries"
    ):

        input_df = pd.DataFrame(
            [user_input]
        )

        if best_model_name in [
            "Linear Regression",
            "Ridge Regression"
        ]:

            pred = best_model.predict(
                scaler.transform(
                    input_df
                )
            )[0]

        else:

            pred = best_model.predict(
                input_df
            )[0]

        st.success(
            f"Predicted Injuries Total = {pred:.2f}"
        )

except:
    st.warning(
        "Train a model first."
    )

# ============================================================
# DOWNLOAD PREDICTIONS
# ============================================================

st.markdown(
    '<div class="section-header">Download Predictions</div>',
    unsafe_allow_html=True
)

try:

    prediction_export = pd.DataFrame({

        "Actual":
            y_test,

        "Predicted":
            best_preds,

        "Residual":
            y_test - best_preds
    })

    csv = (
        prediction_export
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(

        label=
        "📥 Download Prediction Results",

        data=csv,

        file_name=
        "prediction_results.csv",

        mime="text/csv"
    )

except:
    pass
```
from sklearn.preprocessing import MinMaxScaler

try:
import tensorflow as tf

```
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

TF_AVAILABLE = True
```

except:
TF_AVAILABLE = False

```python
# ============================================================
# DEEP LEARNING TAB
# LSTM REGRESSION
# ============================================================

with tab5:

    st.markdown(
        '<div class="section-header">Deep Learning - LSTM Regression</div>',
        unsafe_allow_html=True
    )

    if not TF_AVAILABLE:

        st.error(
            "TensorFlow not installed. Run: pip install tensorflow"
        )

    else:

        dl_df = (
            st.session_state
            .get("engineered_df", df)
            .copy()
        )

        dl_df = prepare_data(
            dl_df
        )

        numeric_cols = get_numeric_cols(
            dl_df
        )

        target_dl = st.selectbox(

            "Select Target",

            numeric_cols,

            key="dl_target"
        )

        epochs = st.slider(

            "Epochs",

            10,

            100,

            30
        )

        batch_size = st.selectbox(

            "Batch Size",

            [16,32,64],

            index=1
        )

        if st.button(
            "🚀 Train LSTM"
        ):

            with st.spinner(
                "Training Deep Learning Model..."
            ):

                X = dl_df.drop(
                    columns=[target_dl]
                )

                y = dl_df[
                    target_dl
                ]

                X_train, X_test, y_train, y_test = (
                    train_test_split(
                        X,
                        y,
                        test_size=0.20,
                        random_state=42
                    )
                )

                scaler_x = MinMaxScaler()

                scaler_y = MinMaxScaler()

                X_train_scaled = (
                    scaler_x.fit_transform(
                        X_train
                    )
                )

                X_test_scaled = (
                    scaler_x.transform(
                        X_test
                    )
                )

                y_train_scaled = (
                    scaler_y.fit_transform(
                        y_train
                        .values
                        .reshape(-1,1)
                    )
                )

                X_train_lstm = (
                    X_train_scaled
                    .reshape(
                        X_train_scaled.shape[0],
                        1,
                        X_train_scaled.shape[1]
                    )
                )

                X_test_lstm = (
                    X_test_scaled
                    .reshape(
                        X_test_scaled.shape[0],
                        1,
                        X_test_scaled.shape[1]
                    )
                )

                model = Sequential([

                    LSTM(
                        64,
                        return_sequences=True,
                        input_shape=(
                            1,
                            X_train.shape[1]
                        )
                    ),

                    Dropout(
                        0.30
                    ),

                    BatchNormalization(),

                    LSTM(
                        32
                    ),

                    Dropout(
                        0.30
                    ),

                    Dense(
                        16,
                        activation="relu"
                    ),

                    Dense(
                        1
                    )
                ])

                model.compile(

                    optimizer="adam",

                    loss="mse",

                    metrics=["mae"]
                )

                early_stop = EarlyStopping(

                    monitor="val_loss",

                    patience=8,

                    restore_best_weights=True
                )

                history = model.fit(

                    X_train_lstm,

                    y_train_scaled,

                    epochs=epochs,

                    batch_size=batch_size,

                    validation_split=0.20,

                    callbacks=[
                        early_stop
                    ],

                    verbose=0
                )

                pred_scaled = (
                    model.predict(
                        X_test_lstm
                    )
                )

                predictions = (
                    scaler_y
                    .inverse_transform(
                        pred_scaled
                    )
                    .flatten()
                )

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )

                r2 = r2_score(
                    y_test,
                    predictions
                )

                st.session_state[
                    "lstm_model"
                ] = model

                st.session_state[
                    "lstm_predictions"
                ] = predictions

                st.session_state[
                    "lstm_y_test"
                ] = y_test

            st.success(
                "LSTM Training Completed"
            )

            c1,c2,c3 = st.columns(3)

            with c1:
                st.metric(
                    "MAE",
                    f"{mae:.4f}"
                )

            with c2:
                st.metric(
                    "RMSE",
                    f"{rmse:.4f}"
                )

            with c3:
                st.metric(
                    "R²",
                    f"{r2:.4f}"
                )

            # -----------------------------------
            # TRAINING HISTORY
            # -----------------------------------

            st.subheader(
                "Training History"
            )

            history_df = pd.DataFrame({

                "Epoch":
                    range(
                        1,
                        len(
                            history.history[
                                "loss"
                            ]
                        ) + 1
                    ),

                "Training Loss":
                    history.history[
                        "loss"
                    ],

                "Validation Loss":
                    history.history[
                        "val_loss"
                    ]
            })

            fig = px.line(

                history_df,

                x="Epoch",

                y=[
                    "Training Loss",
                    "Validation Loss"
                ],

                title=
                "Training vs Validation Loss"
            )

            fig.update_layout(
                **PLOTLY_LAYOUT
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -----------------------------------
            # ACTUAL VS PREDICTED
            # -----------------------------------

            st.subheader(
                "Actual vs Predicted"
            )

            pred_df = pd.DataFrame({

                "Actual":
                    y_test.values,

                "Predicted":
                    predictions
            })

            fig = px.scatter(

                pred_df,

                x="Actual",

                y="Predicted",

                trendline="ols",

                title=
                "LSTM Actual vs Predicted"
            )

            fig.update_layout(
                **PLOTLY_LAYOUT
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -----------------------------------
            # FIRST 100 OBSERVATIONS
            # -----------------------------------

            st.subheader(
                "Prediction Sequence"
            )

            compare_df = pd.DataFrame({

                "Actual":
                    y_test.values[:100],

                "Predicted":
                    predictions[:100]
            })

            fig = px.line(

                compare_df,

                title=
                "First 100 Predictions"
            )

            fig.update_layout(
                **PLOTLY_LAYOUT
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -----------------------------------
            # EXPORT PREDICTIONS
            # -----------------------------------

            csv = (
                pred_df
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )

            st.download_button(

                label=
                "📥 Download LSTM Predictions",

                data=csv,

                file_name=
                "lstm_predictions.csv",

                mime="text/csv"
            )
```
```python
# ============================================================
# FORECASTING TAB
# SARIMA TIME SERIES FORECASTING
# ============================================================

with tab6:

    st.markdown(
        '<div class="section-header">Time Series Forecasting (SARIMA)</div>',
        unsafe_allow_html=True
    )

    if not SARIMA_AVAILABLE:

        st.error(
            "statsmodels not installed. Run: pip install statsmodels"
        )

    elif "crash_date" not in df.columns:

        st.warning(
            "Dataset must contain a crash_date column."
        )

    else:

        if not pd.api.types.is_datetime64_any_dtype(
            df["crash_date"]
        ):
            st.warning(
                "crash_date must be datetime format."
            )

        else:

            numeric_cols = get_numeric_cols(df)

            target_ts = st.selectbox(
                "Forecast Variable",
                numeric_cols,
                index=numeric_cols.index("injuries_total")
                if "injuries_total" in numeric_cols
                else 0,
                key="forecast_target"
            )

            forecast_periods = st.slider(
                "Future Forecast Periods",
                3,
                24,
                12
            )

            frequency = st.selectbox(
                "Aggregation Frequency",
                [
                    "D",
                    "W",
                    "M"
                ]
            )

            if st.button(
                "🚀 Run Forecast"
            ):

                with st.spinner(
                    "Training SARIMA Model..."
                ):

                    ts_df = (
                        df[
                            [
                                "crash_date",
                                target_ts
                            ]
                        ]
                        .copy()
                    )

                    ts_df = (
                        ts_df
                        .set_index(
                            "crash_date"
                        )
                    )

                    ts_series = (
                        ts_df[target_ts]
                        .resample(
                            frequency
                        )
                        .sum()
                        .dropna()
                    )

                    if len(ts_series) < 12:

                        st.error(
                            "Need at least 12 time periods for forecasting."
                        )

                    else:

                        split_idx = int(
                            len(ts_series)
                            * 0.8
                        )

                        train = (
                            ts_series.iloc[
                                :split_idx
                            ]
                        )

                        test = (
                            ts_series.iloc[
                                split_idx:
                            ]
                        )

                        try:

                            model = SARIMAX(

                                train,

                                order=(1,1,1),

                                seasonal_order=(
                                    1,
                                    1,
                                    1,
                                    12
                                ),

                                enforce_stationarity=False,

                                enforce_invertibility=False
                            )

                            fitted_model = (
                                model.fit(
                                    disp=False
                                )
                            )

                            test_forecast = (
                                fitted_model.forecast(
                                    steps=len(test)
                                )
                            )

                            mae = (
                                mean_absolute_error(
                                    test,
                                    test_forecast
                                )
                            )

                            rmse = np.sqrt(
                                mean_squared_error(
                                    test,
                                    test_forecast
                                )
                            )

                            r2 = (
                                r2_score(
                                    test,
                                    test_forecast
                                )
                            )

                            future_forecast = (
                                fitted_model.forecast(
                                    steps=forecast_periods
                                )
                            )

                            st.success(
                                "Forecast Completed"
                            )

                            c1,c2,c3 = st.columns(3)

                            with c1:
                                st.metric(
                                    "MAE",
                                    f"{mae:.2f}"
                                )

                            with c2:
                                st.metric(
                                    "RMSE",
                                    f"{rmse:.2f}"
                                )

                            with c3:
                                st.metric(
                                    "R²",
                                    f"{r2:.2f}"
                                )

                            # ---------------------------------
                            # HISTORICAL TREND
                            # ---------------------------------

                            st.subheader(
                                "Historical Trend"
                            )

                            fig = px.line(

                                ts_series,

                                title=
                                f"{target_ts} Trend Over Time"
                            )

                            fig.update_layout(
                                **PLOTLY_LAYOUT
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            # ---------------------------------
                            # ACTUAL VS TEST FORECAST
                            # ---------------------------------

                            st.subheader(
                                "Model Validation"
                            )

                            validation_df = pd.DataFrame({

                                "Actual":
                                    test,

                                "Forecast":
                                    test_forecast
                            })

                            fig = px.line(

                                validation_df,

                                title=
                                "Actual vs Forecast"
                            )

                            fig.update_layout(
                                **PLOTLY_LAYOUT
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            # ---------------------------------
                            # FUTURE FORECAST
                            # ---------------------------------

                            st.subheader(
                                "Future Forecast"
                            )

                            future_dates = pd.date_range(

                                start=
                                ts_series.index[-1],

                                periods=
                                forecast_periods + 1,

                                freq=
                                frequency
                            )[1:]

                            future_df = pd.DataFrame({

                                "Date":
                                    future_dates,

                                "Forecast":
                                    future_forecast
                            })

                            fig = px.line(

                                future_df,

                                x="Date",

                                y="Forecast",

                                markers=True,

                                title=
                                "Future Accident Forecast"
                            )

                            fig.update_layout(
                                **PLOTLY_LAYOUT
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                            st.dataframe(
                                future_df,
                                use_container_width=True
                            )

                            # ---------------------------------
                            # FORECAST DOWNLOAD
                            # ---------------------------------

                            csv = (
                                future_df
                                .to_csv(
                                    index=False
                                )
                                .encode(
                                    "utf-8"
                                )
                            )

                            st.download_button(

                                label=
                                "📥 Download Forecast",

                                data=csv,

                                file_name=
                                "future_forecast.csv",

                                mime=
                                "text/csv"
                            )

                            st.session_state[
                                "forecast_df"
                            ] = future_df

                        except Exception as e:

                            st.error(
                                f"SARIMA Error: {str(e)}"
                            )
```

