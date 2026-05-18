import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("🚦 Traffic Prediction App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ── EDA SECTION ──────────────────────────────────────────
    st.header("📊 Exploratory Data Analysis")

    # 1. Data Preview
    st.subheader("1. Data Preview")
    st.dataframe(df.head(10))

    # 2. Shape & Types
    st.subheader("2. Dataset Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.write("**Data Types:**")
    st.dataframe(df.dtypes.rename("dtype").reset_index().rename(columns={"index": "Column"}))

    # 3. Statistical Summary
    st.subheader("3. Statistical Summary")
    st.dataframe(df.describe())

    # 4. Missing Values Heatmap
    if df.isnull().sum().sum() > 0:
        st.subheader("4. Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

    # 5. Distribution of Numeric Columns
    st.subheader("5. Feature Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_col = st.selectbox("Select a column to plot", numeric_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[selected_col], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {selected_col}")
    st.pyplot(fig)

    # 6. Correlation Heatmap
    st.subheader("6. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 7. Boxplots for Outliers
    st.subheader("7. Boxplot (Outlier Detection)")
    box_col = st.selectbox("Select a column for boxplot", numeric_cols, key="box")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df[box_col], color="salmon", ax=ax)
    ax.set_title(f"Boxplot of {box_col}")
    st.pyplot(fig)

    # ── MODEL SECTION ─────────────────────────────────────────
    st.header("🤖 Train Gradient Boosting Regressor")

    target = st.selectbox("Select target column", df.columns)
    features = st.multiselect("Select feature columns", [c for c in df.columns if c != target])

    if features:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        n_estimators = st.slider("n_estimators", 50, 500, 100)
        learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)

        if st.button("Train Model"):
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📈 Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", round(mean_absolute_error(y_test, y_pred), 4))
            col2.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
            col3.metric("R² Score", round(r2_score(y_test, y_pred), 4))

            st.subheader("🌟 Feature Importances")
            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.bar_chart(fi)

            st.subheader("🔍 Actual vs Predicted")
            result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            st.line_chart(result_df.reset_index(drop=True))
