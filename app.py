# Traffic_Accident_Intelligence_Full_App.py
# Consolidated Streamlit Application (Starter Full Version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Traffic Accident Intelligence", layout="wide")

st.title("🚦 Traffic Accident Intelligence Dashboard")

def get_numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    tabs = st.tabs([
        "Dashboard",
        "EDA",
        "Feature Engineering",
        "Machine Learning",
        "Deep Learning",
        "Forecasting",
        "Executive Summary"
    ])

    with tabs[0]:
        st.subheader("Dataset Overview")
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing", int(df.isna().sum().sum()))
        st.dataframe(df.head())

    with tabs[1]:
        nums = get_numeric_cols(df)
        if nums:
            col = st.selectbox("Numeric Column", nums)
            st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

    with tabs[2]:
        st.write("Add crash hour/date engineered features here.")

    with tabs[3]:
        nums = get_numeric_cols(df)
        if len(nums) >= 2:
            target = st.selectbox("Target", nums)
            if st.button("Train Random Forest"):
                data = df.copy()
                for c in data.columns:
                    if data[c].dtype == "object":
                        data[c] = LabelEncoder().fit_transform(data[c].astype(str))
                data = data.fillna(0)

                X = data.drop(columns=[target])
                y = data[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = RandomForestRegressor(random_state=42)
                model.fit(X_train, y_train)

                preds = model.predict(X_test)

                st.metric("R²", round(r2_score(y_test, preds), 4))
                st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, preds)), 4))
                st.metric("MAE", round(mean_absolute_error(y_test, preds), 4))

    with tabs[4]:
        st.info("Insert LSTM module from Part 3A here")

    with tabs[5]:
        st.info("Insert SARIMA forecasting module from Part 3B here")

    with tabs[6]:
        st.success("Executive Summary and Best Model Recommendation")

else:
    st.info("Upload a dataset to begin.")
