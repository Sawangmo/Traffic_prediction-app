import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Gradient Boosting Regressor")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

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

            st.subheader("Evaluation Metrics")
            st.metric("MAE", round(mean_absolute_error(y_test, y_pred), 4))
            st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
            st.metric("R² Score", round(r2_score(y_test, y_pred), 4))

            st.subheader("Feature Importances")
            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            st.bar_chart(fi)
