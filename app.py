# app.py

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from xgboost import XGBRegressor

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Time Series
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Traffic Accident Prediction",
    layout="wide"
)

st.title("🚦 Traffic Accident Prediction System")
st.markdown("### Machine Learning • Deep Learning • Time Series Forecasting")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Traffic Accident Dataset (.csv)",
    type=["csv"]
)

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
if uploaded_file:

    # Load data
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully")

    # --------------------------------------------------
    # PREVIEW
    # --------------------------------------------------
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # BASIC INFO
    # --------------------------------------------------
    st.subheader("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    # --------------------------------------------------
    # HANDLE DATE COLUMN
    # --------------------------------------------------
    if 'crash_date' in df.columns:
        try:
            df['crash_date'] = pd.to_datetime(df['crash_date'])
        except:
            pass

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------
    st.subheader("⚙️ Feature Engineering")

    if 'crash_hour' in df.columns:

        df['is_weekend'] = df['crash_day_of_week'].apply(
            lambda x: 1 if x >= 5 else 0
        )

        df['is_night'] = df['crash_hour'].apply(
            lambda x: 1 if x >= 20 or x <= 5 else 0
        )

        df['is_rush_hour'] = df['crash_hour'].apply(
            lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0
        )

        st.success("✅ New features created")

    # --------------------------------------------------
    # NUMERICAL ANALYSIS
    # --------------------------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:

        st.subheader("📈 Numerical Analysis")

        selected_num_col = st.selectbox(
            "Select Numerical Column",
            numeric_cols
        )

        fig1 = px.histogram(
            df,
            x=selected_num_col,
            nbins=30,
            title=f"Distribution of {selected_num_col}"
        )

        st.plotly_chart(fig1, width="stretch")

        fig2 = px.box(
            df,
            y=selected_num_col,
            title=f"Boxplot of {selected_num_col}"
        )

        st.plotly_chart(fig2, width="stretch")

    # --------------------------------------------------
    # CATEGORICAL ANALYSIS
    # --------------------------------------------------
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if categorical_cols:

        st.subheader("📊 Categorical Analysis")

        selected_cat_col = st.selectbox(
            "Select Categorical Column",
            categorical_cols
        )

        value_counts = df[selected_cat_col].value_counts().reset_index()
        value_counts.columns = [selected_cat_col, 'Count']

        fig3 = px.bar(
            value_counts.head(20),
            x='Count',
            y=selected_cat_col,
            orientation='h',
            title=f'{selected_cat_col} Distribution'
        )

        st.plotly_chart(fig3, width="stretch")

    # --------------------------------------------------
    # CORRELATION
    # --------------------------------------------------
    if len(numeric_cols) > 1:

        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig4 = px.imshow(
            corr,
            text_auto=True,
            aspect='auto',
            title='Correlation Matrix'
        )

        st.plotly_chart(fig4, width="stretch")

    # --------------------------------------------------
    # SIDEBAR MODELS
    # --------------------------------------------------
    st.sidebar.title("🤖 Select Model Type")

    model_type = st.sidebar.selectbox(
        "Choose Model",
        [
            'Machine Learning - Regression',
            'Machine Learning - Classification',
            'Deep Learning - LSTM Regression',
            'Deep Learning - LSTM Classification',
            'Time Series Forecasting'
        ]
    )

    # --------------------------------------------------
    # ENCODING
    # --------------------------------------------------
    data = df.copy()

    for col in data.columns:

        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())

    label_encoders = {}

    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # --------------------------------------------------
    # MACHINE LEARNING REGRESSION
    # --------------------------------------------------
    if model_type == 'Machine Learning - Regression':

        st.subheader("📈 Regression Models")

        target = st.selectbox(
            "Select Regression Target",
            numeric_cols
        )

        if st.button("Train Regression Models"):

            X = data.drop(columns=[target])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=42
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor()
            }

            results = []

            for name, model in models.items():

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)

                results.append({
                    'Model': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2 Score': r2
                })

            results_df = pd.DataFrame(results)

            st.dataframe(results_df)

            fig = px.bar(
                results_df,
                x='Model',
                y='R2 Score',
                color='Model',
                title='Regression Model Comparison'
            )

            st.plotly_chart(fig, width="stretch")

    # --------------------------------------------------
    # MACHINE LEARNING CLASSIFICATION
    # --------------------------------------------------
    elif model_type == 'Machine Learning - Classification':

        st.subheader("🚦 Classification Models")

        target = st.selectbox(
            "Select Classification Target",
            data.columns
        )

        if st.button("Train Classification Models"):

            X = data.drop(columns=[target])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=42
            )

            models = {
                'Logistic Regression': LogisticRegression(
                    max_iter=5000,
                    solver='liblinear'
                ),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier()
            }

            results = []
            best_model = None
            best_accuracy = 0

            for name, model in models.items():

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                accuracy = accuracy_score(y_test, preds)

                results.append({
                    'Model': name,
                    'Accuracy': accuracy
                })

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

            results_df = pd.DataFrame(results)

            st.dataframe(results_df)

            fig = px.bar(
                results_df,
                x='Model',
                y='Accuracy',
                color='Model',
                title='Classification Model Comparison'
            )

            st.plotly_chart(fig, width="stretch")

            # Confusion Matrix
            preds = best_model.predict(X_test)
            cm = confusion_matrix(y_test, preds)

            labels = [str(i) for i in range(cm.shape[0])]

            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Viridis'
            )

            st.plotly_chart(fig_cm, width="stretch")

    # --------------------------------------------------
    # LSTM REGRESSION
    # --------------------------------------------------
    elif model_type == 'Deep Learning - LSTM Regression':

        st.subheader("🧠 LSTM Regression")

        target = st.selectbox(
            "Select Regression Target",
            numeric_cols
        )

        if st.button("Train LSTM Regressor"):

            X = data.drop(columns=[target])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=42
            )

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_train_sc = scaler_X.fit_transform(X_train)
            X_test_sc = scaler_X.transform(X_test)

            y_train_sc = scaler_y.fit_transform(
                y_train.values.reshape(-1, 1)
            )

            y_test_sc = scaler_y.transform(
                y_test.values.reshape(-1, 1)
            )

            X_train_lstm = X_train_sc.reshape(
                X_train_sc.shape[0],
                1,
                X_train_sc.shape[1]
            )

            X_test_lstm = X_test_sc.reshape(
                X_test_sc.shape[0],
                1,
                X_test_sc.shape[1]
            )

            model = Sequential([
                LSTM(128, return_sequences=True,
                     input_shape=(1, X_train_sc.shape[1])),
                Dropout(0.3),
                BatchNormalization(),

                LSTM(64),
                Dropout(0.3),

                Dense(32, activation='relu'),
                Dense(1)
            ])

            model.compile(
                optimizer='adam',
                loss='mse'
            )

            early_stop = EarlyStopping(
                patience=10,
                restore_best_weights=True
            )

            history = model.fit(
                X_train_lstm,
                y_train_sc,
                epochs=30,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )

            preds = model.predict(X_test_lstm)

            preds_actual = scaler_y.inverse_transform(preds)

            mae = mean_absolute_error(y_test, preds_actual)
            rmse = np.sqrt(mean_squared_error(y_test, preds_actual))
            r2 = r2_score(y_test, preds_actual)

            st.success("✅ LSTM Training Completed")

            st.write(f"### MAE: {mae:.4f}")
            st.write(f"### RMSE: {rmse:.4f}")
            st.write(f"### R² Score: {r2:.4f}")

    # --------------------------------------------------
    # LSTM CLASSIFICATION
    # --------------------------------------------------
    elif model_type == 'Deep Learning - LSTM Classification':

        st.subheader("🧠 LSTM Classification")

        target = st.selectbox(
            "Select Classification Target",
            data.columns
        )

        if st.button("Train LSTM Classifier"):

            X = data.drop(columns=[target])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=42
            )

            scaler_X = MinMaxScaler()

            X_train_sc = scaler_X.fit_transform(X_train)
            X_test_sc = scaler_X.transform(X_test)

            X_train_lstm = X_train_sc.reshape(
                X_train_sc.shape[0],
                1,
                X_train_sc.shape[1]
            )

            X_test_lstm = X_test_sc.reshape(
                X_test_sc.shape[0],
                1,
                X_test_sc.shape[1]
            )

            model = Sequential([
                LSTM(128, return_sequences=True,
                     input_shape=(1, X_train_sc.shape[1])),
                Dropout(0.3),
                BatchNormalization(),

                LSTM(64),
                Dropout(0.3),

                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            early_stop = EarlyStopping(
                patience=10,
                restore_best_weights=True
            )

            history = model.fit(
                X_train_lstm,
                y_train,
                epochs=30,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stop],
                verbose=0
            )

            preds = model.predict(X_test_lstm)
            preds = (preds > 0.5).astype(int)

            accuracy = accuracy_score(y_test, preds)

            st.success("✅ LSTM Classification Completed")

            st.write(f"### Accuracy: {accuracy:.4f}")

            cm = confusion_matrix(y_test, preds)

            labels = [str(i) for i in range(cm.shape[0])]

            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Viridis'
            )

            st.plotly_chart(fig_cm, width="stretch")

    # --------------------------------------------------
    # TIME SERIES
    # --------------------------------------------------
    elif model_type == 'Time Series Forecasting':

        st.subheader("📅 Time Series Forecasting")

        if 'crash_date' not in df.columns:
            st.error("Dataset must contain 'crash_date' column")

        else:

            numeric_target = st.selectbox(
                'Select Time Series Target',
                numeric_cols
            )

            if st.button("Run Forecasting"):

                ts_data = df.copy()

                ts_data['crash_date'] = pd.to_datetime(
                    ts_data['crash_date']
                )

                ts_data = ts_data.set_index('crash_date')[numeric_target]\
                    .resample('M').sum()

                split = int(len(ts_data) * 0.7)

                train_ts = ts_data[:split]
                test_ts = ts_data[split:]

                model = SARIMAX(
                    train_ts,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12)
                )

                fitted_model = model.fit(disp=False)

                forecast = fitted_model.forecast(
                    steps=len(test_ts)
                )

                mae = mean_absolute_error(test_ts, forecast)
                rmse = np.sqrt(mean_squared_error(test_ts, forecast))
                r2 = r2_score(test_ts, forecast)

                st.success("✅ Forecasting Completed")

                st.write(f"### MAE: {mae:.4f}")
                st.write(f"### RMSE: {rmse:.4f}")
                st.write(f"### R² Score: {r2:.4f}")

                forecast_df = pd.DataFrame({
                    'Date': test_ts.index,
                    'Actual': test_ts.values,
                    'Forecast': forecast.values
                })

                fig = px.line(
                    forecast_df,
                    x='Date',
                    y=['Actual', 'Forecast'],
                    title='SARIMA Forecast'
                )

                st.plotly_chart(fig, width="stretch")

else:

    st.info("📂 Upload a CSV dataset to begin")
