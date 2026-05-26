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
# HELPER: get safe numeric cols (exclude datetime)
# --------------------------------------------------
def get_numeric_cols(dataframe):
    """Return numeric columns excluding datetime columns."""
    return [
        col for col in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[col])
        and not pd.api.types.is_datetime64_any_dtype(dataframe[col])
    ]

def get_categorical_cols(dataframe):
    """Return object/categorical columns excluding datetime."""
    return [
        col for col in dataframe.columns
        if dataframe[col].dtype == 'object'
    ]

def drop_datetime_cols(dataframe):
    """Drop all datetime columns from a dataframe copy."""
    dt_cols = [
        col for col in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[col])
    ]
    return dataframe.drop(columns=dt_cols)

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Uploaded Successfully")

    # --------------------------------------------------
    # HANDLE DATE COLUMN (parse but keep separate)
    # --------------------------------------------------
    if 'crash_date' in df.columns:
        try:
            df['crash_date'] = pd.to_datetime(df['crash_date'])
        except Exception:
            pass

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
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------
    st.subheader("⚙️ Feature Engineering")

    if 'crash_hour' in df.columns:
        if 'crash_day_of_week' in df.columns:
            df['is_weekend'] = df['crash_day_of_week'].apply(
                lambda x: 1 if x >= 5 else 0
            )
        df['is_night'] = df['crash_hour'].apply(
            lambda x: 1 if x >= 20 or x <= 5 else 0
        )
        df['is_rush_hour'] = df['crash_hour'].apply(
            lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0
        )
        st.success("✅ New features created: is_weekend, is_night, is_rush_hour")
    else:
        st.info("ℹ️ No 'crash_hour' column found — skipping feature engineering.")

    # --------------------------------------------------
    # SAFE COLUMN LISTS (no datetime)
    # --------------------------------------------------
    numeric_cols = get_numeric_cols(df)
    categorical_cols = get_categorical_cols(df)

    # --------------------------------------------------
    # NUMERICAL ANALYSIS
    # --------------------------------------------------
    if numeric_cols:
        st.subheader("📈 Numerical Analysis")
        selected_num_col = st.selectbox("Select Numerical Column", numeric_cols)

        fig1 = px.histogram(
            df, x=selected_num_col, nbins=30,
            title=f"Distribution of {selected_num_col}"
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(
            df, y=selected_num_col,
            title=f"Boxplot of {selected_num_col}"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------
    # CATEGORICAL ANALYSIS
    # --------------------------------------------------
    if categorical_cols:
        st.subheader("📊 Categorical Analysis")
        selected_cat_col = st.selectbox("Select Categorical Column", categorical_cols)

        value_counts = df[selected_cat_col].value_counts().reset_index()
        value_counts.columns = [selected_cat_col, 'Count']

        fig3 = px.bar(
            value_counts.head(20),
            x='Count', y=selected_cat_col,
            orientation='h',
            title=f'{selected_cat_col} Distribution'
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --------------------------------------------------
    # CORRELATION
    # --------------------------------------------------
    if len(numeric_cols) > 1:
        st.subheader("🔥 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig4 = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation Matrix')
        st.plotly_chart(fig4, use_container_width=True)

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
    # ENCODING (drop datetime cols first)
    # --------------------------------------------------
    data = drop_datetime_cols(df.copy())

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    # Refresh numeric cols on encoded data
    model_numeric_cols = get_numeric_cols(data)

    # --------------------------------------------------
    # MACHINE LEARNING — REGRESSION
    # --------------------------------------------------
    if model_type == 'Machine Learning - Regression':
        st.subheader("📈 Regression Models")

        target = st.selectbox("Select Regression Target", model_numeric_cols)

        if st.button("Train Regression Models"):
            X = data.drop(columns=[target])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Random Forest': RandomForestRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'XGBoost': XGBRegressor(random_state=42, verbosity=0)
            }

            results = []
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                results.append({
                    'Model': name,
                    'MAE': round(mean_absolute_error(y_test, preds), 4),
                    'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 4),
                    'R2 Score': round(r2_score(y_test, preds), 4)
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            fig = px.bar(
                results_df, x='Model', y='R2 Score',
                color='Model', title='Regression Model Comparison — R² Score'
            )
            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # MACHINE LEARNING — CLASSIFICATION
    # --------------------------------------------------
    elif model_type == 'Machine Learning - Classification':
        st.subheader("🚦 Classification Models")

        target = st.selectbox("Select Classification Target", data.columns.tolist())

        if st.button("Train Classification Models"):
            X = data.drop(columns=[target])
            y = data[target]

            # Ensure binary/label encoding for target
            if y.nunique() > 20:
                st.warning("⚠️ Target has many unique values. Consider using Regression instead.")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            models = {
                'Logistic Regression': LogisticRegression(max_iter=5000, solver='liblinear'),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }

            results = []
            best_model = None
            best_accuracy = 0

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                accuracy = accuracy_score(y_test, preds)
                results.append({'Model': name, 'Accuracy': round(accuracy, 4)})
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            fig = px.bar(
                results_df, x='Model', y='Accuracy',
                color='Model', title='Classification Model Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Confusion Matrix for best model
            st.markdown(f"#### Confusion Matrix — Best Model ({results_df.loc[results_df['Accuracy'].idxmax(), 'Model']})")
            preds = best_model.predict(X_test)
            cm = confusion_matrix(y_test, preds)
            labels = [str(i) for i in sorted(y_test.unique())]

            fig_cm = ff.create_annotated_heatmap(
                z=cm, x=labels, y=labels, colorscale='Viridis'
            )
            fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)

    # --------------------------------------------------
    # DEEP LEARNING — LSTM REGRESSION
    # --------------------------------------------------
    elif model_type == 'Deep Learning - LSTM Regression':
        st.subheader("🧠 LSTM Regression")

        target = st.selectbox("Select Regression Target", model_numeric_cols)

        if st.button("Train LSTM Regressor"):
            with st.spinner("Training LSTM model..."):
                X = data.drop(columns=[target]).values
                y = data[target].values

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()

                X_train_sc = scaler_X.fit_transform(X_train)
                X_test_sc = scaler_X.transform(X_test)
                y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1))

                X_train_lstm = X_train_sc.reshape(X_train_sc.shape[0], 1, X_train_sc.shape[1])
                X_test_lstm = X_test_sc.reshape(X_test_sc.shape[0], 1, X_test_sc.shape[1])

                model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(1, X_train_sc.shape[1])),
                    Dropout(0.3),
                    BatchNormalization(),
                    LSTM(64),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])

                model.compile(optimizer='adam', loss='mse')

                early_stop = EarlyStopping(patience=10, restore_best_weights=True)

                model.fit(
                    X_train_lstm, y_train_sc,
                    epochs=30, batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )

                preds_sc = model.predict(X_test_lstm)
                preds_actual = scaler_y.inverse_transform(preds_sc).flatten()

                mae = mean_absolute_error(y_test, preds_actual)
                rmse = np.sqrt(mean_squared_error(y_test, preds_actual))
                r2 = r2_score(y_test, preds_actual)

            st.success("✅ LSTM Regression Training Completed")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"{mae:.4f}")
            m2.metric("RMSE", f"{rmse:.4f}")
            m3.metric("R² Score", f"{r2:.4f}")

            # Actual vs Predicted plot
            pred_df = pd.DataFrame({'Actual': y_test[:100], 'Predicted': preds_actual[:100]})
            fig_pred = px.line(pred_df, title="Actual vs Predicted (first 100 samples)")
            st.plotly_chart(fig_pred, use_container_width=True)

    # --------------------------------------------------
    # DEEP LEARNING — LSTM CLASSIFICATION
    # --------------------------------------------------
    elif model_type == 'Deep Learning - LSTM Classification':
        st.subheader("🧠 LSTM Classification")

        target = st.selectbox("Select Classification Target", data.columns.tolist())

        if st.button("Train LSTM Classifier"):
            with st.spinner("Training LSTM classifier..."):
                X = data.drop(columns=[target]).values
                y = data[target].values

                # Remap labels to 0-based integers
                unique_labels = np.unique(y)
                label_map = {v: i for i, v in enumerate(unique_labels)}
                y_mapped = np.array([label_map[v] for v in y])
                num_classes = len(unique_labels)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_mapped, test_size=0.3, random_state=42
                )

                scaler_X = MinMaxScaler()
                X_train_sc = scaler_X.fit_transform(X_train)
                X_test_sc = scaler_X.transform(X_test)

                X_train_lstm = X_train_sc.reshape(X_train_sc.shape[0], 1, X_train_sc.shape[1])
                X_test_lstm = X_test_sc.reshape(X_test_sc.shape[0], 1, X_test_sc.shape[1])

                if num_classes == 2:
                    output_units = 1
                    loss_fn = 'binary_crossentropy'
                    output_activation = 'sigmoid'
                else:
                    output_units = num_classes
                    loss_fn = 'sparse_categorical_crossentropy'
                    output_activation = 'softmax'

                model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(1, X_train_sc.shape[1])),
                    Dropout(0.3),
                    BatchNormalization(),
                    LSTM(64),
                    Dropout(0.3),
                    Dense(32, activation='relu'),
                    Dense(output_units, activation=output_activation)
                ])

                model.compile(
                    optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy']
                )

                early_stop = EarlyStopping(patience=10, restore_best_weights=True)

                model.fit(
                    X_train_lstm, y_train,
                    epochs=30, batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )

                if num_classes == 2:
                    preds_prob = model.predict(X_test_lstm)
                    preds = (preds_prob > 0.5).astype(int).flatten()
                else:
                    preds_prob = model.predict(X_test_lstm)
                    preds = np.argmax(preds_prob, axis=1)

                accuracy = accuracy_score(y_test, preds)

            st.success("✅ LSTM Classification Completed")
            st.metric("Accuracy", f"{accuracy:.4f}")

            cm = confusion_matrix(y_test, preds)
            labels = [str(i) for i in range(cm.shape[0])]
            fig_cm = ff.create_annotated_heatmap(
                z=cm, x=labels, y=labels, colorscale='Viridis'
            )
            fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)

    # --------------------------------------------------
    # TIME SERIES FORECASTING
    # --------------------------------------------------
    elif model_type == 'Time Series Forecasting':
        st.subheader("📅 Time Series Forecasting (SARIMA)")

        if 'crash_date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['crash_date']):
            st.error("❌ Dataset must contain a parseable 'crash_date' column.")
        else:
            numeric_target = st.selectbox('Select Time Series Target', numeric_cols)

            if st.button("Run Forecasting"):
                with st.spinner("Fitting SARIMA model..."):
                    ts_df = df[['crash_date', numeric_target]].copy()
                    ts_df = ts_df.set_index('crash_date')
                    ts_data = ts_df[numeric_target].resample('ME').sum()

                    if len(ts_data) < 24:
                        st.warning("⚠️ Less than 24 monthly observations found. SARIMA seasonal fitting may be unreliable.")

                    split = int(len(ts_data) * 0.7)
                    train_ts = ts_data.iloc[:split]
                    test_ts = ts_data.iloc[split:]

                    try:
                        sarima_model = SARIMAX(
                            train_ts,
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fitted_model = sarima_model.fit(disp=False)
                        forecast = fitted_model.forecast(steps=len(test_ts))

                        mae = mean_absolute_error(test_ts, forecast)
                        rmse = np.sqrt(mean_squared_error(test_ts, forecast))
                        r2 = r2_score(test_ts, forecast)

                        st.success("✅ Forecasting Completed")

                        m1, m2, m3 = st.columns(3)
                        m1.metric("MAE", f"{mae:.4f}")
                        m2.metric("RMSE", f"{rmse:.4f}")
                        m3.metric("R² Score", f"{r2:.4f}")

                        forecast_df = pd.DataFrame({
                            'Date': test_ts.index,
                            'Actual': test_ts.values,
                            'Forecast': forecast.values
                        })

                        fig = px.line(
                            forecast_df, x='Date',
                            y=['Actual', 'Forecast'],
                            title='SARIMA Forecast vs Actual'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ SARIMA fitting failed: {e}")

else:
    st.info("📂 Upload a CSV dataset to begin.")
