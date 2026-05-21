import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Time Series
from prophet import Prophet

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Traffic Accident Prediction System",
    layout="wide",
    page_icon="🚦"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🚦 Traffic Accident Prediction System")

st.markdown("""
This application provides:

✅ Exploratory Data Analysis  
✅ Machine Learning Models  
✅ Deep Learning Models  
✅ Time-Series Forecasting  
✅ Interactive Visualizations  

Please upload your dataset to begin.
""")

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Your CSV Dataset",
    type=["csv"]
)

# ---------------------------------------------------
# IF FILE NOT UPLOADED
# ---------------------------------------------------
if uploaded_file is None:

    st.info("👆 Please upload a CSV dataset to continue.")

    st.markdown("""
    ### Dataset Requirements
    Your dataset should contain:
    
    - Numerical columns
    - Categorical columns
    - Target column for prediction
    - Optional date column for time-series forecasting
    """)

# ---------------------------------------------------
# IF FILE IS UPLOADED
# ---------------------------------------------------
else:

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully")

    # ---------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------
    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head())

    # ---------------------------------------------------
    # BASIC INFORMATION
    # ---------------------------------------------------
    st.subheader("📌 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    # ---------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------
    st.subheader("🧹 Missing Values")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        st.dataframe(missing)
    else:
        st.success("No Missing Values Found")

    # ---------------------------------------------------
    # COLUMN TYPES
    # ---------------------------------------------------
    st.subheader("📂 Column Types")

    st.dataframe(df.dtypes.astype(str))

    # ---------------------------------------------------
    # NUMERICAL ANALYSIS
    # ---------------------------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) > 0:

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

        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(
            df,
            y=selected_num_col,
            title=f"Boxplot of {selected_num_col}"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------
    # CATEGORICAL ANALYSIS
    # ---------------------------------------------------
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if len(categorical_cols) > 0:

        st.subheader("📊 Categorical Analysis")

        selected_cat_col = st.selectbox(
            "Select Categorical Column",
            categorical_cols
        )

        value_counts = df[selected_cat_col].value_counts().reset_index()

        value_counts.columns = [selected_cat_col, "Count"]

        value_counts = value_counts.head(20)

        fig3 = px.bar(
            value_counts,
            x="Count",
            y=selected_cat_col,
            orientation="h",
            title=f"{selected_cat_col} Distribution"
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ---------------------------------------------------
    # CORRELATION HEATMAP
    # ---------------------------------------------------
    if len(numeric_cols) > 1:

        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig4 = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(fig4, use_container_width=True)

    # ---------------------------------------------------
    # MACHINE LEARNING SECTION
    # ---------------------------------------------------
    st.subheader("🤖 Machine Learning Models")

    target_column = st.selectbox(
        "🎯 Select Target Column",
        df.columns
    )

    if st.button("Train Machine Learning Models"):

        data = df.copy()

        # Handle Missing Values
        for col in data.columns:

            if data[col].dtype == "object":
                data[col].fillna(data[col].mode()[0], inplace=True)

            else:
                data[col].fillna(data[col].mean(), inplace=True)

        # Encode Categorical Variables
        label_encoders = {}

        for col in data.columns:

            if data[col].dtype == "object":

                le = LabelEncoder()

                data[col] = le.fit_transform(
                    data[col].astype(str)
                )

                label_encoders[col] = le

        # Split Data
        X = data.drop(columns=[target_column])
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Models
        models = {

            "Logistic Regression": LogisticRegression(max_iter=1000),

            "Decision Tree": DecisionTreeClassifier(),

            "Random Forest": RandomForestClassifier()
        }

        results = []

        best_model = None
        best_model_name = ""
        best_accuracy = 0

        # Training Loop
        for name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            results.append({

                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

            if accuracy > best_accuracy:

                best_accuracy = accuracy
                best_model = model
                best_model_name = name

        # Results Table
        results_df = pd.DataFrame(results)

        st.subheader("📋 ML Model Results")

        st.dataframe(results_df)

        # Accuracy Chart
        fig5 = px.bar(
            results_df,
            x="Model",
            y="Accuracy",
            color="Model",
            title="Model Accuracy Comparison"
        )

        st.plotly_chart(fig5, use_container_width=True)

        # Best Model
        st.subheader("🏆 Best Model")

        st.success(
            f"Best Model: {best_model_name} | Accuracy: {best_accuracy:.4f}"
        )

        # Confusion Matrix
        predictions = best_model.predict(X_test)

        cm = confusion_matrix(y_test, predictions)

        fig6 = ff.create_annotated_heatmap(
            z=cm,
            x=[str(i) for i in np.unique(y)],
            y=[str(i) for i in np.unique(y)],
            colorscale="Viridis"
        )

        st.subheader("📉 Confusion Matrix")

        st.plotly_chart(fig6, use_container_width=True)

    # ---------------------------------------------------
    # DEEP LEARNING SECTION
    # ---------------------------------------------------
    st.subheader("🧠 Deep Learning Model")

    if st.button("Train Deep Learning Model"):

        data = df.copy()

        # Encode Data
        for col in data.columns:

            if data[col].dtype == "object":

                le = LabelEncoder()

                data[col] = le.fit_transform(
                    data[col].astype(str)
                )

        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Standardization
        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ANN Model
        model = Sequential()

        model.add(Dense(128, activation="relu", input_shape=(X.shape[1],)))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Train Model
        history = model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        # Predictions
        predictions = model.predict(X_test)

        predictions = (predictions > 0.5).astype(int)

        accuracy = accuracy_score(y_test, predictions)

        st.success(
            f"✅ Deep Learning Model Accuracy: {accuracy:.4f}"
        )

        # Training Graph
        fig7 = plt.figure(figsize=(10, 5))

        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

        plt.legend()

        plt.title("Deep Learning Accuracy")

        st.pyplot(fig7)

    # ---------------------------------------------------
    # TIME SERIES FORECASTING
    # ---------------------------------------------------
    st.subheader("📈 Time-Series Forecasting")

    st.info(
        "Select a Date Column and Numerical Target for Forecasting"
    )

    date_columns = df.columns.tolist()

    selected_date_col = st.selectbox(
        "📅 Select Date Column",
        date_columns
    )

    selected_target_col = st.selectbox(
        "📊 Select Forecast Column",
        numeric_cols
    )

    if st.button("Run Time-Series Forecast"):

        try:

            ts_df = df[[selected_date_col, selected_target_col]]

            ts_df.columns = ["ds", "y"]

            ts_df["ds"] = pd.to_datetime(ts_df["ds"])

            model = Prophet()

            model.fit(ts_df)

            future = model.make_future_dataframe(periods=30)

            forecast = model.predict(future)

            st.subheader("📋 Forecast Results")

            st.dataframe(
                forecast[["ds", "yhat"]].tail(30)
            )

            # Forecast Plot
            fig8 = model.plot(forecast)

            st.pyplot(fig8)

        except Exception as e:

            st.error(
                f"Time-Series Forecasting Error: {e}"
            )
