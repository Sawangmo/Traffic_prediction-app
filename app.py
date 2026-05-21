import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Traffic Accident Prediction", layout="wide")

st.title("📊 Interactive ML Dashboard")
st.markdown("Upload dataset → Explore data → Train ML models")

# ---------------------------------------------------
# UPLOAD DATA
# ---------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully ✅")

    # ---------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ---------------------------------------------------
    # BASIC INFO
    # ---------------------------------------------------
    st.subheader("📌 Dataset Info")

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
        st.success("No missing values found")

    # ---------------------------------------------------
    # COLUMN TYPES
    # ---------------------------------------------------
    st.subheader("📂 Column Types")
    st.dataframe(df.dtypes.astype(str))

    # ---------------------------------------------------
    # NUMERICAL EDA
    # ---------------------------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if numeric_cols:

        st.subheader("📈 Numerical Analysis")

        num_col = st.selectbox("Select Numerical Column", numeric_cols)

        fig1 = px.histogram(df, x=num_col, nbins=30, title=f"Distribution of {num_col}")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.box(df, y=num_col, title=f"Boxplot of {num_col}")
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------
    # CATEGORICAL EDA (FIXED)
    # ---------------------------------------------------
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if cat_cols:

        st.subheader("📊 Categorical Analysis")

        cat_col = st.selectbox("Select Categorical Column", cat_cols)

        value_counts = df[cat_col].value_counts().reset_index()
        value_counts.columns = [cat_col, "Count"]

        value_counts = value_counts.head(20)  # prevent overload

        fig3 = px.bar(
            value_counts,
            x="Count",
            y=cat_col,
            orientation="h",
            title=f"{cat_col} Distribution"
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ---------------------------------------------------
    # CORRELATION HEATMAP
    # ---------------------------------------------------
    if len(numeric_cols) > 1:

        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig4 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig4, use_container_width=True)

    # ---------------------------------------------------
    # MACHINE LEARNING
    # ---------------------------------------------------
    st.subheader("🤖 Machine Learning")

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Models"):

        data = df.copy()

        # ---------------------------
        # Handle missing values
        # ---------------------------
        for col in data.columns:
            if data[col].dtype == "object":
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].mean(), inplace=True)

        # ---------------------------
        # Encode categorical
        # ---------------------------
        encoders = {}

        for col in data.columns:
            if data[col].dtype == "object":
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                encoders[col] = le

        # ---------------------------
        # Split data
        # ---------------------------
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------------------
        # Models
        # ---------------------------
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        results = []

        best_model_name = None
        best_model = None
        best_acc = 0

        # ---------------------------
        # Train + Evaluate
        # ---------------------------
        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted", zero_division=0)
            rec = recall_score(y_test, preds, average="weighted", zero_division=0)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })

            if acc > best_acc:
                best_acc = acc
                best_model_name = name
                best_model = model

        results_df = pd.DataFrame(results)

        st.subheader("📋 Model Results")
        st.dataframe(results_df)

        # ---------------------------
        # Accuracy comparison chart
        # ---------------------------
        fig5 = px.bar(
            results_df,
            x="Model",
            y="Accuracy",
            color="Model",
            title="Model Comparison"
        )

        st.plotly_chart(fig5, use_container_width=True)

        # ---------------------------
        # Best model
        # ---------------------------
        st.subheader("🏆 Best Model")

        st.success(f"Best Model: {best_model_name} \nAccuracy: {best_acc:.4f}")

        # ---------------------------
        # Confusion Matrix
        # ---------------------------
        preds = best_model.predict(X_test)
        cm = confusion_matrix(y_test, preds)

        fig6 = ff.create_annotated_heatmap(
            z=cm,
            x=[str(i) for i in np.unique(y)],
            y=[str(i) for i in np.unique(y)],
            colorscale="Viridis"
        )

        st.plotly_chart(fig6, use_container_width=True)

else:
    st.info("Upload a CSV file to start 🚀")
