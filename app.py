import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Interactive ML Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Interactive Machine Learning Dashboard")
st.markdown("Upload your dataset and perform EDA + Model Evaluation")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully ✅")

    # ---------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------------------------
    # BASIC INFO
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

    st.subheader("📂 Column Data Types")
    st.dataframe(df.dtypes.astype(str))

    # ---------------------------------------------------
    # NUMERICAL EDA
    # ---------------------------------------------------

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) > 0:

        st.subheader("📈 Numerical Feature Distribution")

        selected_num_col = st.selectbox(
            "Select Numerical Column",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=selected_num_col,
            nbins=30,
            title=f"Distribution of {selected_num_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Boxplot
        fig2 = px.box(
            df,
            y=selected_num_col,
            title=f"Boxplot of {selected_num_col}"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------------------------------------
    # CATEGORICAL EDA
    # ---------------------------------------------------

    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if len(categorical_cols) > 0:

        st.subheader("📊 Categorical Feature Analysis")

        selected_cat_col = st.selectbox(
            "Select Categorical Column",
            categorical_cols
        )

        fig3 = px.bar(
            df[selected_cat_col].value_counts().reset_index(),
            x='count',
            y='index',
            orientation='h',
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

    st.subheader("🤖 Machine Learning Model Evaluation")

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    if st.button("Train Models"):

        data = df.copy()

        # ---------------------------------------------------
        # HANDLE MISSING VALUES
        # ---------------------------------------------------

        for col in data.columns:

            if data[col].dtype == 'object':
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].mean())

        # ---------------------------------------------------
        # LABEL ENCODING
        # ---------------------------------------------------

        le = LabelEncoder()

        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = le.fit_transform(data[col].astype(str))

        # ---------------------------------------------------
        # FEATURES & TARGET
        # ---------------------------------------------------

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # ---------------------------------------------------
        # TRAIN TEST SPLIT
        # ---------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ---------------------------------------------------
        # MODELS
        # ---------------------------------------------------

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        results = []

        best_model_name = ""
        best_accuracy = 0

        # ---------------------------------------------------
        # TRAIN & EVALUATE
        # ---------------------------------------------------

        for name, model in models.items():

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test,
                y_pred,
                average='weighted',
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                average='weighted',
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average='weighted',
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
                best_model_name = name

        # ---------------------------------------------------
        # RESULTS TABLE
        # ---------------------------------------------------

        results_df = pd.DataFrame(results)

        st.subheader("📋 Model Performance")
        st.dataframe(results_df)

        # ---------------------------------------------------
        # PERFORMANCE CHART
        # ---------------------------------------------------

        fig5 = px.bar(
            results_df,
            x='Model',
            y='Accuracy',
            color='Model',
            title='Model Accuracy Comparison'
        )

        st.plotly_chart(fig5, use_container_width=True)

        # ---------------------------------------------------
        # BEST MODEL RECOMMENDATION
        # ---------------------------------------------------

        st.subheader("🏆 Best Model Recommendation")

        st.success(
            f"""
            Best Model: {best_model_name}

            Accuracy: {best_accuracy:.4f}
            """
        )

        # ---------------------------------------------------
        # WHY THIS MODEL
        # ---------------------------------------------------

        if best_model_name == "Random Forest":

            st.info("""
            Random Forest performed best because:
            - Handles complex data patterns well
            - Reduces overfitting
            - Works well on large datasets
            - Provides better generalization
            """)

        elif best_model_name == "Decision Tree":

            st.info("""
            Decision Tree performed best because:
            - Easy to interpret
            - Works well with nonlinear data
            - Requires less preprocessing
            """)

        else:

            st.info("""
            Logistic Regression performed best because:
            - Simple and efficient
            - Works well for linearly separable data
            - Faster training time
            """)

        # ---------------------------------------------------
        # CONFUSION MATRIX
        # ---------------------------------------------------

        st.subheader("📉 Confusion Matrix")

        best_model = models[best_model_name]

        y_pred_best = best_model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred_best)

        fig6 = ff.create_annotated_heatmap(
            z=cm,
            x=[str(i) for i in np.unique(y)],
            y=[str(i) for i in np.unique(y)],
            colorscale='Viridis'
        )

        st.plotly_chart(fig6, use_container_width=True)

else:
    st.info("Please upload a CSV dataset to continue.")
