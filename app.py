import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Traffic Accident Predictor", page_icon="🚦", layout="wide")
st.title("🚦 Traffic Accident Prediction & Analysis App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ── EDA SECTION ──────────────────────────────────────────
    st.header("📊 Exploratory Data Analysis")

    st.subheader("1. Data Preview")
    st.dataframe(df.head(10))

    st.subheader("2. Dataset Info")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.write("**Data Types:**")
    st.dataframe(df.dtypes.rename("dtype").reset_index().rename(columns={"index": "Column"}))

    st.subheader("3. Statistical Summary")
    st.dataframe(df.describe())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if df.isnull().sum().sum() > 0:
        st.subheader("4. Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        st.pyplot(fig)

    st.subheader("5. Feature Distributions")
    selected_col = st.selectbox("Select a column to plot", numeric_cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[selected_col], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution of {selected_col}")
    st.pyplot(fig)

    st.subheader("6. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("7. Boxplot (Outlier Detection)")
    box_col = st.selectbox("Select a column for boxplot", numeric_cols, key="box")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df[box_col], color="salmon", ax=ax)
    ax.set_title(f"Boxplot of {box_col}")
    st.pyplot(fig)

    # ── PAST ACCIDENTS SECTION ────────────────────────────────
    st.header("📅 Past Accident Analysis")

    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "year", "month", "time"])]
    accident_count_cols = [c for c in df.columns if any(k in c.lower() for k in ["accident", "crash", "count", "incident", "total"])]

    if date_cols and accident_count_cols:
        date_col = st.selectbox("Select date/time column", date_cols, key="date_col")
        count_col = st.selectbox("Select accident count column", accident_count_cols, key="count_col")

        df_time = df[[date_col, count_col]].dropna()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors="coerce")
        df_time = df_time.dropna().sort_values(date_col)
        df_time_grouped = df_time.groupby(date_col)[count_col].sum().reset_index()

        st.subheader("📉 Accidents Over Time")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_time_grouped[date_col], df_time_grouped[count_col], color="crimson", linewidth=2)
        ax.fill_between(df_time_grouped[date_col], df_time_grouped[count_col], alpha=0.2, color="crimson")
        ax.set_title("Historical Accident Trend", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Accidents")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        total = int(df_time_grouped[count_col].sum())
        peak_date = df_time_grouped.loc[df_time_grouped[count_col].idxmax(), date_col]
        peak_val = int(df_time_grouped[count_col].max())

        col1, col2, col3 = st.columns(3)
        col1.metric("📌 Total Accidents (Historical)", f"{total:,}")
        col2.metric("📍 Peak Date", str(peak_date)[:10])
        col3.metric("🔺 Peak Count", f"{peak_val:,}")

    else:
        st.info("ℹ️ No date or accident count columns detected automatically. Please select manually:")
        date_col = st.selectbox("Select date/year column", df.columns, key="manual_date")
        count_col = st.selectbox("Select accident count column", df.columns, key="manual_count")

        if date_col and count_col:
            df_time = df[[date_col, count_col]].dropna()
            df_time_grouped = df_time.groupby(date_col)[count_col].sum().reset_index()
            st.bar_chart(df_time_grouped.set_index(date_col))

    # ── CAUSE ANALYSIS SECTION ────────────────────────────────
    st.header("🔍 Accident Cause Analysis")

    cause_cols = [c for c in df.columns if any(k in c.lower() for k in
                  ["cause", "reason", "weather", "road", "condition", "type",
                   "light", "speed", "alcohol", "drug", "vehicle", "factor"])]

    if cause_cols:
        cause_col = st.selectbox("Select cause/factor column to analyze", cause_cols)
        cause_counts = df[cause_col].value_counts().head(10)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Top Causes: {cause_col}")
            fig, ax = plt.subplots(figsize=(8, 5))
            cause_counts.plot(kind="bar", ax=ax, color="tomato", edgecolor="black")
            ax.set_title(f"Top 10 Values in '{cause_col}'")
            ax.set_xlabel(cause_col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

        with col2:
            st.subheader("Proportion Breakdown")
            fig, ax = plt.subplots(figsize=(6, 6))
            cause_counts.plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90)
            ax.set_ylabel("")
            ax.set_title(f"Distribution of '{cause_col}'")
            st.pyplot(fig)

        st.subheader("📋 Cause Summary Table")
        cause_df = cause_counts.reset_index()
        cause_df.columns = [cause_col, "Count"]
        cause_df["Percentage"] = (cause_df["Count"] / cause_df["Count"].sum() * 100).round(2)
        st.dataframe(cause_df)

        # Multi-cause analysis
        if len(cause_cols) > 1:
            st.subheader("🔗 Multi-Factor Analysis")
            cause_col2 = st.selectbox("Select a second factor to cross-analyze", 
                                       [c for c in cause_cols if c != cause_col])
            cross_tab = pd.crosstab(df[cause_col], df[cause_col2])
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            ax.set_title(f"{cause_col} vs {cause_col2}")
            st.pyplot(fig)
    else:
        st.info("ℹ️ No cause/factor columns detected. Select manually:")
        manual_cause = st.selectbox("Select a categorical column to analyze as cause", 
                                     df.select_dtypes(include="object").columns.tolist())
        if manual_cause:
            cause_counts = df[manual_cause].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            cause_counts.plot(kind="bar", ax=ax, color="tomato", edgecolor="black")
            ax.set_title(f"Top Values in '{manual_cause}'")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

    # ── MODEL + FUTURE PREDICTION SECTION ────────────────────
    st.header("🤖 Train Model & Predict Future Accidents")

    target = st.selectbox("Select target column (accident count to predict)", df.columns)
    features = st.multiselect("Select feature columns", [c for c in df.columns if c != target])

    if features:
        X = df[features].copy()
        y = df[target].copy()

        # Preprocessing
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna(X[col].median())
        y = y.fillna(y.median())

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        label_encoders = {}
        if cat_cols:
            st.info(f"ℹ️ Auto-encoding categorical columns: {cat_cols}")
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

        if y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        n_estimators = st.slider("n_estimators", 50, 500, 100)
        learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)

        if st.button("Train Model"):
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📈 Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", round(mean_absolute_error(y_test, y_pred), 4))
            col2.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
            col3.metric("R² Score", round(r2_score(y_test, y_pred), 4))

            st.subheader("🌟 Feature Importances")
            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            fi.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
            ax.set_title("Feature Importances")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

            st.subheader("🔍 Actual vs Predicted")
            result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
            st.line_chart(result_df.reset_index(drop=True))

            # ── FUTURE PREDICTION ─────────────────────────────
            st.subheader("🔮 Predict Future Accidents")
            st.write("Adjust the sliders below to simulate future conditions:")

            future_input = {}
            cols = st.columns(min(3, len(features)))
            for i, feat in enumerate(features):
                with cols[i % 3]:
                    if feat in cat_cols:
                        options = label_encoders[feat].classes_.tolist()
                        chosen = st.selectbox(f"{feat}", options, key=f"fut_{feat}")
                        future_input[feat] = label_encoders[feat].transform([chosen])[0]
                    else:
                        min_val = float(X[feat].min())
                        max_val = float(X[feat].max())
                        mean_val = float(X[feat].mean())
                        future_input[feat] = st.slider(
                            f"{feat}", min_val, max_val, mean_val, key=f"fut_{feat}"
                        )

            future_df = pd.DataFrame([future_input])
            future_pred = model.predict(future_df)[0]

            st.markdown("---")
            st.subheader("🚨 Predicted Future Accident Count")
            st.metric(label="Predicted Number of Accidents", value=f"{max(0, round(future_pred, 2)):,}")

            if future_pred > y.mean() * 1.2:
                st.error("⚠️ HIGH RISK: Predicted accidents exceed average by more than 20%. Take preventive measures!")
            elif future_pred > y.mean():
                st.warning("🟡 MODERATE RISK: Predicted accidents are above average.")
            else:
                st.success("🟢 LOW RISK: Predicted accidents are below average.")

            # Future trend chart
            st.subheader("📆 Simulated Future Trend (Next 12 Periods)")
            future_preds = []
            for i in range(1, 13):
                temp = future_df.copy()
                for col in temp.columns:
                    if col not in cat_cols:
                        temp[col] = temp[col] * (1 + i * 0.01)
                future_preds.append(max(0, model.predict(temp)[0]))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, 13), future_preds, marker="o", color="darkorange", linewidth=2)
            ax.fill_between(range(1, 13), future_preds, alpha=0.2, color="darkorange")
            ax.set_title("Simulated Accident Trend Over Next 12 Periods")
            ax.set_xlabel("Future Period")
            ax.set_ylabel("Predicted Accidents")
            ax.axhline(y.mean(), color="red", linestyle="--", label=f"Historical Avg: {y.mean():.1f}")
            ax.legend()
            st.pyplot(fig)
