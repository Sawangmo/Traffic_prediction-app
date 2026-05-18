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
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

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
        plt.close()

    if numeric_cols:
        st.subheader("5. Feature Distributions")
        selected_col = st.selectbox("Select a column to plot", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_col], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)
        plt.close()

        st.subheader("6. Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close()

        st.subheader("7. Boxplot (Outlier Detection)")
        box_col = st.selectbox("Select a column for boxplot", numeric_cols, key="box")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=df[box_col], color="salmon", ax=ax)
        ax.set_title(f"Boxplot of {box_col}")
        st.pyplot(fig)
        plt.close()
    else:
        st.info("ℹ️ No numeric columns found for distribution/correlation plots.")

    # ── PAST ACCIDENTS SECTION ────────────────────────────────
    st.header("📅 Past Accident Analysis")

    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "year", "month", "time"])]
    accident_count_cols = [c for c in df.columns if any(k in c.lower() for k in ["accident", "crash", "count", "incident", "total", "injur", "fatal"])]

    # Fallback to all columns if nothing detected
    date_col_options = date_cols if date_cols else list(df.columns)
    count_col_options = accident_count_cols if accident_count_cols else numeric_cols

    if count_col_options:
        date_col = st.selectbox("Select date/time column", date_col_options, key="date_col")
        count_col = st.selectbox("Select accident count column", count_col_options, key="count_col")

        df_time = df[[date_col, count_col]].dropna().copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors="coerce")
        df_time = df_time.dropna(subset=[date_col]).sort_values(date_col)

        if not df_time.empty:
            df_time_grouped = df_time.groupby(date_col)[count_col].sum().reset_index()

            st.subheader("📉 Accidents Over Time")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_time_grouped[date_col], df_time_grouped[count_col], color="crimson", linewidth=2)
            ax.fill_between(df_time_grouped[date_col], df_time_grouped[count_col], alpha=0.2, color="crimson")
            ax.set_title("Historical Accident Trend", fontsize=14)
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Accidents")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            total = int(df_time_grouped[count_col].sum())
            peak_idx = df_time_grouped[count_col].idxmax()
            peak_date = df_time_grouped.loc[peak_idx, date_col]
            peak_val = int(df_time_grouped.loc[peak_idx, count_col])

            col1, col2, col3 = st.columns(3)
            col1.metric("📌 Total Accidents (Historical)", f"{total:,}")
            col2.metric("📍 Peak Date", str(peak_date)[:10])
            col3.metric("🔺 Peak Count", f"{peak_val:,}")
        else:
            st.warning("⚠️ Could not parse dates in the selected column. Try selecting a different date column.")
    else:
        st.warning("⚠️ No numeric columns available for accident count analysis.")

    # ── CAUSE ANALYSIS SECTION ────────────────────────────────
    st.header("🔍 Accident Cause Analysis")

    cause_cols = [c for c in df.columns if any(k in c.lower() for k in
                  ["cause", "reason", "weather", "road", "condition", "type",
                   "light", "speed", "alcohol", "drug", "vehicle", "factor", "severity"])]

    all_cat_cols = df.select_dtypes(include="object").columns.tolist()
    cause_col_options = cause_cols if cause_cols else all_cat_cols

    if cause_col_options:
        cause_col = st.selectbox("Select cause/factor column to analyze", cause_col_options)
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
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Proportion Breakdown")
            fig, ax = plt.subplots(figsize=(6, 6))
            cause_counts.plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90)
            ax.set_ylabel("")
            ax.set_title(f"Distribution of '{cause_col}'")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("📋 Cause Summary Table")
        cause_df = cause_counts.reset_index()
        cause_df.columns = [cause_col, "Count"]
        cause_df["Percentage"] = (cause_df["Count"] / cause_df["Count"].sum() * 100).round(2)
        st.dataframe(cause_df)

        remaining_cats = [c for c in cause_col_options if c != cause_col]
        if remaining_cats:
            st.subheader("🔗 Multi-Factor Analysis")
            cause_col2 = st.selectbox("Select a second factor to cross-analyze", remaining_cats)
            cross_tab = pd.crosstab(df[cause_col], df[cause_col2])
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            ax.set_title(f"{cause_col} vs {cause_col2}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("ℹ️ No categorical columns found for cause analysis.")

    # ── MODEL + FUTURE PREDICTION SECTION ────────────────────
    st.header("🤖 Train Model & Predict Future Accidents")

    # Only show numeric columns as valid targets
    valid_targets = numeric_cols if numeric_cols else list(df.columns)
    st.info("💡 **Tip:** Select a numeric count column as your target (e.g. injuries_total, accident_count). Avoid date columns.")
    target = st.selectbox("Select target column (accident count to predict)", valid_targets)
    features = st.multiselect("Select feature columns", [c for c in df.columns if c != target])

    if features:
        X = df[features].copy()
        y = df[target].copy()

        # ── Validate & fix target column ──────────────────────
        if y.dtype == "object" or str(y.dtype).startswith("datetime"):
            try:
                y_dt = pd.to_datetime(y, errors="coerce")
                if y_dt.notna().sum() > len(y) * 0.5:
                    st.warning("⚠️ Target looks like a date column. Converting to numeric ordinal. Consider picking a count column instead.")
                    y = y_dt.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
                else:
                    st.warning("⚠️ Target is non-numeric. Label-encoding it.")
                    le_target = LabelEncoder()
                    y = pd.Series(le_target.fit_transform(y.astype(str)))
            except Exception:
                le_target = LabelEncoder()
                y = pd.Series(le_target.fit_transform(y.astype(str)))

        # Fill missing target values safely
        if pd.api.types.is_numeric_dtype(y):
            y = y.fillna(y.median())
        else:
            y = y.fillna(0)

        y = pd.to_numeric(y, errors="coerce").fillna(0)

        # ── Preprocess feature columns ─────────────────────────
        for col in X.columns:
            if str(X[col].dtype).startswith("datetime"):
                X[col] = pd.to_datetime(X[col], errors="coerce").map(
                    lambda x: x.toordinal() if pd.notna(x) else 0
                )
            elif X[col].dtype == "object":
                try:
                    X[col] = pd.to_datetime(X[col], errors="raise").map(
                        lambda x: x.toordinal() if pd.notna(x) else 0
                    )
                except Exception:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
            else:
                X[col] = X[col].fillna(X[col].median())

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        label_encoders = {}
        if cat_cols:
            st.info(f"ℹ️ Auto-encoding categorical columns: {cat_cols}")
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

        # Ensure all features are numeric
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        st.subheader("✅ Preprocessed Feature Preview")
        st.dataframe(X.head())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("n_estimators", 50, 500, 100)
        with col2:
            learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.1)

        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            st.success("✅ Model trained successfully!")

            # ── Evaluation Metrics ─────────────────────────────
            st.subheader("📈 Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", round(mean_absolute_error(y_test, y_pred), 4))
            col2.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
            col3.metric("R² Score", round(r2_score(y_test, y_pred), 4))

            # ── Feature Importances ────────────────────────────
            st.subheader("🌟 Feature Importances")
            fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            fi.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
            ax.set_title("Feature Importances")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Actual vs Predicted ────────────────────────────
            st.subheader("🔍 Actual vs Predicted")
            result_df = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": np.round(y_pred, 2)
            }).reset_index(drop=True)
            st.line_chart(result_df)

            # ── Future Prediction ──────────────────────────────
            st.subheader("🔮 Predict Future Accidents")
            st.write("Adjust values below to simulate future conditions:")

            future_input = {}
            num_cols_ui = min(3, len(features))
            ui_cols = st.columns(num_cols_ui)

            for i, feat in enumerate(features):
                with ui_cols[i % num_cols_ui]:
                    if feat in cat_cols and feat in label_encoders:
                        options = label_encoders[feat].classes_.tolist()
                        chosen = st.selectbox(f"{feat}", options, key=f"fut_{feat}")
                        future_input[feat] = int(label_encoders[feat].transform([chosen])[0])
                    else:
                        min_val = float(X[feat].min())
                        max_val = float(X[feat].max())
                        mean_val = float(X[feat].mean())
                        if min_val == max_val:
                            max_val = min_val + 1.0
                        future_input[feat] = st.slider(
                            f"{feat}", min_val, max_val, mean_val, key=f"fut_{feat}"
                        )

            future_df = pd.DataFrame([future_input])
            future_pred = float(model.predict(future_df)[0])
            future_pred = max(0, round(future_pred, 2))

            st.markdown("---")
            st.subheader("🚨 Predicted Future Accident Count")
            st.metric(label="Predicted Number of Accidents", value=f"{future_pred:,}")

            avg = float(y.mean())
            if future_pred > avg * 1.2:
                st.error("⚠️ HIGH RISK: Predicted accidents exceed historical average by more than 20%. Take preventive measures!")
            elif future_pred > avg:
                st.warning("🟡 MODERATE RISK: Predicted accidents are above historical average.")
            else:
                st.success("🟢 LOW RISK: Predicted accidents are below historical average.")

            # ── Future Trend Chart ─────────────────────────────
            st.subheader("📆 Simulated Future Trend (Next 12 Periods)")
            future_preds = []
            for i in range(1, 13):
                temp = future_df.copy()
                for col in temp.columns:
                    if col not in cat_cols:
                        temp[col] = temp[col] * (1 + i * 0.01)
                future_preds.append(max(0, float(model.predict(temp)[0])))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, 13), future_preds, marker="o", color="darkorange", linewidth=2, label="Predicted")
            ax.fill_between(range(1, 13), future_preds, alpha=0.2, color="darkorange")
            ax.axhline(avg, color="red", linestyle="--", linewidth=1.5, label=f"Historical Avg: {avg:.1f}")
            ax.set_title("Simulated Accident Trend Over Next 12 Periods")
            ax.set_xlabel("Future Period")
            ax.set_ylabel("Predicted Accidents")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Summary Table ──────────────────────────────────
            st.subheader("📋 Future Prediction Summary Table")
            summary_df = pd.DataFrame({
                "Period": [f"Period {i}" for i in range(1, 13)],
                "Predicted Accidents": [round(p, 2) for p in future_preds],
                "vs Historical Avg": [f"{'▲' if p > avg else '▼'} {abs(round(p - avg, 2))}" for p in future_preds]
            })
            st.dataframe(summary_df)
