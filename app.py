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

    # ── PAST ACCIDENTS SECTION ────────────────────────────────
    st.header("📅 Past Accident Analysis")

    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "year", "month", "time"])]
    accident_count_cols = [c for c in df.columns if any(k in c.lower() for k in
                           ["accident", "crash", "count", "incident", "total", "injur", "fatal"])]

    date_col_options = date_cols if date_cols else list(df.columns)
    count_col_options = accident_count_cols if accident_count_cols else numeric_cols

    if count_col_options:
        date_col = st.selectbox("Select date/time column", date_col_options, key="date_col")
        count_col = st.selectbox("Select accident count column", count_col_options, key="count_col")

        df_time = df[[date_col, count_col]].dropna().copy()
        df_time[count_col] = pd.to_numeric(df_time[count_col], errors="coerce").fillna(0)
        df_time[date_col] = pd.to_datetime(df_time[date_col], errors="coerce")
        df_time = df_time.dropna(subset=[date_col]).sort_values(date_col)

        if not df_time.empty:
            df_time_grouped = df_time.groupby(date_col)[count_col].sum().reset_index()
            df_time_grouped[count_col] = pd.to_numeric(df_time_grouped[count_col], errors="coerce").fillna(0)

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
            numeric_series = pd.to_numeric(df_time_grouped[count_col], errors="coerce").fillna(0)
            peak_idx = numeric_series.idxmax()
            peak_date = df_time_grouped.loc[peak_idx, date_col]
            peak_val = int(numeric_series.loc[peak_idx])

            col1, col2, col3 = st.columns(3)
            col1.metric("📌 Total Accidents (Historical)", f"{total:,}")
            col2.metric("📍 Peak Date", str(peak_date)[:10])
            col3.metric("🔺 Peak Count", f"{peak_val:,}")
        else:
            st.warning("⚠️ Could not parse dates. Try a different date column.")

    # ── AREA ANALYSIS SECTION ─────────────────────────────────
    st.header("📍 Accident Area Analysis")

    area_cols = [c for c in df.columns if any(k in c.lower() for k in
                 ["area", "location", "district", "zone", "region", "city",
                  "street", "road", "suburb", "place", "ward", "county", "state"])]
    area_col_options = area_cols if area_cols else df.select_dtypes(include="object").columns.tolist()

    area_col = None
    if area_col_options:
        area_col = st.selectbox("Select area/location column", area_col_options, key="area_col")
        area_counts = df[area_col].value_counts().head(15)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏙️ Top 15 Accident-Prone Areas")
            fig, ax = plt.subplots(figsize=(9, 6))
            area_counts.plot(kind="barh", ax=ax, color="firebrick", edgecolor="black")
            ax.set_title("Areas with Most Accidents")
            ax.set_xlabel("Number of Accidents")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("📊 Area Proportion")
            fig, ax = plt.subplots(figsize=(6, 6))
            area_counts.head(8).plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90)
            ax.set_ylabel("")
            ax.set_title("Top 8 Areas by Accident Share")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("📋 Area Summary Table")
        area_df = area_counts.reset_index()
        area_df.columns = [area_col, "Accident Count"]
        area_df["Percentage"] = (area_df["Accident Count"] / area_df["Accident Count"].sum() * 100).round(2)
        area_df["Risk Level"] = area_df["Percentage"].apply(
            lambda x: "🔴 High" if x > 10 else ("🟡 Medium" if x > 5 else "🟢 Low")
        )
        st.dataframe(area_df)

        top_area = area_counts.index[0]
        top_area_count = int(area_counts.iloc[0])
        st.error(f"🔴 **Accident Hotspot:** `{top_area}` with **{top_area_count:,}** recorded accidents")
    else:
        st.info("ℹ️ No area/location columns detected.")

    # ── TIME ANALYSIS SECTION ─────────────────────────────────
    st.header("⏰ Accident Time Analysis")

    time_cols = [c for c in df.columns if any(k in c.lower() for k in
                 ["time", "hour", "period", "shift", "day", "date", "month", "week"])]
    time_col_options = time_cols if time_cols else list(df.columns)

    time_col = None
    if time_col_options:
        time_col = st.selectbox("Select time column", time_col_options, key="time_col")
        time_series = df[time_col].dropna().copy()
        parsed_time = pd.to_datetime(time_series, errors="coerce")
        has_time = parsed_time.notna().sum() > len(time_series) * 0.3

        if has_time:
            df["_hour"] = parsed_time.dt.hour
            df["_day"] = parsed_time.dt.day_name()
            df["_month"] = parsed_time.dt.month_name()

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🕐 Accidents by Hour of Day")
                hour_counts = df["_hour"].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.bar(hour_counts.index, hour_counts.values, color="steelblue", edgecolor="black")
                ax.set_title("Accident Frequency by Hour")
                ax.set_xlabel("Hour of Day (0-23)")
                ax.set_ylabel("Number of Accidents")
                ax.set_xticks(range(0, 24))
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                peak_hour = int(hour_counts.idxmax())
                st.warning(f"⚠️ **Peak Accident Hour:** {peak_hour}:00 - {peak_hour+1}:00")

            with col2:
                st.subheader("📅 Accidents by Day of Week")
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                day_counts = df["_day"].value_counts().reindex(day_order, fill_value=0)
                fig, ax = plt.subplots(figsize=(9, 4))
                day_counts.plot(kind="bar", ax=ax, color="coral", edgecolor="black")
                ax.set_title("Accident Frequency by Day of Week")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                peak_day = day_counts.idxmax()
                st.warning(f"⚠️ **Most Dangerous Day:** {peak_day}")

            st.subheader("📆 Accidents by Month")
            month_order = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            month_counts = df["_month"].value_counts().reindex(month_order, fill_value=0)
            fig, ax = plt.subplots(figsize=(12, 4))
            month_counts.plot(kind="bar", ax=ax, color="mediumpurple", edgecolor="black")
            ax.set_title("Accident Frequency by Month")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            peak_month = month_counts.idxmax()
            st.warning(f"⚠️ **Most Dangerous Month:** {peak_month}")

        else:
            st.subheader("🕐 Accident Frequency by Time Period")
            time_counts = time_series.value_counts().head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            time_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
            ax.set_title(f"Accidents by {time_col}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.warning(f"⚠️ **Peak Period:** {time_counts.index[0]} with {int(time_counts.iloc[0]):,} accidents")

    # ── WEATHER ANALYSIS SECTION ──────────────────────────────
    st.header("🌦️ Weather Condition Analysis")

    weather_cols = [c for c in df.columns if any(k in c.lower() for k in
                    ["weather", "condition", "climate", "rain", "fog",
                     "visibility", "wind", "snow", "temperature", "humid"])]
    weather_col_options = weather_cols if weather_cols else df.select_dtypes(include="object").columns.tolist()

    weather_col = None
    if weather_col_options:
        weather_col = st.selectbox("Select weather column", weather_col_options, key="weather_col")
        weather_counts = df[weather_col].value_counts().head(10)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌧️ Accidents by Weather Condition")
            fig, ax = plt.subplots(figsize=(9, 5))
            colors = plt.cm.Set2(np.linspace(0, 1, len(weather_counts)))
            weather_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
            ax.set_title("Accident Count by Weather Condition")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("☁️ Weather Proportion")
            fig, ax = plt.subplots(figsize=(6, 6))
            weather_counts.plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90,
                                colors=plt.cm.Set2(np.linspace(0, 1, len(weather_counts))))
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.subheader("📋 Weather Summary Table")
        weather_df = weather_counts.reset_index()
        weather_df.columns = [weather_col, "Accident Count"]
        weather_df["Percentage"] = (weather_df["Accident Count"] / weather_df["Accident Count"].sum() * 100).round(2)
        weather_df["Severity"] = weather_df["Percentage"].apply(
            lambda x: "🔴 Critical" if x > 20 else ("🟡 Moderate" if x > 10 else "🟢 Minor")
        )
        st.dataframe(weather_df)

        worst_weather = weather_counts.index[0]
        worst_count = int(weather_counts.iloc[0])
        st.error(f"🌩️ **Most Dangerous Weather:** `{worst_weather}` — involved in **{worst_count:,}** accidents")

        if area_col and area_col in df.columns:
            st.subheader("🔗 Weather vs Area Heatmap")
            top_areas = df[area_col].value_counts().head(8).index.tolist()
            top_weathers = df[weather_col].value_counts().head(6).index.tolist()
            df_filtered = df[df[area_col].isin(top_areas) & df[weather_col].isin(top_weathers)]
            if not df_filtered.empty:
                cross = pd.crosstab(df_filtered[area_col], df_filtered[weather_col])
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(cross, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
                ax.set_title("Accidents by Area and Weather Condition")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # ── CAUSE ANALYSIS SECTION ────────────────────────────────
    st.header("🔍 Accident Cause Analysis")

    cause_cols = [c for c in df.columns if any(k in c.lower() for k in
                  ["cause", "reason", "type", "light", "speed",
                   "alcohol", "drug", "vehicle", "factor", "severity"])]
    all_cat_cols = df.select_dtypes(include="object").columns.tolist()
    cause_col_options = cause_cols if cause_cols else all_cat_cols

    cause_col = None
    if cause_col_options:
        cause_col = st.selectbox("Select cause/factor column", cause_col_options)
        cause_counts = df[cause_col].value_counts().head(10)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Top Causes: {cause_col}")
            fig, ax = plt.subplots(figsize=(8, 5))
            cause_counts.plot(kind="bar", ax=ax, color="tomato", edgecolor="black")
            ax.set_title(f"Top 10 Values in '{cause_col}'")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Proportion Breakdown")
            fig, ax = plt.subplots(figsize=(6, 6))
            cause_counts.plot(kind="pie", ax=ax, autopct="%1.1f%%", startangle=90)
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        cause_df = cause_counts.reset_index()
        cause_df.columns = [cause_col, "Count"]
        cause_df["Percentage"] = (cause_df["Count"] / cause_df["Count"].sum() * 100).round(2)
        st.dataframe(cause_df)

        remaining = [c for c in cause_col_options if c != cause_col]
        if remaining:
            st.subheader("🔗 Multi-Factor Analysis")
            cause_col2 = st.selectbox("Select second factor", remaining)
            cross_tab = pd.crosstab(df[cause_col], df[cause_col2])
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(cross_tab, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
            ax.set_title(f"{cause_col} vs {cause_col2}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ── RECOMMENDATIONS SECTION ───────────────────────────────
    st.header("💡 Safety Recommendations")

    recommendations = []

    if area_col and area_col in df.columns:
        top3_areas = df[area_col].value_counts().head(3).index.tolist()
        recommendations.append({
            "category": "📍 High-Risk Areas",
            "finding": f"Top accident zones: **{', '.join(str(a) for a in top3_areas)}**",
            "recommendation": (
                "• Deploy additional traffic police and speed cameras in these zones\n"
                "• Install more road signs, speed bumps, and guardrails\n"
                "• Improve road lighting, especially at intersections\n"
                "• Conduct regular road condition maintenance"
            )
        })

    if "_hour" in df.columns:
        peak_h = int(df["_hour"].value_counts().idxmax())
        time_label = "Rush hour (morning)" if 6 <= peak_h <= 9 else \
                     "Rush hour (evening)" if 16 <= peak_h <= 19 else \
                     "Late night / early morning" if peak_h >= 22 or peak_h <= 4 else "Daytime"
        recommendations.append({
            "category": "⏰ Peak Time",
            "finding": f"Most accidents at **{peak_h}:00** ({time_label})",
            "recommendation": (
                "• Increase traffic enforcement during peak hours\n"
                "• Run public awareness campaigns for high-risk time periods\n"
                "• Adjust traffic light timing to reduce congestion\n"
                "• Encourage staggered work/school hours to ease rush hour pressure"
            )
        })
    elif time_col and time_col in df.columns:
        peak_period = df[time_col].value_counts().idxmax()
        recommendations.append({
            "category": "⏰ Peak Period",
            "finding": f"Most accidents during **{peak_period}**",
            "recommendation": (
                "• Increase patrols and monitoring during this period\n"
                "• Issue public safety alerts for high-risk time windows"
            )
        })

    if weather_col and weather_col in df.columns:
        worst_w = df[weather_col].value_counts().idxmax()
        recommendations.append({
            "category": "🌦️ Weather Conditions",
            "finding": f"Most accidents in **{worst_w}** conditions",
            "recommendation": (
                "• Issue weather-based driving advisories and alerts\n"
                "• Enforce reduced speed limits during adverse weather\n"
                "• Improve road drainage to reduce wet/flooded surfaces\n"
                "• Educate drivers on safe driving in poor visibility"
            )
        })

    if cause_col and cause_col in df.columns:
        top_cause = df[cause_col].value_counts().idxmax()
        recommendations.append({
            "category": "🔍 Primary Cause",
            "finding": f"Leading cause: **{top_cause}**",
            "recommendation": (
                "• Launch targeted awareness campaigns addressing this cause\n"
                "• Strengthen penalties for related violations\n"
                "• Introduce mandatory driver education programs\n"
                "• Increase random checks and enforcement operations"
            )
        })

    recommendations.append({
        "category": "🛡️ General Safety",
        "finding": "Overall system-wide improvements",
        "recommendation": (
            "• Implement a real-time accident reporting and alert system\n"
            "• Improve emergency response infrastructure near hotspots\n"
            "• Use predictive analytics to pre-position traffic officers\n"
            "• Collaborate with local government on road infrastructure upgrades\n"
            "• Conduct annual road safety audits on high-risk corridors"
        )
    })

    for rec in recommendations:
        with st.expander(f"{rec['category']} — {rec['finding']}", expanded=True):
            st.markdown(rec["recommendation"])

    rec_text = "\n\n".join(
        [f"{r['category']}\nFinding: {r['finding']}\nRecommendations:\n{r['recommendation']}"
         for r in recommendations]
    )
    st.download_button(
        label="📥 Download Recommendations as TXT",
        data=rec_text,
        file_name="accident_recommendations.txt",
        mime="text/plain"
    )

    # ── MODEL + FUTURE PREDICTION SECTION ────────────────────
    st.header("🤖 Train Model & Predict Future Accidents")

    valid_targets = numeric_cols if numeric_cols else list(df.columns)
    st.info("💡 **Tip:** Select a numeric count column as your target (e.g. injuries_total, accident_count). Avoid date columns.")
    target = st.selectbox("Select target column (accident count to predict)", valid_targets)
    features = st.multiselect("Select feature columns", [c for c in df.columns if c != target])

    if features:
        X = df[features].copy()
        y = df[target].copy()

        if y.dtype == "object" or str(y.dtype).startswith("datetime"):
            try:
                y_dt = pd.to_datetime(y, errors="coerce")
                if y_dt.notna().sum() > len(y) * 0.5:
                    st.warning("⚠️ Target looks like a date. Converting to ordinal.")
                    y = y_dt.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
                else:
                    le_t = LabelEncoder()
                    y = pd.Series(le_t.fit_transform(y.astype(str)))
            except Exception:
                le_t = LabelEncoder()
                y = pd.Series(le_t.fit_transform(y.astype(str)))

        y = pd.to_numeric(y, errors="coerce").fillna(0)

        for col in X.columns:
            if str(X[col].dtype).startswith("datetime"):
                X[col] = pd.to_datetime(X[col], errors="coerce").map(
                    lambda x: x.toordinal() if pd.notna(x) else 0)
            elif X[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(X[col], errors="raise")
                    X[col] = parsed.map(lambda x: x.toordinal() if pd.notna(x) else 0)
                except Exception:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
            else:
                X[col] = X[col].fillna(X[col].median())

        cat_cols = X.select_dtypes(include="object").columns.tolist()
        label_encoders = {}
        if cat_cols:
            st.info(f"ℹ️ Auto-encoding: {cat_cols}")
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

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
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.subheader("🔍 Actual vs Predicted")
            result_df = pd.DataFrame({"Actual": y_test.values, "Predicted": np.round(y_pred, 2)}).reset_index(drop=True)
            st.line_chart(result_df)

            st.subheader("🔮 Predict Future Accidents")
            st.write("Adjust values below to simulate future conditions:")

            future_input = {}
            ui_cols = st.columns(min(3, len(features)))
            for i, feat in enumerate(features):
                with ui_cols[i % min(3, len(features))]:
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
                        future_input[feat] = st.slider(f"{feat}", min_val, max_val, mean_val, key=f"fut_{feat}")

            future_df = pd.DataFrame([future_input])
            future_pred = max(0, round(float(model.predict(future_df)[0]), 2))
            avg = float(y.mean())

            st.markdown("---")
            st.subheader("🚨 Predicted Future Accident Count")
            st.metric("Predicted Number of Accidents", f"{future_pred:,}")

            if future_pred > avg * 1.2:
                st.error("⚠️ HIGH RISK: Exceeds historical average by >20%. Immediate action needed!")
            elif future_pred > avg:
                st.warning("🟡 MODERATE RISK: Above historical average. Caution advised.")
            else:
                st.success("🟢 LOW RISK: Below historical average.")

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

            st.subheader("📋 Future Prediction Summary Table")
            summary_df = pd.DataFrame({
                "Period": [f"Period {i}" for i in range(1, 13)],
                "Predicted Accidents": [round(p, 2) for p in future_preds],
                "vs Historical Avg": [f"{'▲' if p > avg else '▼'} {abs(round(p - avg, 2))}" for p in future_preds]
            })
            st.dataframe(summary_df)
