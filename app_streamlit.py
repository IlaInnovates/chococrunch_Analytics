# app_streamlit.py — FINAL (no MySQL)
# Streamlit dashboard for ChocoCrunch Analytics
# Displays 27 query CSV tables, EDA charts, and product lookup features.

import os, glob
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from io import BytesIO

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="🍫 ChocoCrunch Analytics", layout="wide")
st.title("🍫 ChocoCrunch Analytics — Data Insights Dashboard")
st.markdown("""
Welcome to **ChocoCrunch Analytics** — explore your chocolate product data with
ready-made query outputs, visual analytics, and quick lookup tools.
""")

# ---------------------------
# Sidebar — Folder Selection
# ---------------------------
st.sidebar.header("Data Source")
data_folder = st.sidebar.text_input("Enter path to data folder", value="out")
st.sidebar.info("Place all your q01–q27 .csv files and full_engineered_snapshot.csv inside this folder.")

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data
def load_all_query_csvs(folder):
    """Load all q*.csv query files and full_engineered_snapshot.csv from the folder."""
    pattern = os.path.join(folder, "q*.csv")
    files = sorted(glob.glob(pattern))
    tables = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            df = pd.read_csv(f)
        except Exception:
            df = pd.read_csv(f, encoding="utf-8", engine="python", errors="ignore")
        tables[name] = df

    snapshot = None
    snap_fp = os.path.join(folder, "full_engineered_snapshot.csv")
    if os.path.exists(snap_fp):
        snapshot = pd.read_csv(snap_fp)

    return tables, snapshot


def to_excel_bytes(df):
    buf = BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()

# ---------------------------
# Load Data
# ---------------------------
with st.spinner("📦 Loading query outputs..."):
    try:
        query_tables, df_full = load_all_query_csvs(data_folder)
    except Exception as e:
        st.error(f"❌ Failed to load CSVs: {e}")
        query_tables, df_full = {}, None

st.sidebar.markdown(f"**Query tables found:** {len(query_tables)}")
st.sidebar.markdown(f"**Full snapshot loaded:** {'✅' if df_full is not None else '❌'}")

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["📊 Overview", "📁 Query Tables (q01–q27)", "📈 EDA", "🔍 Product Lookup"])

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tabs[0]:
    st.header("Project Overview")
    st.write("A quick glance at dataset scale and nutritional highlights.")

    c1, c2, c3, c4 = st.columns(4)
    total_rows = len(df_full) if df_full is not None else 0
    c1.metric("Engineered Rows", f"{total_rows}")
    c2.metric("Query Tables", f"{len(query_tables)}")

    if df_full is not None:
        avg_kcal = pd.to_numeric(df_full.get("energy-kcal_value", pd.Series(dtype=float))).mean()
        avg_sugar = pd.to_numeric(df_full.get("sugars_value", pd.Series(dtype=float))).mean()
        ultra_count = df_full[df_full.get("is_ultra_processed", "") == "Yes"].shape[0] \
                      if "is_ultra_processed" in df_full.columns else 0
        c3.metric("Avg Energy (kcal)", f"{avg_kcal:.1f}" if pd.notna(avg_kcal) else "N/A")
        c4.metric("Avg Sugar (g)", f"{avg_sugar:.1f}" if pd.notna(avg_sugar) else "N/A")

        st.subheader("Top Brands by Average Calories")
        if "brand" in df_full.columns and "energy-kcal_value" in df_full.columns:
            brand_stats = (df_full.groupby("brand")["energy-kcal_value"]
                           .mean().sort_values(ascending=False).head(10).reset_index())
            fig = px.bar(brand_stats, x="brand", y="energy-kcal_value",
                         color="energy-kcal_value", color_continuous_scale="YlOrBr",
                         title="Top 10 Brands by Average Energy (kcal)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Navigation:**
- 🗂 Use *Query Tables* to inspect all 27 queries.  
- 📈 Use *EDA* for interactive data visualization.  
- 🔍 Use *Product Lookup* to examine any product code.
""")

# =========================================================
# TAB 2 — QUERY TABLES
# =========================================================
with tabs[1]:
    st.header("All Query Outputs (q01–q27)")
    if not query_tables:
        st.info("⚠️ No q*.csv files found. Place them in the 'out/' folder and reload.")
    else:
        for name, df in query_tables.items():
            with st.expander(f"{name} — {df.shape[0]} rows"):
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "⬇️ Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{name}.csv"
                )
                st.caption(f"Columns: {list(df.columns)}")

# =========================================================
# TAB 3 — EDA (Exploratory Data Analysis)
# =========================================================
with tabs[2]:
    st.header("Exploratory Data Analysis (EDA)")
    if df_full is None:
        st.info("⚠️ Missing full_engineered_snapshot.csv — place it in 'out/' folder.")
    else:
        st.subheader("Filters")
        brands = sorted(df_full["brand"].dropna().unique()[:500]) if "brand" in df_full.columns else []
        brand_sel = st.multiselect("Filter by brand (top 500 shown)", options=brands)
        df_filtered = df_full.copy()
        if brand_sel:
            df_filtered = df_filtered[df_filtered["brand"].isin(brand_sel)]

        # -----------------------
        # Distributions
        # -----------------------
        st.subheader("Distribution Plots")
        c1, c2 = st.columns(2)
        if "energy-kcal_value" in df_filtered.columns:
            with c1:
                fig = px.histogram(df_filtered, x="energy-kcal_value", nbins=40,
                                   title="Energy (kcal per 100g)", color_discrete_sequence=["#8B4513"])
                st.plotly_chart(fig, use_container_width=True)
        if "sugars_value" in df_filtered.columns:
            with c2:
                fig = px.histogram(df_filtered, x="sugars_value", nbins=40,
                                   title="Sugars (g per 100g)", color_discrete_sequence=["#C68642"])
                st.plotly_chart(fig, use_container_width=True)

        # -----------------------
        # Scatter Plot
        # -----------------------
        st.subheader("Calories vs Sugar (Scatter)")
        if {"energy-kcal_value", "sugars_value"}.issubset(df_filtered.columns):
            fig = px.scatter(df_filtered, x="sugars_value", y="energy-kcal_value",
                             hover_data=["product_code", "product_name", "brand"],
                             trendline="ols",
                             title="Calories vs Sugars Correlation")
            st.plotly_chart(fig, use_container_width=True)

        # -----------------------
        # Boxplot
        # -----------------------
        st.subheader("Sugar-to-Carb Ratio by Calorie Category")
        if "sugar_to_carb_ratio" in df_filtered.columns and "calorie_category" in df_filtered.columns:
            fig = px.box(df_filtered, x="calorie_category", y="sugar_to_carb_ratio",
                         color="calorie_category", points="outliers",
                         title="Sugar-to-Carb Ratio Distribution by Calorie Category")
            st.plotly_chart(fig, use_container_width=True)

        # -----------------------
        # Correlation Heatmap
        # -----------------------
        st.subheader("Correlation Heatmap")
        numeric_cols = [c for c in ["energy-kcal_value", "sugars_value",
                                    "fat_value", "carbohydrates_value",
                                    "proteins_value", "sodium_value"]
                        if c in df_filtered.columns]
        if numeric_cols:
            corr = df_filtered[numeric_cols].corr().round(2)
            fig = ff.create_annotated_heatmap(
                z=corr.values.tolist(),
                x=list(corr.columns),
                y=list(corr.index),
                annotation_text=corr.astype(str).values.tolist(),
                colorscale="YlGnBu"
            )
            fig.update_layout(title="Nutrient Correlation Matrix", height=600)
            st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4 — PRODUCT LOOKUP
# =========================================================
with tabs[3]:
    st.header("Product Lookup")
    if df_full is None:
        st.info("⚠️ Snapshot not found. Place full_engineered_snapshot.csv in 'out/'.")
    else:
        code = st.text_input("Enter product_code (exact match)")
        if code:
            res = df_full[df_full["product_code"].astype(str) == str(code)]
            if res.empty:
                st.warning("No product found with that code.")
            else:
                st.success(f"Found {len(res)} record(s)")
                st.dataframe(res.T, use_container_width=True)
                st.download_button(
                    "⬇️ Download Product as Excel",
                    to_excel_bytes(res),
                    file_name=f"product_{code}.xlsx"
                )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("© 2025 ChocoCrunch Analytics | Made with ❤️ in Python + Streamlit")
