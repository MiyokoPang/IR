import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine
import pymysql

# --- 1. Page Setup ---
st.set_page_config(page_title="Thesis Results: Sentiment Ablation", layout="wide", page_icon="🎓")

# --- 2. Load Data ---
@st.cache_data
def load_data():
    engine = create_engine("mysql+pymysql://root:@127.0.0.1/trading_system")

    tables = {
        'Traditional': 'results_ir_models', 
        'Deep Learning': 'results_deep_learning_models', 
        'Trad. Tuned': 'results_hypertuning',
        'DL Tuned': 'results_dl_hypertuning' 
    }

    dfs = []
    for cat, tbl in tables.items():
        try:
            t = pd.read_sql(f"SELECT * FROM {tbl}", con=engine)

            # FIX 1: Lowercase columns FIRST to normalize DB columns (e.g., Ticker -> ticker)
            t.columns = [c.lower() for c in t.columns]

            # FIX 2: Add Category AFTER lowercasing so it stays 'Category' (Capitalized)
            t['Category'] = cat

            dfs.append(t)
        except: pass 

    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)

    # --- CLEANING LOGIC ---
    # 1. Standardize Feature Set Names
    # We map the lowercase DB values to clean UI labels
    feature_map = {
        'price_only': 'Price Only',
        'price_base_sentiment': 'Base Sentiment', 
        'price_full_sentiment': 'Full Sentiment'
    }

    # Handle missing or weirdly cased feature_set columns
    if 'feature_set' in df.columns:
        df['feature_set_clean'] = df['feature_set'].str.lower().map(feature_map).fillna('Other')
    else:
        # Fallback if the feature_set column is missing (parse from model name)
        def fallback_feature(m):
            if 'Price_Full_Sentiment' in m: return 'Full Sentiment'
            if 'Price_Base_Sentiment' in m: return 'Base Sentiment'
            if 'Price_Only' in m: return 'Price Only'
            return 'Other'
        df['feature_set_clean'] = df['model'].apply(fallback_feature)

    # 2. Standardize Architecture Names
    def clean_arch(row):
        m = row['model']
        # Remove feature suffixes
        m = m.replace('_Price_Full_Sentiment', '').replace('_Price_Base_Sentiment', '').replace('_Price_Only', '')
        # Remove Tuning suffix
        m = m.replace('_Tuned', '')
        # Remove Sentiment text if embedded directly
        m = m.replace('Price_Full_Sentiment', '').replace('Price_Base_Sentiment', '').replace('Price_Only', '')
        return m.strip('_')

    df['Architecture'] = df.apply(clean_arch, axis=1)

    # 3. Calculate Baselines (Compare against Price Only of same Architecture)
    # Ensure we group by clean architecture and ticker
    baseline_df = df[df['feature_set_clean'] == 'Price Only'].groupby(['ticker', 'Architecture'])[['rmse', 'directional_accuracy']].mean().reset_index()
    baseline_df.rename(columns={'rmse': 'base_rmse', 'directional_accuracy': 'base_acc'}, inplace=True)

    df = df.merge(baseline_df, on=['ticker', 'Architecture'], how='left')

    # Calculate Improvements
    df['RMSE Imp %'] = ((df['base_rmse'] - df['rmse']) / df['base_rmse']) * 100
    df['Acc Imp %'] = df['directional_accuracy'] - df['base_acc']

    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Data Processing Error: {e}")
    st.stop()

if df.empty:
    st.warning("No data found. Please run your training notebooks to populate the database.")
    st.stop()

# --- 3. Sidebar ---
st.sidebar.title("📊 Analysis Controls")

metric_map = {
    "Directional Accuracy (%)": "directional_accuracy",
    "RMSE (Error)": "rmse",
    "R-Squared": "r2"
}
selected_metric_label = st.sidebar.selectbox("Primary Metric", list(metric_map.keys()))
metric = metric_map[selected_metric_label]

# Sorting: RMSE is better when lower (Ascending), others when higher (Descending)
is_ascending = True if metric == 'rmse' else False

# Filters
# This line caused the error before. It should work now that 'Category' is preserved.
cats = st.sidebar.multiselect("Model Category", df['Category'].unique(), default=df['Category'].unique())
df_filtered = df[df['Category'].isin(cats)]

# --- 4. Main Dashboard ---
st.title(f"🏆 Best Models by {selected_metric_label}")

# KPI Row
best_run = df_filtered.sort_values(metric, ascending=is_ascending).iloc[0]
# Calculate Best Average Model
model_avg_scores = df_filtered.groupby('model')[metric].mean().sort_values(ascending=is_ascending)
best_avg_model = model_avg_scores.index[0]
best_avg_val = model_avg_scores.iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Top Single Run", f"{best_run[metric]:.4f}", f"{best_run['ticker']} ({best_run['Architecture']})")
c2.metric("Best Avg Model", f"{best_avg_val:.4f}", best_avg_model)
c3.metric("Models Analyzed", len(df_filtered))
c4.metric("Unique Tickers", df_filtered['ticker'].nunique())

st.divider()

tab1, tab2, tab3 = st.tabs(["🥇 Leaderboard", "🧪 Sentiment Ablation", "📉 Tuning Impact"])

# === TAB 1: LEADERBOARD ===
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Top 15 Models (Market-Wide Average)")

        # Group by Model configuration to get market-wide average
        # We group by 'model' to differentiate between Tuned and Base versions
        leaderboard = df_filtered.groupby(['model', 'Category', 'feature_set_clean'])[metric].mean().reset_index()
        leaderboard = leaderboard.sort_values(metric, ascending=is_ascending).head(15)

        fig_lead = px.bar(
            leaderboard, x=metric, y='model', orientation='h', color='Category',
            text_auto='.4f',
            title=f"Market-Wide Performance ({selected_metric_label})",
            color_discrete_map={
                'Traditional': '#636EFA', 'Deep Learning': '#EF553B', 
                'Trad. Tuned': '#00CC96', 'DL Tuned': '#AB63FA'
            }
        )
        fig_lead.update_layout(yaxis={'categoryorder':'total descending' if is_ascending else 'total ascending'})
        st.plotly_chart(fig_lead, use_container_width=True)

    with col2:
        st.info("**Analysis Tip**")
        st.markdown(f"""
        This chart ranks models by their **consistency** across all tickers.

        * **{best_avg_model}** is currently the most robust architecture.
        * If **Deep Learning** models are missing from the top, it implies they may be overfitting or struggling with the noise in stock data compared to simpler linear models.
        """)

# === TAB 2: ABLATION ===
with tab2:
    st.subheader(f"Feature Impact: Does Sentiment Help?")

    # Pivot for Heatmap using Cleaned Feature Set
    heat_data = df_filtered.groupby(['Architecture', 'feature_set_clean'])[metric].mean().reset_index()
    heat_pivot = heat_data.pivot(index='Architecture', columns='feature_set_clean', values=metric)

    # Dynamic Colors
    colors = "RdBu_r" if metric == 'rmse' else "Viridis" 

    fig_heat = px.imshow(
        heat_pivot, text_auto=".3f", color_continuous_scale=colors, aspect="auto",
        title=f"Heatmap: {selected_metric_label} by Architecture"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# === TAB 3: TUNING IMPACT ===
with tab3:
    st.subheader("Hypertuning: Base vs Tuned")

    # Identify Tuned vs Base categories
    tuned_mask = df_filtered['Category'].str.contains("Tuned")
    tuned_df = df_filtered[tuned_mask]
    base_df = df_filtered[~tuned_mask]

    if tuned_df.empty:
        st.warning("No Tuned results loaded. Check filters or database.")
    else:
        # Compare overlapping architectures
        common_arch = set(tuned_df['Architecture']).intersection(set(base_df['Architecture']))

        comp_data = []
        for arch in common_arch:
            base_score = base_df[base_df['Architecture']==arch][metric].mean()
            tuned_score = tuned_df[tuned_df['Architecture']==arch][metric].mean()
            comp_data.append({'Architecture': arch, 'Type': 'Base', 'Score': base_score})
            comp_data.append({'Architecture': arch, 'Type': 'Tuned', 'Score': tuned_score})

        comp_df = pd.DataFrame(comp_data)

        if comp_df.empty:
            st.warning("No matching Architectures found between Base and Tuned datasets.")
        else:
            fig_comp = px.bar(
                comp_df, x='Architecture', y='Score', color='Type', barmode='group',
                title=f"Head-to-Head: Base vs Tuned ({selected_metric_label})"
            )
            # Zoom logic for Accuracy to make differences visible
            if metric == 'directional_accuracy':
                min_score = comp_df['Score'].min() * 0.95
                fig_comp.update_yaxes(range=[min_score, 100 if min_score < 90 else min_score + 10])

            st.plotly_chart(fig_comp, use_container_width=True)
