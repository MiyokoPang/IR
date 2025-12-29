import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import pymysql

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Thesis: Sentiment Dashboard", layout="wide", page_icon="📊")

# Custom Color Palette
COLOR_MAP = {
    'Price Only': '#2c3e50',           # Dark Blue/Grey
    'Base Sentiment': '#2980b9',       # Blue
    'Full Sentiment': '#3498db',       # Light Blue
    'VADER': '#e67e22',                # Orange
    'TextBlob': '#f1c40f',             # Yellow
    'FinBERT': '#9b59b6',              # Purple
    'Other': '#95a5a6'
}

# --- 2. MODEL NAMING LOGIC ---
def get_model_family(model_name):
    m = str(model_name).lower().strip()
    m = m.replace('_price_only', '').replace('_price_base_sentiment', '')\
         .replace('_price_full_sentiment', '').replace('_tuned', '')
    
    if 'bilstm' in m: return 'BiLSTM'
    if 'lstm' in m: return 'LSTM'
    if 'gru' in m: return 'GRU'
    if 'cnn' in m: return 'CNN'
    if 'mlp' in m: return 'MLP'
    if 'xgboost' in m: return 'XGBoost'
    if 'random' in m and 'forest' in m: return 'Random Forest'
    if 'arima' in m or 'arimax' in m: return 'ARIMA Family'
    if 'linear' in m and 'regression' in m: return 'Linear Regression'
    if 'ridge' in m: return 'Ridge'
    if 'lasso' in m: return 'Lasso'
    if 'svr' in m: return 'SVR'
    
    return model_name.replace('_', ' ').title()

def get_clean_specific_name(model_name):
    m = str(model_name)
    for suffix in ['_Price_Only', '_Price_Base_Sentiment', '_Price_Full_Sentiment', '_Tuned']:
        m = m.replace(suffix, '')
    return m.replace('_', ' ')

# --- 3. DATA LOADER ---
@st.cache_data(ttl=3600)
def load_and_unify_data():
    dfs = []
    
    def process_table(df, category, is_ablation=False):
        df.columns = [c.lower() for c in df.columns]
        
        # Handle column naming differences
        if is_ablation:
            df.rename(columns={'source': 'Feature Set', 'accuracy': 'directional_accuracy'}, inplace=True)
        else:
            df.rename(columns={'feature_set': 'Feature Set'}, inplace=True)
        
        df['Raw_Model'] = df['model']
        df['Family'] = df['Raw_Model'].apply(get_model_family)
        df['Specific_Model'] = df['Raw_Model'].apply(get_clean_specific_name)
        df['Category'] = category
        return df

    # A. LOCAL DB
    try:
        engine_local = create_engine("mysql+pymysql://root:@127.0.0.1/trading_system")
        try:
            t1 = pd.read_sql("SELECT * FROM results_ir_models", con=engine_local)
            if not t1.empty: dfs.append(process_table(t1, "Traditional", is_ablation=False))
        except: pass
        try:
            t2 = pd.read_sql("SELECT * FROM results_deep_learning_models", con=engine_local)
            if not t2.empty: dfs.append(process_table(t2, "Deep Learning", is_ablation=False))
        except: pass
    except: pass

    # B. REMOTE DB
    try:
        engine_remote = create_engine("mysql+pymysql://tenent:007963@vpn.servercd.co/trading_system")
        try:
            t3 = pd.read_sql("SELECT * FROM results_trad_source_ablation", con=engine_remote)
            if not t3.empty: dfs.append(process_table(t3, "Traditional", is_ablation=True))
        except: pass
        try:
            t4 = pd.read_sql("SELECT * FROM results_dl_source_ablation", con=engine_remote)
            if not t4.empty: dfs.append(process_table(t4, "Deep Learning", is_ablation=True))
        except: pass
    except: pass

    if not dfs: return pd.DataFrame()
    master_df = pd.concat(dfs, ignore_index=True)
    
    master_df['Feature Set'] = master_df['Feature Set'].replace({
        'vader': 'VADER', 'textblob': 'TextBlob', 'finbert': 'FinBERT',
        'price_only': 'Price Only', 'Price_Only': 'Price Only',
        'Price_Base_Sentiment': 'Base Sentiment',
        'Price_Full_Sentiment': 'Full Sentiment'
    })
    
    return master_df

def calculate_improvements(df, metric_col='rmse', is_error_metric=True):
    df = df.copy()
    
    # 1. Try standard baseline (Price Only)
    baselines = df[df['Feature Set'] == 'Price Only'].groupby(['ticker', 'Specific_Model'])[metric_col].mean().reset_index()
    baselines.rename(columns={metric_col: 'base_val'}, inplace=True)
    df = df.merge(baselines, on=['ticker', 'Specific_Model'], how='left')
    
    # 2. Calculate Imp % where baseline exists
    if is_error_metric:
        df['Imp %'] = ((df['base_val'] - df[metric_col]) / df['base_val']) * 100
    else:
        # Avoid div by zero for accuracy/r2
        df['Imp %'] = ((df[metric_col] - df['base_val']) / abs(df['base_val'] + 1e-9)) * 100
    
    # 3. Fallback for models with NO 'Price Only' baseline (Pure Ablation Models)
    for model in df['Specific_Model'].unique():
        mask = df['Specific_Model'] == model
        if df.loc[mask, 'base_val'].isna().all():
            mean_val = df.loc[mask, metric_col].mean()
            if is_error_metric:
                df.loc[mask, 'Imp %'] = ((mean_val - df.loc[mask, metric_col]) / mean_val) * 100
            else:
                df.loc[mask, 'Imp %'] = ((df.loc[mask, metric_col] - mean_val) / abs(mean_val + 1e-9)) * 100
                
    return df

# --- 4. LOAD ---
try:
    df_raw = load_and_unify_data()
except:
    st.error("Data Load Error")
    st.stop()

if df_raw.empty:
    st.warning("No Data Found")
    st.stop()

# --- 5. DASHBOARD UI ---

# Sidebar Filters
st.sidebar.header("🎛️ Filters")
cats = st.sidebar.multiselect("Category", sorted(df_raw['Category'].unique()), default=sorted(df_raw['Category'].unique()))

# Main Layout
st.title("📊 Thesis Results: Sentiment Analysis")

# CONTROLS IN MIDDLE
ctrl_col1, ctrl_col2 = st.columns([2, 1])

with ctrl_col1:
    view = st.radio("Select View:", ["🏆 Leaderboard", "🔬 Drill-Down", "🔥 Heatmap"], horizontal=True)

with ctrl_col2:
    metric_map = {
        "RMSE (Error ↓)": "rmse",
        "MAE (Error ↓)": "mae",
        "Directional Accuracy % (↑)": "directional_accuracy",
        "R-Squared (↑)": "r2"
    }
    metric_label = st.selectbox("Primary Metric", list(metric_map.keys()))
    metric = metric_map[metric_label]
    is_error = metric in ['rmse', 'mae']

st.divider()

# --- RECALCULATE BASED ON SELECTION ---
df = calculate_improvements(df_raw, metric_col=metric, is_error_metric=is_error)
df_filtered = df[df['Category'].isin(cats)]

# --- KPI Section ---
best_sort = True if is_error else False
best_row = df_filtered.sort_values(metric, ascending=best_sort).iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("🏆 Overall Winner", best_row['Specific_Model'])
c2.metric("Feature Set", best_row['Feature Set'])
c3.metric("Value", f"{best_row[metric]:.4f}")
st.divider()

# --- HELPER: HIGHLIGHT CHAMPIONS ---
def highlight_champions(fig, df_source, x_col, y_col):
    """Adds outlines to winners. Green = Global Best. Yellow = Row Best."""
    if df_source.empty: return fig

    if is_error:
        global_best_val = df_source[x_col].min()
    else:
        global_best_val = df_source[x_col].max()
        
    families = df_source[y_col].unique()
    
    for fam in families:
        fam_data = df_source[df_source[y_col] == fam]
        if fam_data.empty: continue
        
        if is_error:
            fam_best_val = fam_data[x_col].min()
        else:
            fam_best_val = fam_data[x_col].max()
            
        winner_row = fam_data[fam_data[x_col] == fam_best_val]
        if winner_row.empty: continue
        
        wx = winner_row.iloc[0][x_col]
        wy = winner_row.iloc[0][y_col]
        
        if fam_best_val == global_best_val:
            glow_color = '#00ff00' # Neon Green
            width = 3
            size = 15
        else:
            glow_color = '#f1c40f' # Gold
            width = 2
            size = 15
            
        fig.add_trace(go.Scatter(
            x=[wx], y=[wy], mode='markers',
            marker=dict(size=size, color='rgba(0,0,0,0)', line=dict(color=glow_color, width=width)),
            showlegend=False, hoverinfo='skip'
        ))
    return fig

# === VIEW 1: LEADERBOARD ===
if view == "🏆 Leaderboard":
    col_table, col_chart = st.columns([1, 2])
    
    with col_table:
        st.subheader("Best Model per Family")
        if is_error:
            best_indices = df_filtered.groupby('Family')[metric].idxmin()
        else:
            best_indices = df_filtered.groupby('Family')[metric].idxmax()
            
        lb = df_filtered.loc[best_indices].sort_values(metric, ascending=best_sort)
        lb_display = lb[['Family', 'Specific_Model', 'Feature Set', metric]].reset_index(drop=True)
        lb_display.index += 1
        
        if is_error:
            st.dataframe(lb_display.style.highlight_min(axis=0, subset=[metric], color='#2ecc71'))
        else:
            st.dataframe(lb_display.style.highlight_max(axis=0, subset=[metric], color='#2ecc71'))

    with col_chart:
        c_head, c_check = st.columns([3, 1])
        with c_head: st.subheader("Performance by Family")
        with c_check: hide_outliers = st.checkbox("Hide Outliers", value=True)
        
        agg_chart = df_filtered.groupby(['Family', 'Feature Set'])[metric].mean().reset_index()
        agg_chart = agg_chart.dropna(subset=[metric])
        
        if hide_outliers:
            threshold = agg_chart[metric].quantile(0.90 if is_error else 0.10)
            if is_error: agg_chart = agg_chart[agg_chart[metric] < threshold]
            else: agg_chart = agg_chart[agg_chart[metric] > threshold]

        agg_chart = agg_chart.sort_values(metric, ascending=not best_sort)

        fig_agg = px.strip(
            agg_chart, x=metric, y='Family', color='Feature Set',
            orientation='h', color_discrete_map=COLOR_MAP, height=600,
            title=f"Comparison: {metric_label}"
        )
        
        fig_agg.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=0.5, color='black')))
        fig_agg = highlight_champions(fig_agg, agg_chart, metric, 'Family')
        fig_agg.update_yaxes(categoryorder='total descending' if not is_error else 'total ascending')
        st.plotly_chart(fig_agg, use_container_width=True)

# === VIEW 2: DRILL DOWN ===
elif view == "🔬 Drill-Down":
    st.subheader("Deep Dive: Specific Model Variants")
    target_family = st.selectbox("Select Model Family", sorted(df_filtered['Family'].unique()))
    
    subset = df_filtered[df_filtered['Family'] == target_family]
    drill_data = subset.groupby(['Specific_Model', 'Feature Set'])[metric].mean().reset_index()
    drill_data = drill_data.dropna(subset=[metric])
    
    if is_error:
        best_local_val = drill_data[metric].min()
    else:
        best_local_val = drill_data[metric].max()

    fig_drill = px.bar(
        drill_data, x='Feature Set', y=metric, color='Feature Set',
        facet_col='Specific_Model',
        color_discrete_map=COLOR_MAP, 
        text_auto='.4f',
        title=f"Variant Analysis: {target_family}"
    )
    
    winner_row = drill_data[drill_data[metric] == best_local_val]
    if not winner_row.empty:
        winning_feature = winner_row.iloc[0]['Feature Set']
        fig_drill.for_each_trace(lambda t: t.update(marker_line_width=3, marker_line_color='#00ff00') 
                                 if t.name == winning_feature else None)

    fig_drill.update_xaxes(matches=None, showticklabels=True)
    fig_drill.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig_drill, use_container_width=True)

# === VIEW 3: HEATMAP ===
elif view == "🔥 Heatmap":
    st.subheader(f"Sentiment Feature Performance: {metric_label}")
    group_col = 'Family'
    
    heat_data = df_filtered.groupby([group_col, 'Feature Set'])['Imp %'].mean().reset_index()
    heat_mat = heat_data.pivot(index=group_col, columns='Feature Set', values='Imp %')
    
    cols = ['Base Sentiment', 'Full Sentiment', 'VADER', 'TextBlob', 'FinBERT']
    valid_cols = [c for c in cols if c in heat_mat.columns]
    heat_mat = heat_mat[valid_cols]
    
    heat_mat['mean'] = heat_mat.mean(axis=1)
    heat_mat = heat_mat.sort_values('mean', ascending=False).drop(columns='mean')
    
    h = max(500, len(heat_mat) * 30 + 100)
    
    if is_error:
        color_label = "Improvement % (Lower Error = Better)"
    else:
        color_label = "Improvement % (Higher = Better)"

    fig_heat = px.imshow(
        heat_mat, 
        text_auto=".1f", 
        color_continuous_scale="RdYlGn",  # Green = High (Improvement), Red = Low (Degradation)
        color_continuous_midpoint=0,      # Force 0 to be the neutral center (Yellow)
        aspect="auto", 
        height=h, 
        origin='lower',
        labels=dict(color=color_label)
    )
    st.plotly_chart(fig_heat, use_container_width=True)