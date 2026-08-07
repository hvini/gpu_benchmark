import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import numpy as np

st.set_page_config(
    page_title="Comprehensive GPU Benchmark",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
        [data-testid="collapsedControl"] { display: none; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .section-title {
            text-align: center;
            font-family: sans-serif;
            color: #1a1a2e;
            margin-top: 50px;
            margin-bottom: 8px;
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .section-subtitle {
            text-align: center;
            color: #666;
            font-family: sans-serif;
            margin-bottom: 20px;
            font-size: 0.95rem;
        }
        .divider {
            border: none;
            border-top: 1px solid #e8e8e8;
            margin: 40px 0 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data_dir = "data"
    all_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception:
            pass

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)

    numeric_cols = [
        'resolution', 'fps_mean', 'fps_std', 'latency_mean_ms', 'latency_std_ms',
        'p95_latency_ms', 'p99_latency_ms', 'fps_per_watt', 'avg_power_W', 'max_power_W',
        'gpu_util_mean_pct', 'gpu_util_max_pct', 'gpu_memory_mean_MB', 'gpu_memory_max_MB'
    ]
    for col in numeric_cols:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    if 'gpu' in combined_df.columns:
        combined_df['gpu'] = combined_df['gpu'].str.replace('NVIDIA GeForce RTX ', '')
        combined_df['gpu'] = combined_df['gpu'].str.replace('NVIDIA RTX ', '')
        combined_df['gpu'] = combined_df['gpu'].str.replace('NVIDIA ', '')

    combined_df['resolution_str'] = combined_df['resolution'].astype(str) + "p"

    # Coefficient of variation (lower = more stable)
    combined_df['fps_cv_pct'] = (combined_df['fps_std'] / combined_df['fps_mean']) * 100
    combined_df['latency_jitter_pct'] = (combined_df['latency_std_ms'] / combined_df['latency_mean_ms']) * 100

    return combined_df

df = load_data()

if df.empty:
    st.error("No benchmark data found.")
    st.stop()

gpu_colors = px.colors.qualitative.Bold
engine_colors = px.colors.qualitative.Vivid
plotly_template = "plotly_white"

gpu_order = sorted(df['gpu'].unique().tolist())

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0;'>
    <h1 style='font-family:sans-serif; font-size:2.4rem; color:#1a1a2e; margin:0; font-weight:800; letter-spacing:-1px;'>
        GPU Benchmark Analysis
    </h1>
    <p style='color:#555; font-size:1.05rem; margin-top:8px;'>
        YOLO11s &middot; Multi-GPU &middot; Multi-Backend &middot; Multi-Precision &middot; Multi-Resolution
    </p>
</div>
<hr class='divider'>
""", unsafe_allow_html=True)

# =========================================================
# SUMMARY METRICS
# =========================================================
cols = st.columns(5)
metrics = [
    ("GPUs", df['gpu'].nunique()),
    ("Backends", df['engine'].nunique()),
    ("Precisions", df['precision'].nunique()),
    ("Resolutions", df['resolution'].nunique()),
    ("Benchmark Runs", len(df)),
]
for col, (label, val) in zip(cols, metrics):
    col.metric(label, val)

# =========================================================
# PLOT 1: FPS SCALING WITH RESOLUTION
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Throughput Scaling by Resolution</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>How FPS degrades as input resolution increases. <em>(Filtered to FP16 Precision)</em></p>", unsafe_allow_html=True)

df_line = df[df['precision'] == 'fp16'].copy()
if not df_line.empty:
    df_line = df_line.groupby(['resolution_str', 'resolution', 'gpu', 'engine'])['fps_mean'].mean().reset_index()
    df_line = df_line.sort_values(by='resolution')

    fig1 = px.line(
        df_line,
        x='resolution_str',
        y='fps_mean',
        color='gpu',
        line_dash='engine',
        markers=True,
        color_discrete_sequence=gpu_colors,
        labels={'fps_mean': 'Mean FPS', 'resolution_str': 'Input Resolution', 'gpu': 'GPU', 'engine': 'Backend'},
        template=plotly_template
    )
    fig1.update_layout(height=500, hovermode='x unified')
    fig1.update_traces(marker=dict(size=10))
    st.plotly_chart(fig1, use_container_width=True)

# =========================================================
# PLOT 2: LATENCY STABILITY
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Latency Stability: Mean vs 95th Percentile</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Points closer to the diagonal = low jitter and consistent inference. <em>(All Data)</em></p>", unsafe_allow_html=True)

fig2 = px.scatter(
    df,
    x='latency_mean_ms',
    y='p95_latency_ms',
    color='gpu',
    symbol='engine',
    size_max=15,
    hover_data=['precision', 'resolution_str'],
    color_discrete_sequence=gpu_colors,
    labels={
        'latency_mean_ms': 'Mean Latency (ms)',
        'p95_latency_ms': '95th Percentile Latency (ms)',
        'gpu': 'GPU',
        'engine': 'Backend'
    },
    template=plotly_template
)
max_val = max(df['latency_mean_ms'].max(), df['p95_latency_ms'].max())
fig2.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, y0=0, x1=max_val, y1=max_val)
fig2.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
fig2.update_layout(height=600)
st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# PLOT 3: IMPACT OF PRECISION
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Impact of Precision / Quantization on Throughput</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Comparing FPS across FP32, FP16, and INT8 for each GPU. <em>(Averaged across Backends, 1280p)</em></p>", unsafe_allow_html=True)

df_bar = df[df['resolution'] == 1280].copy()
if not df_bar.empty:
    df_bar = df_bar.groupby(['gpu', 'precision'])['fps_mean'].mean().reset_index()
    df_bar['precision'] = pd.Categorical(df_bar['precision'], categories=['fp32', 'fp16', 'int8'], ordered=True)
    df_bar = df_bar.sort_values(by='precision')

    fig3 = px.bar(
        df_bar,
        x='fps_mean',
        y='precision',
        color='gpu',
        barmode='group',
        orientation='h',
        color_discrete_sequence=gpu_colors,
        labels={'fps_mean': 'Mean FPS', 'precision': 'Precision Type', 'gpu': 'GPU'},
        template=plotly_template
    )
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# PLOT 4: POWER EFFICIENCY TRADE-OFFS
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Power Efficiency Trade-offs</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Raw throughput vs power draw. Bubble size = FPS/Watt efficiency. <em>(TensorRT, FP16, 640p)</em></p>", unsafe_allow_html=True)

df_bubble = df[(df['engine'] == 'tensorrt') & (df['precision'] == 'fp16') & (df['resolution'] == 640)].copy()

if not df_bubble.empty and 'avg_power_W' in df_bubble.columns and not df_bubble['avg_power_W'].isna().all():
    df_bubble = df_bubble.dropna(subset=['avg_power_W', 'fps_per_watt'])

    fig4 = px.scatter(
        df_bubble,
        x='fps_mean',
        y='avg_power_W',
        size='fps_per_watt',
        color='gpu',
        hover_name='gpu',
        color_discrete_sequence=gpu_colors,
        labels={
            'fps_mean': 'Raw Throughput (FPS)',
            'avg_power_W': 'Average Power Drawn (W)',
            'fps_per_watt': 'Efficiency (FPS/W)',
            'gpu': 'GPU'
        },
        template=plotly_template,
        size_max=40
    )
    fig4.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
    fig4.update_layout(height=600)
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Power consumption data not available for this configuration.")

# =========================================================
# PLOT 5: BACKEND x GPU HEATMAP
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Backend x GPU Throughput Heatmap</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Mean FPS for each GPU + Backend combination. <em>(FP16, 640p)</em></p>", unsafe_allow_html=True)

df_heat = df[(df['precision'] == 'fp16') & (df['resolution'] == 640)].copy()
if not df_heat.empty:
    pivot = df_heat.groupby(['gpu', 'engine'])['fps_mean'].mean().unstack(fill_value=0)
    pivot = pivot.reindex(index=sorted(pivot.index))

    fig5 = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale='Viridis',
        text=np.round(pivot.values, 1),
        texttemplate="%{text}",
        textfont={"size": 14},
        hoverongaps=False,
        colorbar=dict(title='FPS'),
    ))
    fig5.update_layout(
        height=400,
        xaxis_title='Backend',
        yaxis_title='GPU',
        template=plotly_template,
    )
    st.plotly_chart(fig5, use_container_width=True)

# =========================================================
# PLOT 6: GPU UTILIZATION
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>GPU Utilization by Configuration</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Mean GPU utilization % per backend and precision. <em>(640p Resolution)</em></p>", unsafe_allow_html=True)

df_util = df[df['resolution'] == 640].dropna(subset=['gpu_util_mean_pct']).copy()
if not df_util.empty:
    df_util_grp = df_util.groupby(['gpu', 'engine', 'precision'])['gpu_util_mean_pct'].mean().reset_index()
    df_util_grp['config'] = df_util_grp['engine'] + ' / ' + df_util_grp['precision']

    fig6 = px.bar(
        df_util_grp,
        x='config',
        y='gpu_util_mean_pct',
        color='gpu',
        barmode='group',
        color_discrete_sequence=gpu_colors,
        labels={
            'config': 'Backend / Precision',
            'gpu_util_mean_pct': 'Mean GPU Utilization (%)',
            'gpu': 'GPU'
        },
        template=plotly_template
    )
    fig6.update_layout(height=500, xaxis_tickangle=-30)
    fig6.add_hline(y=100, line_dash='dot', line_color='red', annotation_text='100% Saturation')
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("GPU utilization data not available.")

# =========================================================
# PLOT 7: GPU MEMORY FOOTPRINT
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>GPU Memory Footprint</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Mean VRAM usage across backends and precisions. <em>(All Resolutions)</em></p>", unsafe_allow_html=True)

df_mem = df.dropna(subset=['gpu_memory_mean_MB']).copy()
if not df_mem.empty:
    df_mem_grp = df_mem.groupby(['gpu', 'engine', 'precision'])['gpu_memory_mean_MB'].mean().reset_index()
    df_mem_grp['gpu_memory_mean_GB'] = df_mem_grp['gpu_memory_mean_MB'] / 1024
    df_mem_grp['config'] = df_mem_grp['engine'] + ' / ' + df_mem_grp['precision']

    fig7 = px.bar(
        df_mem_grp,
        x='gpu',
        y='gpu_memory_mean_GB',
        color='config',
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={
            'gpu': 'GPU',
            'gpu_memory_mean_GB': 'Mean VRAM Usage (GB)',
            'config': 'Backend / Precision'
        },
        template=plotly_template,
        category_orders={'gpu': gpu_order}
    )
    fig7.update_layout(height=500, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    st.plotly_chart(fig7, use_container_width=True)
else:
    st.info("GPU memory data not available.")

# =========================================================
# PLOT 8: LATENCY DISTRIBUTION (VIOLIN)
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Latency Distribution: Mean, P95 & P99</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Distribution of latency percentiles across all configs per GPU. Wider = higher variance. <em>(All Data)</em></p>", unsafe_allow_html=True)

df_violin = df[['gpu', 'latency_mean_ms', 'p95_latency_ms', 'p99_latency_ms']].dropna().copy()
if not df_violin.empty:
    df_melt = df_violin.melt(
        id_vars='gpu',
        value_vars=['latency_mean_ms', 'p95_latency_ms', 'p99_latency_ms'],
        var_name='metric',
        value_name='latency_ms'
    )
    metric_labels = {
        'latency_mean_ms': 'Mean Latency',
        'p95_latency_ms': 'P95 Latency',
        'p99_latency_ms': 'P99 Latency'
    }
    df_melt['metric'] = df_melt['metric'].map(metric_labels)

    fig8 = px.violin(
        df_melt,
        x='gpu',
        y='latency_ms',
        color='metric',
        box=True,
        points='all',
        color_discrete_sequence=['#4361ee', '#f72585', '#7209b7'],
        labels={
            'gpu': 'GPU',
            'latency_ms': 'Latency (ms)',
            'metric': 'Percentile'
        },
        template=plotly_template,
        category_orders={'gpu': gpu_order}
    )
    fig8.update_layout(height=600, violinmode='group')
    st.plotly_chart(fig8, use_container_width=True)

# =========================================================
# PLOT 9: FPS STABILITY RANKING (CoV)
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>FPS Stability Ranking</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Coefficient of variation (FPS std / FPS mean x 100). Lower = more stable. <em>(FP16)</em></p>", unsafe_allow_html=True)

df_cv = df[df['precision'] == 'fp16'].dropna(subset=['fps_cv_pct']).copy()
if not df_cv.empty:
    df_cv_grp = df_cv.groupby(['gpu', 'engine', 'resolution_str'])['fps_cv_pct'].mean().reset_index()
    df_cv_grp = df_cv_grp.sort_values('fps_cv_pct')

    fig9 = px.bar(
        df_cv_grp,
        x='fps_cv_pct',
        y='gpu',
        color='engine',
        facet_col='resolution_str',
        orientation='h',
        color_discrete_sequence=engine_colors,
        labels={
            'fps_cv_pct': 'FPS Coeff. of Variation (%)',
            'gpu': 'GPU',
            'engine': 'Backend',
            'resolution_str': 'Resolution'
        },
        template=plotly_template
    )
    fig9.update_layout(height=450)
    st.plotly_chart(fig9, use_container_width=True)
else:
    st.info("FPS stability data not available.")

# =========================================================
# PLOT 10: FPS/WATT EFFICIENCY RANKING
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>FPS/Watt Efficiency Ranking</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Energy efficiency across all GPU and backend combinations. Higher = more efficient. <em>(FP16, 640p)</em></p>", unsafe_allow_html=True)

df_eff = df[(df['precision'] == 'fp16') & (df['resolution'] == 640)].dropna(subset=['fps_per_watt']).copy()
if not df_eff.empty:
    df_eff_grp = df_eff.groupby(['gpu', 'engine'])['fps_per_watt'].mean().reset_index()
    df_eff_grp = df_eff_grp.sort_values('fps_per_watt', ascending=False)
    df_eff_grp['label'] = df_eff_grp['gpu'] + ' / ' + df_eff_grp['engine']

    fig10 = px.bar(
        df_eff_grp,
        x='fps_per_watt',
        y='label',
        color='gpu',
        orientation='h',
        color_discrete_sequence=gpu_colors,
        labels={
            'fps_per_watt': 'FPS per Watt',
            'label': 'GPU / Backend',
            'gpu': 'GPU'
        },
        template=plotly_template
    )
    fig10.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig10, use_container_width=True)
else:
    st.info("FPS/Watt efficiency data not available.")

# =========================================================
# PLOT 11: AVERAGE VS PEAK POWER
# =========================================================
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<h2 class='section-title'>Average vs Peak Power Consumption</h2>", unsafe_allow_html=True)
st.markdown("<p class='section-subtitle'>Comparing sustained vs peak power draw per GPU and backend. <em>(FP16)</em></p>", unsafe_allow_html=True)

df_pwr = df[df['precision'] == 'fp16'].dropna(subset=['avg_power_W', 'max_power_W']).copy()
if not df_pwr.empty:
    df_pwr_grp = df_pwr.groupby(['gpu', 'engine'])[['avg_power_W', 'max_power_W']].mean().reset_index()
    df_pwr_melt = df_pwr_grp.melt(
        id_vars=['gpu', 'engine'],
        value_vars=['avg_power_W', 'max_power_W'],
        var_name='power_type',
        value_name='watts'
    )
    df_pwr_melt['power_type'] = df_pwr_melt['power_type'].map({
        'avg_power_W': 'Average Power',
        'max_power_W': 'Peak Power'
    })

    fig11 = px.bar(
        df_pwr_melt,
        x='gpu',
        y='watts',
        color='power_type',
        facet_col='engine',
        barmode='group',
        color_discrete_sequence=['#4cc9f0', '#f72585'],
        labels={
            'gpu': 'GPU',
            'watts': 'Power (W)',
            'power_type': 'Power Type',
            'engine': 'Backend'
        },
        template=plotly_template,
        category_orders={'gpu': gpu_order}
    )
    fig11.update_layout(height=500)
    st.plotly_chart(fig11, use_container_width=True)
else:
    st.info("Power data not available.")
