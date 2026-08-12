import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import numpy as np

st.set_page_config(
    page_title="GPU Benchmark — YOLO11s",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Japanese info-dense grid aesthetic ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}
.stApp { background: #f0f0eb; }
.main .block-container { padding: 0 2rem 4rem 2rem; max-width: 100% !important; }
.main > div:first-child { padding-top: 0 !important; }
[data-testid="collapsedControl"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* page title bar */
.page-bar {
    background: #fff;
    border-bottom: 2px solid #111;
    padding: 22px 28px 20px 28px;
    margin-bottom: 0;
}
.page-title {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: #111;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.page-sub {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.64rem;
    color: #999;
    letter-spacing: 2.5px;
    margin-top: 6px;
    text-transform: uppercase;
    padding-left: 2px;
}
.page-tag {
    display: inline-block;
    background: #c00000;
    color: #fff;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    padding: 3px 9px;
    letter-spacing: 1px;
    margin-right: 12px;
    vertical-align: middle;
    line-height: 1.6;
}

/* KPI metric grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 1px;
    background: #bbb;
    border: 1px solid #bbb;
    margin: 18px 0 0 0;
}
.kpi-cell {
    background: #fff;
    padding: 16px 12px 14px 16px;
}
.kpi-val {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #111;
    line-height: 1;
}
.kpi-val-sm {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #111;
    line-height: 1;
}
.kpi-lbl {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.58rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 6px;
}

/* section header */
.sec-header {
    display: flex;
    align-items: center;
    gap: 14px;
    background: #111;
    color: #fff;
    padding: 9px 16px;
    margin: 48px 0 0 0;
    border-left: 5px solid #c00000;
}
.sec-num {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: #ffffff;
    letter-spacing: 2px;
    flex-shrink: 0;
    opacity: 0.85;
}
.sec-ttl {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.sec-desc {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.66rem;
    color: #888;
    letter-spacing: 1px;
    margin: 4px 0 14px 26px;
    border-left: 4px solid #ddd;
    padding-left: 10px;
}

/* chart label strip */
.clabel {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    background: #fafaf7;
    border: 1px solid #ccc;
    border-bottom: none;
}
.clabel-title {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.66rem;
    font-weight: 600;
    color: #222;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.clabel-tag {
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
    font-size: 0.6rem;
    color: #aaa;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Streamlit tabs — minimal font override, native layout preserved ──── */
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #c00000 !important;
}

/* ── Streamlit selectbox — styled to match grid ────────────────────────── */
.stSelectbox label {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
    font-size: 0.64rem !important;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #555 !important;
}
.stSelectbox [data-baseweb="select"] > div:first-child {
    border-radius: 0 !important;
    border: 1px solid #bbb !important;
    background: #fafaf7 !important;
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
    font-size: 0.72rem !important;
    min-height: 34px !important;
}

/* ── Streamlit radio — styled to match grid ────────────────────────────── */
.stRadio > label {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
    font-size: 0.64rem !important;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #555 !important;
}
.stRadio [data-testid="stWidgetLabel"] p {
    font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
    font-size: 0.64rem !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    def read_all_data(data_dir):
        files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
        rows = []
        for f in files:
            try:
                d = pd.read_csv(f)
                if "_python.csv" in f:
                    d["runtime"] = "Python"
                elif "_native.csv" in f:
                    d["runtime"] = "Native C++"
                else:
                    d["runtime"] = "Unknown"
                rows.append(d)
            except Exception:
                pass
        return rows

    all_rows = read_all_data("../results/phase0_synthetic")
    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)

    for col in ["resolution", "fps_mean", "fps_std", "latency_mean_ms", "latency_std_ms",
                "p95_latency_ms", "p99_latency_ms", "fps_per_watt", "avg_power_W",
                "max_power_W", "gpu_util_mean_pct", "gpu_util_max_pct",
                "gpu_memory_mean_MB", "gpu_memory_max_MB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["gpu"] = (df["gpu"]
                 .str.replace("NVIDIA GeForce RTX ", "", regex=False)
                 .str.replace("NVIDIA RTX ", "", regex=False)
                 .str.replace("NVIDIA ", "", regex=False))
    df["resolution_str"] = df["resolution"].astype(int).astype(str) + "p"
    df["fps_cv_pct"] = (df["fps_std"] / df["fps_mean"]) * 100
    return df


df = load_data()
if df.empty:
    st.error("No benchmark data found.")
    st.stop()

GPU_COLORS   = px.colors.qualitative.Bold
ENG_COLORS   = px.colors.qualitative.Vivid
RT_COLORS    = {"Python": "#2563eb", "Native C++": "#c00000"}
RT_ORDER     = {"runtime": ["Python", "Native C++"]}
GPU_ORDER    = sorted(df["gpu"].unique().tolist())
TMPL         = "plotly_white"
MONO         = "IBM Plex Mono, Courier New, monospace"

# ── Shared chart base layout ─────────────────────────────────────────────────
def base_layout(fig, height=430):
    fig.update_layout(
        height=height,
        template=TMPL,
        font=dict(family=MONO, size=10, color="#222"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(t=32, r=12, b=32, l=12),
        legend=dict(font=dict(family=MONO, size=9)),
    )
    fig.update_xaxes(tickfont=dict(family=MONO, size=9), title_font=dict(family=MONO, size=9), gridcolor="#ebebeb")
    fig.update_yaxes(tickfont=dict(family=MONO, size=9), title_font=dict(family=MONO, size=9), gridcolor="#ebebeb")
    # clean up facet labels "runtime=Python" → "Python"
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig

# ── Layout helpers ────────────────────────────────────────────────────────────
def sec(num, title, desc=""):
    desc_html = (f'<div class="sec-desc">{desc}</div>' if desc else "")
    st.markdown(
        f'<div class="sec-header"><span class="sec-num">§{num:02d}</span>'
        f'<span class="sec-ttl">{title}</span></div>{desc_html}',
        unsafe_allow_html=True,
    )

def clabel(title, tag=""):
    st.markdown(
        f'<div class="clabel"><span class="clabel-title">{title}</span>'
        f'<span class="clabel-tag">{tag}</span></div>',
        unsafe_allow_html=True,
    )

# ── Helper: group & mean keeping runtime separate ─────────────────────────────
def grp(data, by, val="fps_mean"):
    return data.groupby(by)[val].mean().reset_index()


# ════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════════════════════════
n_py  = df[df["runtime"] == "Python"]["runs"].sum() if "runs" in df.columns else len(df[df["runtime"] == "Python"])
n_nat = len(df[df["runtime"] == "Native C++"])
n_gpus = df["gpu"].nunique()
n_eng  = df["engine"].nunique()
n_prec = df["precision"].nunique()
n_res  = df["resolution"].nunique()
n_runs = len(df)

st.markdown(f"""
<div class="page-bar">
    <div class="page-title">
        <span class="page-tag">YOLO11s</span>GPU Benchmark Analysis
    </div>
    <div class="page-sub">Multi-GPU &nbsp;&nbsp;·&nbsp;&nbsp; Multi-Backend &nbsp;&nbsp;·&nbsp;&nbsp; Multi-Precision &nbsp;&nbsp;·&nbsp;&nbsp; Multi-Resolution &nbsp;&nbsp;·&nbsp;&nbsp; Python vs Native C++</div>
</div>
<div class="kpi-grid">
    <div class="kpi-cell"><div class="kpi-val">{n_gpus}</div><div class="kpi-lbl">GPUs Tested</div></div>
    <div class="kpi-cell"><div class="kpi-val">{n_eng}</div><div class="kpi-lbl">Backends</div></div>
    <div class="kpi-cell"><div class="kpi-val">{n_prec}</div><div class="kpi-lbl">Precisions</div></div>
    <div class="kpi-cell"><div class="kpi-val">{n_res}</div><div class="kpi-lbl">Resolutions</div></div>
    <div class="kpi-cell"><div class="kpi-val">2</div><div class="kpi-lbl">Runtimes</div></div>
    <div class="kpi-cell"><div class="kpi-val">{len(df[df["runtime"]=="Python"])}</div><div class="kpi-lbl">Python Runs</div></div>
    <div class="kpi-cell"><div class="kpi-val">{len(df[df["runtime"]=="Native C++"])}</div><div class="kpi-lbl">Native C++ Runs</div></div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# §01  PYTHON vs NATIVE C++ — FPS COMPARISON
# ════════════════════════════════════════════════════════════════════════════
sec(1, "Python vs Native C++ Inference",
    "Direct FPS comparison per backend · GPUs with no data for a backend are simply absent · FP16 · 640p")

d = grp(df[(df["precision"] == "fp16") & (df["resolution"] == 640)],
        ["gpu", "engine", "runtime"])
clabel("FPS by backend — Python vs Native C++", "FP16 · 640p · ALL GPUS")
fig = px.bar(d, x="gpu", y="fps_mean", color="runtime", barmode="group",
             facet_col="engine", color_discrete_map=RT_COLORS,
             labels={"fps_mean": "Mean FPS", "gpu": "GPU", "runtime": "Runtime", "engine": "Backend"},
             category_orders={"gpu": GPU_ORDER, **RT_ORDER})
base_layout(fig, 400)
st.plotly_chart(fig, use_container_width=True)

# Speedup heatmap + trend in 2-column grid
c1, c2 = st.columns(2)

# ── Speedup heatmap ──
df_py  = df[df["runtime"] == "Python"]
df_nat = df[df["runtime"] == "Native C++"]
py_fps  = df_py[df_py["precision"]=="fp16"].groupby(["gpu","engine","resolution"])["fps_mean"].mean()
nat_fps = df_nat[df_nat["precision"]=="fp16"].groupby(["gpu","engine","resolution"])["fps_mean"].mean()
speedup_all = pd.DataFrame()
if not py_fps.empty and not nat_fps.empty:
    su = (nat_fps / py_fps).dropna().reset_index()
    su.columns = ["gpu", "engine", "resolution", "speedup"]
    su["resolution_str"] = su["resolution"].astype(int).astype(str) + "p"
    speedup_all = su

with c1:
    # Resolution selector sits ABOVE the label strip — label stays flush on chart
    res_labels = [f"{int(r)}p" for r in sorted(speedup_all["resolution"].unique().tolist())]
    res_values = sorted(speedup_all["resolution"].unique().tolist())
    sel_res_lbl = st.radio("Resolution", options=res_labels, horizontal=True, key="su_res",
                           label_visibility="collapsed")
    sel_res = res_values[res_labels.index(sel_res_lbl)]
    su_f = speedup_all[speedup_all["resolution"] == sel_res]
    if not su_f.empty:
        piv = su_f.pivot_table(index="gpu", columns="engine", values="speedup").reindex(index=sorted(su_f["gpu"].unique()))
        txt = [[f"{v:.2f}×" if not np.isnan(v) else "N/A" for v in row] for row in piv.values]
        clabel(f"Native C++ speedup over Python (×) — {sel_res_lbl}", "FP16 · HEATMAP")
        fig_su = go.Figure(go.Heatmap(
            z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
            colorscale=[[0,"#d62728"],[0.5,"#fff"],[1,"#2ca02c"]], zmid=1,
            text=txt, texttemplate="%{text}", textfont={"size": 13, "family": MONO},
            hoverongaps=False, colorbar=dict(title="Speedup ×", tickfont=dict(family=MONO, size=9)),
        ))
        fig_su.update_layout(height=310, template=TMPL,
                             font=dict(family=MONO, size=10), paper_bgcolor="#fff",
                             margin=dict(t=10, r=8, b=24, l=8),
                             xaxis_title="Backend", yaxis_title="GPU")
        st.plotly_chart(fig_su, use_container_width=True)

# ── Speedup trend ──
with c2:
    clabel("Speedup trend by resolution — TensorRT FP16", "NATIVE ÷ PYTHON · PER GPU")
    if not speedup_all.empty:
        su_trt = speedup_all[speedup_all["engine"] == "tensorrt"].sort_values("resolution")
        if not su_trt.empty:
            fig_sl = px.line(su_trt, x="resolution_str", y="speedup", color="gpu", markers=True,
                             color_discrete_sequence=GPU_COLORS,
                             labels={"speedup": "Speedup (×)", "resolution_str": "Resolution", "gpu": "GPU"})
            fig_sl.add_hline(y=1, line_dash="dash", line_color="#aaa", annotation_text="1× baseline")
            fig_sl.update_traces(marker=dict(size=9))
            base_layout(fig_sl, 360)
            fig_sl.update_layout(hovermode="x unified", margin=dict(t=10, r=8, b=24, l=8))
            st.plotly_chart(fig_sl, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# §02  THROUGHPUT SCALING BY RESOLUTION
# Each backend gets its own panel. Runtime encoded as line dash style.
# ════════════════════════════════════════════════════════════════════════════
sec(2, "Throughput Scaling by Resolution",
    "FPS vs resolution per backend · solid = Python · dashed = Native C++ · FP16")

d = (df[df["precision"] == "fp16"]
     .groupby(["resolution_str", "resolution", "gpu", "engine", "runtime"])["fps_mean"]
     .mean().reset_index().sort_values("resolution"))
d["series"] = d["runtime"]  # line_dash maps to this

clabel("FPS vs resolution — all backends", "FP16 · ALL GPUS · PYTHON & NATIVE C++")
fig = px.line(d, x="resolution_str", y="fps_mean", color="gpu", line_dash="series",
              facet_col="engine", markers=True, color_discrete_sequence=GPU_COLORS,
              labels={"fps_mean": "FPS", "resolution_str": "Resolution",
                      "gpu": "GPU", "series": "Runtime", "engine": "Backend"},
              category_orders={"gpu": GPU_ORDER, **RT_ORDER})
base_layout(fig, 400)
fig.update_traces(marker=dict(size=8))
fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# §03  BACKEND & PRECISION COMPARISON
# ════════════════════════════════════════════════════════════════════════════
sec(3, "Backend & Precision Comparison",
    "Which backend and precision give the most FPS? · FP16 · 640p · TensorRT for precision")

# Backend comparison — Python left, Native right
clabel("Backend comparison — Python vs Native C++", "FP16 · 640p · ALL GPUS")
d = grp(df[(df["precision"] == "fp16") & (df["resolution"] == 640)], ["gpu", "engine", "runtime"])
fig = px.bar(d, x="engine", y="fps_mean", color="gpu", barmode="group",
             facet_col="runtime", color_discrete_sequence=GPU_COLORS,
             labels={"fps_mean": "Mean FPS", "engine": "Backend", "gpu": "GPU"},
             category_orders={"gpu": GPU_ORDER, **RT_ORDER})
base_layout(fig, 420)
st.plotly_chart(fig, use_container_width=True)

# Precision — TRT only (precision is a TRT-specific knob)
clabel("Precision impact — TensorRT only", "640p · FP32 / FP16 / INT8 · PYTHON & NATIVE C++")
d = grp(df[(df["engine"] == "tensorrt") & (df["resolution"] == 640)], ["gpu", "precision", "runtime"])
d["precision"] = pd.Categorical(d["precision"], ["fp32", "fp16", "int8"], ordered=True)
d = d.sort_values("precision")
fig = px.bar(d, x="precision", y="fps_mean", color="gpu", barmode="group",
             facet_col="runtime", color_discrete_sequence=GPU_COLORS,
             labels={"fps_mean": "Mean FPS", "precision": "Precision", "gpu": "GPU"},
             category_orders={"gpu": GPU_ORDER, **RT_ORDER})
base_layout(fig, 400)
st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# §04  POWER & EFFICIENCY
# Bubble charts separated into two columns — no cross-runtime averaging.
# ════════════════════════════════════════════════════════════════════════════
sec(4, "Power & Efficiency",
    "Throughput vs power draw · bubble size = FPS/W efficiency · TensorRT · FP16 · 640p")

c1, c2 = st.columns(2)

for col, rt in [(c1, "Python"), (c2, "Native C++")]:
    with col:
        clabel(f"Power efficiency — {rt}", "TRT · FP16 · 640p")
        d = df[(df["runtime"] == rt) & (df["engine"] == "tensorrt") &
               (df["precision"] == "fp16") & (df["resolution"] == 640)].dropna(subset=["avg_power_W", "fps_per_watt"])
        if not d.empty:
            fig = px.scatter(d, x="fps_mean", y="avg_power_W", size="fps_per_watt",
                             color="gpu", hover_name="gpu",
                             hover_data={"fps_per_watt": ":.2f"},
                             color_discrete_sequence=GPU_COLORS,
                             labels={"fps_mean": "Mean FPS", "avg_power_W": "Avg Power (W)",
                                     "fps_per_watt": "FPS/W", "gpu": "GPU"},
                             size_max=40)
            base_layout(fig, 380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No power data for {rt}.")

# FPS/Watt ranking — both runtimes, runtime in label
clabel("FPS / Watt efficiency ranking", "FP16 · 640p · ALL BACKENDS · PYTHON & NATIVE C++")
d = grp(df[(df["precision"] == "fp16") & (df["resolution"] == 640)].dropna(subset=["fps_per_watt"]),
        ["gpu", "engine", "runtime"], "fps_per_watt").sort_values("fps_per_watt")
d["label"] = d["gpu"] + "  /  " + d["engine"]
fig = px.bar(d, x="fps_per_watt", y="label", color="runtime", barmode="group",
             orientation="h", color_discrete_map=RT_COLORS,
             labels={"fps_per_watt": "FPS per Watt", "label": "GPU / Backend", "runtime": "Runtime"},
             category_orders={**RT_ORDER})
base_layout(fig, max(400, len(d["label"].unique()) * 22))
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# §05  LATENCY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
sec(5, "Latency Analysis",
    "Consistency and distribution of inference latency · FP16 · 640p or TensorRT")

# Mean vs P95 scatter — one panel per runtime
clabel("Mean vs P95 latency — diagonal = perfect consistency", "FP16 · 640p · PYTHON & NATIVE C++")
d = df[(df["precision"] == "fp16") & (df["resolution"] == 640)].dropna(subset=["latency_mean_ms", "p95_latency_ms"])
max_v = max(d["latency_mean_ms"].max(), d["p95_latency_ms"].max()) * 1.05
fig = px.scatter(d, x="latency_mean_ms", y="p95_latency_ms", color="gpu", symbol="engine",
                 facet_col="runtime", size_max=14, hover_data=["engine"],
                 color_discrete_sequence=GPU_COLORS,
                 labels={"latency_mean_ms": "Mean Latency (ms)", "p95_latency_ms": "P95 Latency (ms)",
                         "gpu": "GPU", "engine": "Backend"},
                 category_orders={**RT_ORDER})
for i in range(len(d["runtime"].unique())):
    fig.add_shape(type="line", row=1, col=i+1,
                  line=dict(dash="dash", color="#aaa"), x0=0, y0=0, x1=max_v, y1=max_v)
fig.update_traces(marker=dict(size=11, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")))
base_layout(fig, 460)
st.plotly_chart(fig, use_container_width=True)

# Latency violin — TRT FP16, one panel per runtime
clabel("Latency distribution — Mean · P95 · P99", "TENSORRT · FP16 · ALL RESOLUTIONS")
d_v = df[(df["precision"] == "fp16") & (df["engine"] == "tensorrt")][
    ["gpu", "runtime", "latency_mean_ms", "p95_latency_ms", "p99_latency_ms"]].dropna()
if not d_v.empty:
    d_m = d_v.melt(id_vars=["gpu", "runtime"],
                   value_vars=["latency_mean_ms", "p95_latency_ms", "p99_latency_ms"],
                   var_name="pct", value_name="ms")
    d_m["pct"] = d_m["pct"].map({"latency_mean_ms": "Mean", "p95_latency_ms": "P95", "p99_latency_ms": "P99"})
    fig = px.violin(d_m, x="gpu", y="ms", color="pct", facet_row="runtime",
                    box=True, points="all",
                    color_discrete_sequence=["#2563eb", "#c00000", "#7209b7"],
                    labels={"gpu": "GPU", "ms": "Latency (ms)", "pct": "Percentile"},
                    category_orders={"gpu": GPU_ORDER, **RT_ORDER})
    base_layout(fig, 640)
    fig.update_layout(violinmode="group")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# §06  HARDWARE METRICS
# ════════════════════════════════════════════════════════════════════════════
sec(6, "Hardware Metrics",
    "GPU utilization and VRAM usage · each tab or panel isolates one runtime")

# Heatmap — tabs act as the top navigation; clabel lives INSIDE each tab, flush on its chart
d_heat = df[(df["precision"] == "fp16") & (df["resolution"] == 640)]
heat_tabs = st.tabs(["Python", "Native C++", "Diff (Native − Python)"])
pivots = {}
for i, rt in enumerate(["Python", "Native C++"]):
    drt = d_heat[d_heat["runtime"] == rt]
    if not drt.empty:
        piv = drt.groupby(["gpu", "engine"])["fps_mean"].mean().unstack()
        piv = piv.reindex(index=sorted(piv.index))
        pivots[rt] = piv
        txt = [[f"{v:.0f}" if not np.isnan(v) else "N/A" for v in row] for row in piv.values]
        with heat_tabs[i]:
            clabel(f"GPU × Backend — {rt} Runtime", "FP16 · 640p · MEAN FPS")
            fig_h = go.Figure(go.Heatmap(
                z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
                colorscale="Viridis", text=txt, texttemplate="%{text}",
                textfont={"size": 13, "family": MONO}, hoverongaps=False,
                colorbar=dict(title="FPS", tickfont=dict(family=MONO, size=9)),
            ))
            fig_h.update_layout(height=380, template=TMPL,
                                font=dict(family=MONO, size=10), paper_bgcolor="#fff",
                                margin=dict(t=0, r=8, b=24, l=8),
                                xaxis_title="Backend", yaxis_title="GPU")
            st.plotly_chart(fig_h, use_container_width=True)

if "Python" in pivots and "Native C++" in pivots:
    pa, pb = pivots["Python"].align(pivots["Native C++"], fill_value=np.nan)
    diff = pb - pa
    txt_d = [[(f"+{v:.0f}" if v >= 0 else f"{v:.0f}") if not np.isnan(v) else "N/A" for v in row] for row in diff.values]
    with heat_tabs[2]:
        clabel("Diff: Native C++ minus Python", "FP16 · 640p · GREEN = NATIVE FASTER · RED = PYTHON FASTER")
        fig_d = go.Figure(go.Heatmap(
            z=diff.values, x=diff.columns.tolist(), y=diff.index.tolist(),
            colorscale="RdYlGn", zmid=0, text=txt_d, texttemplate="%{text}",
            textfont={"size": 13, "family": MONO}, hoverongaps=False,
            colorbar=dict(title="ΔFPS", tickfont=dict(family=MONO, size=9)),
        ))
        fig_d.update_layout(height=380, template=TMPL,
                            font=dict(family=MONO, size=10), paper_bgcolor="#fff",
                            margin=dict(t=0, r=8, b=24, l=8),
                            xaxis_title="Backend", yaxis_title="GPU")
        st.plotly_chart(fig_d, use_container_width=True)

# GPU utilization + memory in 2-column grid
c1, c2 = st.columns(2)

with c1:
    clabel("GPU utilization by backend", "FP16 · 640p · PYTHON & NATIVE C++")
    d = df[(df["resolution"] == 640) & (df["precision"] == "fp16")].dropna(subset=["gpu_util_mean_pct"])
    if not d.empty:
        d_g = grp(d, ["gpu", "engine", "runtime"], "gpu_util_mean_pct")
        fig = px.bar(d_g, x="engine", y="gpu_util_mean_pct", color="gpu",
                     barmode="group", facet_col="runtime", color_discrete_sequence=GPU_COLORS,
                     labels={"engine": "Backend", "gpu_util_mean_pct": "GPU Util (%)", "gpu": "GPU"},
                     category_orders={"gpu": GPU_ORDER, **RT_ORDER})
        base_layout(fig, 380)
        fig.add_hline(y=100, line_dash="dot", line_color="#c00000", annotation_text="100%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("GPU utilization data unavailable.")

with c2:
    clabel("GPU memory footprint", "ALL RESOLUTIONS · BACKEND / PRECISION · PYTHON ONLY")
    # Memory is model/precision dependent; show Python as reference (same model weights)
    d = df_py.dropna(subset=["gpu_memory_mean_MB"])
    if not d.empty:
        d_g = grp(d, ["gpu", "engine", "precision"], "gpu_memory_mean_MB")
        d_g["gpu_memory_mean_GB"] = d_g["gpu_memory_mean_MB"] / 1024
        d_g["cfg"] = d_g["engine"] + " / " + d_g["precision"]
        fig = px.bar(d_g, x="gpu", y="gpu_memory_mean_GB", color="cfg", barmode="group",
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     labels={"gpu": "GPU", "gpu_memory_mean_GB": "VRAM (GB)", "cfg": "Backend / Precision"},
                     category_orders={"gpu": GPU_ORDER})
        base_layout(fig, 380)
        fig.update_layout(legend=dict(orientation="h", y=1.08, x=1, xanchor="right", font=dict(size=8)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("GPU memory data unavailable.")


# ════════════════════════════════════════════════════════════════════════════
# §07  STABILITY & POWER CONSUMPTION
# ════════════════════════════════════════════════════════════════════════════
sec(7, "Stability & Power Consumption",
    "FPS coefficient of variation · average vs peak power draw · FP16")

# FPS stability CoV — faceted by resolution, Python vs Native with color
clabel("FPS stability — coefficient of variation", "FP16 · LOWER = MORE STABLE · FACET = RESOLUTION")
d = df[df["precision"] == "fp16"].dropna(subset=["fps_cv_pct"])
d_g = grp(d, ["gpu", "engine", "resolution_str", "runtime"], "fps_cv_pct").sort_values("fps_cv_pct")
d_g["label"] = d_g["gpu"] + " [" + d_g["runtime"].str.replace("Native C++", "Native") + "]"
fig = px.bar(d_g, x="fps_cv_pct", y="label", color="engine",
             facet_col="resolution_str", orientation="h",
             color_discrete_sequence=ENG_COLORS,
             labels={"fps_cv_pct": "FPS CoV (%)", "label": "GPU [Runtime]",
                     "engine": "Backend", "resolution_str": "Resolution"})
base_layout(fig, max(460, len(d_g["label"].unique()) * 18))
st.plotly_chart(fig, use_container_width=True)

# Power avg vs peak — Python vs Native in two columns
c1, c2 = st.columns(2)
for col, rt in [(c1, "Python"), (c2, "Native C++")]:
    with col:
        clabel(f"Avg vs peak power — {rt}", "FP16 · FACET = BACKEND")
        d = (df[(df["runtime"] == rt) & (df["precision"] == "fp16")]
             .dropna(subset=["avg_power_W", "max_power_W"]))
        if not d.empty:
            d_g = d.groupby(["gpu", "engine"])[["avg_power_W", "max_power_W"]].mean().reset_index()
            d_m = d_g.melt(id_vars=["gpu", "engine"], value_vars=["avg_power_W", "max_power_W"],
                           var_name="ptype", value_name="watts")
            d_m["ptype"] = d_m["ptype"].map({"avg_power_W": "Avg", "max_power_W": "Peak"})
            fig = px.bar(d_m, x="gpu", y="watts", color="ptype", facet_col="engine",
                         barmode="group", color_discrete_sequence=["#2563eb", "#c00000"],
                         labels={"gpu": "GPU", "watts": "Power (W)",
                                 "ptype": "Power Type", "engine": "Backend"},
                         category_orders={"gpu": GPU_ORDER})
            base_layout(fig, 380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No power data for {rt}.")
