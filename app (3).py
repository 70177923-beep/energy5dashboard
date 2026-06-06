import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OWID Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- base ---------- */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main { background: #0b1a2b !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2137 0%, #0a3d2e 100%) !important;
    border-right: 1px solid #1e3a50;
}

/* ---------- text ---------- */
h1,h2,h3,h4,h5,h6,p,span,label,
[data-testid="stMarkdownContainer"] * { color: #e0f0ff !important; }

/* ---------- metric cards ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0e3460 0%, #0a5c45 100%);
    border: 1px solid #1e5a7a;
    border-radius: 14px;
    padding: 18px 20px !important;
}
[data-testid="stMetricValue"] { font-size: 2rem !important; color: #00e5ff !important; }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #7ecfea !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ---------- selectbox / slider ---------- */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] { background: transparent !important; }
.stSelectbox select, .stSlider { color: #00e5ff !important; }

/* ---------- tabs ---------- */
[data-testid="stTabs"] button {
    background: #0e2a40 !important;
    border-radius: 10px 10px 0 0 !important;
    color: #7ecfea !important;
    border: 1px solid #1e3a50 !important;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg,#0e3460,#0a5c45) !important;
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
}

/* ---------- buttons ---------- */
.stButton > button {
    background: linear-gradient(90deg,#0077b6,#00b386) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700;
    padding: 10px 28px !important;
}
.stButton > button:hover { opacity: 0.88; }

/* ---------- success / info boxes ---------- */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ---------- sidebar labels ---------- */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label { color: #7ecfea !important; font-size:0.78rem; }

/* ---------- divider ---------- */
hr { border-color: #1e3a50 !important; }

/* ── KPI top card strip ── */
.kpi-card {
    background: linear-gradient(135deg,#0e3460 0%,#0a5c45 100%);
    border: 1px solid #1e5a7a;
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
}
.kpi-card .kpi-label { font-size:0.75rem; color:#7ecfea; margin-bottom:4px; }
.kpi-card .kpi-value { font-size:1.9rem; font-weight:800; color:#00e5ff; }
.kpi-card .kpi-sub   { font-size:0.75rem; color:#a0cce0; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    return pd.read_csv(url)

def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

with st.spinner("⏳ Loading OWID Energy Data..."):
    df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 24px;'>
        <div style='font-size:2.2rem;'>⚡</div>
        <div style='font-size:1rem; font-weight:800; color:#00e5ff; letter-spacing:1px;'>ENERGY</div>
        <div style='font-size:0.68rem; color:#7ecfea; letter-spacing:2px;'>ANALYTICS DASHBOARD</div>
        <div style='font-size:0.62rem; color:#4a8fa8; margin-top:4px;'>SAP ID: 70177923</div>
    </div>
    <hr style='margin-bottom:20px;'>
    """, unsafe_allow_html=True)

    pages = ["🏠 Overview", "🌿 Renewables", "🛢️ Fossil Fuels", "🤖 ML Prediction"]
    page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem;color:#7ecfea;margin-bottom:8px;'>FILTERS</div>", unsafe_allow_html=True)

    countries = ["World"] + sorted(df["country"].dropna().unique().tolist())
    selected_country = st.selectbox("Country", countries)

    ymin, ymax = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider("Year Range", ymin, ymax, (2000, 2023))

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.7rem;color:#4a8fa8;text-align:center;'>Data: Our World in Data<br>{df.shape[0]:,} records · {df['country'].nunique()} countries</div>", unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
filtered = df[
    (df["country"] == selected_country) &
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
].copy()

# ── Helper: dark matplotlib style ────────────────────────────────────────────
def dark_fig(figsize=(12, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0b1a2b")
    ax.set_facecolor("#0e2a40")
    ax.tick_params(colors="#7ecfea", labelsize=8)
    ax.xaxis.label.set_color("#7ecfea")
    ax.yaxis.label.set_color("#7ecfea")
    ax.title.set_color("#00e5ff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a50")
    ax.grid(color="#1e3a50", linewidth=0.5, alpha=0.7)
    return fig, ax

def dark_fig2(figsize=(12, 4)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#0b1a2b")
    for ax in axes:
        ax.set_facecolor("#0e2a40")
        ax.tick_params(colors="#7ecfea", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a50")
        ax.grid(color="#1e3a50", linewidth=0.5, alpha=0.7)
    return fig, axes

# ── Safe last value ───────────────────────────────────────────────────────────
def last_val(col, fmt=".1f", fallback="N/A"):
    if col not in filtered.columns:
        return fallback
    s = filtered[col].dropna()
    if s.empty:
        return fallback
    return f"{s.iloc[-1]:{fmt}}"

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown(f"<h2 style='color:#00e5ff;margin-bottom:4px;'>🌍 {selected_country} — Energy Overview</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#4a8fa8;font-size:0.82rem;margin-top:0;'>Period: {year_range[0]} – {year_range[1]}</p>", unsafe_allow_html=True)

    # ── Top KPI strip ─────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    kpis = [
        (k1, "renewables_share_energy", "Renewable Share", "%", ".1f"),
        (k2, "fossil_share_energy",     "Fossil Share",    "%", ".1f"),
        (k3, "energy_per_capita",       "Energy / Capita", " kWh", ".0f"),
        (k4, "solar_share_elec",        "Solar Share",     "%", ".1f"),
    ]
    for col_ui, field, label, unit, fmt in kpis:
        val = last_val(field, fmt)
        display = f"{val}{unit}" if val != "N/A" else "N/A"
        col_ui.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{display}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main trend chart + donut ──────────────────────────────────────────────
    left, right = st.columns([2, 1])

    with left:
        st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>ENERGY MIX TRENDS</div>", unsafe_allow_html=True)
        trend_map = {
            "renewables_share_energy": ("#00e5ff", "Renewables"),
            "fossil_share_energy":     ("#ff6b35", "Fossil"),
            "solar_share_elec":        ("#ffe066", "Solar"),
            "wind_share_elec":         ("#56cfe1", "Wind"),
            "hydro_share_elec":        ("#48bfe3", "Hydro"),
        }
        avail = safe_cols(filtered, list(trend_map.keys()))
        if avail:
            fig, ax = dark_fig((10, 3.8))
            for col in avail:
                d = filtered[["year", col]].dropna()
                clr, lbl = trend_map[col]
                ax.plot(d["year"], d[col], color=clr, linewidth=2.2, label=lbl)
                ax.fill_between(d["year"], d[col], alpha=0.08, color=clr)
            ax.set_xlabel("Year"); ax.set_ylabel("Share (%)")
            ax.set_title("")
            leg = ax.legend(fontsize=8, framealpha=0.2, labelcolor="white")
            leg.get_frame().set_facecolor("#0e2a40")
            st.pyplot(fig)
        else:
            st.info("No trend data available.")

    with right:
        st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>ENERGY MIX BREAKDOWN</div>", unsafe_allow_html=True)
        donut_fields = safe_cols(filtered, [
            "renewables_share_energy", "fossil_share_energy",
            "nuclear_share_energy", "solar_share_energy",
            "wind_share_energy"
        ])
        if donut_fields:
            vals, lbls, clrs = [], [], []
            cmap = {"renewables_share_energy": "#00e5ff",
                    "fossil_share_energy":     "#ff6b35",
                    "nuclear_share_energy":    "#a855f7",
                    "solar_share_energy":      "#ffe066",
                    "wind_share_energy":       "#56cfe1"}
            for f in donut_fields:
                v = filtered[f].dropna()
                if not v.empty:
                    vals.append(v.iloc[-1])
                    lbls.append(f.replace("_share_energy","").replace("_share","").capitalize())
                    clrs.append(cmap.get(f,"#888"))
            if vals:
                fig2, ax2 = plt.subplots(figsize=(3.8, 3.8))
                fig2.patch.set_facecolor("#0e2a40")
                ax2.set_facecolor("#0e2a40")
                wedges, _ = ax2.pie(vals, colors=clrs, startangle=90,
                                    wedgeprops=dict(width=0.55, edgecolor="#0b1a2b", linewidth=2))
                ax2.text(0, 0, f"{vals[0]:.1f}%\n{lbls[0]}", ha="center", va="center",
                         color="#00e5ff", fontsize=11, fontweight="bold")
                ax2.legend(wedges, lbls, loc="lower center", bbox_to_anchor=(0.5,-0.08),
                           ncol=2, fontsize=7, framealpha=0, labelcolor="white")
                st.pyplot(fig2)

    # ── Second row: bar + line ────────────────────────────────────────────────
    st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin:14px 0 6px;'>ELECTRICITY SOURCES BY YEAR</div>", unsafe_allow_html=True)
    bar_fields = safe_cols(filtered, ["solar_share_elec","wind_share_elec","hydro_share_elec","nuclear_share_elec"])

    if bar_fields:
        bar_df = filtered[["year"] + bar_fields].dropna()
        if not bar_df.empty:
            fig3, ax3 = dark_fig((12, 3.2))
            bar_colors = ["#ffe066","#56cfe1","#48bfe3","#a855f7"]
            width = 0.18; x = np.arange(len(bar_df))
            for i, (f, c) in enumerate(zip(bar_fields, bar_colors)):
                ax3.bar(x + i*width, bar_df[f], width, color=c,
                        alpha=0.85, label=f.replace("_share_elec","").capitalize())
            ax3.set_xticks(x + width); ax3.set_xticklabels(bar_df["year"].astype(int), rotation=45, fontsize=7)
            ax3.set_ylabel("Share (%)")
            leg3 = ax3.legend(fontsize=8, framealpha=0.2, labelcolor="white")
            leg3.get_frame().set_facecolor("#0e2a40")
            st.pyplot(fig3)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RENEWABLES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌿 Renewables":

    st.markdown("<h2 style='color:#00e5ff;'>🌿 Renewable Energy Deep Dive</h2>", unsafe_allow_html=True)

    # Top 10 countries comparison
    latest_year = df["year"].max()
    latest_df = df[df["year"] == latest_year][["country","renewables_share_energy","solar_share_elec","wind_share_elec"]].dropna()
    top10 = latest_df.nlargest(10, "renewables_share_energy")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>TOP 10 COUNTRIES — RENEWABLE SHARE</div>", unsafe_allow_html=True)
        fig4, ax4 = dark_fig((6, 4))
        bars = ax4.barh(top10["country"], top10["renewables_share_energy"],
                        color="#00e5ff", alpha=0.85, edgecolor="#0b1a2b")
        ax4.set_xlabel("Renewable Share (%)")
        for bar, val in zip(bars, top10["renewables_share_energy"]):
            ax4.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                     f"{val:.1f}%", va="center", color="#00e5ff", fontsize=8)
        st.pyplot(fig4)

    with col_b:
        st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>SOLAR vs WIND — TOP 10</div>", unsafe_allow_html=True)
        fig5, ax5 = dark_fig((6, 4))
        x5 = np.arange(len(top10)); w=0.35
        ax5.bar(x5-w/2, top10["solar_share_elec"], w, color="#ffe066", alpha=0.85, label="Solar")
        ax5.bar(x5+w/2, top10["wind_share_elec"],  w, color="#56cfe1", alpha=0.85, label="Wind")
        ax5.set_xticks(x5); ax5.set_xticklabels(top10["country"], rotation=45, ha="right", fontsize=7)
        ax5.set_ylabel("Share (%)")
        leg5 = ax5.legend(fontsize=8, framealpha=0.2, labelcolor="white")
        leg5.get_frame().set_facecolor("#0e2a40")
        st.pyplot(fig5)

    # Selected country trend
    st.markdown(f"<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin:14px 0 6px;'>RENEWABLE GROWTH — {selected_country}</div>", unsafe_allow_html=True)
    ren_cols = safe_cols(filtered, ["renewables_share_energy","solar_share_elec","wind_share_elec","hydro_share_elec"])
    if ren_cols:
        fig6, ax6 = dark_fig((12, 3))
        cmap6 = {"renewables_share_energy":"#00e5ff","solar_share_elec":"#ffe066",
                  "wind_share_elec":"#56cfe1","hydro_share_elec":"#48bfe3"}
        for c in ren_cols:
            d = filtered[["year",c]].dropna()
            ax6.plot(d["year"], d[c], color=cmap6.get(c,"#aaa"), linewidth=2,
                     label=c.replace("_share_elec","").replace("_share_energy","").capitalize())
            ax6.fill_between(d["year"], d[c], alpha=0.08, color=cmap6.get(c,"#aaa"))
        ax6.set_xlabel("Year"); ax6.set_ylabel("%")
        leg6 = ax6.legend(fontsize=8, framealpha=0.2, labelcolor="white")
        leg6.get_frame().set_facecolor("#0e2a40")
        st.pyplot(fig6)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — FOSSIL FUELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛢️ Fossil Fuels":

    st.markdown("<h2 style='color:#ff6b35;'>🛢️ Fossil Fuel Analysis</h2>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col_ui, field, label in [
        (c1,"fossil_share_energy","Fossil Share %"),
        (c2,"coal_share_energy","Coal Share %"),
        (c3,"gas_share_energy","Gas Share %"),
    ]:
        val = last_val(field, ".1f")
        col_ui.markdown(f"""
        <div class='kpi-card' style='background:linear-gradient(135deg,#3d0f0f,#2a1a0a);border-color:#5a2a1a;'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value' style='color:#ff6b35;'>{val}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>FOSSIL SHARE TREND</div>", unsafe_allow_html=True)
        fos_cols = safe_cols(filtered, ["fossil_share_energy","coal_share_energy","oil_share_energy","gas_share_energy"])
        if fos_cols:
            fig7, ax7 = dark_fig((6, 3.5))
            fcolors = {"fossil_share_energy":"#ff6b35","coal_share_energy":"#8b4513",
                       "oil_share_energy":"#cd853f","gas_share_energy":"#daa520"}
            for c in fos_cols:
                d = filtered[["year",c]].dropna()
                ax7.plot(d["year"], d[c], color=fcolors.get(c,"#aaa"), linewidth=2,
                         label=c.replace("_share_energy","").capitalize())
                ax7.fill_between(d["year"], d[c], alpha=0.07, color=fcolors.get(c,"#aaa"))
            ax7.set_xlabel("Year"); ax7.set_ylabel("Share (%)")
            leg7 = ax7.legend(fontsize=8, framealpha=0.2, labelcolor="white")
            leg7.get_frame().set_facecolor("#0e2a40")
            st.pyplot(fig7)

    with col_right:
        st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>TOP 10 FOSSIL-DEPENDENT COUNTRIES</div>", unsafe_allow_html=True)
        latest_yr = df["year"].max()
        fos_top = df[df["year"]==latest_yr][["country","fossil_share_energy"]].dropna().nlargest(10,"fossil_share_energy")
        fig8, ax8 = dark_fig((6, 3.5))
        ax8.barh(fos_top["country"], fos_top["fossil_share_energy"], color="#ff6b35", alpha=0.85, edgecolor="#0b1a2b")
        ax8.set_xlabel("Fossil Share (%)")
        st.pyplot(fig8)

    # Fossil vs Renewable global comparison
    st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin:14px 0 6px;'>GLOBAL FOSSIL vs RENEWABLE TRANSITION</div>", unsafe_allow_html=True)
    world_df = df[df["country"]=="World"][["year","fossil_share_energy","renewables_share_energy"]].dropna()
    if not world_df.empty:
        fig9, ax9 = dark_fig((12, 3))
        ax9.fill_between(world_df["year"], world_df["fossil_share_energy"], alpha=0.4, color="#ff6b35", label="Fossil")
        ax9.fill_between(world_df["year"], world_df["renewables_share_energy"], alpha=0.4, color="#00e5ff", label="Renewable")
        ax9.plot(world_df["year"], world_df["fossil_share_energy"], color="#ff6b35", linewidth=2)
        ax9.plot(world_df["year"], world_df["renewables_share_energy"], color="#00e5ff", linewidth=2)
        ax9.set_xlabel("Year"); ax9.set_ylabel("Share (%)")
        leg9 = ax9.legend(fontsize=9, framealpha=0.2, labelcolor="white")
        leg9.get_frame().set_facecolor("#0e2a40")
        st.pyplot(fig9)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — ML PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Prediction":

    st.markdown("<h2 style='color:#00e5ff;'>🤖 ML Renewable Share Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4a8fa8;'>Random Forest model trained on global OWID energy data</p>", unsafe_allow_html=True)

    candidate_features = [
        "fossil_share_energy","fossil_share_elec","energy_per_capita",
        "solar_share_elec","wind_share_elec","hydro_share_elec",
        "low_carbon_share_energy","electricity_share_energy","year"
    ]
    target = "renewables_share_energy"
    feats  = safe_cols(df, candidate_features)
    ml_df  = df[feats + [target]].dropna()

    if len(ml_df) > 100:
        X = ml_df[feats]; y = ml_df[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        sc  = StandardScaler()
        mdl = RandomForestRegressor(100, random_state=42)
        mdl.fit(sc.fit_transform(X_tr), y_tr)
        preds = mdl.predict(sc.transform(X_te))
        r2   = r2_score(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))

        # Model metrics
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"<div class='kpi-card'><div class='kpi-label'>R² Score</div><div class='kpi-value'>{r2:.4f}</div></div>", unsafe_allow_html=True)
        m2.markdown(f"<div class='kpi-card'><div class='kpi-label'>RMSE</div><div class='kpi-value'>{rmse:.4f}</div></div>", unsafe_allow_html=True)
        m3.markdown(f"<div class='kpi-card'><div class='kpi-label'>Training Rows</div><div class='kpi-value'>{len(X_tr):,}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Feature importance + actual vs predicted
        left, right = st.columns([1,1])

        with left:
            st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>FEATURE IMPORTANCE</div>", unsafe_allow_html=True)
            fi = pd.Series(mdl.feature_importances_, index=feats).sort_values()
            fig10, ax10 = dark_fig((6, 3.5))
            colors_fi = ["#00e5ff" if v == fi.max() else "#1e5a7a" for v in fi]
            ax10.barh(fi.index, fi.values, color=colors_fi, edgecolor="#0b1a2b")
            ax10.set_xlabel("Importance")
            for i, (idx, val) in enumerate(fi.items()):
                ax10.text(val+0.003, i, f"{val:.3f}", va="center", color="#7ecfea", fontsize=7)
            st.pyplot(fig10)

        with right:
            st.markdown("<div style='color:#7ecfea;font-size:0.82rem;font-weight:600;margin-bottom:6px;'>ACTUAL vs PREDICTED</div>", unsafe_allow_html=True)
            fig11, ax11 = dark_fig((6, 3.5))
            ax11.scatter(y_te, preds, color="#00e5ff", alpha=0.35, s=12, edgecolors="none")
            mn_v = min(y_te.min(), preds.min())
            mx_v = max(y_te.max(), preds.max())
            ax11.plot([mn_v,mx_v],[mn_v,mx_v], color="#ff6b35", linewidth=1.5, linestyle="--")
            ax11.set_xlabel("Actual (%)"); ax11.set_ylabel("Predicted (%)")
            st.pyplot(fig11)

        # Interactive prediction
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='color:#7ecfea;font-size:0.9rem;font-weight:700;margin-bottom:12px;'>🎛️ INTERACTIVE PREDICTOR — Adjust values to predict renewable share</div>", unsafe_allow_html=True)

        inputs = {}
        cols_grid = st.columns(3)
        for i, feat in enumerate(feats):
            mn  = float(X[feat].min())
            mx  = float(X[feat].max())
            med = float(X[feat].median())
            inputs[feat] = cols_grid[i%3].slider(feat.replace("_"," ").title(), mn, mx, med, key=feat)

        if st.button("🔮 Predict Renewable Share"):
            inp_arr = sc.transform([list(inputs.values())])
            pred = mdl.predict(inp_arr)[0]
            st.markdown(f"""
            <div class='kpi-card' style='margin-top:16px;'>
                <div class='kpi-label'>Predicted Renewable Energy Share</div>
                <div class='kpi-value' style='font-size:2.8rem;'>{pred:.2f}%</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.error("ML model ke liye data insufficient hai.")
