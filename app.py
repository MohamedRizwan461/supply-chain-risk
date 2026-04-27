import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components
import json

st.set_page_config(
    page_title="AI Supply Chain Risk Predictor",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme ─────────────────────────────────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = True

D = st.session_state.dark

BG      = "#0d1b2a"  if D else "#f0f4f8"
SIDEBAR = "#0f2035"  if D else "#ffffff"
CARD    = "#1a2f4a"  if D else "#ffffff"
SURF    = "#152a42"  if D else "#f1f5f9"
TEXT    = "#e2e8f0"  if D else "#1a2f4a"
SUB     = "#7fa8d0"  if D else "#5a7a9a"
BDR     = "#2a4060"  if D else "#dde8f0"
INP     = "#0d1b2a"  if D else "#ffffff"
BLUE    = "#4a90d9"  if D else "#2563eb"
GREEN   = "#27ae60"  if D else "#059669"
AMBER   = "#f39c12"  if D else "#d97706"
RED     = "#e74c3c"  if D else "#dc2626"

# ── Inject only what's needed for native Streamlit components ─────────────────
# Self-hiding container prevents the <style> text from rendering visibly
st.markdown(f"""<div id="_sc_s"><style>
#_sc_s{{display:none!important}}
.stApp{{background:{BG}!important}}
.main .block-container{{padding:2rem 2.5rem!important;max-width:1400px!important}}
#MainMenu,footer{{visibility:hidden!important}}
header{{background:transparent!important}}
[data-testid="stHeader"]{{background:transparent!important}}
[data-testid="collapsedControl"]{{display:flex!important;visibility:visible!important;opacity:1!important;color:#fff!important;background:{BLUE}!important;border-radius:8px!important;padding:8px!important;top:14px!important;left:14px!important;z-index:999999!important;box-shadow:0 2px 8px rgba(0,0,0,0.3)!important}}
[data-testid="collapsedControl"] svg{{fill:#fff!important;color:#fff!important;width:22px!important;height:22px!important}}
[data-testid="stSidebarCollapseButton"]{{display:flex!important;visibility:visible!important}}
[data-testid="stSidebar"]{{background:{SIDEBAR}!important;border-right:1px solid {BDR}!important}}
[data-testid="stSidebarContent"]{{background:{SIDEBAR}!important}}
[data-testid="stSidebar"] label{{color:{SUB}!important;font-size:.82rem!important}}
[data-testid="stSidebar"] [data-testid="stSelectbox"]>div>div{{background:{INP}!important;border:1px solid {BDR}!important;color:{TEXT}!important;border-radius:8px!important}}
[data-testid="stSidebar"] input{{background:{INP}!important;border:1px solid {BDR}!important;color:{TEXT}!important;border-radius:8px!important}}
[data-testid="stSlider"] [data-baseweb="thumb"]{{background:{BLUE}!important;border-color:{BLUE}!important}}
[data-testid="stToggle"] label{{color:{TEXT}!important}}
[data-testid="stToggle"] [role="switch"][aria-checked="true"]{{background:{BLUE}!important}}
[data-testid="stButton"]>button{{background:{BLUE}!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:600!important;font-size:.95rem!important;padding:12px!important;width:100%!important}}
[data-testid="stDownloadButton"]>button{{background:{SURF}!important;color:{BLUE}!important;border:1.5px solid {BLUE}!important;border-radius:8px!important;font-weight:600!important}}
[data-testid="stFileUploadDropzone"]{{background:{SURF}!important;border:2px dashed {BDR}!important;border-radius:12px!important}}
[data-testid="stDataFrame"]{{border:1px solid {BDR}!important;border-radius:10px!important;overflow:hidden!important}}
[data-testid="stMetric"]{{background:{CARD}!important;border:1px solid {BDR}!important;border-radius:12px!important;padding:16px 20px!important}}
[data-testid="stMetricLabel"]{{color:{SUB}!important;font-size:.8rem!important}}
[data-testid="stMetricValue"]{{color:{TEXT}!important;font-size:1.6rem!important;font-weight:700!important}}
[data-testid="stTabs"] [data-baseweb="tab-list"],
[data-testid="stTabs"] [role="tablist"]{{background:{SURF}!important;border-radius:12px!important;padding:4px!important;gap:4px!important;border:none!important}}
[data-testid="stTabs"] [data-baseweb="tab"],
[data-testid="stTabs"] [role="tab"]{{background:transparent!important;color:{SUB}!important;border:none!important;border-radius:8px!important;font-size:.88rem!important;font-weight:500!important;padding:8px 18px!important}}
[data-testid="stTabs"] [aria-selected="true"]{{background:{BLUE}!important;color:#fff!important;font-weight:600!important}}
[data-testid="stTabs"] [data-baseweb="tab-border"],
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{{display:none!important}}
[data-testid="stMarkdownContainer"] p{{color:{TEXT}!important}}
</style></div>""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    return (joblib.load("best_supply_chain_model.pkl"),
            joblib.load("scaler.pkl"),
            joblib.load("label_encoders.pkl"))

model, scaler, encoders = load_assets()

def predict_one(st_, wc, ge, sr, tm, rr):
    df = pd.DataFrame({
        "shipment_time":        [st_],
        "weather_condition":    [encoders["weather_condition"].transform([wc])[0]],
        "geopolitical_event":   [encoders["geopolitical_event"].transform([ge])[0]],
        "supplier_reliability": [sr],
        "transportation_mode":  [encoders["transportation_mode"].transform([tm])[0]],
        "route_risk":           [rr],
    })
    return float(model.predict(scaler.transform(df))[0])

def risk_info(days):
    if days < 2:  return "🟢 Low Risk",      GREEN, "ok"
    if days < 5:  return "🟡 Moderate Risk", AMBER, "warn"
    return              "🔴 High Risk",      RED,   "danger"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    tog = st.toggle("🌙 Dark Mode", value=st.session_state.dark)
    if tog != st.session_state.dark:
        st.session_state.dark = tog
        st.rerun()

    st.markdown(f"""
    <div style="margin:14px 0 6px 0;font-size:1.05rem;font-weight:700;color:{TEXT};">
      🚛 Shipment Details
    </div>
    <hr style="border:none;border-top:1px solid {BDR};margin:0 0 10px 0;">
    """, unsafe_allow_html=True)

    s_time = st.number_input("🕐 Transit Time (Days)", min_value=2, max_value=14, value=7)
    s_rel  = st.slider("🏭 Supplier Reliability", 0.30, 1.00, 0.85, 0.01)
    r_risk = st.slider("🗺️ Route Risk Level (1–9)", 1, 9, 3)
    weath  = st.selectbox("🌤️ Weather Condition",  ["Clear","Rainy","Snow","Storm"])
    geo    = st.selectbox("🌍 Geopolitical Event",  ["None","Sanction","Strike","Tariff"])
    trans  = st.selectbox("🚢 Transportation Mode", ["Air","Rail","Road","Sea"])

    st.markdown(f'<hr style="border:none;border-top:1px solid {BDR};margin:12px 0;">', unsafe_allow_html=True)
    btn = st.button("🔍 Predict Delay", use_container_width=True)

# ── Header (always dark — branding) ──────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1b2a 0%,#1a3a5c 60%,#0d2137 100%);
            border-radius:16px;padding:36px 40px;text-align:center;
            margin-bottom:24px;border:1px solid #2a4a6a;">
  <div style="font-size:2rem;font-weight:800;color:#fff;margin-bottom:10px;line-height:1.2;">
    🚛 AI-Driven Supply Chain Risk Predictor
  </div>
  <div style="color:#94b8d8;font-size:.9rem;margin:4px 0;">
    Final Review &nbsp;|&nbsp; Complete System: Single Prediction · Batch Processing · Model Analytics
  </div>
  <div style="color:#6a9abf;font-size:.82rem;margin-top:4px;">
    Governors State University &nbsp;·&nbsp; Powered by Gradient Boosting
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Shipment Prediction",
    "📂 Batch Prediction (CSV Upload)",
    "📊 Model Analytics"
])

# helper: section title
def sec(icon, title):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;font-size:1rem;font-weight:700;
                color:{TEXT};padding-bottom:10px;border-bottom:2px solid {BDR};
                margin:24px 0 16px 0;">
      {icon} {title}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if btn:
        days = predict_one(s_time, weath, geo, s_rel, trans, r_risk)
        label, color, cls = risk_info(days)

        c1, c2, c3 = st.columns(3)
        c1.metric("⏱️ Predicted Delay",      f"{days:.1f} days")
        c2.metric("⚠️ Risk Level",            label)
        c3.metric("🏭 Supplier Reliability",  f"{s_rel:.2f}")

        st.markdown("<br>", unsafe_allow_html=True)
        L, R = st.columns(2)

        with L:
            sec("📋", "Shipment Summary")
            for icon, key, val in [
                ("🕐","Transit Time",        f"{s_time} days"),
                ("🌤️","Weather Condition",   weath),
                ("🌍","Geopolitical Event",  geo),
                ("🏭","Supplier Reliability",f"{s_rel:.2f}"),
                ("🚢","Transportation Mode", trans),
                ("🗺️","Route Risk",          f"{r_risk} / 9"),
            ]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:9px 0;border-bottom:1px solid {BDR};font-size:.88rem;">
                  <span style="color:{SUB};">{icon} {key}</span>
                  <span style="font-weight:600;color:{TEXT};">{val}</span>
                </div>""", unsafe_allow_html=True)

        with R:
            sec("📌", "Recommended Actions")
            ac = f"border-left:4px solid {color};background:{SURF};border-radius:0 10px 10px 0;padding:11px 15px;margin-bottom:8px;font-size:.85rem;color:{TEXT};font-weight:500;"

            if cls == "danger":
                st.markdown(f'<div style="{ac}">🔴 <b>High Risk — Immediate action required</b></div>', unsafe_allow_html=True)
                if weath in ["Storm","Snow"]:
                    st.markdown(f'<div style="{ac}">⚡ Re-route away from weather-affected corridor</div>', unsafe_allow_html=True)
                if geo in ["Strike","Sanction","Tariff"]:
                    st.markdown(f'<div style="{ac}">🏭 Activate backup supplier outside risk zone</div>', unsafe_allow_html=True)
                if r_risk >= 7:
                    st.markdown(f'<div style="{ac}">🗺️ Re-evaluate route — consider multimodal switch</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="{ac}">🏪 Increase safety stock · Notify downstream teams</div>', unsafe_allow_html=True)
            elif cls == "warn":
                st.markdown(f'<div style="{ac}">🟡 <b>Moderate Risk — Monitor closely</b></div>', unsafe_allow_html=True)
                for txt in ["🕐 Check shipment status every 6 hours",
                            "🔔 Notify customer of potential delay",
                            "🏭 Keep backup supplier on standby",
                            "📈 Reassess route conditions daily"]:
                    st.markdown(f'<div style="{ac}">{txt}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="{ac}">🟢 <b>Low Risk — No action required</b></div>', unsafe_allow_html=True)
                for txt in ["✅ Proceed with standard JIT schedule",
                            "👁️ Continue routine shipment tracking",
                            "📦 Confirm departure window with carrier"]:
                    st.markdown(f'<div style="{ac}">{txt}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BDR};border-radius:14px;
                    padding:60px 32px;text-align:center;color:{SUB};font-size:.95rem;">
          🔍<br><br>
          <span style="color:{TEXT};font-weight:600;">Configure parameters in the sidebar and click Predict Delay</span><br><br>
          Set transit time, weather, geopolitical events, supplier reliability,<br>
          transportation mode and route risk to get an AI-powered delay forecast.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    sec("📂", "Batch Shipment Prediction")
    st.markdown(f'<span style="color:{SUB};font-size:.88rem;">Upload a CSV to predict risk across multiple shipments at once.</span>',
                unsafe_allow_html=True)

    sample = pd.DataFrame({
        "shipment_time":        [5, 14, 8, 3, 12],
        "weather_condition":    ["Clear","Storm","Rainy","Clear","Snow"],
        "geopolitical_event":   ["None","Strike","Tariff","None","Sanction"],
        "supplier_reliability": [0.90, 0.30, 0.70, 0.95, 0.45],
        "transportation_mode":  ["Road","Sea","Rail","Air","Road"],
        "route_risk":           [2, 9, 5, 1, 7]
    })
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("⬇️ Download CSV Template", data=sample.to_csv(index=False),
                       file_name="shipment_template.csv", mime="text/csv")
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your shipment CSV", type=["csv"])

    if uploaded:
        df_b  = pd.read_csv(uploaded, keep_default_na=False)
        req   = ["shipment_time","weather_condition","geopolitical_event",
                 "supplier_reliability","transportation_mode","route_risk"]
        miss  = [c for c in req if c not in df_b.columns]
        if miss:
            st.error(f"Missing columns: {miss}")
        else:
            preds, labels = [], []
            for _, row in df_b.iterrows():
                p = predict_one(row["shipment_time"], row["weather_condition"],
                                row["geopolitical_event"], row["supplier_reliability"],
                                row["transportation_mode"], row["route_risk"])
                l, _, _ = risk_info(p)
                preds.append(round(p, 2))
                labels.append(l)
            df_b["Predicted Delay (Days)"] = preds
            df_b["Risk Level"]             = labels

            low  = labels.count("🟢 Low Risk")
            mod  = labels.count("🟡 Moderate Risk")
            high = labels.count("🔴 High Risk")

            sec("📊", "Summary")
            s_card = f"background:{CARD};border:1px solid {BDR};border-radius:12px;padding:18px 14px;text-align:center;"
            m1, m2, m3, m4 = st.columns(4)
            for col, val, lbl, clr in [
                (m1, len(df_b), "Total Shipments", TEXT),
                (m2, low,  "🟢 Low Risk",  GREEN),
                (m3, mod,  "🟡 Moderate",  AMBER),
                (m4, high, "🔴 High Risk", RED),
            ]:
                col.markdown(f"""
                <div style="{s_card}">
                  <div style="font-size:2rem;font-weight:700;color:{clr};">{val}</div>
                  <div style="font-size:.78rem;color:{SUB};margin-top:4px;">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df_b, use_container_width=True)

            sec("📉", "Predicted Delay Distribution")
            st.bar_chart(pd.DataFrame({
                "Shipment": [f"S{i+1}" for i in range(len(preds))],
                "Delay (Days)": preds
            }).set_index("Shipment"))

            st.download_button("⬇️ Download Predictions Report",
                               data=df_b.to_csv(index=False),
                               file_name="predictions_report.csv", mime="text/csv")
    else:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BDR};border-radius:14px;
                    padding:56px 32px;text-align:center;color:{SUB};font-size:.95rem;">
          📂<br><br>
          <span style="color:{TEXT};font-weight:600;">Drop your CSV file here</span><br><br>
          Download the template above, fill in your shipment data, then upload.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    sec("📊", "Model Benchmark Results")

    mc = f"border-radius:14px;padding:24px 20px;text-align:center;border:1px solid {BDR};"
    c1, c2, c3 = st.columns(3)

    c1.markdown(f"""
    <div style="background:{CARD};{mc}">
      <div style="font-size:.88rem;font-weight:600;color:{BLUE};margin-bottom:14px;">Linear Regression</div>
      <div style="font-size:2.6rem;font-weight:800;color:{BLUE};line-height:1;margin-bottom:6px;">0.675</div>
      <div style="font-size:.75rem;color:{SUB};">MAE (Days)</div>
      <div style="font-size:.75rem;color:{SUB};margin-bottom:8px;">RMSE: 0.808 days</div>
      <span style="background:{SURF};color:{SUB};border-radius:20px;padding:3px 12px;font-size:.72rem;font-weight:700;">Baseline</span>
    </div>""", unsafe_allow_html=True)

    c2.markdown(f"""
    <div style="background:{CARD};{mc}">
      <div style="font-size:.88rem;font-weight:600;color:{BLUE};margin-bottom:14px;">Random Forest</div>
      <div style="font-size:2.6rem;font-weight:800;color:{BLUE};line-height:1;margin-bottom:6px;">0.464</div>
      <div style="font-size:.75rem;color:{SUB};">MAE (Days)</div>
      <div style="font-size:.75rem;color:{SUB};margin-bottom:8px;">RMSE: 0.582 days</div>
      <span style="background:{SURF};color:{BLUE};border-radius:20px;padding:3px 12px;font-size:.72rem;font-weight:700;">Strong</span>
    </div>""", unsafe_allow_html=True)

    c3.markdown(f"""
    <div style="background:{CARD};border:2px solid {GREEN};border-radius:14px;padding:24px 20px;text-align:center;">
      <div style="font-size:.88rem;font-weight:600;color:{GREEN};margin-bottom:14px;">Gradient Boosting</div>
      <div style="font-size:2.6rem;font-weight:800;color:{GREEN};line-height:1;margin-bottom:6px;">0.422</div>
      <div style="font-size:.75rem;color:{SUB};">MAE (Days)</div>
      <div style="font-size:.75rem;color:{SUB};margin-bottom:8px;">RMSE: 0.533 days</div>
      <span style="background:{"#1a3828" if D else "#ecfdf5"};color:{GREEN};border-radius:20px;padding:3px 12px;font-size:.72rem;font-weight:700;">✅ Deployed</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        sec("📉", "MAE Comparison")
        st.bar_chart(pd.DataFrame({
            "Model": ["Gradient Boosting","Random Forest","Linear Regression"],
            "MAE":   [0.422, 0.464, 0.675]
        }).set_index("Model"))
    with ch2:
        sec("📉", "RMSE Comparison")
        st.bar_chart(pd.DataFrame({
            "Model": ["Gradient Boosting","Random Forest","Linear Regression"],
            "RMSE":  [0.533, 0.582, 0.808]
        }).set_index("Model"))

    sec("🔁", "End-to-End System Pipeline")

    for num, title, desc in [
        ("1", "Data Ingestion",   f"5,000 row CSV dataset &nbsp;·&nbsp; <b style='color:{TEXT};'>Pandas</b>"),
        ("2", "Preprocessing",   f"Label Encoding + Standard Scaling &nbsp;·&nbsp; <b style='color:{TEXT};'>Scikit-learn</b>"),
        ("3", "Model Training",  f"Linear Regression, Random Forest, Gradient Boosting &nbsp;·&nbsp; <b style='color:{TEXT};'>Scikit-learn</b>"),
        ("4", "Deployment",      f"Best model serialized as .pkl files &nbsp;·&nbsp; <b style='color:{TEXT};'>Joblib</b>"),
        ("5", "User Interface",  f"Real-time single prediction dashboard &nbsp;·&nbsp; <b style='color:{TEXT};'>Streamlit</b>"),
        ("6", "Batch Processing",f"Multi-shipment CSV prediction + export &nbsp;·&nbsp; <b style='color:{TEXT};'>Pandas + Streamlit</b>"),
    ]:
        st.markdown(f"""
        <div style="background:{CARD};border:1px solid {BDR};border-left:4px solid {BLUE};
                    border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:10px;
                    display:flex;align-items:flex-start;gap:16px;">
          <div style="background:{BLUE};color:#fff;width:28px;height:28px;border-radius:50%;
                      display:flex;align-items:center;justify-content:center;
                      font-weight:700;font-size:.82rem;flex-shrink:0;margin-top:1px;">{num}</div>
          <div>
            <div style="font-weight:600;font-size:.9rem;color:{BLUE};margin-bottom:3px;">{title}</div>
            <div style="font-size:.8rem;color:{SUB};">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:20px 0 8px 0;font-size:.75rem;color:{SUB};
            border-top:1px solid {BDR};margin-top:32px;">
  AI-Driven Supply Chain Risk Prediction System &nbsp;·&nbsp;
  Final Review &nbsp;·&nbsp; Governors State University &nbsp;·&nbsp;
  Powered by Gradient Boosting
</div>""", unsafe_allow_html=True)
