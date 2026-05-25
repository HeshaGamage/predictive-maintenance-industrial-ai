import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="IFS-Inspired Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# ── Load all assets (cached so they only load once) ──────────
@st.cache_resource
def load_model():
    return joblib.load('models/xgb_model.pkl')

@st.cache_resource
def load_explainer():
    return joblib.load('models/shap_explainer.pkl')

@st.cache_data
def load_data():
    test = pd.read_csv('data/processed/test_features.csv')
    fleet = pd.read_csv('data/processed/fleet_status.csv')
    feature_cols = joblib.load('models/feature_cols.pkl')
    shap_values = joblib.load('models/shap_values_test.pkl')
    with open('data/processed/work_orders_cache.json', 'r') as f:
        work_orders = json.load(f)
    return test, fleet, feature_cols, shap_values, work_orders

model = load_model()
explainer = load_explainer()
test, fleet, feature_cols, shap_values, work_orders = load_data()
test_last = test.groupby('engine_id').last().reset_index()





def risk_color(risk_level):
    return {
        'CRITICAL': '🔴',
        'HIGH':     '🟠',
        'MEDIUM':   '🟡',
        'LOW':      '🟢'
    }.get(risk_level, '⚪')

def risk_badge(risk_level):
    colors = {
        'CRITICAL': '#ff4444',
        'HIGH':     '#ff8800',
        'MEDIUM':   '#ffcc00',
        'LOW':      '#44bb44'
    }
    color = colors.get(risk_level, '#888888')
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:10px;font-weight:bold;font-size:12px">{risk_level}</span>'

def parse_engine_id(value):
    """Normalize engine identifiers like 34 or 'ENGINE-034' to int."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().upper()
        if cleaned.startswith("ENGINE-"):
            cleaned = cleaned.split("-", 1)[1]
        return int(cleaned)
    return int(value)

def engine_label(engine_id_num):
    return f"ENGINE-{int(engine_id_num):03d}"


# Normalize engine IDs once so UI logic works for both numeric and ENGINE-xxx formats
fleet['engine_id_num'] = fleet['engine_id'].apply(parse_engine_id)
test['engine_id_num'] = test['engine_id'].apply(parse_engine_id)
test_last['engine_id_num'] = test_last['engine_id'].apply(parse_engine_id)






st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Gear_icon.svg/240px-Gear_icon.svg.png", width=60)
st.sidebar.title("Predictive Maintenance")
st.sidebar.caption("Industrial AI — IFS.ai inspired")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["🏭 Fleet Overview", "🔍 Engine Detail", "📋 Work Order"]
)

st.sidebar.divider()
st.sidebar.caption("NASA CMAPSS FD001 Dataset")
st.sidebar.caption("XGBoost + SHAP + Llama 3")




if page == "🏭 Fleet Overview":

    st.title("⚙️ Fleet Health Dashboard")
    st.caption("Real-time risk monitoring across all 100 engines — powered by Industrial AI")

    # ── Top metrics row ──
    col1, col2, col3, col4 = st.columns(4)

    critical = len(fleet[fleet['risk_level'] == 'CRITICAL'])
    high     = len(fleet[fleet['risk_level'] == 'HIGH'])
    medium   = len(fleet[fleet['risk_level'] == 'MEDIUM'])
    low      = len(fleet[fleet['risk_level'] == 'LOW'])

    col1.metric("🔴 Critical", critical, help="Failure within 15 cycles")
    col2.metric("🟠 High Risk", high,     help="Failure within 30 cycles")
    col3.metric("🟡 Medium",   medium,    help="Failure within 60 cycles")
    col4.metric("🟢 Healthy",  low,       help="More than 60 cycles remaining")

    st.divider()

    # ── Risk distribution chart ──
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Risk Distribution")
        risk_counts = fleet['risk_level'].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = ['#ff4444', '#ff8800', '#ffcc00', '#44bb44']
        labels = [l for l in ['CRITICAL','HIGH','MEDIUM','LOW'] if l in risk_counts.index]
        sizes  = [risk_counts[l] for l in labels]
        ax.pie(sizes, labels=labels, colors=colors[:len(labels)],
               autopct='%1.0f%%', startangle=90)
        ax.set_title("Fleet Risk Breakdown")
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Predicted RUL Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(fleet['predicted_rul'], bins=20,
                color='steelblue', edgecolor='white')
        ax.axvline(15, color='#ff4444', linestyle='--', label='Critical threshold (15)')
        ax.axvline(30, color='#ff8800', linestyle='--', label='High threshold (30)')
        ax.axvline(60, color='#ffcc00', linestyle='--', label='Medium threshold (60)')
        ax.set_xlabel('Predicted Remaining Useful Life (cycles)')
        ax.set_ylabel('Number of engines')
        ax.set_title('Fleet RUL Distribution')
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

    st.divider()

    # ── Fleet table ──
    st.subheader("All Engines — sorted by risk")

    # Add emoji column for display
    display_fleet = fleet.copy()
    display_fleet['Risk'] = display_fleet['risk_level'].apply(
        lambda x: f"{risk_color(x)} {x}"
    )
    display_fleet = display_fleet.rename(columns={
        'engine_id':    'Engine',
        'current_cycle':'Current Cycle',
        'predicted_rul':'Predicted RUL (cycles)'
    })[['Engine', 'Current Cycle', 'Predicted RUL (cycles)', 'Risk']]

    st.dataframe(
        display_fleet,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ── ROI calculator ──
    st.subheader("💰 Estimated Business Impact")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        hourly_cost = st.number_input(
            "Downtime cost per hour ($)",
            min_value=1000,
            max_value=500000,
            value=50000,
            step=1000
        )
    with col_b:
        avg_downtime = st.number_input(
            "Avg downtime per failure (hours)",
            min_value=1,
            max_value=72,
            value=8
        )
    with col_c:
        engines_saved = critical + high
        downtime_avoided = engines_saved * avg_downtime
        savings = downtime_avoided * hourly_cost
        st.metric(
            "Estimated savings this cycle",
            f"${savings:,.0f}",
            help=f"{engines_saved} engines flagged × {avg_downtime}hrs × ${hourly_cost:,}/hr"
        )


elif page == "🔍 Engine Detail":

    st.title("🔍 Engine Detail View")
    st.caption("Drill into individual engine health and understand what's driving the prediction")

    # Engine selector
    fleet_sorted = fleet.sort_values('predicted_rul')
    engine_options = fleet_sorted['engine_id_num'].tolist()
    selected_engine = st.selectbox(
        "Select Engine",
        engine_options,
        format_func=lambda x: f"{engine_label(x)}  |  "
                               f"{risk_color(fleet[fleet['engine_id_num']==x]['risk_level'].values[0])} "
                               f"{fleet[fleet['engine_id_num']==x]['risk_level'].values[0]}  |  "
                               f"RUL: {fleet[fleet['engine_id_num']==x]['predicted_rul'].values[0]:.0f} cycles"
    )

    engine_row = fleet[fleet['engine_id_num'] == selected_engine].iloc[0]
    engine_idx = test_last[test_last['engine_id_num'] == selected_engine].index[0]
    engine_idx_arr = test_last.index.get_loc(engine_idx)

    # ── Engine header ──
    col1, col2, col3 = st.columns(3)
    col1.metric("Engine ID",        engine_label(selected_engine))
    col2.metric("Predicted RUL",    f"{engine_row['predicted_rul']:.0f} cycles")
    col3.metric("Risk Level",       f"{risk_color(engine_row['risk_level'])} {engine_row['risk_level']}")

    st.divider()

    # ── Sensor history ──
    st.subheader("Sensor History Over Engine Lifetime")

    engine_history = test[test['engine_id_num'] == selected_engine].sort_values('cycle')

    key_sensors_display = {
        's4_mean_10':  'LPT Outlet Temp (10-cycle avg)',
        's11_mean_10': 'HPC Static Pressure (10-cycle avg)',
        's7_mean_10':  'HPC Outlet Pressure (10-cycle avg)',
        's12_mean_10': 'Fuel Flow Ratio (10-cycle avg)',
    }

    available = {k: v for k, v in key_sensors_display.items() if k in engine_history.columns}

    fig, axes = plt.subplots(len(available), 1,
                              figsize=(10, len(available) * 2.5))
    if len(available) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, available.items()):
        ax.plot(engine_history['cycle'], engine_history[col],
                linewidth=1.2, color='steelblue')
        ax.set_ylabel(label, fontsize=8)
        ax.set_xlabel('Cycle')
        ax.grid(True, alpha=0.3)

    axes[0].set_title(f'{engine_label(selected_engine)} — Key sensor trends')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── SHAP waterfall ──
    st.subheader("Why is this engine flagged? — SHAP Explanation")
    st.caption("Each bar shows how much a sensor pushed the prediction toward or away from failure")

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[engine_idx_arr],
            base_values=explainer.expected_value,
            data=test_last.iloc[engine_idx_arr][feature_cols].values,
            feature_names=feature_cols
        ),
        max_display=10,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── Navigate to work order ──
    engine_key = engine_label(selected_engine)
    if engine_key in work_orders:
        if st.button("📋 View Work Order for this Engine", type="primary"):
            st.session_state['selected_engine'] = selected_engine
            st.switch_page = "📋 Work Order"
            st.info("Go to **Work Order** in the sidebar to see the generated maintenance plan.")
    else:
        st.info("This engine is LOW or MEDIUM risk. No work order generated yet.")

elif page == "📋 Work Order":

    st.title("📋 Maintenance Work Order")
    st.caption("AI-generated maintenance plan — powered by Llama 3 + SHAP analysis")

    # Engine selector — only flagged engines
    flagged_engines = [k for k in work_orders.keys()]
    flagged_engines_sorted = sorted(
        flagged_engines,
        key=lambda x: work_orders[x]['predicted_rul']
    )

    selected_wo_engine = st.selectbox(
        "Select Engine",
        flagged_engines_sorted,
        format_func=lambda x: f"{x}  |  "
                               f"{risk_color(work_orders[x]['risk_level'])} "
                               f"{work_orders[x]['risk_level']}  |  "
                               f"RUL: {work_orders[x]['predicted_rul']} cycles"
    )

    wo = work_orders[selected_wo_engine]

    # ── Work order header ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Asset ID",        wo['asset_id'])
    col2.metric("Priority",        wo['priority'].upper())
    col3.metric("Estimated Hours", f"{wo['estimated_hours']} hrs")
    col4.metric("Technician Level", wo['required_technician_skill'])

    st.divider()

    # ── Technician summary ──
    st.subheader("📱 Summary for Technician")
    st.info(wo['summary_for_technician'])

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🔧 Affected Components")
        for component in wo.get('affected_components', []):
            st.write(f"• {component}")

        st.subheader("🔩 Recommended Parts")
        for part in wo.get('recommended_parts', []):
            st.write(f"• {part}")

    with col_right:
        st.subheader("⚠️ Safety Precautions")
        for precaution in wo.get('safety_precautions', []):
            st.write(f"• {precaution}")

        st.subheader("📊 Prediction Details")
        st.write(f"**Predicted RUL:** {wo['predicted_rul']} cycles")
        st.write(f"**Failure window:** {wo.get('predicted_failure_window', 'N/A')}")
        st.write(f"**Current cycle:** {wo.get('current_cycle', 'N/A')}")

    st.divider()

    # ── Send to field service button ──
    col_btn, col_status = st.columns([1, 3])

    with col_btn:
        send_clicked = st.button("🚀 Send to Field Service", type="primary")

    if send_clicked:
        # Save to submitted work orders log
        submitted_path = 'data/processed/submitted_work_orders.json'

        if os.path.exists(submitted_path):
            with open(submitted_path, 'r') as f:
                submitted = json.load(f)
        else:
            submitted = {}

        submitted[selected_wo_engine] = wo

        with open(submitted_path, 'w') as f:
            json.dump(submitted, f, indent=2)

        with col_status:
            st.success(
                f"✅ Work order for {selected_wo_engine} submitted to Field Service Management system. "
                f"Technician ({wo['required_technician_skill']}) has been notified."
            )

    st.divider()

    # ── Raw JSON toggle ──
    with st.expander("View raw work order JSON"):
        st.json(wo)