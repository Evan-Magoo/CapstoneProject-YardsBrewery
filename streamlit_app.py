
import json
import time
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------

st.set_page_config(
    page_title="Can Inspection Monitor",
    page_icon="🏭",
    layout="wide"
)

DATA_FILE = (
    Path(__file__).resolve().parent
    / "shared_data"
    / "live_metrics.json"
)


# ----------------------------------------
# DASHBOARD STYLING
# ----------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #10141c;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #303b4d;
    }

    [data-testid="stMetricLabel"] {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------
# HEADER
# ----------------------------------------

st.title("🏭 CAN QUALITY INSPECTION MONITOR")
st.caption("Live production monitoring dashboard")

st.divider()


# ----------------------------------------
# AUTOMATIC REFRESH
# ----------------------------------------

st_autorefresh(
    interval=1000,
    key="dashboard_refresh"
)


# ----------------------------------------
# READ SHARED METRICS
# ----------------------------------------

def read_metrics():

    default_metrics = {
        "total_inspected": 0,
        "graded_count": 0,
        "pass_count": 0,
        "fail_count": 0,
        "pending_count": 0,
        "pass_rate": 0.0,
        "last_updated": "Waiting for inspection system"
    }

    if not DATA_FILE.exists():
        return default_metrics

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        return {**default_metrics, **metrics}

    except (json.JSONDecodeError, OSError):
        return default_metrics


metrics = read_metrics()


# ----------------------------------------
# PRODUCTION METRICS
# ----------------------------------------

st.subheader("Production Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Cans Inspected",
        f"{metrics['total_inspected']:,}"
    )

with col2:
    st.metric(
        "Pass Count",
        f"{metrics['pass_count']:,}"
    )

with col3:
    st.metric(
        "Fail Count",
        f"{metrics['fail_count']:,}"
    )

with col4:
    st.metric(
        "Pass Rate",
        f"{metrics['pass_rate']:.2f}%"
    )


# ----------------------------------------
# GRADING INFORMATION
# ----------------------------------------

st.divider()

st.subheader("Inspection Summary")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Completed Grades",
        f"{metrics['graded_count']:,}"
    )

with col2:
    st.metric(
        "Pending Results",
        f"{metrics['pending_count']:,}"
    )


# ----------------------------------------
# QUALITY PROGRESS
# ----------------------------------------

st.subheader("Production Quality")

pass_rate = float(metrics["pass_rate"])

st.progress(
    min(max(pass_rate / 100, 0.0), 1.0)
)

st.write(f"**Pass Rate: {pass_rate:.2f}%**")


# ----------------------------------------
# SYSTEM INFORMATION
# ----------------------------------------

st.divider()

st.subheader("System Information")

st.write(
    f"**Last Data Update:** "
    f"{metrics['last_updated']}"
)

if DATA_FILE.exists():
    st.success("Inspection data file detected.")
else:
    st.warning(
        "Waiting for the OpenCV inspection application..."
    )

st.caption(
    "Dashboard refresh interval: 1 second"
)
