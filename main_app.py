import streamlit as st
from head_shoulders_detector import HeadShouldersDetector
from double_top_bottom_detector import DoubleTopBottomDetector
from wedge_detector import WedgeDetector

detectors = [
    HeadShouldersDetector(),
    DoubleTopBottomDetector(),
    WedgeDetector()
]

st.sidebar.title("Select Pattern Detector")
app_mode = st.sidebar.radio(
    "Choose a pattern detection tool:",
    [d.get_name() for d in detectors]
)

selected_detector = next(d for d in detectors if d.get_name() == app_mode)

params = selected_detector.sidebar_params(st)

selected_detector.run_detection(st, params)
