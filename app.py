import streamlit as st
from tabs.tab_video import render_tab_video
from tabs.tab_map import render_tab_map

st.set_page_config(
    page_title="Kiểm tra ùn tắc giao thông",
    layout="wide"
)

tab1, tab2 = st.tabs([
    "Mô phỏng giao thông",
    "Bản đồ & đèn tín hiệu"
])

with tab1:
    render_tab_video()

with tab2:
    render_tab_map()
