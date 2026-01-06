import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Giao thông D3", layout="wide")
st.title("Màn hình video giao thông D3")

VIDEO_PATH = Path(
    "/Users/nguyenduchung/Downloads/Vehicle_Speed_Estimation/Ket_qua/Tắc.mp4"
)

if not VIDEO_PATH.exists():
    st.error("Không tìm thấy video")
    st.stop()

video_bytes = VIDEO_PATH.read_bytes()

st.video(video_bytes, format="video/mp4")
st.caption("Thông thoáng")
