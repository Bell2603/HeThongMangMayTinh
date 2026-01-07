import streamlit as st
import pandas as pd
from pathlib import Path

def render_tab_video():
    st.header("🎥 Mô phỏng giao thông bằng video")

    BASE_DIR = Path("/Users/nguyenduchung/Downloads/Vehicle_Speed_Estimation/Ket_qua")

    video_files = sorted(BASE_DIR.glob("*.mp4"))

    if not video_files:
        st.warning("❌ Không có video")
        return

    # Chọn video
    video_name = st.selectbox(
        "Chọn video mô phỏng",
        [v.name for v in video_files]
    )

    video_path = BASE_DIR / video_name
    csv_path = video_path.with_suffix(".csv")   # cùng tên nhưng đuôi .csv

    # Hiển thị video
    st.video(video_path.read_bytes())

    # Kiểm tra CSV
    if not csv_path.exists():
        st.error(f"❌ Không tìm thấy CSV: {csv_path.name}")
        return

    df = pd.read_csv(csv_path)

    # Kiểm tra cột CI
    if "CI" not in df.columns:
        st.error("❌ CSV không có cột CI")
        st.write("Cột hiện có:", df.columns.tolist())
        return

    avg_CI = df["CI"].mean()

    st.metric("Chỉ số ùn tắc (CI)", round(avg_CI, 3))

    if avg_CI < 0.3:
        st.success("🟢 Thông thoáng")
    elif avg_CI < 0.6:
        st.warning("🟡 Đông")
    else:
        st.error("🔴 Ùn tắc")

    # Hiển thị chi tiết
    with st.expander("📊 Chi tiết CI theo frame"):
        st.dataframe(df)
