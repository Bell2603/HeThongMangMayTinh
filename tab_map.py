import streamlit as st
from PIL import Image

def render_tab_map():
    st.header("🚦 Điều khiển đèn giao thông")

    img = Image.open("bach_khoa_map.png")
    st.image(img, use_container_width=True)

    junctions = {
        "C1": (30, 40),
        "C5": (25, 35),
        "D3": (20, 30),
    }

    for name, (g, r) in junctions.items():
        with st.expander(f"Nút {name}"):
            green = st.slider("Đèn xanh (giây)", 10, 120, g)
            red = st.slider("Đèn đỏ (giây)", 10, 120, r)
            st.write(f"Tổng chu kỳ: {green + red} giây")
