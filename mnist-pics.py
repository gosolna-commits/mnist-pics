#mnist-pics.py
# streamlit run mnist-pics.py

import streamlit as st
import numpy as np
import os

from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas


import os


st.write(os.getcwd())

# =========================================
# SKAPA MAPpar
# =========================================

for i in range(10):
    os.makedirs(f"data/{i}", exist_ok=True)


# =========================================
# STREAMLIT
# =========================================

st.set_page_config(page_title="MNIST Drawing App")

st.title("Rita en siffra och spara till dataset")

label = st.selectbox(
    "Vilken siffra ritar du?",
    list(range(10))
)


# =========================================
# CANVAS
# =========================================

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


# =========================================
# SPARA
# =========================================

if st.button("Spara bild"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data

        # RGBA -> grayscale
        img = Image.fromarray(
            (img[:, :, 0]).astype(np.uint8)
        )

        # Resize till MNIST-format
        img = img.resize((28, 28))

        # =====================================
        # FÖRSÖK SPARA LOKALT/SERVER
        # =====================================

        try:

            os.makedirs(f"data/{label}", exist_ok=True)

            count = len(os.listdir(f"data/{label}"))

            filename = f"data/{label}/{count}.png"

            img.save(filename)

            st.success(f"Sparade: {filename}")

        except Exception as e:

            st.warning(f"Kunde inte spara på servern: {e}")

        # =====================================
        # DOWNLOAD-KNAPP
        # =====================================

        buffer = BytesIO()

        img.save(buffer, format="PNG")

        buffer.seek(0)

        st.download_button(
            label="Ladda ner bilden",
            data=buffer,
            file_name=f"mnist_{label}.png",
            mime="image/png"
        )

        # Visa bilden
        st.image(
            img,
            caption="28x28-bild",
            width=150
        )

    else:

        st.warning("Rita en siffra först")
      
        

