#mnist_pics.py

import streamlit as st
import numpy as np
import cv2

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from streamlit_drawable_canvas import st_canvas

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# =========================================
# STREAMLIT
# =========================================

st.set_page_config(page_title="MNIST MLP + CNN", layout="centered")

st.title("Rita en siffra (0–9)")


# =========================================
# LADDA DATA + TRÄNA MODELLER
# =========================================

@st.cache_resource
def load_models():

    mnist = fetch_openml("mnist_784", version=1)

    X = mnist.data.astype(np.float32).to_numpy()
    y = mnist.target.astype(np.int32)

    X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =========================================
    # MLP
    # =========================================

    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=20,
        random_state=42,
        verbose=True
    )

    mlp.fit(X_train, y_train)

    mlp_pred = mlp.predict(X_test)

    mlp_acc = accuracy_score(y_test, mlp_pred)

    # =========================================
    # CNN
    # =========================================

    X_train_cnn = X_train.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test.reshape(-1, 28, 28, 1)

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    cnn = Sequential([

        Conv2D(
            32,
            (3,3),
            activation="relu",
            input_shape=(28,28,1)
        ),

        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation="relu"),

        MaxPooling2D((2,2)),

        Flatten(),

        Dense(128, activation="relu"),

        Dense(10, activation="softmax")
    ])

    cnn.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    cnn.fit(
        X_train_cnn,
        y_train_cat,
        epochs=3,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    loss, cnn_acc = cnn.evaluate(
        X_test_cnn,
        y_test_cat,
        verbose=0
    )

    return mlp, cnn, mlp_acc, cnn_acc


with st.spinner("Tränar modeller första gången..."):

    mlp_model, cnn_model, mlp_acc, cnn_acc = load_models()


st.success(f"MLP accuracy: {mlp_acc:.4f}")
st.success(f"CNN accuracy: {cnn_acc:.4f}")


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
# PREDIKTION
# =========================================

if st.button("Skicka"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data

        # RGBA -> grayscale
        img = cv2.cvtColor(
            img.astype(np.uint8),
            cv2.COLOR_RGBA2GRAY
        )

        # Resize till MNIST-format
        img = cv2.resize(img, (28, 28))

        # Blur
        img = cv2.GaussianBlur(img, (3,3), 0)

        # Normalisera
        img = img / 255.0

        # Kontroll om tom canvas
        if np.sum(img) < 5:

            st.warning("Rita en siffra först")

        else:

            # =========================================
            # MLP
            # =========================================

            img_flat = img.reshape(1, 784)

            mlp_probs = mlp_model.predict_proba(img_flat)[0]

            mlp_pred = np.argmax(mlp_probs)

            mlp_conf = mlp_probs[mlp_pred] * 100


            # =========================================
            # CNN
            # =========================================

            img_cnn = img.reshape(1, 28, 28, 1)

            cnn_probs = cnn_model.predict(
                img_cnn,
                verbose=0
            )[0]

            cnn_pred = np.argmax(cnn_probs)

            cnn_conf = cnn_probs[cnn_pred] * 100


            # =========================================
            # RESULTAT
            # =========================================

            st.markdown(
                f"## MLP: {mlp_pred} ({mlp_conf:.1f}%)"
            )

            st.markdown(
                f"## CNN: {cnn_pred} ({cnn_conf:.1f}%)"
            )

            st.image(
                img,
                width=150,
                caption="28x28-bild som modellerna ser"
            )