import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

st.set_page_config(page_title="Classificador de Dígits", layout="centered")
st.title("🔢 Classificador de Dígits Manuscrits 🧠")
st.markdown("Puja una imatge en blanc i negre (28x28) d’un dígit i la IA et dirà quin número veu (0-9).")

uploaded_file = st.file_uploader("📤 Pujar imatge (jpg, png)", type=["jpg", "jpeg", "png"])

if not os.path.exists("model_mnist_digits.json") or not os.path.exists("model_mnist_digits.weights.h5"):
    st.error("❌ El model no s'ha trobat. Assegura't que els fitxers JSON i WEIGHTS estiguin pujats correctament al teu repositori.")
else:
    with open("model_mnist_digits.json", "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    model.load_weights("model_mnist_digits.weights.h5")

    if uploaded_file:
        try:
            # Obrim i processem la imatge
            image = Image.open(uploaded_file).convert("L").resize((28, 28))  # L = escala de grisos
            st.image(image, caption='📷 Imatge pujada', use_container_width=True)

            img_array = np.array(image) / 255.0  # Normalitzem
            img_array = np.expand_dims(img_array, axis=0)  # Afegim dimensió batch
            img_array = np.expand_dims(img_array, axis=-1)  # Afegim canal

            prediction = model.predict(img_array)
            predicted_digit = int(np.argmax(prediction))

            st.success(f"✏️ Aquesta imatge és el número **{predicted_digit}**")

        except UnidentifiedImageError:
            st.error("❌ No s'ha pogut llegir la imatge. Si us plau, puja un arxiu .jpg o .png vàlid.")
