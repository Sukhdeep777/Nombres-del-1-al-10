import streamlit as st
from tensorflow.keras.models import model_from_json
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

# Configuración de la página de Streamlit
st.set_page_config(page_title="Clasificador de Dígitos", layout="centered")
st.title("🔢 Clasificador de Dígitos Manuscritos 🧠")
st.markdown("Sube una imagen en blanco y negro (28x28) de un dígito y la IA te dirá qué número ve (0-9).")

# Cargar los archivos del modelo si están disponibles
if not os.path.exists("model_mnist_digits_complex.json") or not os.path.exists("model_mnist_digits_complex.weights.h5"):
    st.error("❌ El modelo no se ha encontrado. Asegúrate de que los archivos JSON y WEIGHTS estén correctamente subidos.")
else:
    with open("model_mnist_digits_complex.json", "r") as json_file:
        model_json = json_file.read()

    # Cargar el modelo y los pesos
    model = model_from_json(model_json)
    model.load_weights("model_mnist_digits_complex.weights.h5")

    # Subir archivo
    uploaded_file = st.file_uploader("📤 Subir imagen (jpg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            # Abrir y procesar la imagen
            image = Image.open(uploaded_file).convert("L").resize((28, 28))  # Convertir a escala de grises
            st.image(image, caption='📷 Imagen subida', use_container_width=True)

            # Preprocesar la imagen
            img_array = np.array(image) / 255.0  # Normalizar
            img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch
            img_array = np.expand_dims(img_array, axis=-1)  # Añadir canal

            # Realizar la predicción
            prediction = model.predict(img_array)
            predicted_digit = int(np.argmax(prediction))

            # Mostrar el resultado
            st.success(f"✏️ Esta imagen es el número **{predicted_digit}**")

        except UnidentifiedImageError:
            st.error("❌ No se ha podido leer la imagen. Por favor, sube un archivo .jpg o .png válido.")
