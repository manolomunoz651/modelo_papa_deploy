import streamlit as st
from PIL import Image
import pandas as pd

import numpy as np
from ultralytics import YOLO



@st.cache_data
def cargar_imagen(image_file):
    img = Image.open(image_file)
    return img

@st.cache_data
def modelo(archivo_imagen):
    # Abrir con PIL
    img = Image.open(archivo_imagen).convert("RGB")

    # Convertir a array de NumPy
    img_np = np.array(img)
    # Run inference
    model = YOLO("pesos_papas_best.pt")
    results = model(img_np)

    # Get the annotated image
    annotated_img = results[0].plot()  # This returns a numpy array

    # Convert numpy array to PIL Image
    #pil_img = Image.fromarray(annotated_img)

    return annotated_img

def main():
    st.title("Enfermedades en papa")
    menu = ["Modelo"]
    eleccion = st.sidebar.selectbox("Menú", menu)

    if eleccion == "Modelo":
        st.subheader("Imagen")
        archivo_imagen = st.file_uploader("Subir Imágenes", type=["png", "jpg", "jpeg"])
        if archivo_imagen is not None:
            detalles_archivo = {"nombre_archivo": archivo_imagen.name,
                                "tipo_archivo": archivo_imagen.type,
                                "tamaño_archivo": archivo_imagen.size}
            st.write(detalles_archivo)
            st.image(cargar_imagen(archivo_imagen), width=250)
            st.image(modelo(archivo_imagen), width=500)




if __name__ == '__main__':
    main()


def cargar_imagen(image_file):
    img = Image.open(image_file)
    return img