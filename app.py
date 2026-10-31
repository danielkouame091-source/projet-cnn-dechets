import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (128, 128)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dechets_cnn.keras")

@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

st.set_page_config(page_title="Tri des déchets", page_icon="♻️")
st.title("♻️ Tri automatique des déchets")
st.write(
    "Télécharge une photo d’un déchet. Le modèle prédit s’il s’agit de "
    "**plastique**, **papier**, **verre** ou **organique**."
)

uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image envoyée", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index])

    st.subheader("Résultat")
    st.success(f"Catégorie prédite : **{predicted_class}**")
    st.write(f"Confiance : **{confidence:.2%}**")

    st.subheader("Scores par catégorie")
    for name, score in zip(class_names, prediction):
        st.write(f"{name} : {score:.2%}")
        st.progress(float(score))
