import streamlit as st
from PIL import Image
from predictor import load_artifacts, predict_image

st.set_page_config(page_title="Voiture ou Bateau ?", page_icon="🚗")

@st.cache_resource
def charger():
    return load_artifacts(".")

model, scaler, model_name = charger()

st.title("Classification d'images : Voiture ou Bateau")
st.caption("Atelier de classification supervisée — Pr. Abdelouahed Sabri")
st.write(f"Modèle utilisé (le plus performant lors des tests) : **{model_name}**")

fichier = st.file_uploader("Charger une image", type=["jpg", "jpeg", "png", "bmp"])
if fichier is not None:
    try:
        image = Image.open(fichier)
        st.image(image, caption="Image chargée", use_container_width=True)
        classe, probs = predict_image(image, model, scaler)
        st.success(f"Classe prédite : **{classe}**")
        if probs:
            st.write("Confiance du modèle :")
            for nom, p in sorted(probs.items(), key=lambda t: -t[1]):
                st.progress(p, text=f"{nom} : {p:.1%}")
    except Exception as e:
        st.error(f"Impossible de traiter cette image : {e}")
else:
    st.info("Charge une image de voiture ou de bateau pour obtenir la prédiction.")
