import joblib, numpy as np
from features import prepare_image, extract

CLASSES = {0: "Voiture", 1: "Bateau"}

def load_artifacts(folder="."):
    model = joblib.load(f"{folder}/model.pkl")
    scaler = joblib.load(f"{folder}/scaler.pkl")
    with open(f"{folder}/model_name.txt", encoding="utf-8") as f:
        name = f.read().strip()
    return model, scaler, name

def predict_image(pil_image, model, scaler):
    """PIL image -> (classe, {classe: probabilité} ou None)."""
    rgb = np.array(pil_image.convert("RGB"))
    x = scaler.transform(extract(prepare_image(rgb)).reshape(1, -1))
    label = int(model.predict(x)[0])
    probs = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)[0]
        probs = {CLASSES[int(c)]: float(v) for c, v in zip(model.classes_, p)}
    return CLASSES[label], probs
