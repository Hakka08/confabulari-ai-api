from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import joblib
from deep_translator import GoogleTranslator
from langdetect import detect
import numpy as np

# ---------------------------------------------------------
# INIZIALIZZAZIONE APP
# ---------------------------------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "API attiva"}


# ---------------------------------------------------------
# CARICAMENTO MODELLI
# ---------------------------------------------------------

embedder = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
clf = joblib.load("classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Modelli caricati con successo.")


# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if text == "":
            return jsonify({"error": "Campo 'text' mancante"}), 400

        # Traduzione verso inglese
        text_en = GoogleTranslator(source="auto", target="en").translate(text)

        # Lingua originale
        lang = detect(text)

        # EMBEDDING (768-d)
        emb = embedder.encode(
            [text_en],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # SCALING + PCA
        emb_scaled = scaler.transform(emb)
        emb_pca = pca.transform(emb_scaled)

        # CLASSIFICAZIONE
        pred_class = clf.predict(emb_pca)[0]
        category = label_encoder.inverse_transform([pred_class])[0]

        return jsonify({
            "category": category,
            "language": lang,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# LOCALE
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
