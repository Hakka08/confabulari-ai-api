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

print("Carico embedder da HuggingFace...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

print("Carico scaler, pca, classifier, label encoder...")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
clf = joblib.load("classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Modelli caricati con successo.")


# ---------------------------------------------------------
# PREDICT ENDPOINT
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if text == "":
            return jsonify({"error": "Campo 'text' mancante"}), 400

        # Traduzione auto → inglese
        text_en = GoogleTranslator(source="auto", target="en").translate(text)

        # Lingua originale
        lang = detect(text)

        # EMBEDDING
        emb = embedder.encode([text_en],
                              convert_to_numpy=True,
                              normalize_embeddings=True)

        # SCALING + PCA
        emb_scaled = scaler.transform(emb)
        emb_pca = pca.transform(emb_scaled)

        # CLASSIFICAZIONE
        pred_class = clf.predict(emb_pca)[0]
        category = label_encoder.inverse_transform([pred_class])[0]

        return jsonify({
            "category": category,
            "language": lang,
            "translated_text": text_en
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------
# RUN LOCALE (Railway ignora questa parte)
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
