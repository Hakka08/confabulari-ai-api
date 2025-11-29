from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import joblib
from deep_translator import GoogleTranslator
from langdetect import detect

app = Flask(__name__)     # <--- PRIMA COSA DA FARE: crea l'app Flask

@app.route("/")
def home():
    return {"message": "API attiva"}


print("Carico il modello HuggingFace...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

print("Carico scaler, pca, label encoder...")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Modelli caricati.")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        frase = data.get("text", "")

        # Traduzione verso inglese
        frase_en = GoogleTranslator(source='auto', target='en').translate(frase)

        # Lingua originale
        lang = detect(frase)

        # Embedding
        emb = embedder.encode(
            [frase_en],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        emb_scaled = scaler.transform(emb)
        emb_pca = pca.transform(emb_scaled)

        pred = label_encoder.inverse_transform(
            [label_encoder.classes_[pca.predict(emb_pca)][0]]
        )

        return jsonify({
            "category": pred[0],
            "language": lang
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
