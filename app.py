from flask import Flask, request, jsonify
import joblib
from deep_translator import GoogleTranslator
from langdetect import detect
import os

# Carica il modello una volta sola
pipeline = joblib.load("pipelineWEB.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        frase = data.get("text", "")

        sentence = GoogleTranslator(source='auto', target='en').translate(frase)
        language_detected = detect(frase)

        embedding = pipeline["embedder"].encode(
            [sentence],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embedding_scaled = pipeline["scaler"].transform(embedding)
        embedding_pca = pipeline["pca"].transform(embedding_scaled)
        prediction = pipeline["classifier"].predict(embedding_pca)
        category = pipeline["label_encoder"].inverse_transform(prediction)[0]

        return jsonify({
            "category": category,
            "language": language_detected
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
