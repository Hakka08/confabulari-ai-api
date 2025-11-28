{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from flask import Flask, request, jsonify\
import joblib\
from deep_translator import GoogleTranslator\
from langdetect import detect\
\
# \uc0\u55357 \u56613  CARICA IL MODELLO UNA VOLTA SOLA\
pipeline = joblib.load("pipelineWEB.pkl")\
\
app = Flask(__name__)\
\
@app.route("/predict", methods=["POST"])\
def predict():\
    try:\
        data = request.json\
        frase = data.get("text", "")\
\
        # Traduzione \uc0\u8594  inglese\
        sentence = GoogleTranslator(source='auto', target='en').translate(frase)\
\
        # Riconoscimento lingua originale\
        language_detected = detect(frase)\
\
        # Embedding + PCA + Classificazione\
        embedding = pipeline["embedder"].encode([sentence], \
                                                convert_to_numpy=True,\
                                                normalize_embeddings=True)\
        embedding_scaled = pipeline["scaler"].transform(embedding)\
        embedding_pca = pipeline["pca"].transform(embedding_scaled)\
        prediction = pipeline["classifier"].predict(embedding_pca)\
        category = pipeline["label_encoder"].inverse_transform(prediction)[0]\
\
        return jsonify(\{\
            "category": category,\
            "language": language_detected\
        \})\
\
    except Exception as e:\
        return jsonify(\{"error": str(e)\}), 500\
\
# Render vuole questo\
if __name__ == "__main__":\
    app.run(host="0.0.0.0", port=10000)\
}