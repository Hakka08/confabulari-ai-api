import joblib
import requests
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?export=download&id=176bL6Ez1X2K9JMe9qZ0idO2lxE98at78"
MODEL_PATH = "pipelineWEB.pkl"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Scarico il modello da Google Drive...")
        response = requests.get(MODEL_URL, stream=True)

        if response.status_code != 200:
            raise Exception(f"Errore download: {response.status_code}")

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("Modello scaricato correttamente!")


print("Controllo modello...")
download_model()
pipeline = joblib.load(MODEL_PATH)
print("Modello caricato!")


@app.route("/")
def home():
    return "API online!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("text", "")
    prediction = pipeline.predict([data])[0]
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
