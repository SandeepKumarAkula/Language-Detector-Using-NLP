from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model and vectorizer
try:
    with open("language_model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Home route for testing if the server is running
@app.route("/", methods=["GET"])
def home():
    return "Language Detection API is running! Use /detect to send a POST request."

# Language detection route
@app.route("/detect", methods=["POST"])
def detect_language():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        text_vectorized = vectorizer.transform([text])  # Convert text to numerical form
        detected_language = model.predict(text_vectorized)[0]  # Predict language

        return jsonify({"language": detected_language})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
