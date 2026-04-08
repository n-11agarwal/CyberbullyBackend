from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

app = Flask(__name__)

MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATASET_FILE = "final_hateXplain.csv"

def load_or_train_model():
    global model, vectorizer

    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        print("✅ Loading saved model...")
        model = pickle.load(open(MODEL_FILE, 'rb'))
        vectorizer = pickle.load(open(VECTORIZER_FILE, 'rb'))

    else:
        print("🔄 Training new model...")

        data = pd.read_csv(DATASET_FILE)

        # ✅ YOUR COLUMN
        X = data['comment'].str.lower()
        y = data['label']

        # Vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
        X_vec = vectorizer.fit_transform(X)

        # Model
        model = LogisticRegression(max_iter=500, class_weight='balanced')
        model.fit(X_vec, y)

        # Save
        pickle.dump(model, open(MODEL_FILE, 'wb'))
        pickle.dump(vectorizer, open(VECTORIZER_FILE, 'wb'))

        print("✅ Model trained & saved!")

load_or_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data['text'].strip().lower()

        vec = vectorizer.transform([text])
        result = model.predict(vec)[0]

        # 🔥 IMPORTANT MAPPING
        if result in ["offensive", "hatespeech"]:
            final_result = "bullying"
        else:
            final_result = "not_bullying"

        return jsonify({
            "prediction": final_result
        })

    except Exception as e:
        return jsonify({
            "prediction": "error",
            "message": str(e)
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    print("🚀 Server running...")
    app.run(host="0.0.0.0", port=5000, debug=True)