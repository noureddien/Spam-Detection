import joblib
from preprocess import transform_text
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "weights"

# Load saved weights
tfidf = joblib.load(MODEL_DIR / "vectorizer.pkl")
model = joblib.load(MODEL_DIR / "best_model.pkl")
def classify_message(message: str) -> str:
    # 1. Preprocess
    cleaned = transform_text(message)
    
    # 2. Vectorize
    vectorized = tfidf.transform([cleaned]).toarray()
    
    # 3. Predict
    prediction = model.predict(vectorized)[0]
    
    return "Spam" if prediction == 1 else "Ham (Legitimate)"

# Test execution
if __name__ == "__main__":
    test_msg = "Guaranteed loan up to $5,000 with no credit check! Apply now at cash-fast-now.com."
    print("Input Message:", test_msg)
    print("Result:", classify_message(test_msg))