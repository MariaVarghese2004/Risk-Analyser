import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# -------------------------
# Risk mapping
# -------------------------
HIGH_RISK = [
    "Cardiology", "Oncology", "Neurology", "Neurosurgery",
    "Pulmonology", "Emergency / Critical Care", "Urology"
]

LOW_RISK = [
    "Dermatology", "Ophthalmology", "General Practice",
    "Dentistry", "ENT - Otolaryngology", "Psychiatry",
    "Gastroenterology", "Pediatrics"
]

def label_risk(specialty):
    if specialty in HIGH_RISK:
        return 1  # HIGH
    elif specialty in LOW_RISK:
        return 0  # LOW
    else:
        return 0  # default LOW unless you want a "MEDIUM" class

# -------------------------
# Load dataset
# -------------------------
# Download mtsamples from Kaggle:
# https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
df = pd.read_csv("mtsamples.csv")

# Add risk label
df["risk_label"] = df["medical_specialty"].apply(label_risk)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["transcription"], df["risk_label"],
    test_size=0.2, random_state=42, stratify=df["risk_label"]
)

# -------------------------
# Pipeline (Vectorizer + Model)
# -------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=500))
])

pipeline.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
acc = pipeline.score(X_test, y_test)
print(f"Validation Accuracy: {acc:.2f}")

# -------------------------
# Save model
# -------------------------
joblib.dump(pipeline, "transcriptions_risk_model.pkl")
print("✅ Model saved as transcriptions_risk_model.pkl")
