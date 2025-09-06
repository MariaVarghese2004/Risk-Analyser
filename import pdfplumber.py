import pdfplumber
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressio
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Extract text from PDF
def extract_pdf_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# Example dataset (replace with your own extracted reports)
data = {
    "note_text": [
        "Patient has hypertension and chest pain.",
        "Routine checkup, no major health issues.",
        "Diabetic with complications, insulin dependent.",
        "Healthy individual, no chronic conditions.",
        "Kidney failure, dialysis required urgently."
    ],
    "label": [1, 0, 1, 0, 1]  # 1 = High Risk, 0 = Low Risk
}
df = pd.DataFrame(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["note_text"], df["label"], test_size=0.2, random_state=42
)

# Build pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=200))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "risk_model.pkl")

# Example: Predict risk from a new PDF
pdf_text = extract_pdf_text("sample_medical_report.pdf")
model = joblib.load("risk_model.pkl")
risk_pred = model.predict([pdf_text])[0]
risk_prob = model.predict_proba([pdf_text])[0]

print("Predicted Risk:", "HIGH" if risk_pred == 1 else "LOW")
print("Probabilities:", {"low": risk_prob[0], "high": risk_prob[1]})
