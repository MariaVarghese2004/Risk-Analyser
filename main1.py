import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os


# -----------------------------
# PDF Extraction Helpers
# -----------------------------
def extract_text_from_pdf(pdf_path):
    """Try extracting text directly (works if it's a digital PDF)."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_scanned_pdf(pdf_path):
    """Fallback OCR method for scanned PDFs."""
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text.strip()


def smart_extract_pdf(pdf_path):
    """Try direct text extraction; fallback to OCR if blank."""
    text = extract_text_from_pdf(pdf_path)
    if len(text.strip()) == 0:
        print("[Warning] No text found, switching to OCR...")
        text = extract_text_from_scanned_pdf(pdf_path)
    return text


# -----------------------------
# Train Risk Prediction Model
# -----------------------------
print("Loading dataset...")
df = pd.read_csv("mtsamples.csv")

# Create synthetic "risk" labels
df["label"] = df["transcription"].str.contains(
    r"cancer|critical|failure|severe", case=False, na=False
).astype(int)

X = df["transcription"].fillna("")
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
    ("clf", LogisticRegression(max_iter=500))
])

print("Training model...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Evaluation:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "transcriptions_risk_model.pkl")


# -----------------------------
# Run Prediction on PDF
# -----------------------------
sample_pdf = "maria.pdf"  # replace with your PDF file

if os.path.exists(sample_pdf):
    print(f"\nExtracting from PDF: {sample_pdf}")
    extracted_text = smart_extract_pdf(sample_pdf)

    print("\n--- Extracted Text Preview ---")
    print(extracted_text[:500])  # show first 500 chars

    model = joblib.load("transcriptions_risk_model.pkl")
    pred = model.predict([extracted_text])[0]
    prob = model.predict_proba([extracted_text])[0]

    print("\n--- Prediction Result ---")
    print("Risk:", "HIGH" if pred == 1 else "LOW")
    print("Probabilities:", {"low": prob[0], "high": prob[1]})
else:
    print("⚠️ sample_report.pdf not found. Please add a PDF file in this directory.")
