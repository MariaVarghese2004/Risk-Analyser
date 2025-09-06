from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os
from io import BytesIO
from PIL import Image

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load your trained model
model = joblib.load("transcriptions_risk_model.pkl")

# -----------------------------
# PDF Extraction Helpers
# -----------------------------
def extract_text_from_pdf_file(file_bytes):
    text = ""
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_scanned_pdf_file(file_bytes):
    images = convert_from_path(BytesIO(file_bytes))
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text.strip()

def smart_extract_pdf_file(file_bytes):
    text = extract_text_from_pdf_file(file_bytes)
    if len(text.strip()) == 0:
        text = extract_text_from_scanned_pdf_file(file_bytes)
    return text

# -----------------------------
# FastAPI Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    extracted_text = smart_extract_pdf_file(contents)
    
    pred = model.predict([extracted_text])[0]
    prob = model.predict_proba([extracted_text])[0]

    result_text = f"Risk: {'HIGH' if pred == 1 else 'LOW'} | Accuracy: {max(prob):.2f}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": result_text
    })
