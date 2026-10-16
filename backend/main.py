
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "phishlens_tfidf_logreg.joblib"


# --------------------------------------------------
# 2. Load the trained machine learning model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="PhishLens AI",
    description=(
        "An educational email classification API "
        "using TF-IDF and Logistic Regression."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# 4. Configure CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


# --------------------------------------------------
# 5. Define the request format
# --------------------------------------------------

class EmailRequest(BaseModel):
    body: str = Field(
        min_length=1,
        max_length=100_000,
        description="The email body to analyze.",
    )


# --------------------------------------------------
# 6. Health check endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "PhishLens AI",
        "status": "running",
        "model_loaded": True,
    }


# --------------------------------------------------
# 7. Email analysis endpoint
# --------------------------------------------------

@app.post("/analyze")
def analyze_email(request: EmailRequest):
    email_text = request.body.strip()

    if not email_text:
        raise HTTPException(
            status_code=422,
            detail="Email body cannot be empty.",
        )

    try:
        prediction = int(model.predict([email_text])[0])
        probabilities = model.predict_proba([email_text])[0]

        classes = list(model.classes_)
        positive_index = classes.index(1)

        positive_probability = float(
            probabilities[positive_index]
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Email analysis failed.",
        )

    if prediction == 1:
        classification = "suspicious"
        message = (
            "The email belongs to the suspicious "
            "class in the training dataset."
        )
    else:
        classification = "safe"
        message = (
            "The model classified this email as safe. "
            "This does not guarantee that it is harmless."
        )

    return {
        "project": "PhishLens AI",
        "classification": classification,
        "prediction": prediction,
        "suspicious_class_probability": round(
            positive_probability, 4
        ),
        "message": message,
        "disclaimer": (
            "This is an educational ML classification result, "
            "not a security guarantee. Class 1 combines phishing, "
            "spam, and scam emails in the source dataset."
        ),
    }