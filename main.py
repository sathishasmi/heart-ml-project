from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load model
import os

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "heart_model.pkl")
    model = joblib.load(model_path)
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)
    model = None

# Home page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict_form", response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...)
):
    try:
        # Create dataframe
        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        # Prediction
        if model:
            prediction = model.predict(input_data)
            result = "Heart Disease Detected ❤️" if prediction[0] == 1 else "No Heart Disease ✅"
        else:
            result = "Model not loaded"

        return HTMLResponse("TEST WORKING")

    except Exception as e:
        return HTMLResponse(content=f"Error: {str(e)}")
