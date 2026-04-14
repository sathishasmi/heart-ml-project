from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
import joblib

templates = Jinja2Templates(directory="templates")

app = FastAPI()

# Load model once (important)
try:
    model = joblib.load("heart_model.pkl")
except Exception as e:
    print("Model loading error:", e)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request,
        context={
            "request": request,
            "prediction": None
        }
    )

from fastapi.responses import HTMLResponse

@app.post("/predict_form", response_class=HTMLResponse)
def predict_form(
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
    data = pd.DataFrame([{
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

    prediction = model.predict(data)[0]

    result = "Heart Disease Detected " if prediction == 1 else "No Heart Disease "

    return templates.TemplateResponse(
    name="index.html",
    request=request,
    context={
        "request": request,
        "prediction": result
    }
    )