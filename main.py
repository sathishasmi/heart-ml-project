from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import pandas as pd
import joblib

templates = Jinja2Templates(directory="templates")

app = FastAPI()

# Load model once (important)
model = joblib.load("heart_model.pkl")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_form")
def predict_form(
    age = int(age)
    sex = int(sex)
    cp = int(cp)
    trestbps = int(trestbps)
    chol = int(chol)
    fbs = int(fbs)
    restecg = int(restecg)
    thalach = int(thalach)
    exang = int(exang)
    oldpeak = float(oldpeak)
    slope = int(slope)
    ca = int(ca)
    thal = int(thal)
):
    
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

    prediction = model.predict(input_data)
    
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": result}
    )
