import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = Path(__file__).parent
MODEL_PATH = HERE / "welfake_tfidf_linsvc.joblib"

app = FastAPI(title="Fake News Detector", version="1.0.0")


try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    print(f"model failed to load: {exc}")
    model = None


class Article(BaseModel):
    title: str = ""
    text: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}




@app.post("/predict")
def predict(article: Article):
    if model is None:
        raise HTTPException(status_code=503, detail="Model failed to load")


    doc = (article.title + " " + article.text).strip()
    if not doc:
        raise HTTPException(status_code=422, detail="Send a title or some text")


    pred = int(model.predict([doc])[0])
    p_real, p_fake = model.predict_proba([doc])[0]


    return {
        "prediction": pred,"verdict": "fake" if pred == 1 else "real","p_real": float(p_real),"p_fake": float(p_fake),
    }


app.mount("/", StaticFiles(directory=HERE / "static", html=True), name="static")