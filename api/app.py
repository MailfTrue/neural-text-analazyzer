import spacy
from fastapi import FastAPI, Body

app = FastAPI()
emotion_spacy = spacy.load("models/emotion_model")


@app.post("/")
async def check(text: str = Body(...)):
    result = emotion_spacy(text)
    return {k: round(v, 3) for k, v in result.cats.items()}
