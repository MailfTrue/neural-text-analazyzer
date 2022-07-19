import spacy
from fastapi import FastAPI

app = FastAPI()
emotion_spacy = spacy.load("models/emotion_model")


@app.get("/")
@app.post("/")
async def check(text: str):
    result = emotion_spacy(text)
    return {k: round(v, 3) for k, v in result.cats.items()}
