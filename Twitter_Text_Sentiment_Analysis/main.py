from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import predict_sentiment

# Create FastAPI instance
app = FastAPI()


# Define request body model
class SentenceInput(BaseModel):
    sentence: str


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Character Counter API"}


# Endpoint to count characters
@app.post("/count-chars")
async def count_characters(input: SentenceInput):
    char_count = len(input.sentence)
    return {"sentence": input.sentence, "character_count": char_count}


@app.post("/predict-sentiment")
async def count_characters(input: SentenceInput):
    return {"sentence": input.sentence, "sentiment": predict_sentiment(input.sentence)}


# To run the app, use this command in terminal:
# uvicorn main:app --host 0.0.0.0 --port 8000
