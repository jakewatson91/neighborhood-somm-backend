import json
import random
import os
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq 
from supabase import create_client, Client
from huggingface_hub import InferenceClient

from .prompts import SOMMELIER_SYSTEM_PROMPT

load_dotenv()

llm_client = AsyncGroq(
    api_key=os.getenv("GROQ_API_KEY")
    )

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserPreferences(BaseModel):
    vibe: str
    type: str = "Any"
    maxPrice: float = 100.0
    shuffle: bool = False
    excludeIds: List[int] = []

# New Model for the "Shuffle" endpoint
class NoteRequest(BaseModel):
    vibe: str
    wine: dict

# --- HELPER: ASYNC LLM GENERATION ---
# FIX 2: Added 'async' keyword here
async def generate_sommelier_note(vibe: str, wine: dict):
    try:
        completion = await llm_client.chat.completions.create(
        model="moonshotai/kimi-k2-instruct-0905",
        messages=[
            {
                "role": "system",
                "content": SOMMELIER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"User Vibe: {vibe}\nWine: {json.dumps(wine)}"
            }
        ],
        temperature=0.9,
        max_tokens=125
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        return f"A perfect match for your vibe: {wine['title']}."

@app.post("/find-wine")
async def find_wine(prefs: UserPreferences):
    print(f"\n🔎 SEARCHING | Vibe: '{prefs.vibe}' | Type: {prefs.type} | Max Price: ${prefs.maxPrice}")

    vibe_vector = hf_client.feature_extraction(
        text=prefs.vibe,
        model="sentence-transformers/all-MiniLM-L6-v2"
        ).tolist()

    # 2. Ask Supabase to do the math and filtering
    # We ask for 20 so we have a pool to "Shuffle" from
    rpc_params = {'query_embedding': vibe_vector, 'match_count': 20}
    response = supabase.rpc('match_wines', rpc_params).execute()

    candidates = response.data

    if prefs.shuffle:
        pool = [w for w in candidates if w['id'] not in prefs.excludeIds]
        wine = random.choice(pool) if pool else candidates[0]
    else:
        wine = candidates[0]

    ai_note = await generate_sommelier_note(prefs.vibe, wine)

    print(f"Wine: {wine}")
    print(f"\nNote: {ai_note}")
    return {"wine": wine, "note": ai_note}

# --- FIX 4: NEW ENDPOINT FOR SHUFFLING ---
# Frontend calls this when user clicks "Find me something else"
@app.post("/get-note")
async def get_note_endpoint(req: NoteRequest):
    note = await generate_sommelier_note(req.vibe, req.wine)
    return {"note": note}

@app.get("/")
async def root():
    return {"status": "online", "endpoints": ["/find-wine", "/get-note"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)