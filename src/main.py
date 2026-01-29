import json
import random
import os
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq 

from prompts import SOMMELIER_SYSTEM_PROMPT

load_dotenv()

# SETUP GROQ
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = AsyncGroq(api_key=GROQ_API_KEY)

print("\n🧠 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FIX 1: LOAD THE VECTOR FILE ---
BASE_DIR = Path(__file__).resolve().parent.parent
# Make sure this points to the file created by build_vector.py
VECTOR_DATA_PATH = BASE_DIR / "src/vector_inventory.json"

try:
    with open(VECTOR_DATA_PATH, "r") as f:
        INVENTORY = json.load(f)
    print(f"✅ API loaded {len(INVENTORY)} wines from {VECTOR_DATA_PATH}")
except FileNotFoundError:
    print(f"❌ API Error: Could not find inventory at {VECTOR_DATA_PATH}. Run build_vector.py!")
    INVENTORY = []

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
        completion = await client.chat.completions.create(
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
        max_tokens=200
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq Error: {e}")
        return f"A perfect match for your vibe: {wine['title']}."

@app.post("/find-wine")
async def find_wine(prefs: UserPreferences):
    print(f"\n🔎 SEARCHING | Vibe: '{prefs.vibe}' | Type: {prefs.type} | Max Price: ${prefs.maxPrice}")

    candidates = [
        w for w in INVENTORY 
        if w.get("price", 0) <= prefs.maxPrice and
        (w.get("product_type", "").lower() == "wine")
    ]
    
    # Fallback if filters are too strict
    if not candidates:
        candidates = INVENTORY

    # VECTOR MATH
    vibe_vector = embedding_model.encode(prefs.vibe).reshape(1, -1)
    candidate_vectors = np.array([w["embedding"] for w in candidates])
    scores = cosine_similarity(vibe_vector, candidate_vectors)[0]

    # SORTING
    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:10]

    if prefs.shuffle:
        candidate_pool = [
            idx for idx in top_indices
            if candidates[idx]['id'] not in prefs.excludeIds
        ]
        
        # 3. Pick random
        if candidate_pool:
            wine_idx = random.choice(candidate_pool)
            print(f"🎲 SHUFFLE: Picked random from {len(candidate_pool)} options (Excluded {len(prefs.excludeIds)})")
        else:
            # If they've seen all 20, just show the #1 result again or reset
            wine_idx = sorted_indices[0]
            print("⚠️ SHUFFLE: Pool exhausted, resetting to #1")

    else:
        wine_idx = top_indices[0]
    
    wine = candidates[wine_idx]

    formatted_wine = {
            "id": int(wine['id']),
            "title": wine['title'],
            "handle": wine['handle'],
            "price": wine['price'],
            "image_url": wine['image_url'],
            "product_type": wine['product_type'],
            "description": wine['description'],
            "tags": wine['tags'],
            "features": wine['features'],
            "match_score": float(scores[wine_idx])
        } 
    # Generate note ONLY for the #1 result to start
    ai_note = await generate_sommelier_note(prefs.vibe, formatted_wine)

    print(f"Wine: {formatted_wine}")
    print(f"\nNote: {ai_note}")
    return {"wine": formatted_wine, "note": ai_note}

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
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)