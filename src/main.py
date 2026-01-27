import json
import random
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq 

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
                    "content": """
                    You are a hip sommelier. Be cool but not over the top.
                    Explain your reasoning for your recommendation as it specifically relates to the user preferences. 
                    Be casual. Mention the user's vibe and food pairings when applicable.
                    """
                },
                {
                    "role": "user",
                    "content": f"User Vibe: {vibe}\nWine: {wine['title']}\nDescription: {wine.get('description', '')}"
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

    # --- FIX 3: RETURN THE LIST (TOP 50) ---
    # We slice [:50] so the response isn't huge, but gives plenty of shuffle options
    top_indices = sorted_indices[:50]
    
    results = []
    for idx in top_indices:
        w = candidates[idx].copy()
        # Remove heavy vector data before sending to frontend
        if 'embedding' in w: del w['embedding']
        # Add score for debugging/UI
        w['match_score'] = float(scores[idx])
        results.append(w)
    
    # Generate note ONLY for the #1 result to start
    first_wine = results[0]
    ai_note = await generate_sommelier_note(prefs.vibe, first_wine)

    print(f"Found wine: {first_wine.get('title')}") 
    print(f"Wine features: {first_wine.get('features')} | {first_wine.get('tags')}")
    print(f"Reasoning: {ai_note}")

    return {
        "results": results, # Frontend will store this array
        "note": ai_note     # Note for index 0
    }

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