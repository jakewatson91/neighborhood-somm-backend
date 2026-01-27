import json
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# SETUP
url = "YOUR_SUPABASE_URL"
key = "YOUR_SUPABASE_SERVICE_ROLE_KEY" # Needed for writing data
supabase: Client = create_client(url, key)

# MODEL (Matches the database vector size)
model = SentenceTransformer('all-MiniLM-L6-v2') 

def upload():
    with open('data/final_enriched_inventory.json', 'r') as f:
        inventory = json.load(f)

    print(f"🚀 Uploading {len(inventory)} wines...")

    for wine in inventory:
        # 1. Create the Semantic String (The "Vibe" source)
        # We mix Tags + Description + Inferred Features
        text_to_embed = f"{wine['title']} {wine['clean_desc']} {' '.join(wine.get('tags', []))} {wine.get('inferred_features', {}).get('body', '')}"
        
        # 2. Vectorize
        embedding = model.encode(text_to_embed).tolist()

        # 3. Insert
        data = {
            "title": wine['title'],
            "price": wine['price'],
            "tags": wine['tags'],
            "description": wine['clean_desc'],
            "image_url": wine['image_url'],
            "product_handle": wine['handle'],
            "inferred_features": wine['inferred_features'],
            "embedding": embedding
        }
        
        supabase.table("wines").insert(data).execute()

    print("✅ Done!")

if __name__ == "__main__":
    upload()