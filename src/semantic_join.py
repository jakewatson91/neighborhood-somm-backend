import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

def strip_html(desc):
    import re
    clean_desc = re.sub(r'<[^>]+>', ' ', desc)
    clean_desc = " ".join(clean_desc.split())
    return clean_desc

def semantic_join():
    print("⏳ Loading Models & Data...")
    # 1. Load the Model (Free, Local, Fast)
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

    # 2. Load "The Shelf" (Neighborhood Wines)
    BASE_DIR = Path(__file__).resolve().parent.parent
    with open(BASE_DIR / "data" / "neighborhood_full_inventory.json", 'r') as f:
        shelf_data = json.load(f)
    
    # Create simple text representations for the Shelf
    # "Producer + Title + Body" gives the best context
    shelf_texts = [
        f"{item.get('vendor', '')} {item['title']} {item.get('body_html', '')}" 
        for item in shelf_data
    ]

    # 3. Load "The Encyclopedia" (X-Wines)
    # We only need a subset to match against (Title + Grapes + Region)
    encyclopedia_df = pd.read_csv('data/x-wines/XWines_Slim_1K_wines.csv')
    
    # Create text representations for the Encyclopedia
    encyclopedia_texts = [
        f"{row['WineryName']} {row['WineName']} {row['RegionName']} {row['Grapes']}" 
        for _, row in encyclopedia_df.iterrows()
    ]

    print("🧠 Vectorizing Datasets (This might take a minute)...")
    shelf_embeddings = model.encode(shelf_texts, convert_to_tensor=True)
    encyclopedia_embeddings = model.encode(encyclopedia_texts, convert_to_tensor=True)

    print("🔗 Performing Semantic Join...")
    # Find the top 1 closest match in the Encyclopedia for every Shelf Item
    hits = util.semantic_search(shelf_embeddings, encyclopedia_embeddings, top_k=1)

    enriched_inventory = []

    for i, hit in enumerate(hits):
        match_score = hit[0]['score']
        match_idx = hit[0]['corpus_id']
        
        item = shelf_data[i]

        item['description'] = strip_html(item['description'])
        
        # QUALITY GATE: Only enrich if similarity is > 60%
        # Otherwise, we might be matching a "Gift Card" to a "Merlot"
        if match_score > 0.6:
            knowledge = encyclopedia_df.iloc[match_idx]
            
            # Enrich the item
            item['inferred_features'] = {
                'match_confidence': float(match_score),
                'grape': knowledge['Grapes'],
                'acidity': knowledge['Acidity'],
                'body': knowledge['Body'],
                'pairings': knowledge.get('Harmonize', []) # If available
            }
            # Add these keywords to the tags for easier UI filtering later
            if 'tags' not in item: item['tags'] = []
            if isinstance(item['tags'], list):
                item['tags'].append(f"Grape: {knowledge['Grapes']}")
        else:
            item['inferred_features'] = None

        enriched_inventory.append(item)

    # 4. Save the "Smart" Inventory
    with open(BASE_DIR / "data" / "final_enriched_inventory.json", 'w') as f:
        json.dump(enriched_inventory, f, indent=4)

    print(f"🎉 Success! Enriched inventory saved. (Checked {len(shelf_data)} items)")

if __name__ == "__main__":
    inv = semantic_join()
    print(inv[0])