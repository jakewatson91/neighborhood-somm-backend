import json
import re
import ast
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder, util

def strip_html(text):
    if not isinstance(text, str): return ""
    p = re.compile(r'<.*?>')
    return p.sub('', text)

def semantic_join():
    print("⏳ Loading Pipeline...")
    # Fast Retriever
    retriever = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    # Smart Reranker
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    with open(BASE_DIR / "data/neighborhood_full_inventory.json", 'r') as f:
        raw_inventory = json.load(f)
    encyclopedia_df = pd.read_csv(BASE_DIR / "data/x-wines/XWines_Slim_1K_wines.csv").fillna('')

    # 1. PREP ENCYCLOPEDIA (Nomic Documents)
    print("🧠 Indexing Encyclopedia...")
    enc_texts = [
        f"search_document: {r['Type']} wine. {r['Grapes']}. {r['WineryName']} {r['WineName']}"
        for _, r in encyclopedia_df.iterrows()
    ]
    enc_vecs = retriever.encode(enc_texts, convert_to_tensor=True)

    # 2. MATCHING LOOP
    enriched_inventory = []
    shelf_items = [i for i in raw_inventory if i.get('product_type') == "Wine" and i.get('images')]

    print(f"🔍 Processing {len(shelf_items)} wines...")
    for item in shelf_items:
        # Construct Search Query
        tags = " ".join(item.get('tags', []))
        query = f"search_query: {tags} {item.get('title','')}"
        
        # A. RETRIEVE TOP 10 (Nomic)
        query_vec = retriever.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_vec, enc_vecs, top_k=10)[0]
        
        # B. RE-RANK (BGE Reranker)
        candidate_indices = [hit['corpus_id'] for hit in hits]
        # Prepare pairs for the judge: [ [Query, Doc1], [Query, Doc2]... ]
        pairs = [
            [f"{tags} {item['title']}", enc_texts[idx].replace("search_document: ", "")] 
            for idx in candidate_indices
        ]
        
        scores = reranker.predict(pairs)
        best_idx = candidate_indices[scores.argmax()]
        
        # 3. MERGE WINNER DATA
        row = encyclopedia_df.iloc[best_idx]
        try:
            pairings = ast.literal_eval(row['Harmonize']) if isinstance(row['Harmonize'], str) else []
        except: pairings = []

        item['inferred_features'] = {
            'body': row['Body'],
            'acidity': row['Acidity'],
            'type': row['Type'],
            'grape': row['Grapes'],
            'pairings': pairings
        }
        
        # 4. FINAL APP EMBEDDING
        search_blob = f"{item['title']} {row['Type']} {row['Grapes']} {' '.join(pairings)}"
        item['embedding'] = retriever.encode(f"search_document: {search_blob}").tolist()
        enriched_inventory.append(item)

    # SAVE
    with open(BASE_DIR / "data/final_enriched_inventory.json", 'w') as f:
        json.dump(enriched_inventory, f, indent=4, ensure_ascii=False)

    print("🎉 Enrichment Complete.")

if __name__ == "__main__":
    semantic_join()