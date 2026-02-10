import json
import os 
import re
import pickle
import ast
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from supabase import create_client, Client
import kagglehub
from kagglehub import KaggleDatasetAdapter
from dotenv import load_dotenv

load_dotenv()

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(URL, KEY)

BASE_DIR = Path(__file__).resolve().parent.parent

retriever = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
reranker = CrossEncoder('BAAI/bge-reranker-base')
final_vector_model = SentenceTransformer('all-MiniLM-L6-v2')

def strip_html(text):
    if not isinstance(text, str): return ""
    p = re.compile(r'<.*?>')
    return p.sub('', text)

def load_ref_data():
    CACHE_PATH = BASE_DIR / "data/XWines_Full_100K_wines.pkl"

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            data = pickle.load(f)
        return data['df'], data['embeddings'], data['enc_texts']
    
    df = pd.read_csv(BASE_DIR / "data/XWines_Full_100K_wines.zip", compression='zip', low_memory=False).fillna('')
    
    print("Embedding reference df...")
    enc_texts = [
        f"search_document: {r['Type']} wine. {r['Grapes']}. {r['WineryName']} {r['WineName']}. Features: {r['Body']} {r['Acidity']} acidity."
        for _, r in df.iterrows()
    ]
    
    embeddings = retriever.encode(enc_texts, convert_to_tensor=True, show_progress_bar=True)

    cache_payload = {'df': df, 'embeddings': embeddings, 'enc_texts': enc_texts}

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache_payload, f)

    return df, embeddings, enc_texts
 
def enrich_inventory():
    print("Loading models...")

    with open(BASE_DIR / "data/neighborhood_full_inventory.json", 'r') as f:
        raw_inventory = json.load(f)

    # reference_df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,
    #     "rogerioxavier/x-wines-slim-version",
    #     "XWines_Slim_1K_wines.csv"
    # ).fillna('')

    df, embeddings, enc_texts = load_ref_data()

    cellar = [i for i in raw_inventory if i.get('product_type') == "Wine" and i.get('images')]
    
    print(f"🔍 Processing {len(cellar)} wines...")
    batch = []
    batch_size = 50
    count = 0
    for bottle in cellar:
        try:
            tags = " ".join(bottle.get('tags', []))
            desc = strip_html(bottle.get('body_html',''))
            query = f"search_query: {desc} {tags} {bottle.get('title','')}"
            
            query_vec = retriever.encode(query, convert_to_tensor=True)
            hits = util.semantic_search(query_vec, embeddings, top_k=10)[0]
            candidate_indices = [hit['corpus_id'] for hit in hits]

            query_clean = f"{desc} {tags} {bottle['title']}"
            pairs = [
                [query_clean.replace("search_query: ", ""), enc_texts[i].replace("search_document: ", "")] 
                for i in candidate_indices
                ]

            scores = reranker.predict(pairs)
            row = df.iloc[candidate_indices[scores.argmax()]]

            # Clean and structure data
            try:
                pairings = ast.literal_eval(row['Harmonize']) if isinstance(row['Harmonize'], str) else []
            except:
                pairings = []

            features = {
                'body': row['Body'],
                'acidity': row['Acidity'],
                'grape': row['Grapes'],
                'type': row['Type'],
                'pairings': pairings
            }

            # final vectorization
            for_search = f"{bottle['title']} {desc} {tags} {features['grape']} {features['type']} {features['body']} {features['acidity']} {' '.join(pairings)}"
            embedding = final_vector_model.encode(for_search).tolist()

            # upload to supabase
            wine_data = {
                "id": bottle['id'],
                "title": bottle['title'],
                "handle": bottle['handle'],
                "price": float(bottle['variants'][0]['price']),
                "image_url": bottle['images'][0]['src'],
                "product_type": bottle.get('product_type', 'Wine'),
                "description": desc,
                "tags": bottle.get('tags', []),
                "features": features,
                "embedding": embedding
            }

            batch.append(wine_data)

            if len(batch) == batch_size:
                supabase.table("wines").upsert(batch).execute()
                count += len(batch)
                batch = []
                print(f"[{count}/{len(cellar)}] Done.")

        except Exception as e:
            print(f"Failed to process wine {bottle['title']} with error: {e}")
            continue

    if batch:
        supabase.table("wines").upsert(batch).execute()
        count += len(batch)
        batch = []
    
    print(f"[{count}/{len(cellar)}] Done ✅")

if __name__ == "__main__":
    enrich_inventory()