import json
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer

def strip_html(text):
    p = re.compile(r'<.*?>')
    # Use re.sub to replace all occurrences of the pattern with an empty string
    return p.sub('', text)

def build_vector():
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_path = BASE_DIR / "data" / "final_enriched_inventory.json"
    output_path = BASE_DIR / "src" / "vector_inventory.json"

    print("Loading runtime model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    clean_inv = []

    with open(input_path, 'r') as f:
        data = json.load(f)

    for w in data:
        try:
            # Flattening the nested Shopify structure to match SQL columns
            clean_wine = {
                "id": w['id'],
                "title": w['title'],
                "handle": w['handle'],
                "price": float(w['variants'][0]['price']),
                "image_url": w['images'][0]['src'] if w.get('images') else "",
                "product_type": w.get('product_type', 'Wine'),
                "description": strip_html(w.get('body_html', '')),
                "tags": w.get('tags', []),
                "features": w.get('inferred_features', {})
            }

            features = clean_wine.get('features') or {}
            for_search = (
                f"{clean_wine['title']} "
                f"{clean_wine['description']} "
                f"{' '.join(clean_wine['tags'])} "
                f"{features.get('body', '')} "
                f"{features.get('grape', '')} "
                f"{features.get('acidity', '')} "
                f"{features.get('pairings', '')}"
            ).strip()

            clean_wine['embedding'] = model.encode(for_search).tolist()
            clean_inv.append(clean_wine)
        
        except (IndexError, KeyError, ValueError):
            continue

    with open(output_path, 'w') as f:
        json.dump(clean_inv, f, indent=4, ensure_ascii=False)

    print(f"Built {len(clean_inv)} vectorized wines and saved to {output_path}")

if __name__ == "__main__":
    build_vector()