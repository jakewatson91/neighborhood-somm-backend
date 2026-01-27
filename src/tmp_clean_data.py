import json

def clean_wine_data():
    with open("data/final_enriched_inventory.json", "r") as f:
        raw_data = json.load(f)

    clean_inventory = []

    for w in raw_data:
        try:
            # Flattening the nested Shopify structure to match SQL columns
            clean_wine = {
                "id": w['id'],
                "title": w['title'],
                "handle": w['handle'],
                "price": float(w['variants'][0]['price']),
                "image_url": w['images'][0]['src'] if w.get('images') else "",
                "product_type": w.get('product_type', 'Wine'),
                "description": w.get('body_html', '').replace('<p>', '').replace('</p>', ''), # Basic HTML strip
                "tags": w.get('tags', []),
                "inferred_features": w.get('inferred_features', {})
            }
            clean_inventory.append(clean_wine)
        except (IndexError, KeyError, ValueError):
            continue 

    with open("clean_inventory.json", "w") as f:
        json.dump(clean_inventory, f, indent=4)
    
    print(f"✅ Reformulated {len(clean_inventory)} wines into clean_inventory.json")

if __name__ == "__main__":
    clean_wine_data()