import pandas as pd
import json
import re

def clean_html(raw_html):
    """One-liner regex to strip tags. Fast and sufficient for Shopify descriptions."""
    if not raw_html or pd.isna(raw_html):
        return ""
    return re.sub(r'<[^>]+>', '', str(raw_html)).strip()

def process_neighborhood_fast():
    # 1. Load Data
    with open("data/neighborhood_full_inventory.json", 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)

    # 2. The "Pandas Way" (No loops!)
    # Extract Price (handle missing/empty lists safely)
    df['price'] = df['variants'].apply(lambda x: float(x[0]['price']) if x and len(x) > 0 else None)
    
    # Extract Image
    df['image_url'] = df['images'].apply(lambda x: x[0]['src'] if x and len(x) > 0 else None)
    
    # Clean Body Text
    df['clean_desc'] = df['body_html'].apply(clean_html)

    # 3. Select only what you need
    # Keep 'tags' as a list! (See explanation below)
    final_df = df[['id', 'title', 'handle', 'vendor', 'product_type', 'tags', 'price', 'image_url', 'clean_desc']]
    
    print(f"✅ Processed {len(final_df)} wines.")
    return final_df

if __name__ == "__main__":
    df = process_neighborhood_fast()
    print(df.head(1).T) # Transpose to see the structure clearly