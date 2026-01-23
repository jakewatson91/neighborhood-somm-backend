import cloudscraper
import json
import time
import random

scraper = cloudscraper.create_scraper(
    browser={'browser': 'chrome', 'platform': 'darwin', 'desktop': True}
)

def harvest_neighborhood():
    base_url = "https://neighborhoodwines.com/products.json"
    
    all_products = []
    page = 1
    
    print("🚀 Starting JSON Harvest...")
    
    while True:
        # Shopify allows up to 250 items per page
        params = {'limit': 250, 'page': page}
        
        print(f"   📄 Fetching Page {page}...", end=" ")
        
        try:
            # We still add a tiny sleep just to be polite
            time.sleep(random.uniform(1, 2))
            
            response = scraper.get(base_url, params=params)
            data = response.json()
            
            products = data.get("products", [])
            
            if not products:
                print("Done! (Empty page)")
                break
                
            count = len(products)
            print(f"Found {count} wines.")
            
            all_products.extend(products)
            page += 1
            
        except Exception as e:
            print(f"\n❌ Error on page {page}: {e}")
            break
            
    # Save the motherlode
    filename = 'neighborhood_full_inventory.json'
    with open(filename, 'w') as f:
        json.dump(all_products, f, indent=4)
        
    print(f"\n🎉 SUCCESS: Harvested {len(all_products)} total wines.")
    print(f"💾 Saved to '{filename}'")
    
    # Quick Preview of what you got
    if all_products:
        print("\n🔍 Sample Data (First Item):")
        first = all_products[0]
        print(f"   Name: {first.get('title')}")
        print(f"   Tags: {first.get('tags')}")
        print(f"   Description length: {len(first.get('body_html', ''))} chars")

if __name__ == "__main__":
    harvest_neighborhood()