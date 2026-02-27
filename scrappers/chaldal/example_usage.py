#!/usr/bin/env python3
"""
Example usage of the Chaldal scraper
Shows different ways to use the scraper
"""

from chaldal_scraper import ChaldalScraper, save_to_json
import json

def example_1_basic_usage():
    """Example 1: Basic scraping"""
    print("\n=== Example 1: Basic Scraping ===")
    scraper = ChaldalScraper()
    result = scraper.scrape_url('https://chaldal.com/coffees')
    
    if 'error' not in result:
        print(f"Found {result['total_products']} products")
        for i, product in enumerate(result['products'][:3], 1):
            print(f"\n{i}. {product['name']}")
            print(f"   Price: {product['price']}")
            print(f"   Size: {product['quantity']}")
            print(f"   Delivery: {product['delivery_time']}")


def example_2_custom_output():
    """Example 2: Custom output file"""
    print("\n=== Example 2: Custom Output File ===")
    scraper = ChaldalScraper()
    result = scraper.scrape_url('https://chaldal.com/coffees')
    
    if 'error' not in result:
        saved_file = save_to_json(result, 'coffees.json')
        print(f"Saved to: {saved_file}")


def example_3_process_results():
    """Example 3: Process and filter results"""
    print("\n=== Example 3: Filter Results ===")
    scraper = ChaldalScraper()
    result = scraper.scrape_url('https://chaldal.com/coffees')
    
    if 'error' not in result:
        products = result['products']
        
        # Try to find products with specific price range
        # (prices are in format "৳275" so we need to parse them)
        print(f"\nTotal products: {len(products)}")
        
        # Get product names and prices
        print("\nAll Products:")
        for product in products:
            print(f"  - {product['name']}: {product['price']}")


def example_4_batch_scraping():
    """Example 4: Scrape multiple pages"""
    print("\n=== Example 4: Batch Scraping Multiple URLs ===")
    
    urls = [
        'https://chaldal.com/coffees',
        'https://chaldal.com/food',
    ]
    
    scraper = ChaldalScraper(delay_between_requests=2.0)
    all_results = []
    
    for url in urls:
        print(f"\nScraping: {url}")
        result = scraper.scrape_url(url)
        
        if 'error' not in result:
            all_results.append(result)
            print(f"  ✓ Found {result['total_products']} products")
        else:
            print(f"  ✗ Error: {result['error']}")
    
    # Save all results
    combined_data = {
        'timestamp': all_results[0]['timestamp'] if all_results else None,
        'total_urls_scraped': len(all_results),
        'total_products': sum(r['total_products'] for r in all_results),
        'results': all_results
    }
    
    saved_file = save_to_json(combined_data, 'batch_results.json')
    print(f"\n✓ Batch scraping complete. Saved to: {saved_file}")


if __name__ == '__main__':
    print("Chaldal Scraper - Usage Examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_1_basic_usage()
        example_2_custom_output()
        example_3_process_results()
        # example_4_batch_scraping()  # Uncomment to run batch scraping
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have installed the required packages:")
        print("  pip install -r requirements.txt")
