# Chaldal.com Web Scraper

A Python script to scrape product data from any page on Chaldal.com and export it in multiple formats (JSON, CSV, Excel).

## Features

- ✅ Scrape any Chaldal.com product page
- ✅ Extract product name, price, quantity, delivery time, and image URL
- ✅ Support for **both regular and discounted prices**
- ✅ Output data in **3 formats: JSON, CSV, Excel**
- ✅ Respectful scraping with configurable delays
- ✅ Error handling and logging
- ✅ Works with the `torch-env` conda environment

## Requirements

- Python 3.7+
- requests
- beautifulsoup4
- lxml
- openpyxl (for Excel export)

## Installation

### Step 1: Activate the torch-env conda environment

```bash
conda activate torch-env
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install requests beautifulsoup4 lxml
```

## Usage

## Usage

### Basic usage (default JSON format):
```bash
python chaldal_scraper.py https://chaldal.com/coffees
```

### With custom output filename:
```bash
python chaldal_scraper.py https://chaldal.com/coffees output.json
```

### Export to CSV format:
```bash
python chaldal_scraper.py https://chaldal.com/coffees output.csv --format csv
```

### Export to Excel format:
```bash
python chaldal_scraper.py https://chaldal.com/fresh-vegetable vegetables.xlsx --format excel
```

### Without explicit filename (auto-generated):
```bash
python chaldal_scraper.py https://chaldal.com/coffees --format csv
# Generates: chaldal_products_20260227_190005.csv
```

### Examples of URLs you can scrape:
- https://chaldal.com/coffees
- https://chaldal.com/beverages
- https://chaldal.com/food
- https://chaldal.com/fresh-vegetable
- https://chaldal.com/search?q=milk
- Any other Chaldal.com product page

## Output Formats

### JSON Format (Default)
Clean, structured JSON with metadata:

```json
{
  "url": "https://chaldal.com/coffees",
  "timestamp": "2024-02-27T10:30:45.123456",
  "total_products": 26,
  "products": [
    {
      "name": "Nestle Nescafe Classic Instant Coffee Jar",
      "price": "৳ 275",
      "discounted_price": "N/A",
      "original_price": "N/A",
      "quantity": "45 gm",
      "delivery_time": "1 hr",
      "image_url": "https://i.chaldn.com/...",
      "product_url": "N/A"
    },
    ...
  ]
}
```

### CSV Format
Comma-separated values in spreadsheet-compatible format:

```
delivery_time,discounted_price,image_url,name,original_price,price,product_url,quantity
1 hr,N/A,https://...,Nestle Coffee Mate Coffee Creamer Jar,N/A,৳ 400,N/A,400 gm
1 hr,৳ 49,https://...,Fulkopi (Cauliflower),৳ 59,৳ 49,N/A,each
...
```

### Excel Format (XLSX)
Two sheets with formatted data:
1. **Products Sheet** - All product data with styled headers (purple background)
2. **Metadata Sheet** - URL, timestamp, and product count

Features:
- Colored header row (purple with white text)
- Automatic column width adjustment
- Wrapped text for better readability
- Meta sheet with scraping information

## Customization

You can modify the scraper by editing the `ChaldalScraper` class:

### Change the delay between requests:
```python
scraper = ChaldalScraper(delay_between_requests=2.0)  # 2 seconds delay
```

### Use it as a library in your own code:
```python
from chaldal_scraper import ChaldalScraper, save_to_json, save_to_csv, save_to_excel

scraper = ChaldalScraper()
result = scraper.scrape_url('https://chaldal.com/coffees')

# Process the results
for product in result['products']:
    print(f"{product['name']} - {product['price']}")

# Save in different formats
save_to_json(result, 'output.json')
save_to_csv(result, 'output.csv')
save_to_excel(result, 'output.xlsx')
```

### Programmatically save to specific format:
```python
from chaldal_scraper import ChaldalScraper, save_to_csv, save_to_excel

scraper = ChaldalScraper()
result = scraper.scrape_url('https://chaldal.com/fresh-vegetable')

# Save with both discounted and original prices
if result['products']:
    # CSV for data analysis
    save_to_csv(result, 'vegetables.csv')
    
    # Excel for reporting
    save_to_excel(result, 'vegetables.xlsx')

## Notes

- The script includes a 1-second delay between requests to be respectful to the server
- All output is UTF-8 encoded to properly handle Bengali characters (৳)
- Timestamps are in ISO format for easy parsing
- The scraper uses User-Agent headers to identify itself

## Troubleshooting

### openpyxl not installed?
If you get an error about openpyxl when using Excel format:
```bash
pip install openpyxl
```

### No products found?
The script includes fallback parsers that try multiple selectors. If it still doesn't find products:
1. Check if the page structure has changed
2. Update the CSS selectors in the `extract_products()` method
3. Check console logs for detailed information

### Connection errors?
- Ensure you have internet connectivity
- The server might be blocking requests (try increasing the delay)
- Check if the URL is correct

### CSV encoding issues?
CSV files are saved with UTF-8 encoding to properly handle Bengali characters. If you're using Excel on Windows and see encoding issues:
1. Open Excel
2. Go to Data → Get Data → From File → CSV
3. Select your file and set encoding to UTF-8

## License

For educational purposes only. Respect the website's terms of service and robots.txt.
