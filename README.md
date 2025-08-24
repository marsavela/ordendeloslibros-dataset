# OrderOfBooks Dataset - Wikipedia Series Matching

A comprehensive dataset that matches Wikipedia's bestselling book series with detailed ordering information from orderofbooks.com, achieving 99.2% match rate with support for multiple ordering types (publication, chronological, companion series, etc.).

## 📊 **Final Results**

- **Wikipedia Series**: 129 bestselling book series
- **OrderOfBooks Series**: 2,750 series scraped
- **Match Rate**: 99.2% (128/129 series successfully matched)
- **Multiple Orderings**: 3 series with multiple ordering variants

### Series with Multiple Orderings:

1. **The Chronicles of Narnia**: 4 orderings (publication, chronological, companion, world)
2. **Alex Cross**: 3 orderings (publication + 2 variants)
3. **A Series of Unfortunate Events**: 2 orderings (publication + variant)

## � **Quick Start**

### To run the complete process:

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

2. **Set up your API key:**

   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Scrape the data:**

   ```bash
   python run_scraper.py
   ```

4. **Run the matching:**

   ```bash
   python run_matcher.py
   ```

5. **Validate results:**
   ```bash
   python validate_dataset.py
   ```

### 🇪🇸 **Spanish Edition Finder (NEW!)**

Find Spanish editions of books using multiple bibliographic APIs:

1. **Run example searches:**

   ```bash
   python example_spanish_finder.py
   ```

2. **Process entire dataset for Spanish editions:**
   ```bash
   python process_spanish_editions.py
   ```

**Features:**

- Multi-API search (ISBNdb, Open Library, Google Books)
- Intelligent title variants and normalization
- Confidence scoring system
- OpenAI validation for official translations
- Support for Spanish and Latin American publishers

That's it! You'll get a `matched_series_final.json` file with 99.2% match rate.

## �📁 **File Structure**

```
├── run_scraper.py                   # Scrapes OrderOfBooks data (multiple orderings support)
├── run_matcher.py                   # Matches Wikipedia series with OrderOfBooks data
├── find_spanish_editions.py         # 🇪🇸 Spanish edition finder (bibliographic search)
├── process_spanish_editions.py      # 🇪🇸 Process dataset for Spanish editions
├── example_spanish_finder.py        # 🇪🇸 Example Spanish edition searches
├── matched_series_final.json        # Final results (99.2% match rate)
├── best_selling_book_series.json    # Wikipedia source data
├── index.json                       # Master index of all series/authors
└── data/
    └── series/                      # 2,750 individual series files
        ├── harry-potter.json
        ├── the-chronicles-of-narnia.json
        └── ...
```

## 🚀 **Usage**

### Scraping OrderOfBooks Data

```bash
# Scrape all series with multiple orderings
python scrape_orderofbooks.py --only-series

# Scrape specific series
python scrape_orderofbooks.py --only-series --include "harry-potter,narnia"

# Resume interrupted scrape with high concurrency
python scrape_orderofbooks.py --resume --concurrency 30
```

### Matching Series

```bash
# Run comprehensive matching
python match_series_final.py

# Custom settings
python match_series_final.py \
  --wikipedia-data best_selling_book_series.json \
  --confidence-threshold 0.3 \
  --output custom_results.json
```

## 📋 **Dependencies**

```bash
pip install playwright aiohttp beautifulsoup4 tqdm python-slugify rapidfuzz openai
python -m playwright install chromium
```

## 🏗 **Data Structure**

### Series with Multiple Orderings

```json
{
  "wikipedia_info": {
    "series_name": "The Chronicles of Narnia",
    "authors": ["C. S. Lewis"],
    "approximate_sales": "120 million"
  },
  "orderofbooks_info": {
    "orderings": {
      "publication": {
        "heading": "Publication Order of The Chronicles Of Narnia Books",
        "books": [
          {
            "title": "The Lion, the Witch, and the Wardrobe",
            "year": 1950,
            "index": 1
          },
          { "title": "Prince Caspian", "year": 1951, "index": 2 }
        ]
      },
      "chronological": {
        "heading": "Chronological Order of The Chronicles Of Narnia Books",
        "books": [
          { "title": "The Magician's Nephew", "year": 1955, "index": 1 },
          {
            "title": "The Lion, the Witch, and the Wardrobe",
            "year": 1950,
            "index": 2
          }
        ]
      }
    }
  },
  "match_info": {
    "confidence": 1.0,
    "match_type": "exact",
    "notes": "4 orderings available"
  }
}
```

## 📈 **Matching Statistics**

- **Total Wikipedia series**: 129
- **Successfully matched**: 128 (99.2%)
- **Match types**:
  - Exact matches: 53 (41.1%)
  - Fuzzy matches: 75 (58.1%)
  - No matches: 1 (0.8%)

### Unmatched Series

Only 1 series remains unmatched: Japanese language series not available on orderofbooks.com.

## 🎯 **Key Features**

### Enhanced Scraper

- Captures **all ordering types** from orderofbooks.com pages
- Detects publication vs chronological vs companion orderings
- Concurrent processing for 2,750+ series
- Resume functionality for large scrapes

### Advanced Matcher

- Fuzzy string matching with rapidfuzz
- GPT validation for ambiguous matches
- Author similarity scoring
- Known series mappings for edge cases
- Comprehensive statistics and reporting

## 🔧 **Technical Highlights**

1. **Multiple Orderings Detection**: Automatically identifies different ordering types from page headings
2. **High Accuracy**: 99.2% match rate through intelligent fuzzy matching + validation
3. **Scalable Architecture**: Handles 2,750+ series with concurrent processing
4. **Robust Data Structure**: Supports both old and new data formats for compatibility

## 📚 **Example Series Coverage**

Major bestselling series successfully matched:

- Harry Potter (7 books)
- Goosebumps (62+ books)
- Chronicles of Narnia (7 books + variants)
- Twilight (4 books)
- Hunger Games (5 books)
- Percy Jackson (7+ books)
- And 122 more...

---

**Created**: August 2025  
**Match Rate**: 99.2% (128/129 series)  
**Data Source**: Wikipedia + orderofbooks.com
