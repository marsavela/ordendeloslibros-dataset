#!/usr/bin/env python3
"""
Process Existing Dataset for Spanish Editions

Processes the matched_series_final.json dataset and finds Spanish editions
for all books in the Wikipedia bestselling series.
"""

import json
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import re

from find_spanish_editions import SpanishEditionFinder, BookQuery

class DatasetSpanishProcessor:
    """Process the existing dataset to find Spanish editions"""
    
    def __init__(self):
        self.finder = SpanishEditionFinder()
        self.processed_books = []
        self.failed_books = []
        
    def extract_books_from_series(self, series: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract individual books from a series entry"""
        books = []
        
        # Get basic series info
        wikipedia_info = series.get('wikipedia_info', {})
        orderofbooks_info = series.get('orderofbooks_info')
        
        if not wikipedia_info or not orderofbooks_info:
            return books  # Skip series without proper data
        
        series_title = wikipedia_info.get('series_name', '')
        authors = wikipedia_info.get('authors', [])
        author = authors[0] if authors else 'Unknown'
        
        # Get year from Wikipedia years (e.g., "1997-2007")
        default_year = 2000
        years_str = wikipedia_info.get('years', '')
        if years_str:
            year_match = re.search(r'\d{4}', years_str)
            if year_match:
                default_year = int(year_match.group())
        
        # Process orderings from orderofbooks_info
        orderings = orderofbooks_info.get('orderings', {})
        
        # Process each ordering type (publication, chronological, etc.)
        for ordering_type, ordering_data in orderings.items():
            if isinstance(ordering_data, dict) and 'books' in ordering_data:
                books_list = ordering_data['books']
                
                for book in books_list:
                    if isinstance(book, dict) and book.get('title'):
                        # Extract book info
                        book_title = book['title']
                        
                        # Try to extract publication year
                        pub_year = default_year
                        pub_date = book.get('publication_date', '') or book.get('published', '')
                        if pub_date:
                            year_match = re.search(r'\d{4}', str(pub_date))
                            if year_match:
                                pub_year = int(year_match.group())
                        
                        book_entry = {
                            'series_title': series_title,
                            'book_title': book_title,
                            'author': author,
                            'year': pub_year,
                            'series_info': {
                                'wikipedia_title': series_title,
                                'ordering_type': ordering_type
                            },
                            'raw_book_data': book
                        }
                        books.append(book_entry)
        
        return books
    
    async def process_book(self, book_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single book to find Spanish edition"""
        query = BookQuery(
            title=book_info['book_title'],
            author=book_info['author'],
            year=book_info['year']
        )
        
        print(f"🔍 Processing: {book_info['book_title']} by {book_info['author']}")
        
        try:
            result = await self.finder.find_spanish_edition(query)
            
            return {
                'original_book': book_info,
                'spanish_search': {
                    'query': {
                        'title': query.title,
                        'author': query.author, 
                        'year': query.year
                    },
                    'best_match': result.best_match.__dict__ if result.best_match else None,
                    'alternates': [alt.__dict__ for alt in result.alternates],
                    'notes': result.notes,
                    'search_timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"❌ Error processing {book_info['book_title']}: {e}")
            return {
                'original_book': book_info,
                'spanish_search': {
                    'error': str(e),
                    'search_timestamp': datetime.now().isoformat()
                }
            }
    
    async def process_dataset(self, input_file: str = 'matched_series_final.json', 
                            output_file: str = 'series_with_spanish_editions.json',
                            limit: int = None):
        """Process the entire dataset"""
        print(f"📚 Loading dataset from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Handle both old and new dataset formats
        if isinstance(dataset, list):
            wikipedia_series = dataset
        else:
            wikipedia_series = dataset.get('wikipedia_series', [])
            
        print(f"Found {len(wikipedia_series)} Wikipedia series")
        
        # Extract all books
        all_books = []
        for series in wikipedia_series:
            books = self.extract_books_from_series(series)
            all_books.extend(books)
        
        print(f"📖 Extracted {len(all_books)} individual books")
        
        if limit:
            all_books = all_books[:limit]
            print(f"⚡ Limited to first {limit} books for testing")
        
        # Process books in batches to avoid overwhelming APIs
        batch_size = 10
        processed_results = []
        
        for i in range(0, len(all_books), batch_size):
            batch = all_books[i:i+batch_size]
            print(f"\n🔄 Processing batch {i//batch_size + 1}/{(len(all_books)-1)//batch_size + 1}")
            
            batch_results = await asyncio.gather(
                *[self.process_book(book) for book in batch],
                return_exceptions=True
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"❌ Batch processing error: {result}")
                else:
                    processed_results.append(result)
            
            # Small delay between batches
            await asyncio.sleep(2)
        
        # Compile results
        output_data = {
            'metadata': {
                'source_file': input_file,
                'processed_timestamp': datetime.now().isoformat(),
                'total_books': len(all_books),
                'processed_books': len(processed_results),
                'spanish_matches_found': sum(1 for r in processed_results 
                                           if r.get('spanish_search', {}).get('best_match'))
            },
            'books_with_spanish_editions': processed_results
        }
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved results to {output_file}")
        print(f"📊 Spanish matches found: {output_data['metadata']['spanish_matches_found']}/{len(processed_results)}")
        
        return output_data

async def main():
    """Main execution function"""
    processor = DatasetSpanishProcessor()
    
    # Process with a small limit first for testing
    print("🧪 Running test with 5 books...")
    await processor.process_dataset(limit=5, output_file='spanish_test_results.json')
    
    # Uncomment to process full dataset:
    # print("🚀 Processing full dataset...")
    # await processor.process_dataset()

if __name__ == "__main__":
    asyncio.run(main())
