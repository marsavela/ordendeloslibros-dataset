#!/usr/bin/env python3
"""
Example usage of Spanish Edition Finder

Demonstrates how to search for Spanish editions of books.
"""

import asyncio
import json
from find_spanish_editions import SpanishEditionFinder, BookQuery

async def example_searches():
    """Run example searches for Spanish editions"""
    finder = SpanishEditionFinder()
    
    # Example books to search for
    example_books = [
        BookQuery("Harry Potter and the Philosopher's Stone", "J.K. Rowling", 1997),
        BookQuery("The Lord of the Rings", "J.R.R. Tolkien", 1954),
        BookQuery("The Chronicles of Narnia", "C.S. Lewis", 1950),
        BookQuery("Dune", "Frank Herbert", 1965),
        BookQuery("The Hobbit", "J.R.R. Tolkien", 1937)
    ]
    
    results = []
    
    print("🔍 Spanish Edition Finder - Example Searches")
    print("=" * 60)
    
    for i, query in enumerate(example_books):
        print(f"\n📚 Search {i+1}: {query.title} by {query.author} ({query.year})")
        print("-" * 50)
        
        try:
            result = await finder.find_spanish_edition(query)
            results.append(result)
            
            if result.best_match:
                print(f"✅ Found Spanish edition:")
                print(f"   Title: {result.best_match.title_es}")
                if result.best_match.subtitle_es:
                    print(f"   Subtitle: {result.best_match.subtitle_es}")
                print(f"   Publisher: {result.best_match.publisher}")
                print(f"   Year: {result.best_match.published_date}")
                print(f"   ISBN: {result.best_match.isbn_13 or result.best_match.isbn_10}")
                print(f"   Confidence: {result.best_match.confidence:.2f}")
                print(f"   Official: {result.best_match.official}")
                print(f"   Source: {result.best_match.source}")
            else:
                print(f"❌ No Spanish edition found")
            
            print(f"📝 Notes: {result.notes}")
            
        except Exception as e:
            print(f"❌ Error searching for {query.title}: {e}")
    
    # Save detailed results
    output_file = "spanish_editions_examples.json"
    detailed_results = []
    
    for result in results:
        detailed_results.append({
            'query': {
                'title': result.query.title,
                'author': result.query.author,
                'year': result.query.year
            },
            'best_match': result.best_match.__dict__ if result.best_match else None,
            'alternates': [alt.__dict__ for alt in result.alternates],
            'notes': result.notes
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to {output_file}")
    print(f"📊 Found Spanish editions for {sum(1 for r in results if r.best_match)}/{len(results)} books")

if __name__ == "__main__":
    asyncio.run(example_searches())
