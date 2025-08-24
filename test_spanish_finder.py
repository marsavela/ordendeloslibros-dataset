#!/usr/bin/env python3
"""
Quick test of Spanish edition finder
"""

import asyncio
from find_spanish_editions import SpanishEditionFinder, BookQuery

async def quick_test():
    print("🧪 Quick Test of Spanish Edition Finder")
    print("=" * 50)
    
    finder = SpanishEditionFinder()
    
    # Test with a popular book
    query = BookQuery(
        title="Harry Potter and the Philosopher's Stone",
        author="J.K. Rowling", 
        year=1997
    )
    
    print(f"Searching for: {query.title}")
    print(f"Author: {query.author}")
    print(f"Year: {query.year}")
    print()
    
    try:
        result = await finder.find_spanish_edition(query)
        
        if result.best_match:
            print("✅ FOUND SPANISH EDITION!")
            print(f"Spanish Title: {result.best_match.title_es}")
            print(f"Publisher: {result.best_match.publisher}")
            print(f"ISBN: {result.best_match.isbn_13 or 'N/A'}")
            print(f"Confidence: {result.best_match.confidence:.2f}")
            print(f"Source: {result.best_match.source}")
            print(f"Validation: {result.best_match.official}")
        else:
            print("❌ No Spanish edition found")
        
        print(f"\nNotes: {result.notes}")
        print(f"Alternates found: {len(result.alternates)}")
        
        print("\n💾 Cache test - running same search again...")
        result2 = await finder.find_spanish_edition(query)
        print("✅ Cache test completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())
