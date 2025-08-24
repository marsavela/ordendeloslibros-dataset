#!/usr/bin/env python3
"""
Test the enhanced Spanish finder with book-level caching
"""

import asyncio
from enhanced_spanish_finder import EnhancedSpanishFinder, BookQuery

async def test_book_caching():
    """Test book-level caching functionality"""
    print("🧪 Testing Enhanced Spanish Finder with Book-Level Caching")
    print("=" * 60)
    
    finder = EnhancedSpanishFinder()
    
    # Test with a book we know should have Spanish editions
    query = BookQuery(
        title="Harry Potter and the Philosopher's Stone",
        author="J.K. Rowling",
        year=1997
    )
    
    print(f"\n📖 Searching for: {query.title}")
    print(f"👤 Author: {query.author}")
    print(f"📅 Year: {query.year}")
    
    # First search - should hit APIs and cache result
    print("\n🔍 FIRST SEARCH (should use APIs):")
    result1 = await finder.find_spanish_edition(query)
    
    print(f"\n✅ Search completed!")
    print(f"Best match: {result1.best_match.spanish_candidate.title if result1.best_match else 'None'}")
    print(f"Alternates: {len(result1.alternates)}")
    print(f"Notes: {result1.notes}")
    
    # Second search - should use book cache
    print("\n🔍 SECOND SEARCH (should use book cache):")
    result2 = await finder.find_spanish_edition(query)
    
    print(f"\n✅ Search completed!")
    print(f"Best match: {result2.best_match.spanish_candidate.title if result2.best_match else 'None'}")
    print(f"Alternates: {len(result2.alternates)}")
    print(f"Notes: {result2.notes}")
    
    # Verify cache is working
    if result1.best_match and result2.best_match:
        match1_title = result1.best_match.spanish_candidate.title
        match2_title = result2.best_match.spanish_candidate.title
        print(f"\n🔍 Cache verification:")
        print(f"First result:  {match1_title}")
        print(f"Second result: {match2_title}")
        print(f"Results match: {match1_title == match2_title}")
    
    print(f"\n📊 Cache files in .cache directory:")
    import os
    cache_files = [f for f in os.listdir('.cache') if f.startswith('book_')]
    print(f"Book cache files: {len(cache_files)}")
    for f in cache_files[:3]:  # Show first 3
        print(f"  - {f}")

if __name__ == "__main__":
    asyncio.run(test_book_caching())
