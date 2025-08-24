#!/usr/bin/env python3
"""
Extended test of Spanish edition finder with multiple books
"""

import asyncio
from find_spanish_editions import SpanishEditionFinder, BookQuery

async def extended_test():
    print("🧪 Extended Test of Spanish Edition Finder")
    print("=" * 60)
    
    finder = SpanishEditionFinder()
    
    # Test with several popular books
    test_books = [
        BookQuery("Harry Potter and the Philosopher's Stone", "J.K. Rowling", 1997),
        BookQuery("The Lord of the Rings", "J.R.R. Tolkien", 1954),
        BookQuery("The Chronicles of Narnia", "C.S. Lewis", 1950),
        BookQuery("Dune", "Frank Herbert", 1965),
        BookQuery("The Hobbit", "J.R.R. Tolkien", 1937)
    ]
    
    results = []
    found_count = 0
    
    for i, query in enumerate(test_books, 1):
        print(f"\n📚 Test {i}/{len(test_books)}: {query.title}")
        print("-" * 50)
        
        try:
            result = await finder.find_spanish_edition(query)
            results.append(result)
            
            if result.best_match:
                found_count += 1
                print(f"✅ Spanish Title: {result.best_match.title_es}")
                print(f"   Publisher: {result.best_match.publisher or 'N/A'}")
                print(f"   ISBN: {result.best_match.isbn_13 or result.best_match.isbn_10 or 'N/A'}")
                print(f"   Confidence: {result.best_match.confidence:.2f}")
                print(f"   Validation: {result.best_match.official}")
                print(f"   Alternates: {len(result.alternates)}")
            else:
                print("❌ No Spanish edition found")
                
            print(f"📝 Notes: {result.notes}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Books tested: {len(test_books)}")
    print(f"Spanish editions found: {found_count}")
    print(f"Success rate: {(found_count/len(test_books)*100):.1f}%")
    print(f"Cache files created: Check ./cache/ directory")

if __name__ == "__main__":
    asyncio.run(extended_test())
