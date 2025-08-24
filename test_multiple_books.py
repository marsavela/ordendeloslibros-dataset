#!/usr/bin/env python3
"""
Test multiple books with caching
"""

import asyncio
from enhanced_spanish_finder import EnhancedSpanishFinder, BookQuery

async def test_multiple_books():
    """Test multiple books with caching"""
    print("🧪 Testing Multiple Books with Caching")
    print("=" * 50)
    
    finder = EnhancedSpanishFinder()
    
    books = [
        BookQuery("Harry Potter and the Philosopher's Stone", "J.K. Rowling", 1997),
        BookQuery("Charlie and the Chocolate Factory", "Roald Dahl", 1964),
        BookQuery("The BFG", "Roald Dahl", 1982)
    ]
    
    print(f"📚 Testing {len(books)} books...")
    
    # First round - should hit APIs and cache
    print(f"\n🔍 FIRST ROUND (should use APIs):")
    for i, query in enumerate(books, 1):
        print(f"\n📖 Book {i}: {query.title}")
        result = await finder.find_spanish_edition(query)
        best_title = result.best_match.spanish_candidate.title if result.best_match else "None"
        print(f"   Spanish: {best_title}")
    
    # Second round - should use cache
    print(f"\n🔍 SECOND ROUND (should use book cache):")
    for i, query in enumerate(books, 1):
        print(f"\n📖 Book {i}: {query.title}")
        result = await finder.find_spanish_edition(query)
        best_title = result.best_match.spanish_candidate.title if result.best_match else "None"
        print(f"   Spanish: {best_title}")
    
    # Check cache files
    print(f"\n📊 Cache Summary:")
    import os
    author_files = [f for f in os.listdir('.cache') if f.startswith('author_')]
    book_files = [f for f in os.listdir('.cache') if f.startswith('book_')]
    print(f"Author cache files: {len(author_files)}")
    print(f"Book cache files: {len(book_files)}")
    
    for author_file in author_files:
        print(f"  📁 {author_file}")
    for book_file in book_files:
        print(f"  📖 {book_file}")

if __name__ == "__main__":
    asyncio.run(test_multiple_books())
