#!/usr/bin/env python3
"""
Example usage of the OrderOfBooks dataset
"""

import json
from pathlib import Path

def example_usage():
    """Demonstrate how to use the matched series dataset"""
    
    print("📚 OrderOfBooks Dataset Usage Examples\n")
    
    # Load the final results
    with open('matched_series_final.json', 'r') as f:
        results = json.load(f)
    
    # Example 1: Find series with multiple orderings
    print("🔄 Series with Multiple Orderings:")
    for result in results:
        oob_info = result.get('orderofbooks_info', {})
        if oob_info and 'orderings' in oob_info:
            orderings = oob_info['orderings']
            if len(orderings) > 1:
                wiki_info = result['wikipedia_info']
                series_name = wiki_info['series_name']
                sales = wiki_info.get('approximate_sales', 'Unknown')
                
                print(f"\n📖 {series_name}")
                print(f"   Sales: {sales}")
                print(f"   Available orderings:")
                
                for order_type, order_data in orderings.items():
                    heading = order_data.get('heading', order_type)
                    book_count = len(order_data.get('books', []))
                    print(f"   • {heading} ({book_count} books)")
    
    # Example 2: Compare publication vs chronological order
    print(f"\n\n📑 Publication vs Chronological Order Example:")
    
    # Find Chronicles of Narnia
    narnia = None
    for result in results:
        if result['wikipedia_info']['series_name'] == 'The Chronicles of Narnia':
            narnia = result
            break
    
    if narnia and 'orderings' in narnia['orderofbooks_info']:
        orderings = narnia['orderofbooks_info']['orderings']
        
        if 'publication' in orderings and 'chronological' in orderings:
            pub_books = orderings['publication']['books'][:3]  # First 3 books
            chron_books = orderings['chronological']['books'][:3]  # First 3 books
            
            print(f"\n🎯 Chronicles of Narnia - Order Comparison:")
            print(f"\n   Publication Order:")
            for book in pub_books:
                print(f"   {book['index']}. {book['title']} ({book['year']})")
            
            print(f"\n   Chronological Order:")
            for book in chron_books:
                print(f"   {book['index']}. {book['title']} ({book['year']})")
    
    # Example 3: Top bestselling series with orderofbooks data
    print(f"\n\n💰 Top Bestselling Series with OrderOfBooks Data:")
    
    matched_series = [r for r in results if r.get('orderofbooks_info') is not None]
    # Sort by sales (convert sales string to number for sorting)
    matched_series.sort(key=lambda x: x['wikipedia_info'].get('approximate_sales_int', 0), reverse=True)
    
    for i, result in enumerate(matched_series[:10], 1):
        wiki_info = result['wikipedia_info']
        series_name = wiki_info['series_name']
        sales = wiki_info.get('approximate_sales', 'Unknown')
        authors = ', '.join(wiki_info.get('authors', ['Unknown']))
        
        # Count total orderings and books
        oob_info = result['orderofbooks_info']
        orderings = oob_info.get('orderings', {})
        total_orderings = len(orderings)
        
        print(f"   {i:2d}. {series_name}")
        print(f"       Author(s): {authors}")
        print(f"       Sales: {sales}")
        print(f"       Available orderings: {total_orderings}")
    
    print(f"\n✨ This dataset provides comprehensive ordering information")
    print(f"   for {len(matched_series)} bestselling book series!")

if __name__ == '__main__':
    example_usage()
