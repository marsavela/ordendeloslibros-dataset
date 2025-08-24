#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv

def test_isbndb_pagination():
    load_dotenv()
    isbndb_key = os.getenv('ISBNDB_API_KEY')
    
    if not isbndb_key:
        print("No ISBNdb API key found")
        return
    
    headers = {'Authorization': isbndb_key}
    
    # Test multiple pages for J.K. Rowling
    all_books = []
    spanish_books = []
    
    for page in [1, 2]:
        url = f'https://api2.isbndb.com/author/J.K.%20Rowling?page={page}&pageSize=100'
        print(f'Fetching page {page}...')
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                books = data.get('books', [])
                print(f'  Got {len(books)} books')
                
                all_books.extend(books)
                
                page_spanish = [b for b in books if b.get('language') == 'es']
                spanish_books.extend(page_spanish)
                print(f'  Spanish books on this page: {len(page_spanish)}')
                
                if len(books) < 100:
                    print('  Reached end (less than 100 books)')
                    break
            else:
                print(f'  Error: {response.status_code} - {response.text[:100]}')
                break
        except Exception as e:
            print(f'  Exception: {e}')
            break
    
    print(f'\nTotal books retrieved: {len(all_books)}')
    print(f'Total Spanish books: {len(spanish_books)}')
    
    if spanish_books:
        print('\nAll Spanish books:')
        for i, book in enumerate(spanish_books):
            title = book.get('title', 'Unknown')
            publisher = book.get('publisher', 'Unknown')
            year = book.get('date_published', 'Unknown')[:4] if book.get('date_published') else 'Unknown'
            print(f'{i+1}. {title[:60]}... ({year}) - {publisher}')

if __name__ == "__main__":
    test_isbndb_pagination()
