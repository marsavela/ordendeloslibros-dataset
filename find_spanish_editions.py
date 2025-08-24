#!/usr/bin/env python3
"""
Spanish Edition Finder

Expert bibliographic assistant that finds Spanish editions of books given {title, author, year}.
Uses ISBNdb, Open Library, Google Books APIs with sophisticated scoring and OpenAI validation.
"""

import os
import re
import json
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import unicodedata
from urllib.parse import quote
from openai import OpenAI
import requests
from dotenv import load_dotenv
import hashlib
from pathlib import Path

# Load environment variables
load_dotenv()

@dataclass
class BookQuery:
    """Input book to find Spanish edition for"""
    title: str
    author: str
    year: int
    
@dataclass 
class SpanishEdition:
    """Spanish edition result"""
    title_es: str
    subtitle_es: Optional[str] = None
    language: str = "es"
    isbn_13: Optional[str] = None
    isbn_10: Optional[str] = None
    publisher: Optional[str] = None
    published_date: Optional[str] = None
    official: str = "unknown"  # valid_official, likely_translation_but_official, mismatch
    source: str = ""
    confidence: float = 0.0
    source_url: Optional[str] = None

@dataclass
class SearchResult:
    """Complete search result"""
    query: BookQuery
    best_match: Optional[SpanishEdition]
    alternates: List[SpanishEdition]
    raw: Dict[str, List[Dict]]
    notes: str

class SpanishEditionFinder:
    """Main class for finding Spanish editions"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.isbndb_key = os.getenv('ISBNDB_API_KEY')
        self.google_books_key = os.getenv('GOOGLE_BOOKS_API_KEY') 
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Spanish/Latin American publishers for scoring boost
        self.spanish_publishers = {
            'planeta', 'alfaguara', 'anagrama', 'salamandra', 'tusquets',
            'seix barral', 'destino', 'circe', 'minotauro', 'booket',
            'debolsillo', 'ediciones b', 'grijalbo', 'martinez roca',
            'sudamericana', 'emece', 'lumen', 'sexto piso', 'acantilado',
            'impedimenta', 'errata naturae', 'libros del asteroide'
        }
        
    def _get_cache_key(self, query: BookQuery) -> str:
        """Generate cache key for a query"""
        # Normalize inputs for consistent caching
        title_norm = self.normalize_text(query.title)
        author_norm = self.normalize_text(query.author)
        
        # Create hash of normalized query
        key_string = f"{title_norm}|{author_norm}|{query.year}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for a cache key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, query: BookQuery) -> Optional[SearchResult]:
        """Load result from cache if available"""
        cache_key = self._get_cache_key(query)
        cache_file = self._get_cache_file(cache_key)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    
                # Check if cache is still valid (not older than 30 days)
                cached_time = datetime.fromisoformat(cached_data.get('cached_at', '2000-01-01'))
                if (datetime.now() - cached_time).days < 30:
                    print(f"💾 Loading from cache: {query.title}")
                    
                    # Reconstruct SearchResult from cached data
                    best_match = None
                    if cached_data.get('best_match'):
                        best_match = SpanishEdition(**cached_data['best_match'])
                    
                    alternates = []
                    for alt_data in cached_data.get('alternates', []):
                        alternates.append(SpanishEdition(**alt_data))
                    
                    return SearchResult(
                        query=BookQuery(**cached_data['query']),
                        best_match=best_match,
                        alternates=alternates,
                        raw=cached_data.get('raw', {}),
                        notes=cached_data.get('notes', '')
                    )
            except Exception as e:
                print(f"⚠️  Cache read error for {query.title}: {e}")
                # Delete corrupt cache file
                cache_file.unlink()
        
        return None
    
    def _save_to_cache(self, query: BookQuery, result: SearchResult):
        """Save result to cache"""
        cache_key = self._get_cache_key(query)
        cache_file = self._get_cache_file(cache_key)
        
        try:
            cache_data = {
                'query': asdict(result.query),
                'best_match': asdict(result.best_match) if result.best_match else None,
                'alternates': [asdict(alt) for alt in result.alternates],
                'raw': result.raw,
                'notes': result.notes,
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Saved to cache: {query.title}")
            
        except Exception as e:
            print(f"⚠️  Cache write error for {query.title}: {e}")
        
    def is_likely_spanish_title(self, title: str) -> bool:
        """Check if a title is likely Spanish"""
        if not title:
            return False
        
        # Spanish words and patterns that commonly appear in book titles
        spanish_indicators = [
            'el ', 'la ', 'los ', 'las ', 'un ', 'una ', 'de ', 'del ', 'y ',
            'señor', 'señora', 'niño', 'niña', 'historia', 'aventuras',
            'crónicas', 'libro', 'mundo', 'tiempo', 'vida', 'muerte',
            'amor', 'guerra', 'paz', 'rey', 'reina', 'príncipe', 'princesa'
        ]
        
        title_lower = title.lower()
        return any(indicator in title_lower for indicator in spanish_indicators)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        # Remove accents and convert to lowercase
        text = unicodedata.normalize('NFD', text.lower())
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Remove special characters except spaces and alphanumeric
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def generate_search_variants(self, title: str) -> List[str]:
        """Generate 3-5 search variants of the title"""
        variants = [title]
        
        # Remove subtitle after colon or dash
        base_title = re.split(r'[:\-–—]', title)[0].strip()
        if base_title != title:
            variants.append(base_title)
        
        # Remove series markers like "Book 1", "(Series Name)", etc.
        clean_title = re.sub(r'\s*\([^)]*\)\s*', '', title).strip()
        clean_title = re.sub(r'\s*book\s*\d+.*$', '', clean_title, flags=re.IGNORECASE).strip()
        if clean_title and clean_title != title:
            variants.append(clean_title)
            
        # Add version without "The" at beginning
        no_the = re.sub(r'^the\s+', '', title, flags=re.IGNORECASE).strip()
        if no_the != title:
            variants.append(no_the)
            
        # Add version with Spanish articles
        spanish_articles = ['el', 'la', 'los', 'las', 'un', 'una']
        for article in spanish_articles:
            variants.append(f"{article} {base_title}")
        
        return list(set(variants))[:5]  # Max 5 variants
    
    async def search_isbndb(self, title: str, author: str) -> List[Dict]:
        """Search ISBNdb API for ISBN candidates"""
        if not self.isbndb_key:
            return []
            
        results = []
        variants = self.generate_search_variants(title)
        
        async with aiohttp.ClientSession() as session:
            for variant in variants:
                try:
                    query = f"{variant} {author}".strip()
                    url = f"https://api2.isbndb.com/books/{quote(query)}"
                    headers = {'Authorization': self.isbndb_key}
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'books' in data:
                                results.extend(data['books'])
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"ISBNdb search error: {e}")
        
        return results
    
    async def lookup_openlibrary_isbn(self, isbn: str) -> Optional[Dict]:
        """Lookup book by ISBN on Open Library"""
        try:
            url = f"https://openlibrary.org/isbn/{isbn}.json"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            print(f"Open Library ISBN lookup error: {e}")
        return None
    
    async def search_openlibrary_title(self, title: str, author: str) -> List[Dict]:
        """Search Open Library by title and author"""
        results = []
        variants = self.generate_search_variants(title)
        
        async with aiohttp.ClientSession() as session:
            for variant in variants:
                try:
                    query = f'title:"{variant}" AND author:"{author}"'
                    url = f"https://openlibrary.org/search.json?q={quote(query)}&lang=spa&limit=10"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'docs' in data:
                                results.extend(data['docs'])
                    
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"Open Library search error: {e}")
        
        return results
    
    async def search_google_books_isbn(self, isbn: str) -> Optional[Dict]:
        """Search Google Books by ISBN"""
        try:
            query = f"isbn:{isbn}"
            url = f"https://www.googleapis.com/books/v1/volumes?q={quote(query)}&langRestrict=es"
            if self.google_books_key:
                url += f"&key={self.google_books_key}"
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'items' in data and data['items']:
                            return data['items'][0]  # Take first result
        except Exception as e:
            print(f"Google Books ISBN search error: {e}")
        return None
    
    async def search_google_books_title(self, title: str, author: str) -> List[Dict]:
        """Search Google Books by title and author"""
        results = []
        variants = self.generate_search_variants(title)
        
        async with aiohttp.ClientSession() as session:
            for variant in variants:
                try:
                    query = f'intitle:"{variant}" inauthor:"{author}"'
                    url = f"https://www.googleapis.com/books/v1/volumes?q={quote(query)}&langRestrict=es&maxResults=10"
                    if self.google_books_key:
                        url += f"&key={self.google_books_key}"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if 'items' in data:
                                results.extend(data['items'])
                    
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"Google Books search error: {e}")
        
        return results
    
    def calculate_confidence(self, edition: SpanishEdition, query: BookQuery, source_data: Dict) -> float:
        """Calculate confidence score for an edition"""
        score = 0.0
        
        # Language match
        if edition.language in ['es', 'spa', 'spanish']:
            score += 0.5
        
        # Spanish title patterns
        if self.is_likely_spanish_title(edition.title_es):
            score += 0.3
        
        # ISBN cross-validation
        if edition.isbn_13 or edition.isbn_10:
            score += 0.4
        
        # Author similarity (simple check for now)
        if query.author.lower() in str(source_data.get('authors', '')).lower():
            score += 0.2
        
        # Year proximity
        if edition.published_date and query.year:
            try:
                pub_year = int(re.search(r'\d{4}', edition.published_date).group())
                if abs(pub_year - query.year) <= 3:
                    score += 0.05
            except (AttributeError, ValueError):
                pass
        
        # Spanish publisher bonus
        if edition.publisher:
            pub_lower = edition.publisher.lower()
            if any(sp in pub_lower for sp in self.spanish_publishers):
                score += 0.05
        
        return min(score, 1.0)
    
    async def validate_with_openai(self, spanish_title: str, original_title: str, author: str) -> str:
        """Validate Spanish title with OpenAI"""
        if not self.openai_client:
            return "unknown"
            
        try:
            prompt = f"""Validate if the following Spanish title is a faithful and official edition title of the book '{original_title}' by {author}.

Spanish title: "{spanish_title}"
Original title: "{original_title}"
Author: {author}

Answer ONLY with one of: valid_official, likely_translation_but_official, or mismatch."""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().lower()
            valid_responses = ['valid_official', 'likely_translation_but_official', 'mismatch']
            
            return result if result in valid_responses else "unknown"
            
        except Exception as e:
            print(f"OpenAI validation error: {e}")
            return "unknown"
    
    def extract_spanish_edition(self, data: Dict, source: str) -> Optional[SpanishEdition]:
        """Extract Spanish edition data from API response"""
        
        title_es = ""
        subtitle_es = None
        isbn_13 = None
        isbn_10 = None
        publisher = None
        published_date = None
        language = "unknown"
        source_url = None
        
        if source == 'isbndb':
            # ISBNdb format
            title_es = data.get('title', '')
            subtitle_es = data.get('title_long', '').replace(title_es, '').strip()
            isbn_13 = data.get('isbn13')
            isbn_10 = data.get('isbn10') 
            publisher = data.get('publisher')
            published_date = data.get('date_published')
            language = data.get('language', 'unknown')
            
        elif source == 'openlibrary':
            # Open Library format
            title_es = data.get('title', '')
            subtitle_es = data.get('subtitle')
            isbn_13 = data.get('isbn_13', [None])[0] if data.get('isbn_13') else None
            isbn_10 = data.get('isbn_10', [None])[0] if data.get('isbn_10') else None
            publisher = ', '.join(data.get('publishers', [])) if data.get('publishers') else None
            published_date = data.get('publish_date')
            languages = data.get('languages', [])
            if languages:
                language = ', '.join([lang.get('key', '').replace('/languages/', '') for lang in languages if isinstance(lang, dict)])
            
        elif source == 'google_books':
            # Google Books format
            vol_info = data.get('volumeInfo', {})
            title_es = vol_info.get('title', '')
            subtitle_es = vol_info.get('subtitle')
            
            isbns = vol_info.get('industryIdentifiers', [])
            for isbn_info in isbns:
                if isbn_info.get('type') == 'ISBN_13':
                    isbn_13 = isbn_info.get('identifier')
                elif isbn_info.get('type') == 'ISBN_10':
                    isbn_10 = isbn_info.get('identifier')
            
            publisher = vol_info.get('publisher')
            published_date = vol_info.get('publishedDate')
            language = vol_info.get('language', 'unknown')
            source_url = vol_info.get('infoLink')
        
        # Only return if we have a title and it's potentially Spanish
        # Be more lenient with language detection since many APIs don't properly tag Spanish
        if title_es:
            is_spanish_lang = language in ['es', 'spa', 'spanish']
            is_spanish_publisher = publisher and any(sp in publisher.lower() for sp in self.spanish_publishers)
            is_spanish_title = self.is_likely_spanish_title(title_es)
            
            # Accept if any of these conditions are met:
            if is_spanish_lang or is_spanish_publisher or is_spanish_title or language == 'unknown':
                return SpanishEdition(
                    title_es=title_es,
                    subtitle_es=subtitle_es,
                    language="es" if (is_spanish_lang or is_spanish_publisher or is_spanish_title) else language,
                    isbn_13=isbn_13,
                    isbn_10=isbn_10,
                    publisher=publisher,
                    published_date=published_date,
                    source=source,
                    source_url=source_url
                )
        
        return None
    
    async def find_spanish_edition(self, query: BookQuery) -> SearchResult:
        """Main method to find Spanish edition of a book"""
        print(f"🔍 Searching for Spanish edition of '{query.title}' by {query.author} ({query.year})")
        
        # Try to load from cache first
        cached_result = self._load_from_cache(query)
        if cached_result:
            return cached_result
        
        print("🌐 Not in cache, searching APIs...")
        
        raw_results = {
            'isbndb': [],
            'openlibrary': [],
            'google_books': []
        }
        
        candidates = []
        
        # Step 1: Search ISBNdb for ISBNs
        print("📚 Searching ISBNdb...")
        isbndb_results = await self.search_isbndb(query.title, query.author)
        raw_results['isbndb'] = isbndb_results
        
        # Step 2: Cross-lookup by ISBN
        isbns_to_check = set()
        for book in isbndb_results:
            # Filter by author similarity and year range
            if query.author.lower() in str(book.get('authors', '')).lower():
                if book.get('date_published'):
                    try:
                        book_year = int(re.search(r'\d{4}', book['date_published']).group())
                        if abs(book_year - query.year) <= 3:
                            if book.get('isbn13'):
                                isbns_to_check.add(book['isbn13'])
                            if book.get('isbn10'):
                                isbns_to_check.add(book['isbn10'])
                    except (AttributeError, ValueError):
                        pass
        
        print(f"🔢 Found {len(isbns_to_check)} ISBNs to check")
        
        # Cross-lookup ISBNs in Open Library and Google Books
        for isbn in list(isbns_to_check)[:10]:  # Limit to avoid rate limits
            ol_result = await self.lookup_openlibrary_isbn(isbn)
            if ol_result:
                raw_results['openlibrary'].append(ol_result)
                edition = self.extract_spanish_edition(ol_result, 'openlibrary')
                if edition:
                    candidates.append(edition)
            
            gb_result = await self.search_google_books_isbn(isbn)
            if gb_result:
                raw_results['google_books'].append(gb_result)
                edition = self.extract_spanish_edition(gb_result, 'google_books')
                if edition:
                    candidates.append(edition)
        
        # Step 3: Fallback title searches
        if not candidates:
            print("🔄 Fallback: Searching by title...")
            
            ol_results = await self.search_openlibrary_title(query.title, query.author)
            raw_results['openlibrary'].extend(ol_results)
            for result in ol_results:
                edition = self.extract_spanish_edition(result, 'openlibrary')
                if edition:
                    candidates.append(edition)
            
            gb_results = await self.search_google_books_title(query.title, query.author)
            raw_results['google_books'].extend(gb_results)
            for result in gb_results:
                edition = self.extract_spanish_edition(result, 'google_books')
                if edition:
                    candidates.append(edition)
        
        # Step 4: Score candidates
        print(f"📊 Scoring {len(candidates)} candidates...")
        for candidate in candidates:
            candidate.confidence = self.calculate_confidence(candidate, query, {})
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        # Step 5: Validate best candidate with OpenAI
        best_match = None
        if candidates:
            top_candidate = candidates[0]
            print(f"🤖 Validating top candidate with OpenAI...")
            validation = await self.validate_with_openai(
                top_candidate.title_es, 
                query.title, 
                query.author
            )
            top_candidate.official = validation
            best_match = top_candidate
        
        # Prepare notes
        notes = []
        if best_match:
            if best_match.isbn_13 or best_match.isbn_10:
                notes.append("Matched by ISBN")
            if best_match.official in ['valid_official', 'likely_translation_but_official']:
                notes.append("validated by OpenAI as official")
            notes.append(f"found via {best_match.source}")
        
        notes_str = "; ".join(notes) if notes else "No Spanish edition found"
        
        result = SearchResult(
            query=query,
            best_match=best_match,
            alternates=candidates[1:5] if len(candidates) > 1 else [],  # Top 4 alternates
            raw=raw_results,
            notes=notes_str
        )
        
        # Save to cache
        self._save_to_cache(query, result)
        
        return result

def main():
    """Example usage"""
    finder = SpanishEditionFinder()
    
    # Example search
    query = BookQuery(
        title="The Chronicles of Narnia: The Lion, the Witch and the Wardrobe",
        author="C.S. Lewis", 
        year=1950
    )
    
    async def run_search():
        result = await finder.find_spanish_edition(query)
        print("\n" + "="*60)
        print("📖 SEARCH RESULT")
        print("="*60)
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    
    asyncio.run(run_search())

if __name__ == "__main__":
    main()
