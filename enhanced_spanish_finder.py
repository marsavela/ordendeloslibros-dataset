#!/usr/bin/env python3
"""
Enhanced Spanish Edition Finder - Author Catalog + OpenAI Batch Matching
Prototype implementation of the recommended strategy.
"""

import os
import json
import asyncio
import aiohttp
import hashlib
import unicodedata
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from urllib.parse import quote
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

load_dotenv()

@dataclass
class BookQuery:
    """Query for finding Spanish editions of an English book"""
    title: str
    author: str
    year: Optional[int] = None

@dataclass
class SpanishBookCandidate:
    """Spanish book candidate from ISBNdb author catalog"""
    title: str
    isbn: str
    publisher: str
    year: str
    raw_data: Dict

@dataclass
class BookMatch:
    """Matched English-Spanish book pair"""
    english_title: str
    spanish_candidate: SpanishBookCandidate
    confidence: float
    reasoning: str
    match_method: str  # 'openai_batch' or 'fallback'

@dataclass
class SearchResult:
    """Result of a Spanish edition search"""
    query: BookQuery
    best_match: Optional[BookMatch]
    alternates: List[BookMatch]
    raw: Dict[str, Any]
    notes: str

class EnhancedSpanishFinder:
    """Enhanced Spanish edition finder using author catalogs + OpenAI batch matching"""
    
    def __init__(self):
        self.isbndb_key = os.getenv('ISBNDB_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.openai = OpenAI(api_key=self.openai_key) if self.openai_key else None
        self.author_catalogs = {}  # Memory cache for author Spanish catalogs
        self.cache_dir = Path('.cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl_days = 30  # Cache author catalogs for 30 days
        
    def get_author_cache_path(self, author: str) -> Path:
        """Get cache file path for author"""
        author_hash = hashlib.md5(author.encode()).hexdigest()
        return self.cache_dir / f"author_{author_hash}.json"
    
    def load_author_from_cache(self, author: str) -> Optional[List[SpanishBookCandidate]]:
        """Load author Spanish catalog from cache if valid"""
        cache_path = self.get_author_cache_path(author)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check cache expiry
            cached_date = datetime.fromisoformat(cache_data['cached_at'])
            if datetime.now() - cached_date > timedelta(days=self.cache_ttl_days):
                print(f"  💾 Cache expired for {author}")
                return None
            
            # Convert back to SpanishBookCandidate objects
            candidates = []
            for book_data in cache_data['spanish_books']:
                candidate = SpanishBookCandidate(
                    title=book_data['title'],
                    isbn=book_data['isbn'],
                    publisher=book_data['publisher'],
                    year=book_data['year'],
                    raw_data=book_data['raw_data']
                )
                candidates.append(candidate)
            
            print(f"  💾 Loaded {len(candidates)} Spanish books from cache for {author}")
            return candidates
            
        except Exception as e:
            print(f"  ❌ Cache load failed for {author}: {e}")
            return None
    
    def save_author_to_cache(self, author: str, spanish_books: List[SpanishBookCandidate]):
        """Save author Spanish catalog to cache"""
        cache_path = self.get_author_cache_path(author)
        
        try:
            # Convert to serializable format
            serializable_books = []
            for candidate in spanish_books:
                serializable_books.append({
                    'title': candidate.title,
                    'isbn': candidate.isbn,
                    'publisher': candidate.publisher,
                    'year': candidate.year,
                    'raw_data': candidate.raw_data
                })
            
            cache_data = {
                'author': author,
                'spanish_books': serializable_books,
                'total_count': len(spanish_books),
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"  💾 Cached {len(spanish_books)} Spanish books for {author}")
            
        except Exception as e:
            print(f"  ❌ Cache save failed for {author}: {e}")
    
    # Book-level caching methods
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison and caching"""
        if not text:
            return ""
        # Remove accents and convert to lowercase
        text = unicodedata.normalize('NFD', text.lower())
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        # Remove special characters except spaces and alphanumeric
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def get_book_cache_key(self, query: BookQuery) -> str:
        """Generate cache key for a book query"""
        # Normalize inputs for consistent caching
        title_norm = self.normalize_text(query.title)
        author_norm = self.normalize_text(query.author)
        
        # Create hash of normalized query
        key_string = f"{title_norm}|{author_norm}|{query.year or ''}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_book_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a book query"""
        return self.cache_dir / f"book_{cache_key}.json"
    
    def load_book_from_cache(self, query: BookQuery) -> Optional[SearchResult]:
        """Load book search result from cache if available"""
        cache_key = self.get_book_cache_key(query)
        cache_path = self.get_book_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid (not older than 30 days)
            cached_time = datetime.fromisoformat(cached_data.get('cached_at', '2000-01-01'))
            if (datetime.now() - cached_time).days < 30:
                print(f"💾 Loading book from cache: {query.title}")
                
                # Reconstruct SearchResult from cached data
                best_match = None
                if cached_data.get('best_match'):
                    match_data = cached_data['best_match']
                    candidate_data = match_data['spanish_candidate']
                    spanish_candidate = SpanishBookCandidate(
                        title=candidate_data['title'],
                        isbn=candidate_data['isbn'],
                        publisher=candidate_data['publisher'],
                        year=candidate_data['year'],
                        raw_data=candidate_data['raw_data']
                    )
                    best_match = BookMatch(
                        english_title=match_data['english_title'],
                        spanish_candidate=spanish_candidate,
                        confidence=match_data['confidence'],
                        reasoning=match_data['reasoning'],
                        match_method=match_data['match_method']
                    )
                
                alternates = []
                for alt_data in cached_data.get('alternates', []):
                    candidate_data = alt_data['spanish_candidate']
                    spanish_candidate = SpanishBookCandidate(
                        title=candidate_data['title'],
                        isbn=candidate_data['isbn'],
                        publisher=candidate_data['publisher'],
                        year=candidate_data['year'],
                        raw_data=candidate_data['raw_data']
                    )
                    alternates.append(BookMatch(
                        english_title=alt_data['english_title'],
                        spanish_candidate=spanish_candidate,
                        confidence=alt_data['confidence'],
                        reasoning=alt_data['reasoning'],
                        match_method=alt_data['match_method']
                    ))
                
                return SearchResult(
                    query=BookQuery(
                        title=cached_data['query']['title'],
                        author=cached_data['query']['author'],
                        year=cached_data['query'].get('year')
                    ),
                    best_match=best_match,
                    alternates=alternates,
                    raw=cached_data.get('raw', {}),
                    notes=cached_data.get('notes', '')
                )
                
        except Exception as e:
            print(f"⚠️  Book cache read error for {query.title}: {e}")
            # Delete corrupt cache file
            cache_path.unlink(missing_ok=True)
        
        return None
    
    def save_book_to_cache(self, query: BookQuery, result: SearchResult):
        """Save book search result to cache"""
        cache_key = self.get_book_cache_key(query)
        cache_path = self.get_book_cache_path(cache_key)
        
        try:
            # Convert BookMatch objects to serializable format
            def serialize_book_match(match: BookMatch) -> dict:
                return {
                    'english_title': match.english_title,
                    'spanish_candidate': {
                        'title': match.spanish_candidate.title,
                        'isbn': match.spanish_candidate.isbn,
                        'publisher': match.spanish_candidate.publisher,
                        'year': match.spanish_candidate.year,
                        'raw_data': match.spanish_candidate.raw_data
                    },
                    'confidence': match.confidence,
                    'reasoning': match.reasoning,
                    'match_method': match.match_method
                }
            
            cache_data = {
                'query': {
                    'title': result.query.title,
                    'author': result.query.author,
                    'year': result.query.year
                },
                'best_match': serialize_book_match(result.best_match) if result.best_match else None,
                'alternates': [serialize_book_match(alt) for alt in result.alternates],
                'raw': result.raw,
                'notes': result.notes,
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Saved book to cache: {query.title}")
            
        except Exception as e:
            print(f"⚠️  Book cache write error for {query.title}: {e}")
    
    async def find_spanish_edition(self, query: BookQuery) -> SearchResult:
        """Find Spanish edition for a single book with caching"""
        # Try to load from cache first
        cached_result = self.load_book_from_cache(query)
        if cached_result:
            return cached_result
        
        print(f"🌐 Not in cache, searching for: {query.title} by {query.author}")
        
        # Get author's Spanish catalog
        spanish_catalog = await self.get_author_spanish_catalog(query.author)
        
        if not spanish_catalog:
            # No Spanish books found for this author
            result = SearchResult(
                query=query,
                best_match=None,
                alternates=[],
                raw={'spanish_catalog_size': 0},
                notes=f"No Spanish books found for author {query.author}"
            )
            self.save_book_to_cache(query, result)
            return result
        
        # Filter to most relevant candidates (top 50)
        top_candidates = self.filter_most_relevant_candidates(spanish_catalog, [query.title])
        
        if not top_candidates:
            # No relevant candidates found
            result = SearchResult(
                query=query,
                best_match=None,
                alternates=[],
                raw={'spanish_catalog_size': len(spanish_catalog), 'filtered_candidates': 0},
                notes=f"No relevant Spanish candidates found among {len(spanish_catalog)} books"
            )
            self.save_book_to_cache(query, result)
            return result
        
        # Use OpenAI to match single book
        matches = self.match_with_openai_batch([query.title], top_candidates)
        
        # Matches are already BookMatch objects
        book_matches = matches
        
        # Separate best match from alternates
        best_match = book_matches[0] if book_matches else None
        alternates = book_matches[1:] if len(book_matches) > 1 else []
        
        result = SearchResult(
            query=query,
            best_match=best_match,
            alternates=alternates,
            raw={
                'spanish_catalog_size': len(spanish_catalog),
                'filtered_candidates': len(top_candidates),
                'openai_matches': len(book_matches)
            },
            notes=f"Found {len(book_matches)} matches from {len(spanish_catalog)} Spanish books by {query.author}"
        )
        
        # Save to cache
        self.save_book_to_cache(query, result)
        
        return result
        
    async def get_author_spanish_catalog(self, author: str) -> List[SpanishBookCandidate]:
        """Get complete Spanish book catalog for an author from ISBNdb"""
        # Check memory cache first
        if author in self.author_catalogs:
            print(f"  💾 Using memory cache for {author}")
            return self.author_catalogs[author]
        
        # Check disk cache
        cached_books = self.load_author_from_cache(author)
        if cached_books is not None:
            self.author_catalogs[author] = cached_books
            return cached_books
            
        if not self.isbndb_key:
            return []
            
        print(f"🔍 Building Spanish catalog for {author}...")
        spanish_books = []
        page = 1
        total_books = 0
        
        async with aiohttp.ClientSession() as session:
            while True:
                url = f"https://api2.isbndb.com/author/{quote(author)}?pageSize=100&page={page}"
                headers = {'Authorization': self.isbndb_key}
                
                try:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            books = data.get('books', [])
                            total_books += len(books)
                            
                            # Filter Spanish books
                            page_spanish = [
                                book for book in books 
                                if book.get('language') == 'es'
                            ]
                            
                            for book in page_spanish:
                                candidate = SpanishBookCandidate(
                                    title=book.get('title', ''),
                                    isbn=book.get('isbn13', book.get('isbn', '')),
                                    publisher=book.get('publisher', ''),
                                    year=book.get('date_published', '')[:4] if book.get('date_published') else '',
                                    raw_data=book
                                )
                                spanish_books.append(candidate)
                            
                            print(f"  📄 Page {page}: {len(books)} total, {len(page_spanish)} Spanish")
                            
                            if len(books) < 100:
                                break
                            page += 1
                        else:
                            print(f"  ❌ Error {response.status}: {await response.text()}")
                            break
                            
                    await asyncio.sleep(0.6)  # Rate limiting for ISBNdb
                    
                except Exception as e:
                    print(f"  ❌ Request failed: {e}")
                    break
        
        print(f"  ✅ Found {len(spanish_books)} Spanish books out of {total_books} total")
        
        # Cache both in memory and disk
        self.author_catalogs[author] = spanish_books
        self.save_author_to_cache(author, spanish_books)
        
        return spanish_books
    
    def format_candidates_for_openai(self, candidates: List[SpanishBookCandidate]) -> str:
        """Format Spanish candidates for OpenAI prompt"""
        formatted = []
        for i, candidate in enumerate(candidates):
            formatted.append(
                f"{i+1}. \"{candidate.title}\" ({candidate.year}) - {candidate.publisher}"
            )
        return "\\n".join(formatted)
    
    def filter_most_relevant_candidates(self, candidates: List[SpanishBookCandidate], 
                                       english_books: List[str], max_candidates: int = 50) -> List[SpanishBookCandidate]:
        """Filter to most relevant Spanish candidates for OpenAI matching"""
        if len(candidates) <= max_candidates:
            return candidates
        
        # Score candidates by relevance
        scored_candidates = []
        english_keywords = set()
        
        # Extract keywords from English titles
        for title in english_books:
            words = title.lower().replace("'", "").split()
            english_keywords.update([w for w in words if len(w) > 3])
        
        for candidate in candidates:
            score = 0.0
            title_lower = candidate.title.lower()
            
            # Score by keyword overlap
            for keyword in english_keywords:
                if keyword in title_lower:
                    score += 1.0
            
            # Score by publisher reputation (Spanish publishers)
            publisher = candidate.publisher.lower()
            if any(pub in publisher for pub in ['salamandra', 'alfaguara', 'empúries', 'destino']):
                score += 2.0
            
            # Score by recency (prefer newer editions)
            if candidate.year and candidate.year.isdigit():
                year = int(candidate.year)
                if year >= 2000:
                    score += 1.0
                if year >= 2010:
                    score += 1.0
            
            scored_candidates.append((score, candidate))
        
        # Sort by score and take top candidates
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [candidate for _, candidate in scored_candidates[:max_candidates]]
    
    def match_with_openai_batch(self, english_books: List[str], 
                                     spanish_candidates: List[SpanishBookCandidate]) -> List[BookMatch]:
        """Use OpenAI to batch match English books to Spanish candidates"""
        if not self.openai or not english_books or not spanish_candidates:
            return []
        
        print(f"🤖 Matching {len(english_books)} books with {len(spanish_candidates)} Spanish candidates...")
        
        # Filter to most relevant candidates to avoid token limit
        max_candidates = 50
        relevant_candidates = self.filter_most_relevant_candidates(
            spanish_candidates, english_books, max_candidates
        )
        
        if len(spanish_candidates) > max_candidates:
            print(f"  📊 Filtered to {len(relevant_candidates)} most relevant candidates")
        
        # Format books for prompt
        english_list = "\\n".join([f"{i+1}. \"{title}\"" for i, title in enumerate(english_books)])
        spanish_list = self.format_candidates_for_openai(relevant_candidates)
        
        prompt = f"""You are a bibliographic expert specializing in Spanish translations of English books.

ENGLISH BOOKS TO MATCH:
{english_list}

SPANISH CANDIDATES:
{spanish_list}

For each English book, find its official Spanish translation from the candidates.
Return a JSON array where each element has:
- "english_index": index of English book (1-based)
- "spanish_index": index of Spanish candidate (1-based) or null if no match
- "confidence": confidence score 0.0-1.0
- "reasoning": brief explanation of why this is/isn't a match

Consider:
- Translation patterns (e.g., "Philosopher's Stone" = "piedra filosofal")
- Publication chronology
- Publisher reputation for translations
- Series context and numbering

Return only valid JSON array."""

        try:
            response = self.openai.responses.create(
                model="gpt-5-nano",
                input=prompt
            )
            
            result_text = response.output_text
            matches_data = json.loads(result_text)
            
            matches = []
            for match_data in matches_data:
                english_idx = match_data.get('english_index', 1) - 1
                spanish_idx = match_data.get('spanish_index')
                
                if (0 <= english_idx < len(english_books) and 
                    spanish_idx and 0 < spanish_idx <= len(relevant_candidates)):
                    
                    match = BookMatch(
                        english_title=english_books[english_idx],
                        spanish_candidate=relevant_candidates[spanish_idx - 1],
                        confidence=match_data.get('confidence', 0.0),
                        reasoning=match_data.get('reasoning', ''),
                        match_method='openai_batch'
                    )
                    matches.append(match)
            
            print(f"  ✅ OpenAI found {len(matches)} matches")
            return matches
            
        except Exception as e:
            print(f"  ❌ OpenAI matching failed: {e}")
            return []
    
    def find_spanish_editions_enhanced(self, books_by_author: Dict[str, List[str]]) -> Dict[str, List[BookMatch]]:
        """Enhanced Spanish edition finder using author catalogs + OpenAI batch matching"""
        results = {}
        
        for author, english_books in books_by_author.items():
            print(f"\\n📚 Processing {len(english_books)} books by {author}")
            
            # Step 1: Get author's Spanish catalog
            spanish_catalog = asyncio.run(self.get_author_spanish_catalog(author))
            
            if not spanish_catalog:
                print(f"  ⚠️  No Spanish books found for {author}")
                results[author] = []
                continue
            
            # Step 2: Batch match with OpenAI
            matches = self.match_with_openai_batch(english_books, spanish_catalog)
            results[author] = matches
            
            # Summary
            matched_count = len(matches)
            print(f"  📊 Matched {matched_count}/{len(english_books)} books")
            
            for match in matches:
                print(f"    ✅ \"{match.english_title}\" → \"{match.spanish_candidate.title}\" ({match.confidence:.2f})")
        
        return results

# Demo/Test function
def test_enhanced_finder():
    """Test the enhanced Spanish finder with sample data"""
    finder = EnhancedSpanishFinder()
    
    # Test data: books grouped by author
    test_books = {
        'J.K. Rowling': [
            "Harry Potter and the Philosopher's Stone",
            'Harry Potter and the Chamber of Secrets', 
            'Harry Potter and the Prisoner of Azkaban'
        ],
        'Roald Dahl': [
            'Charlie and the Chocolate Factory',
            'The Witches',
            'Matilda'
        ]
    }
    
    print("🚀 Testing Enhanced Spanish Edition Finder")
    results = finder.find_spanish_editions_enhanced(test_books)
    
    # Save results
    output = {
        'method': 'enhanced_author_catalog_openai',
        'results': {}
    }
    
    for author, matches in results.items():
        output['results'][author] = [
            {
                'english_title': match.english_title,
                'spanish_title': match.spanish_candidate.title,
                'spanish_publisher': match.spanish_candidate.publisher,
                'spanish_year': match.spanish_candidate.year,
                'confidence': match.confidence,
                'reasoning': match.reasoning,
                'method': match.match_method
            }
            for match in matches
        ]
    
    with open('enhanced_spanish_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\\n✅ Results saved to enhanced_spanish_results.json")
    return results

if __name__ == "__main__":
    test_enhanced_finder()
