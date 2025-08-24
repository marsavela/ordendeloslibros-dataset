#!/usr/bin/env python3
"""
Series Matching Script - CLEAN VERSION
=====================================

This script matches book series from Wikipedia with series from orderofbooks.com
and creates a combined dataset with complete book information.

Usage:
    python match_series_clean.py [--openai-api-key YOUR_KEY] [--verbose]

Requirements:
    pip install rapidfuzz openai python-dotenv
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import argparse
import os

try:
    from rapidfuzz import fuzz, process
except ImportError:
    print("Please install rapidfuzz: pip install rapidfuzz")
    exit(1)

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI not available. Install with: pip install openai")

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

@dataclass
class SeriesMatch:
    wikipedia_series: dict
    orderofbooks_series: Optional[dict]
    confidence: float
    match_type: str  # 'exact', 'fuzzy', 'gpt_confirmed', 'gpt_rejected', 'no_match'
    notes: str = ""

class SeriesMatcher:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_client = None
        if openai_api_key and HAS_OPENAI:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        self.orderofbooks_data = []
        self.wikipedia_data = []
        self.matches = []
        
    def load_data(self, orderofbooks_path: str = "orderofbooks_series.json", 
                  wikipedia_path: str = "best_selling_book_series.json"):
        """Load data from both sources"""
        # Load orderofbooks data
        if Path(orderofbooks_path).exists():
            with open(orderofbooks_path, 'r', encoding='utf-8') as f:
                self.orderofbooks_data = json.load(f)
            print(f"Loaded {len(self.orderofbooks_data)} series from orderofbooks")
        else:
            print(f"Warning: {orderofbooks_path} not found")
            
        # Load wikipedia data
        if Path(wikipedia_path).exists():
            with open(wikipedia_path, 'r', encoding='utf-8') as f:
                wiki_data = json.load(f)
                # Extract series from the nested structure
                self.wikipedia_data = []
                if 'ranges' in wiki_data:
                    for range_data in wiki_data['ranges']:
                        if 'series' in range_data:
                            for series in range_data['series']:
                                if series.get('series'):  # has series name
                                    self.wikipedia_data.append(series)
                print(f"Loaded {len(self.wikipedia_data)} series from Wikipedia")
        else:
            print(f"Warning: {wikipedia_path} not found")
            
    def normalize_series_name(self, name: str) -> str:
        """Normalize series name for comparison"""
        if not name:
            return ""
            
        # Convert to lowercase
        name = name.lower()
        
        # Remove orderofbooks specific prefixes - very aggressive
        name = re.sub(r'^(publication|chronological)\s+order\s+of\s+(the\s+)?', '', name)
        name = re.sub(r'^bookshots:\s*', '', name)  # Remove bookshots prefix
        
        # Remove common suffixes more aggressively
        name = re.sub(r'\s+(books?|series|novels?|stories?|collection)(\s+books?)?\s*$', '', name)
        name = re.sub(r'\s+(short\s+stories?|novellas?|collections?|miscellaneous)\s*$', '', name)
        name = re.sub(r'\s+(graphic\s+novels?)\s*$', '', name)
        name = re.sub(r'\s+(mystery|mysteries)\s*$', '', name)
        name = re.sub(r'\s+order\s*$', '', name)
        name = re.sub(r'\s+trilogy\s*$', '', name)  # Remove trilogy suffix but keep for comparison
        
        # Remove articles and common words
        name = re.sub(r'^(the|a|an)\s+', '', name)
        name = re.sub(r'\s+(the|a|an)\s+', ' ', name)
        
        # Handle special cases that should match
        if 'harry potter' in name:
            name = 'harry potter'
        elif 'alex cross' in name:
            name = 'alex cross'
        elif 'jack reacher' in name:
            name = 'jack reacher'
        elif 'dark tower' in name:
            name = 'dark tower'
        elif 'doc savage' in name:
            name = 'doc savage'
        elif 'james bond' in name:
            name = 'james bond'
        elif 'nancy drew' in name:
            name = 'nancy drew'
        elif 'hardy boys' in name:
            name = 'hardy boys'
        elif 'goosebumps' in name:
            name = 'goosebumps'
        elif 'percy jackson' in name:
            name = 'percy jackson'
        elif 'wimpy kid' in name:
            name = 'diary wimpy kid'
        elif 'magic tree house' in name:
            name = 'magic tree house'
        elif 'captain underpants' in name:
            name = 'captain underpants'
        elif 'artemis fowl' in name:
            name = 'artemis fowl'
        elif 'inheritance cycle' in name or 'eragon' in name:
            name = 'inheritance cycle'
        elif 'hunger games' in name:
            name = 'hunger games'
        elif 'twilight' in name:
            name = 'twilight'
        elif 'chronicles narnia' in name or 'narnia' in name:
            name = 'chronicles narnia'
        elif 'fifty shades' in name:
            name = 'fifty shades'
        elif 'game of thrones' in name or 'song of ice and fire' in name:
            name = 'song ice fire'
        elif 'wheel of time' in name:
            name = 'wheel time'
        elif 'discworld' in name:
            name = 'discworld'
        elif 'left behind' in name:
            name = 'left behind'
        elif 'little house' in name and 'prairie' in name:
            name = 'little house prairie'
        elif 'winnie the pooh' in name or 'winnie-the-pooh' in name:
            name = 'winnie pooh'
        elif 'his dark materials' in name:
            name = 'his dark materials'
        elif 'hitchhiker' in name and 'galaxy' in name:
            name = 'hitchhiker guide galaxy'
        elif 'foundation' in name and ('asimov' in name.lower() or 'trilogy' in name):
            name = 'foundation'
        elif 'dune' in name:
            name = 'dune'
        elif 'outlander' in name:
            name = 'outlander'
        elif 'bridget jones' in name:
            name = 'bridget jones'
        elif 'redwall' in name:
            name = 'redwall'
        elif 'animorphs' in name:
            name = 'animorphs'
        elif 'warriors' in name and ('erin hunter' in name or 'cats' in name):
            name = 'warriors'
        elif 'shadowhunter' in name:
            name = 'shadowhunter'
        elif 'divergent' in name:
            name = 'divergent'
        elif 'maze runner' in name:
            name = 'maze runner'
        elif 'vampir' in name and ('chronicles' in name or 'anne rice' in name):
            name = 'vampire chronicles'
        elif 'millennium' in name and ('stieg' in name or 'larsson' in name):
            name = 'millennium'
        elif 'shannara' in name:
            name = 'shannara'
        elif 'sword truth' in name:
            name = 'sword truth'
        elif 'dragonlance' in name:
            name = 'dragonlance'
        elif 'dragonriders' in name and 'pern' in name:
            name = 'dragonriders pern'
        elif 'cosmere' in name or 'sanderson' in name:
            name = 'cosmere'
        elif 'witcher' in name:
            name = 'witcher'
        elif 'thrawn' in name:
            name = 'thrawn'
        elif 'riftwar' in name:
            name = 'riftwar'
        elif 'dirk pitt' in name:
            name = 'dirk pitt'
        elif 'robert langdon' in name or ('dan brown' in name and 'angel' in name):
            name = 'robert langdon'
        elif 'horrible histories' in name:
            name = 'horrible histories'
        elif 'rainbow magic' in name:
            name = 'rainbow magic'
        elif 'southern vampire' in name:
            name = 'southern vampire mysteries'
        elif 'no 1 ladies' in name or 'ladies detective agency' in name:
            name = 'no 1 ladies detective agency'
        elif 'berenstain bears' in name:
            name = 'berenstain bears'
        elif 'curious george' in name:
            name = 'curious george'
        elif 'paddington' in name:
            name = 'paddington'
        elif 'clifford' in name and ('red dog' in name or 'big red' in name):
            name = 'clifford big red dog'
        elif 'noddy' in name:
            name = 'noddy'
        elif 'mr men' in name or 'mr. men' in name:
            name = 'mr men'
        elif 'little critter' in name:
            name = 'little critter'
        elif 'peter rabbit' in name:
            name = 'peter rabbit'
        elif 'railway series' in name or 'thomas tank engine' in name:
            name = 'railway series'
        elif 'american girl' in name:
            name = 'american girl'
        elif 'babysitters club' in name or 'baby-sitters club' in name:
            name = 'babysitters club'
        elif 'geronimo stilton' in name:
            name = 'geronimo stilton'
        elif 'choose your own adventure' in name:
            name = 'choose your own adventure'
        elif 'sweet valley' in name:
            name = 'sweet valley high'
        elif 'fear street' in name:
            name = 'fear street'
        elif 'magic school bus' in name:
            name = 'magic school bus'
        elif 'series unfortunate events' in name:
            name = 'series unfortunate events'
        elif 'where\'s wally' in name or 'where\'s waldo' in name:
            name = 'where wally'
        elif 'all creatures great small' in name:
            name = 'all creatures great small'
        elif 'men mars women venus' in name:
            name = 'men mars women venus'
        elif 'chicken soup soul' in name:
            name = 'chicken soup soul'
        elif 'frank merriwell' in name:
            name = 'frank merriwell'
        elif 'vampire hunter d' in name:
            name = 'vampire hunter d'
        elif 'southern vampire' in name:
            name = 'southern vampire'
        else:
            # Remove extra whitespace and punctuation for non-special cases
            name = re.sub(r'[^\w\s]', ' ', name)
            name = re.sub(r'\s+', ' ', name).strip()
        
        return name
        
    def find_author_matches(self, orderofbooks_series: dict, wikipedia_series: dict) -> float:
        """Calculate author similarity between series"""
        oob_author = orderofbooks_series.get('author', '').lower()
        wiki_authors = wikipedia_series.get('authors', [])
        
        if not wiki_authors or not oob_author:
            return 0.0
            
        # Check if any Wikipedia author matches the orderofbooks author
        best_author_match = 0.0
        for wiki_author in wiki_authors:
            if isinstance(wiki_author, str):
                wiki_author_clean = wiki_author.lower()
                
                # Exact match (case insensitive)
                if oob_author == wiki_author_clean:
                    return 1.0
                
                # Direct fuzzy comparison
                similarity = fuzz.ratio(oob_author, wiki_author_clean) / 100.0
                best_author_match = max(best_author_match, similarity)
                
                # Check if last names match (for pen names)
                oob_parts = oob_author.split()
                wiki_parts = wiki_author_clean.split()
                if oob_parts and wiki_parts and len(oob_parts) > 1 and len(wiki_parts) > 1:
                    if oob_parts[-1] == wiki_parts[-1] and len(oob_parts[-1]) > 2:  # Same last name (not initials)
                        best_author_match = max(best_author_match, 0.8)
                    
                # Check partial matches for compound names (stricter)
                for oob_part in oob_parts:
                    if len(oob_part) > 3:  # Only match longer name parts
                        for wiki_part in wiki_parts:
                            if len(wiki_part) > 3 and (oob_part in wiki_part or wiki_part in oob_part):
                                best_author_match = max(best_author_match, 0.6)
                
        return best_author_match
        
    def is_author_compatible(self, orderofbooks_series: dict, wikipedia_series: dict) -> bool:
        """Check if authors are compatible (not completely different)"""
        author_sim = self.find_author_matches(orderofbooks_series, wikipedia_series)
        
        # If we have high author similarity, they're compatible
        if author_sim > 0.7:
            return True
            
        # Special cases for known compatible situations
        oob_author = orderofbooks_series.get('author', '').lower()
        wiki_authors = [a.lower() for a in wikipedia_series.get('authors', [])]
        
        # Same person, different attribution styles
        if 'j. k. rowling' in wiki_authors and 'joanne rowling' in oob_author:
            return True
        if 'stephen king' in wiki_authors and 'stephen king' in oob_author:
            return True
        if 'james patterson' in wiki_authors and 'james patterson' in oob_author:
            return True
            
        # If author similarity is very low, they're incompatible
        if author_sim < 0.2:
            return False
            
        return True
        
    def ask_gpt_for_match_confirmation(self, oob_series: dict, wiki_series: dict, confidence: float) -> Tuple[bool, str]:
        """Use GPT-5-nano to confirm if two series are the same"""
        if not self.openai_client:
            return False, "OpenAI not available"
            
        oob_name = oob_series.get('series_name', '')
        oob_author = oob_series.get('author', '')
        oob_books = [book.get('title', '') for book in oob_series.get('books', [])[:5]]  # First 5 books
        
        wiki_name = wiki_series.get('series', '')
        wiki_authors = wiki_series.get('authors', [])
        
        prompt = f"""Are these two book series the same series?

Series 1 (from orderofbooks.com):
- Name: {oob_name}
- Author: {oob_author}
- Sample books: {', '.join(oob_books)}

Series 2 (from Wikipedia):
- Name: {wiki_name}
- Authors: {', '.join(wiki_authors) if wiki_authors else 'Unknown'}

Confidence from fuzzy matching: {confidence:.2f}

Please answer with just "YES" or "NO", followed by a brief explanation."""

        try:
            response = self.openai_client.responses.create(
                model="gpt-5-nano",
                input=prompt
            )
            
            answer = response.output_text.strip()
            is_match = answer.upper().startswith('YES')
            return is_match, answer
            
        except Exception as e:
            return False, f"OpenAI error: {str(e)}"
            
    def match_series(self, confidence_threshold: float = 0.5, verbose: bool = False):
        """Match Wikipedia series with orderofbooks data"""
        print(f"\\nMatching {len(self.wikipedia_data)} Wikipedia series with {len(self.orderofbooks_data)} orderofbooks series")
        print(f"Confidence threshold: {confidence_threshold}")
        
        # Create a list of normalized orderofbooks series for faster searching
        oob_choices = []
        for i, oob_series in enumerate(self.orderofbooks_data):
            oob_name = oob_series.get('series_name', '')
            normalized = self.normalize_series_name(oob_name)
            oob_choices.append((normalized, i, oob_name))
        
        for i, wiki_series in enumerate(self.wikipedia_data):
            if i % 20 == 0:
                print(f"Processing Wikipedia series {i+1}/{len(self.wikipedia_data)}")
                
            wiki_name = wiki_series.get('series', '')
            wiki_authors = wiki_series.get('authors', [])
            normalized_wiki = self.normalize_series_name(wiki_name)
            
            if verbose:
                print(f"\\n--- Processing #{i+1}: '{wiki_name}' by {wiki_authors} ---")
                print(f"Normalized: '{normalized_wiki}'")
            
            if not normalized_wiki:
                if verbose:
                    print("❌ SKIP: Empty normalized series name")
                self.matches.append(SeriesMatch(
                    wikipedia_series=wiki_series,
                    orderofbooks_series=None,
                    confidence=0.0,
                    match_type='no_match',
                    notes='Empty Wikipedia series name'
                ))
                continue
            
            # Find best match using rapidfuzz
            best_matches = process.extract(
                normalized_wiki, 
                [choice[0] for choice in oob_choices],
                scorer=fuzz.token_sort_ratio,
                limit=5
            )
            
            if verbose:
                print(f"Top 5 fuzzy matches:")
                for match_name, score, _ in best_matches:
                    # Find original name
                    orig_name = next(orig for norm, _, orig in oob_choices if norm == match_name)
                    print(f"  {score:.1f}% - '{orig_name}' (normalized: '{match_name}')")
            
            if not best_matches:
                if verbose:
                    print("❌ NO MATCHES: No similar series found")
                self.matches.append(SeriesMatch(
                    wikipedia_series=wiki_series,
                    orderofbooks_series=None,
                    confidence=0.0,
                    match_type='no_match',
                    notes='No similar series found in orderofbooks'
                ))
                continue
                
            best_match, best_score, _ = best_matches[0]
            confidence = best_score / 100.0
            
            # Find the corresponding orderofbooks series
            oob_index = next(i for norm, i, _ in oob_choices if norm == best_match)
            oob_series = self.orderofbooks_data[oob_index]
            
            # STRICT AUTHOR COMPATIBILITY CHECK
            if not self.is_author_compatible(oob_series, wiki_series):
                if verbose:
                    print(f"❌ AUTHOR MISMATCH: Incompatible authors - skipping")
                    print(f"   OOB: '{oob_series.get('author', '')}'")
                    print(f"   Wiki: {wiki_authors}")
                self.matches.append(SeriesMatch(
                    wikipedia_series=wiki_series,
                    orderofbooks_series=None,
                    confidence=0.0,
                    match_type='no_match',
                    notes=f'Author incompatible: {oob_series.get("author", "")} vs {wiki_authors}'
                ))
                continue
            
            # Also check author similarity
            author_similarity = self.find_author_matches(oob_series, wiki_series)
            
            if verbose:
                print(f"Best match: {best_score:.1f}% confidence")
                print(f"Author similarity: {author_similarity:.2f}")
                print(f"OOB Author: '{oob_series.get('author', '')}'")
                print(f"Wiki Authors: {wiki_authors}")
            
            # Adjust confidence based on author match
            original_confidence = confidence
            if author_similarity > 0.8:
                confidence = min(1.0, confidence + 0.15)  # Strong boost for good author match
            elif author_similarity > 0.6:
                confidence = min(1.0, confidence + 0.1)   # Medium boost
            elif author_similarity < 0.2 and confidence > 0.7:
                confidence -= 0.4  # Very strong penalty for author mismatch on high-confidence matches
            elif author_similarity < 0.4 and confidence > 0.6:
                confidence -= 0.25  # Strong penalty for poor author match
                
            # Additional penalty for suspicious matches (common words that don't really match)
            wiki_norm = normalized_wiki
            oob_norm = self.normalize_series_name(oob_series.get('series_name', ''))
            
            # Penalize matches that are just common words
            common_words = {'chronicles', 'mysteries', 'collection', 'stories', 'novels', 'trilogy', 'series', 'diaries', 'vampire', 'girl', 'american', 'little', 'critter', 'geniuses', 'dog'}
            wiki_words = set(wiki_norm.split())
            oob_words = set(oob_norm.split())
            
            shared_words = wiki_words.intersection(oob_words)
            non_common_shared = shared_words - common_words
            
            # If only common words match, heavily penalize
            if len(shared_words) > 0 and len(non_common_shared) == 0:
                confidence -= 0.4  # Very heavy penalty for matches based only on common words
                if verbose:
                    print(f"⚠️  COMMON WORD PENALTY: Match based only on common word(s): {shared_words}")
            elif len(non_common_shared) == 1 and len(shared_words) <= 2:
                # Only one meaningful word matches
                confidence -= 0.2
                if verbose:
                    print(f"⚠️  WEAK MATCH PENALTY: Only one meaningful word matches: {non_common_shared}")
            
            # Special penalty for obviously wrong matches
            if ('american girl' in wiki_norm and 'american vampire' in oob_norm) or \
               ('little critter' in wiki_norm and 'little geniuses' in oob_norm) or \
               ('dork diaries' in wiki_norm and 'dog diaries' in oob_norm):
                confidence -= 0.5
                if verbose:
                    print(f"⚠️  OBVIOUS MISMATCH PENALTY: These are clearly different series")
                
            if verbose:
                if confidence != original_confidence:
                    print(f"Adjusted confidence: {original_confidence:.2f} -> {confidence:.2f}")
                
            if confidence >= confidence_threshold:
                # High confidence match
                match_type = 'exact' if confidence >= 0.95 else 'fuzzy'
                if verbose:
                    print(f"✅ MATCH ACCEPTED: High confidence ({confidence:.2f}) - {match_type}")
                self.matches.append(SeriesMatch(
                    wikipedia_series=wiki_series,
                    orderofbooks_series=oob_series,
                    confidence=confidence,
                    match_type=match_type,
                    notes=f'Author similarity: {author_similarity:.2f}'
                ))
            elif confidence >= 0.4 and self.openai_client:
                # Medium confidence - ask GPT
                if verbose:
                    print(f"🤔 ASKING GPT-5-nano: Medium confidence ({confidence:.2f})")
                is_match, gpt_response = self.ask_gpt_for_match_confirmation(oob_series, wiki_series, confidence)
                if is_match:
                    if verbose:
                        print(f"✅ GPT CONFIRMED: {gpt_response}")
                    self.matches.append(SeriesMatch(
                        wikipedia_series=wiki_series,
                        orderofbooks_series=oob_series,
                        confidence=confidence,
                        match_type='gpt_confirmed',
                        notes=f'GPT: {gpt_response}'
                    ))
                else:
                    if verbose:
                        print(f"❌ GPT REJECTED: {gpt_response}")
                    self.matches.append(SeriesMatch(
                        wikipedia_series=wiki_series,
                        orderofbooks_series=None,
                        confidence=confidence,
                        match_type='gpt_rejected',
                        notes=f'GPT: {gpt_response}'
                    ))
            else:
                # Low confidence - no match
                if verbose:
                    print(f"❌ NO MATCH: Low confidence ({confidence:.2f})")
                self.matches.append(SeriesMatch(
                    wikipedia_series=wiki_series,
                    orderofbooks_series=None,
                    confidence=confidence,
                    match_type='no_match',
                    notes=f'Low confidence: {confidence:.2f}'
                ))
                
    def create_combined_dataset(self) -> List[dict]:
        """Create combined dataset with matched series"""
        combined = []
        
        for match in self.matches:
            if match.orderofbooks_series:
                # Found a match - combine the data
                combined_series = {
                    'wikipedia_info': {
                        'series_name': match.wikipedia_series.get('series'),
                        'authors': match.wikipedia_series.get('authors', []),
                        'original_language': match.wikipedia_series.get('original_language'),
                        'installments': match.wikipedia_series.get('installments'),
                        'years': match.wikipedia_series.get('years'),
                        'approximate_sales': match.wikipedia_series.get('approximate_sales'),
                        'approximate_sales_int': match.wikipedia_series.get('approximate_sales_int')
                    },
                    'orderofbooks_info': {
                        'series_name': match.orderofbooks_series.get('series_name'),
                        'author': match.orderofbooks_series.get('author'),
                        'books': match.orderofbooks_series.get('books', [])
                    },
                    'match_info': {
                        'confidence': match.confidence,
                        'match_type': match.match_type,
                        'notes': match.notes
                    }
                }
            else:
                # No match found - keep Wikipedia data only
                combined_series = {
                    'wikipedia_info': {
                        'series_name': match.wikipedia_series.get('series'),
                        'authors': match.wikipedia_series.get('authors', []),
                        'original_language': match.wikipedia_series.get('original_language'),
                        'installments': match.wikipedia_series.get('installments'),
                        'years': match.wikipedia_series.get('years'),
                        'approximate_sales': match.wikipedia_series.get('approximate_sales'),
                        'approximate_sales_int': match.wikipedia_series.get('approximate_sales_int')
                    },
                    'orderofbooks_info': None,
                    'match_info': {
                        'confidence': match.confidence,
                        'match_type': match.match_type,
                        'notes': match.notes
                    }
                }
            
            combined.append(combined_series)
            
        return combined
        
    def print_statistics(self):
        """Print matching statistics"""
        total = len(self.matches)
        matched = sum(1 for m in self.matches if m.orderofbooks_series is not None)
        
        match_types = {}
        for match in self.matches:
            match_types[match.match_type] = match_types.get(match.match_type, 0) + 1
            
        print(f"\\n=== MATCHING STATISTICS ===")
        print(f"Total Wikipedia series: {len(self.wikipedia_data)}")
        print(f"Total orderofbooks series: {len(self.orderofbooks_data)}")
        print(f"Successfully matched: {matched}/{total} ({matched/total*100:.1f}%)")
        print(f"\\nMatch types:")
        for match_type, count in match_types.items():
            print(f"  {match_type}: {count} ({count/total*100:.1f}%)")
            
        # Show some examples of high-confidence matches
        high_confidence = [m for m in self.matches if m.confidence > 0.8 and m.orderofbooks_series]
        if high_confidence:
            print(f"\\nHigh-confidence matches (confidence > 0.8):")
            for match in high_confidence[:10]:
                wiki_name = match.wikipedia_series.get('series', '')
                oob_name = match.orderofbooks_series.get('series_name', '')
                print(f"  '{wiki_name}' <-> '{oob_name}' (confidence: {match.confidence:.2f})")

def main():
    parser = argparse.ArgumentParser(description="Match Wikipedia book series with OrderOfBooks data")
    parser.add_argument('--openai-api-key', help='OpenAI API key for GPT assistance')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, 
                       help='Minimum confidence for automatic matching (default: 0.5)')
    parser.add_argument('--orderofbooks-file', default='orderofbooks_series.json',
                       help='Path to orderofbooks JSON file')
    parser.add_argument('--wikipedia-file', default='best_selling_book_series.json',
                       help='Path to Wikipedia JSON file')
    parser.add_argument('--output', default='matched_series.json',
                       help='Output file for matched series')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output to see matching details')
    
    args = parser.parse_args()
    
    # Get OpenAI API key from args or environment
    api_key = args.openai_api_key
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        print("Using OpenAI API key for GPT-5-nano assistance")
    else:
        print("No OpenAI API key found. Medium-confidence matches will not use GPT validation.")
    
    # Initialize matcher
    matcher = SeriesMatcher(openai_api_key=api_key)
    
    # Load data
    matcher.load_data(args.orderofbooks_file, args.wikipedia_file)
    
    if not matcher.orderofbooks_data:
        print("No orderofbooks data found. Please run scrape_orderofbooks_series.py first.")
        return
        
    # Match series
    matcher.match_series(confidence_threshold=args.confidence_threshold, verbose=args.verbose)
    
    # Create combined dataset
    combined_data = matcher.create_combined_dataset()
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    print(f"\\nSaved combined dataset to {args.output}")
    
    # Print statistics
    matcher.print_statistics()

if __name__ == "__main__":
    main()
