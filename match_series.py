#!/usr/bin/env python3
"""
Series Matching Script
=====================

This script matches book series from Wikipedia with series from orderofbooks.com
and creates a combined dataset with complete book information.

Usage:
    python match_series.py [--openai-api-key YOUR_KEY]

Requirements:
    pip install rapidfuzz openai
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
    orderofbooks_series: dict
    wikipedia_series: Optional[dict]
    confidence: float
    match_type: str  # 'exact', 'fuzzy', 'manual', 'no_match'
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
        
        # Remove common prefixes/suffixes more aggressively
        name = re.sub(r'^(publication|chronological)\s+order\s+of\s+', '', name)
        name = re.sub(r'\s+(books?|series|novels?|stories?|collection)\s*$', '', name)
        name = re.sub(r'\s+(short\s+stories?|novellas?|collections?|miscellaneous)\s*$', '', name)
        name = re.sub(r'\s+order\s*$', '', name)
        
        # Remove articles and common words
        name = re.sub(r'^(the|a|an)\s+', '', name)
        name = re.sub(r'\s+(the|a|an)\s+', ' ', name)
        
        # Remove extra whitespace and punctuation
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
        
    def calculate_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two series names"""
        norm1 = self.normalize_series_name(name1)
        norm2 = self.normalize_series_name(name2)
        
        # Try different similarity metrics
        ratio = fuzz.ratio(norm1, norm2)
        token_sort = fuzz.token_sort_ratio(norm1, norm2)
        token_set = fuzz.token_set_ratio(norm1, norm2)
        partial = fuzz.partial_ratio(norm1, norm2)
        
        # Return the highest score
        return max(ratio, token_sort, token_set, partial) / 100.0
        
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
                # Direct comparison
                similarity = fuzz.ratio(oob_author, wiki_author.lower()) / 100.0
                best_author_match = max(best_author_match, similarity)
                
                # Check if last names match (for pen names)
                oob_parts = oob_author.split()
                wiki_parts = wiki_author.lower().split()
                if oob_parts and wiki_parts:
                    if oob_parts[-1] == wiki_parts[-1]:  # Same last name
                        best_author_match = max(best_author_match, 0.8)
                    
                # Check partial matches for compound names
                for oob_part in oob_parts:
                    if len(oob_part) > 2:  # Avoid matching initials
                        for wiki_part in wiki_parts:
                            if oob_part in wiki_part or wiki_part in oob_part:
                                best_author_match = max(best_author_match, 0.6)
                
        return best_author_match
        
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
            
    def match_series(self, confidence_threshold: float = 0.8, verbose: bool = False):
        """Match Wikipedia series with orderofbooks data"""
        print(f"\\nMatching Wikipedia series with orderofbooks data, confidence threshold: {confidence_threshold}")
        
        # Create a list of normalized orderofbooks series for faster searching
        oob_choices = []
        for i, oob_series in enumerate(self.orderofbooks_data):
            oob_name = oob_series.get('series_name', '')
            normalized = self.normalize_series_name(oob_name)
            oob_choices.append((normalized, i, oob_name))
        
        if verbose:
            print(f"\\nOrderofbooks series available for matching:")
            for norm, i, orig in oob_choices[:20]:  # Show first 20
                print(f"  '{orig}' -> '{norm}'")
            if len(oob_choices) > 20:
                print(f"  ... and {len(oob_choices) - 20} more")
        
        for i, wiki_series in enumerate(self.wikipedia_data):
            if i % 10 == 0:
                print(f"Processing Wikipedia series {i+1}/{len(self.wikipedia_data)}")
                
            wiki_name = wiki_series.get('series', '')
            wiki_authors = wiki_series.get('authors', [])
            normalized_wiki = self.normalize_series_name(wiki_name)
            
            if verbose:  # Show all series processing
                print(f"\\n--- Processing #{i+1}: '{wiki_name}' by {wiki_authors} ---")
                print(f"Normalized: '{normalized_wiki}'")
            
            if not normalized_wiki:
                if verbose:
                    print("SKIP: Empty normalized series name")
                self.matches.append(SeriesMatch(
                    orderofbooks_series=None,
                    wikipedia_series=wiki_series,
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
                limit=5  # Show top 5 candidates
            )
            
            if verbose:
                print(f"Top 5 fuzzy matches:")
                for match_name, score, _ in best_matches:
                    # Find original name
                    orig_name = next(orig for norm, _, orig in oob_choices if norm == match_name)
                    print(f"  {score:.1f}% - '{orig_name}' (normalized: '{match_name}')")
            
            if not best_matches:
                if verbose:
                    print("NO MATCHES: No similar series found")
                self.matches.append(SeriesMatch(
                    orderofbooks_series=None,
                    wikipedia_series=wiki_series,
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
            
            # Also check author similarity
            author_similarity = self.find_author_matches(oob_series, wiki_series)
            
            if verbose:
                print(f"Best match: {best_score}% confidence")
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
                confidence -= 0.3  # Strong penalty for author mismatch on high-confidence matches
            elif author_similarity < 0.4 and confidence > 0.6:
                confidence -= 0.15  # Medium penalty
                
            if verbose:
                print(f"Adjusted confidence: {original_confidence:.2f} -> {confidence:.2f}")
                
            if confidence >= confidence_threshold:
                # High confidence match
                if verbose:
                    print(f"✅ MATCH ACCEPTED: High confidence ({confidence:.2f})")
                self.matches.append(SeriesMatch(
                    orderofbooks_series=oob_series,
                    wikipedia_series=wiki_series,
                    confidence=confidence,
                    match_type='fuzzy' if confidence < 0.95 else 'exact',
                    notes=f'Author similarity: {author_similarity:.2f}'
                ))
            elif confidence >= 0.5 and self.openai_client:
                # Medium confidence - ask GPT
                if verbose:
                    print(f"🤔 ASKING GPT-5-nano: Medium confidence ({confidence:.2f})")
                is_match, gpt_response = self.ask_gpt_for_match_confirmation(oob_series, wiki_series, confidence)
                if is_match:
                    if verbose:
                        print(f"✅ GPT CONFIRMED: {gpt_response}")
                    self.matches.append(SeriesMatch(
                        orderofbooks_series=oob_series,
                        wikipedia_series=wiki_series,
                        confidence=confidence,
                        match_type='gpt_confirmed',
                        notes=f'GPT: {gpt_response}'
                    ))
                else:
                    if verbose:
                        print(f"❌ GPT REJECTED: {gpt_response}")
                    self.matches.append(SeriesMatch(
                        orderofbooks_series=None,
                        wikipedia_series=wiki_series,
                        confidence=confidence,
                        match_type='gpt_rejected',
                        notes=f'GPT: {gpt_response}'
                    ))
            else:
                # Low confidence - no match
                if verbose:
                    print(f"❌ NO MATCH: Low confidence ({confidence:.2f})")
                self.matches.append(SeriesMatch(
                    orderofbooks_series=None,
                    wikipedia_series=wiki_series,
                    confidence=confidence,
                    match_type='no_match',
                    notes=f'Low confidence: {confidence:.2f}'
                ))
            
            if verbose and i < 10:
                print(f"Top 5 fuzzy matches:")
                for match_name, score, _ in best_matches:
                    # Find original name
                    orig_name = next(orig for norm, idx, orig in wiki_choices if norm == match_name)
                    print(f"  {score:3d}% '{orig_name}' ('{match_name}')")
            
            if not best_matches:
                if verbose and i < 10:
                    print("RESULT: No similar series found")
                self.matches.append(SeriesMatch(
                    orderofbooks_series=oob_series,
                    wikipedia_series=None,
                    confidence=0.0,
                    match_type='no_match',
                    notes='No similar series found'
                ))
                continue
                
            best_match, best_score, _ = best_matches[0]
            confidence = best_score / 100.0
            
            # Find the corresponding Wikipedia series
            wiki_index = next(i for norm, i, _ in wiki_choices if norm == best_match)
            wiki_series = self.wikipedia_data[wiki_index]
            
            # Also check author similarity
            author_similarity = self.find_author_matches(oob_series, wiki_series)
            
            if verbose and i < 10:
                print(f"Best match: '{wiki_series.get('series', '')}' (confidence: {confidence:.3f})")
                print(f"Author similarity: {author_similarity:.3f} (OOB: '{oob_author}' vs Wiki: {wiki_series.get('authors', [])})")
            
            # Adjust confidence based on author match
            original_confidence = confidence
            if author_similarity > 0.8:
                confidence = min(1.0, confidence + 0.15)  # Strong boost for good author match
            elif author_similarity > 0.6:
                confidence = min(1.0, confidence + 0.1)   # Medium boost
            elif author_similarity < 0.2 and confidence > 0.7:
                confidence -= 0.3  # Strong penalty for author mismatch on high-confidence matches
            elif author_similarity < 0.4 and confidence > 0.6:
                confidence -= 0.15  # Medium penalty
                
            if verbose and i < 10:
                if confidence != original_confidence:
                    print(f"Adjusted confidence: {original_confidence:.3f} -> {confidence:.3f}")
                
            if confidence >= confidence_threshold:
                # High confidence match
                match_type = 'fuzzy' if confidence < 0.95 else 'exact'
                if verbose and i < 10:
                    print(f"RESULT: HIGH CONFIDENCE MATCH ({match_type})")
                self.matches.append(SeriesMatch(
                    orderofbooks_series=oob_series,
                    wikipedia_series=wiki_series,
                    confidence=confidence,
                    match_type=match_type,
                    notes=f'Author similarity: {author_similarity:.2f}'
                ))
            elif confidence >= 0.5 and self.openai_client:
                # Medium confidence - ask GPT
                if verbose and i < 10:
                    print(f"CHECKING with GPT-5-nano (confidence: {confidence:.3f})")
                is_match, gpt_response = self.ask_gpt_for_match_confirmation(oob_series, wiki_series, confidence)
                if is_match:
                    if verbose and i < 10:
                        print(f"RESULT: GPT CONFIRMED MATCH")
                    self.matches.append(SeriesMatch(
                        orderofbooks_series=oob_series,
                        wikipedia_series=wiki_series,
                        confidence=confidence,
                        match_type='gpt_confirmed',
                        notes=f'GPT: {gpt_response}'
                    ))
                else:
                    if verbose and i < 10:
                        print(f"RESULT: GPT REJECTED - {gpt_response[:50]}...")
                    self.matches.append(SeriesMatch(
                        orderofbooks_series=oob_series,
                        wikipedia_series=None,
                        confidence=confidence,
                        match_type='gpt_rejected',
                        notes=f'GPT: {gpt_response}'
                    ))
            else:
                # Low confidence - no match
                if verbose and i < 10:
                    print(f"RESULT: LOW CONFIDENCE NO MATCH ({confidence:.3f})")
                self.matches.append(SeriesMatch(
                    orderofbooks_series=oob_series,
                    wikipedia_series=None,
                    confidence=confidence,
                    match_type='no_match',
                    notes=f'Low confidence: {confidence:.2f}'
                ))
                
    def create_combined_dataset(self) -> List[dict]:
        """Create combined dataset with matched series"""
        combined = []
        
        for match in self.matches:
            if match.wikipedia_series:
                # Found a match - combine the data
                combined_series = {
                    'series_name': match.orderofbooks_series.get('series_name'),
                    'author': match.orderofbooks_series.get('author'),
                    'books': match.orderofbooks_series.get('books', []),
                    'wikipedia_info': {
                        'series_name': match.wikipedia_series.get('series'),
                        'authors': match.wikipedia_series.get('authors', []),
                        'original_language': match.wikipedia_series.get('original_language'),
                        'installments': match.wikipedia_series.get('installments'),
                        'years': match.wikipedia_series.get('years'),
                        'approximate_sales': match.wikipedia_series.get('approximate_sales'),
                        'approximate_sales_int': match.wikipedia_series.get('approximate_sales_int')
                    },
                    'match_info': {
                        'confidence': match.confidence,
                        'match_type': match.match_type,
                        'notes': match.notes
                    }
                }
            else:
                # No match found - keep orderofbooks data only
                combined_series = {
                    'series_name': match.orderofbooks_series.get('series_name'),
                    'author': match.orderofbooks_series.get('author'),
                    'books': match.orderofbooks_series.get('books', []),
                    'wikipedia_info': None,
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
        print(f"Successfully matched: {matched} ({matched/total*100:.1f}%)")
        print(f"\\nMatch types:")
        for match_type, count in match_types.items():
            print(f"  {match_type}: {count} ({count/total*100:.1f}%)")
            
        # Show some examples of high-confidence matches
        high_confidence = [m for m in self.matches if m.confidence > 0.9 and m.orderofbooks_series]
        if high_confidence:
            print(f"\\nHigh-confidence matches (confidence > 0.9):")
            for match in high_confidence[:5]:
                wiki_name = match.wikipedia_series.get('series', '')
                oob_name = match.orderofbooks_series.get('series_name', '')
                print(f"  '{wiki_name}' <-> '{oob_name}' (confidence: {match.confidence:.2f})")

def main():
    parser = argparse.ArgumentParser(description="Match book series from Wikipedia and OrderOfBooks")
    parser.add_argument('--openai-api-key', help='OpenAI API key for GPT assistance')
    parser.add_argument('--confidence-threshold', type=float, default=0.8, 
                       help='Minimum confidence for automatic matching (default: 0.8)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output to debug matching process')
    parser.add_argument('--orderofbooks-file', default='orderofbooks_series.json',
                       help='Path to orderofbooks JSON file')
    parser.add_argument('--wikipedia-file', default='best_selling_book_series.json',
                       help='Path to Wikipedia JSON file')
    parser.add_argument('--output', default='matched_series.json',
                       help='Output file for matched series')
    
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
