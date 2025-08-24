#!/usr/bin/env python3
"""
Comprehensive Series Matching Script
===================================

This script matches Wikipedia best-selling series with comprehensive OrderOfBooks data,
supporting both publication and chronological ordering where available.

The script processes the new comprehensive orderofbooks data structure from index.json
and individual series files, then matches them with Wikipedia series.

Usage:
    python match_series_comprehensive.py [--openai-api-key YOUR_KEY] [--verbose]
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
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
class SeriesOrdering:
    """Represents a single ordering type for a series"""
    order_type: str  # 'publication', 'chronological', 'other'
    series_data: dict
    books: List[dict]

@dataclass
class ComprehensiveSeriesMatch:
    """Enhanced match that can contain multiple orderings"""
    wikipedia_series: dict
    orderofbooks_matches: List[SeriesOrdering]  # Can have multiple orderings
    best_confidence: float
    match_type: str  # 'exact', 'fuzzy', 'gpt_confirmed', 'gpt_rejected', 'no_match'
    notes: str = ""

class ComprehensiveSeriesMatcher:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_client = None
        if openai_api_key and HAS_OPENAI:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        self.orderofbooks_series = []  # Processed series from data/series/
        self.wikipedia_data = []
        self.matches = []
        
    def load_orderofbooks_data(self, index_path: str = "index.json", data_dir: str = "data/series"):
        """Load comprehensive orderofbooks data from index and individual files"""
        index_file = Path(index_path)
        data_path = Path(data_dir)
        
        if not index_file.exists():
            print(f"Warning: {index_path} not found. Please run the comprehensive scraper first.")
            return
            
        if not data_path.exists():
            print(f"Warning: {data_dir} not found. Please run the comprehensive scraper first.")
            return
            
        # Load index to get series list
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        series_list = index_data.get('series', [])
        print(f"Found {len(series_list)} series in index")
        
        # Load individual series files
        loaded_count = 0
        for series_ref in series_list:
            series_slug = series_ref.get('slug')
            if not series_slug:
                continue
                
            series_file = data_path / f"{series_slug}.json"
            if series_file.exists():
                try:
                    with open(series_file, 'r', encoding='utf-8') as f:
                        series_data = json.load(f)
                    self.orderofbooks_series.append(series_data)
                    loaded_count += 1
                except Exception as e:
                    print(f"Error loading {series_file}: {e}")
                    
        print(f"Successfully loaded {loaded_count} series files from orderofbooks")
        
    def load_wikipedia_data(self, wikipedia_path: str = "best_selling_book_series.json"):
        """Load Wikipedia bestselling series data"""
        if not Path(wikipedia_path).exists():
            print(f"Warning: {wikipedia_path} not found")
            return
            
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
        
    def normalize_series_name(self, name: str) -> str:
        """Normalize series name for comparison - enhanced version"""
        if not name:
            return ""
            
        # Convert to lowercase
        name = name.lower()
        
        # Handle specific known mappings first
        series_mappings = {
            'harry potter': 'harry potter',
            'jack reacher': 'jack reacher', 
            'alex cross': 'alex cross',
            'dark tower': 'dark tower',
            'doc savage': 'doc savage',
            'james bond': 'james bond',
            'nancy drew': 'nancy drew',
            'hardy boys': 'hardy boys',
            'goosebumps': 'goosebumps',
            'percy jackson': 'percy jackson',
            'diary of a wimpy kid': 'diary wimpy kid',
            'diary wimpy kid': 'diary wimpy kid',
            'magic tree house': 'magic tree house',
            'captain underpants': 'captain underpants',
            'artemis fowl': 'artemis fowl',
            'inheritance cycle': 'inheritance cycle',
            'eragon': 'inheritance cycle',  # Alternative name
            'hunger games': 'hunger games',
            'twilight': 'twilight',
            'chronicles of narnia': 'chronicles narnia',
            'narnia': 'chronicles narnia',
            'fifty shades': 'fifty shades',
            'game of thrones': 'song ice fire',
            'song of ice and fire': 'song ice fire',
            'wheel of time': 'wheel time',
            'discworld': 'discworld',
            'left behind': 'left behind',
            'little house on the prairie': 'little house prairie',
            'winnie the pooh': 'winnie pooh',
            'winnie-the-pooh': 'winnie pooh',
            'his dark materials': 'his dark materials',
            'hitchhiker\'s guide to the galaxy': 'hitchhiker guide galaxy',
            'foundation': 'foundation',
            'dune': 'dune',
            'outlander': 'outlander',
            'bridget jones': 'bridget jones',
            'redwall': 'redwall',
            'animorphs': 'animorphs',
            'warriors': 'warriors',
            'shadowhunter chronicles': 'shadowhunter',
            'divergent': 'divergent',
            'vampire chronicles': 'vampire chronicles',
            'millennium': 'millennium',
            'shannara': 'shannara',
            'sword of truth': 'sword truth',
            'dragonlance': 'dragonlance',
            'dragonriders of pern': 'dragonriders pern',
            'cosmere': 'cosmere',
            'witcher': 'witcher',
            'thrawn': 'thrawn',
            'riftwar': 'riftwar',
            'dirk pitt': 'dirk pitt',
            'robert langdon': 'robert langdon',
            'horrible histories': 'horrible histories',
            'rainbow magic': 'rainbow magic',
            'southern vampire mysteries': 'southern vampire',
            'no. 1 ladies\' detective agency': 'no 1 ladies detective',
            'berenstain bears': 'berenstain bears',
            'curious george': 'curious george',
            'paddington': 'paddington',
            'clifford the big red dog': 'clifford big red dog',
            'noddy': 'noddy',
            'mr. men': 'mr men',
            'little critter': 'little critter',
            'peter rabbit': 'peter rabbit',
            'railway series': 'railway series',
            'thomas the tank engine': 'railway series',
            'american girl': 'american girl',
            'baby-sitters club': 'babysitters club',
            'babysitters club': 'babysitters club',
            'geronimo stilton': 'geronimo stilton',
            'choose your own adventure': 'choose your own adventure',
            'sweet valley high': 'sweet valley',
            'fear street': 'fear street',
            'magic school bus': 'magic school bus',
            'series of unfortunate events': 'series unfortunate events',
            'where\'s wally': 'where wally',
            'where\'s waldo': 'where wally',
            'all creatures great and small': 'all creatures great small',
            'men are from mars, women are from venus': 'men mars women venus',
            'chicken soup for the soul': 'chicken soup soul',
            'frank merriwell': 'frank merriwell',
            '39 clues': '39 clues',
            'maze runner': 'maze runner',
            'miss marple': 'miss marple',
            'hercule poirot': 'hercule poirot',
        }
        
        # Check for direct mappings first
        for key, value in series_mappings.items():
            if key in name:
                return value
                
        # If no direct mapping, apply general normalization
        # Remove common prefixes/suffixes
        name = re.sub(r'^(publication|chronological)\s+order\s+of\s+(the\s+)?', '', name)
        name = re.sub(r'^bookshots:\s*', '', name)  # Remove bookshots prefix
        
        # Remove common suffixes
        name = re.sub(r'\s+(books?|series|novels?|stories?|collection)(\s+books?)?\s*$', '', name)
        name = re.sub(r'\s+(short\s+stories?|novellas?|collections?|miscellaneous)\s*$', '', name)
        name = re.sub(r'\s+(graphic\s+novels?)\s*$', '', name)
        name = re.sub(r'\s+(mystery|mysteries)\s*$', '', name)
        name = re.sub(r'\s+order\s*$', '', name)
        
        # Remove articles and common words
        name = re.sub(r'^(the|a|an)\s+', '', name)
        name = re.sub(r'\s+(the|a|an)\s+', ' ', name)
        
        # Clean up extra whitespace and punctuation
        name = re.sub(r'[^\w\s]', ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
        
    def group_series_by_base_name(self) -> Dict[str, List[dict]]:
        """Group orderofbooks series by their base name to identify different orderings"""
        grouped = {}
        
        for series in self.orderofbooks_series:
            name = series.get('name', '')
            normalized = self.normalize_series_name(name)
            
            if normalized not in grouped:
                grouped[normalized] = []
            grouped[normalized].append(series)
            
        return grouped
        
    def determine_ordering_type(self, series_name: str) -> str:
        """Determine the ordering type from series name
        
        For orderofbooks.com, the default order is typically publication order
        unless explicitly stated otherwise in the series name.
        """
        name_lower = series_name.lower()
        if 'chronological' in name_lower or 'chronology' in name_lower:
            return 'chronological'
        elif 'publication order' in name_lower:
            return 'publication'
        else:
            # Default assumption: orderofbooks.com uses publication order
            return 'publication'
            
    def find_author_matches(self, oob_series: dict, wikipedia_series: dict) -> float:
        """Calculate author similarity between series"""
        oob_authors = oob_series.get('authors', [])
        wiki_authors = wikipedia_series.get('authors', [])
        
        if not wiki_authors or not oob_authors:
            return 0.0
            
        # Get author names from orderofbooks format
        oob_author_names = []
        for author in oob_authors:
            if isinstance(author, dict):
                oob_author_names.append(author.get('name', '').lower())
            elif isinstance(author, str):
                oob_author_names.append(author.lower())
                
        # Compare with Wikipedia authors
        best_match = 0.0
        for wiki_author in wiki_authors:
            if isinstance(wiki_author, str):
                wiki_author_lower = wiki_author.lower()
                for oob_author in oob_author_names:
                    if oob_author and wiki_author_lower:
                        similarity = fuzz.ratio(oob_author, wiki_author_lower) / 100.0
                        best_match = max(best_match, similarity)
                        
                        # Check for last name matches
                        oob_parts = oob_author.split()
                        wiki_parts = wiki_author_lower.split()
                        if oob_parts and wiki_parts:
                            if oob_parts[-1] == wiki_parts[-1]:  # Same last name
                                best_match = max(best_match, 0.8)
                                
        return best_match
        
    def ask_gpt_for_match_confirmation(self, oob_series: dict, wiki_series: dict, confidence: float) -> Tuple[bool, str]:
        """Use GPT-5-nano to confirm if two series are the same"""
        if not self.openai_client:
            return False, "OpenAI not available"
            
        oob_name = oob_series.get('name', '')
        oob_authors = oob_series.get('authors', [])
        oob_author_names = [a.get('name', '') if isinstance(a, dict) else str(a) for a in oob_authors]
        oob_books = [book.get('title', '') for book in oob_series.get('books', [])[:5]]
        
        wiki_name = wiki_series.get('series', '')
        wiki_authors = wiki_series.get('authors', [])
        
        prompt = f"""Are these two book series the same series?

Series 1 (from orderofbooks.com):
- Name: {oob_name}
- Authors: {', '.join(oob_author_names)}
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
        """Match Wikipedia series with comprehensive orderofbooks data"""
        print(f"\\nMatching {len(self.wikipedia_data)} Wikipedia series with {len(self.orderofbooks_series)} orderofbooks series")
        print(f"Confidence threshold: {confidence_threshold}")
        
        # Group orderofbooks series by normalized name
        grouped_series = self.group_series_by_base_name()
        print(f"Grouped into {len(grouped_series)} unique series names")
        
        # Create search index
        search_choices = []
        for normalized_name, series_list in grouped_series.items():
            if normalized_name:  # Skip empty names
                search_choices.append((normalized_name, series_list))
        
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
                self.matches.append(ComprehensiveSeriesMatch(
                    wikipedia_series=wiki_series,
                    orderofbooks_matches=[],
                    best_confidence=0.0,
                    match_type='no_match',
                    notes='Empty Wikipedia series name'
                ))
                continue
            
            # Find best match using rapidfuzz
            best_matches = process.extract(
                normalized_wiki, 
                [choice[0] for choice in search_choices],
                scorer=fuzz.token_sort_ratio,
                limit=5
            )
            
            if verbose:
                print(f"Top 5 fuzzy matches:")
                for match_name, score, _ in best_matches:
                    print(f"  {score:.1f}% - '{match_name}'")
            
            if not best_matches:
                if verbose:
                    print("❌ NO MATCHES: No similar series found")
                self.matches.append(ComprehensiveSeriesMatch(
                    wikipedia_series=wiki_series,
                    orderofbooks_matches=[],
                    best_confidence=0.0,
                    match_type='no_match',
                    notes='No similar series found in orderofbooks'
                ))
                continue
                
            best_match_name, best_score, _ = best_matches[0]
            confidence = best_score / 100.0
            
            # Get all orderings for this series
            matched_series_list = next(series_list for norm_name, series_list in search_choices if norm_name == best_match_name)
            
            # Calculate author similarity and create orderings
            orderings = []
            best_author_similarity = 0.0
            
            for series_data in matched_series_list:
                author_similarity = self.find_author_matches(series_data, wiki_series)
                best_author_similarity = max(best_author_similarity, author_similarity)
                
                order_type = self.determine_ordering_type(series_data.get('name', ''))
                ordering = SeriesOrdering(
                    order_type=order_type,
                    series_data=series_data,
                    books=series_data.get('books', [])
                )
                orderings.append(ordering)
            
            if verbose:
                print(f"Best match: {best_score:.1f}% confidence")
                print(f"Author similarity: {best_author_similarity:.2f}")
                print(f"Found {len(orderings)} orderings:")
                for ordering in orderings:
                    print(f"  - {ordering.order_type}: {len(ordering.books)} books")
            
            # Adjust confidence based on author match
            original_confidence = confidence
            if best_author_similarity > 0.8:
                confidence = min(1.0, confidence + 0.15)
            elif best_author_similarity > 0.6:
                confidence = min(1.0, confidence + 0.1)
            elif best_author_similarity < 0.2 and confidence > 0.7:
                confidence -= 0.3
            elif best_author_similarity < 0.4 and confidence > 0.6:
                confidence -= 0.15
                
            if verbose and confidence != original_confidence:
                print(f"Adjusted confidence: {original_confidence:.2f} -> {confidence:.2f}")
                
            # Determine match type and create match
            if confidence >= confidence_threshold:
                match_type = 'exact' if confidence >= 0.95 else 'fuzzy'
                if verbose:
                    print(f"✅ MATCH ACCEPTED: High confidence ({confidence:.2f}) - {match_type}")
                    
                self.matches.append(ComprehensiveSeriesMatch(
                    wikipedia_series=wiki_series,
                    orderofbooks_matches=orderings,
                    best_confidence=confidence,
                    match_type=match_type,
                    notes=f'Author similarity: {best_author_similarity:.2f}, {len(orderings)} orderings'
                ))
                
            elif confidence >= 0.4 and self.openai_client:
                # Use the first/best series for GPT validation
                primary_series = orderings[0].series_data if orderings else {}
                
                if verbose:
                    print(f"🤔 ASKING GPT-5-nano: Medium confidence ({confidence:.2f})")
                    
                is_match, gpt_response = self.ask_gpt_for_match_confirmation(primary_series, wiki_series, confidence)
                
                if is_match:
                    if verbose:
                        print(f"✅ GPT CONFIRMED: {gpt_response}")
                    self.matches.append(ComprehensiveSeriesMatch(
                        wikipedia_series=wiki_series,
                        orderofbooks_matches=orderings,
                        best_confidence=confidence,
                        match_type='gpt_confirmed',
                        notes=f'GPT: {gpt_response}'
                    ))
                else:
                    if verbose:
                        print(f"❌ GPT REJECTED: {gpt_response}")
                    self.matches.append(ComprehensiveSeriesMatch(
                        wikipedia_series=wiki_series,
                        orderofbooks_matches=[],
                        best_confidence=confidence,
                        match_type='gpt_rejected',
                        notes=f'GPT: {gpt_response}'
                    ))
                    
            else:
                if verbose:
                    print(f"❌ NO MATCH: Low confidence ({confidence:.2f})")
                self.matches.append(ComprehensiveSeriesMatch(
                    wikipedia_series=wiki_series,
                    orderofbooks_matches=[],
                    best_confidence=confidence,
                    match_type='no_match',
                    notes=f'Low confidence: {confidence:.2f}'
                ))
                
    def create_comprehensive_dataset(self) -> List[dict]:
        """Create comprehensive dataset with multiple orderings"""
        combined = []
        
        for match in self.matches:
            if match.orderofbooks_matches:
                # Found matches - include all orderings
                orderings_data = {}
                for ordering in match.orderofbooks_matches:
                    orderings_data[ordering.order_type] = {
                        'series_name': ordering.series_data.get('name'),
                        'authors': ordering.series_data.get('authors', []),
                        'books': ordering.books,
                        'image': ordering.series_data.get('image'),
                        'source': ordering.series_data.get('source')
                    }
                
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
                        'orderings': orderings_data,
                        'available_orderings': list(orderings_data.keys())
                    },
                    'match_info': {
                        'confidence': match.best_confidence,
                        'match_type': match.match_type,
                        'notes': match.notes
                    }
                }
            else:
                # No match found
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
                        'confidence': match.best_confidence,
                        'match_type': match.match_type,
                        'notes': match.notes
                    }
                }
            
            combined.append(combined_series)
            
        return combined
        
    def print_statistics(self):
        """Print comprehensive matching statistics"""
        total = len(self.matches)
        matched = sum(1 for m in self.matches if m.orderofbooks_matches)
        
        match_types = {}
        ordering_counts = {}
        
        for match in self.matches:
            match_types[match.match_type] = match_types.get(match.match_type, 0) + 1
            
            if match.orderofbooks_matches:
                num_orderings = len(match.orderofbooks_matches)
                ordering_counts[num_orderings] = ordering_counts.get(num_orderings, 0) + 1
                
        print(f"\\n=== COMPREHENSIVE MATCHING STATISTICS ===")
        print(f"Total Wikipedia series: {len(self.wikipedia_data)}")
        print(f"Total orderofbooks series: {len(self.orderofbooks_series)}")
        print(f"Successfully matched: {matched}/{total} ({matched/total*100:.1f}%)")
        
        print(f"\\nMatch types:")
        for match_type, count in match_types.items():
            print(f"  {match_type}: {count} ({count/total*100:.1f}%)")
            
        print(f"\\nOrdering availability:")
        for num_orderings, count in ordering_counts.items():
            print(f"  {num_orderings} ordering(s): {count} series")
            
        # Show examples of matches with multiple orderings
        multi_ordering = [m for m in self.matches if len(m.orderofbooks_matches) > 1]
        if multi_ordering:
            print(f"\\nSeries with multiple orderings ({len(multi_ordering)}):")
            for match in multi_ordering[:5]:
                wiki_name = match.wikipedia_series.get('series', '')
                orderings = [o.order_type for o in match.orderofbooks_matches]
                print(f"  '{wiki_name}': {orderings}")

def main():
    parser = argparse.ArgumentParser(description="Match Wikipedia series with comprehensive OrderOfBooks data")
    parser.add_argument('--openai-api-key', help='OpenAI API key for GPT assistance')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, 
                       help='Minimum confidence for automatic matching (default: 0.5)')
    parser.add_argument('--index-file', default='index.json',
                       help='Path to orderofbooks index JSON file')
    parser.add_argument('--data-dir', default='data/series',
                       help='Path to orderofbooks series data directory')
    parser.add_argument('--wikipedia-file', default='best_selling_book_series.json',
                       help='Path to Wikipedia JSON file')
    parser.add_argument('--output', default='matched_series_comprehensive.json',
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
    matcher = ComprehensiveSeriesMatcher(openai_api_key=api_key)
    
    # Load data
    matcher.load_orderofbooks_data(args.index_file, args.data_dir)
    matcher.load_wikipedia_data(args.wikipedia_file)
    
    if not matcher.orderofbooks_series:
        print("No orderofbooks data found. Please run the comprehensive scraper first:")
        print("python scrape_orderofbooks.py --only-series --limit-series 1000")
        return
        
    # Match series
    matcher.match_series(confidence_threshold=args.confidence_threshold, verbose=args.verbose)
    
    # Create comprehensive dataset
    combined_data = matcher.create_comprehensive_dataset()
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    print(f"\\nSaved comprehensive dataset to {args.output}")
    
    # Print statistics
    matcher.print_statistics()

if __name__ == "__main__":
    main()
