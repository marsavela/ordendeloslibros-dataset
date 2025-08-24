#!/usr/bin/env python3
"""
Quick validation script to verify the final dataset integrity
"""

import json
from pathlib import Path

def validate_dataset():
    """Validate the final matched dataset"""
    
    # Check if all required files exist
    required_files = [
        'matched_series_final.json',
        'best_selling_book_series.json', 
        'index.json',
        'data/series'
    ]
    
    print("🔍 Validating dataset integrity...")
    
    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ Missing: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")
    
    # Load and validate final results
    try:
        with open('matched_series_final.json', 'r') as f:
            results = json.load(f)
        
        total_series = len(results)
        matched_series = sum(1 for r in results if r.get('orderofbooks_info') is not None)
        match_rate = (matched_series / total_series) * 100
        
        # Count multiple orderings
        multiple_orderings = []
        for result in results:
            oob_info = result.get('orderofbooks_info', {})
            if oob_info and 'orderings' in oob_info:
                orderings = oob_info['orderings']
                if len(orderings) > 1:
                    wiki_name = result.get('wikipedia_info', {}).get('series_name', 'Unknown')
                    multiple_orderings.append((wiki_name, list(orderings.keys())))
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total Wikipedia series: {total_series}")
        print(f"   Successfully matched: {matched_series}")
        print(f"   Match rate: {match_rate:.1f}%")
        print(f"   Series with multiple orderings: {len(multiple_orderings)}")
        
        if multiple_orderings:
            print(f"\n📚 Multiple Orderings Found:")
            for series_name, orderings in multiple_orderings:
                print(f"   • {series_name}: {orderings}")
        
        # Validate series files count
        series_dir = Path('data/series')
        series_files = list(series_dir.glob('*.json'))
        print(f"\n📁 Series Files: {len(series_files)} JSON files")
        
        print(f"\n🎉 Dataset validation successful!")
        print(f"   Ready for use with {match_rate:.1f}% match rate")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == '__main__':
    validate_dataset()
