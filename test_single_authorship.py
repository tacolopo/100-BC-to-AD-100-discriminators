#!/usr/bin/env python3
"""
Test Single Authorship Analysis
===============================

This script demonstrates how the single authorship analyzer would work
using some of our existing texts to simulate the Pauline letters scenario.
"""

import os
import shutil
from pathlib import Path
from single_authorship_analyzer import SingleAuthorshipAnalyzer

def create_test_corpus():
    """Create test corpora to demonstrate the analysis."""
    
    # Create test directories
    test_dir = Path("test_corpora")
    test_dir.mkdir(exist_ok=True)
    
    # Test 1: Single author corpus (Dionysius of Halicarnassus - has 14 texts)
    single_author_dir = test_dir / "single_author_test"
    single_author_dir.mkdir(exist_ok=True)
    
    dionysius_dir = Path("Dionysius of Halicarnassus")
    if dionysius_dir.exists():
        # Copy some texts from Dionysius (single author)
        texts = list(dionysius_dir.glob("*.txt"))[:5]  # Take first 5 texts
        for i, text_file in enumerate(texts):
            dest = single_author_dir / f"text_{i+1:02d}.txt"
            shutil.copy2(text_file, dest)
        print(f"Created single author test corpus with {len(texts)} texts from Dionysius")
    
    # Test 2: Multiple author corpus (mix different authors)
    multi_author_dir = test_dir / "multiple_author_test"
    multi_author_dir.mkdir(exist_ok=True)
    
    # Collect texts from different authors
    author_dirs = ["Homer", "Strabo", "Chariton", "Cebes", "Longinus"]
    text_count = 0
    for author in author_dirs:
        author_path = Path(author)
        if author_path.exists():
            texts = list(author_path.glob("*.txt"))[:1]  # Take 1 text per author
            for text_file in texts:
                dest = multi_author_dir / f"{author}_{text_file.name}"
                shutil.copy2(text_file, dest)
                text_count += 1
    print(f"Created multiple author test corpus with {text_count} texts from different authors")
    
    return single_author_dir, multi_author_dir

def main():
    """Run test analysis."""
    print("=== Testing Single Authorship Analyzer ===")
    print()
    
    # Create test corpora
    single_dir, multi_dir = create_test_corpus()
    
    # Initialize analyzer
    analyzer = SingleAuthorshipAnalyzer()
    
    # Test 1: Single author corpus
    print("\n" + "="*60)
    print("TEST 1: EXPECTED SINGLE AUTHOR (Dionysius texts)")
    print("="*60)
    prob1, interp1 = analyzer.analyze_corpus(single_dir)
    
    # Test 2: Multiple author corpus  
    print("\n" + "="*60)
    print("TEST 2: EXPECTED MULTIPLE AUTHORS (Mixed authors)")
    print("="*60)
    prob2, interp2 = analyzer.analyze_corpus(multi_dir)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF TESTS")
    print("="*60)
    print(f"Single Author Test:    {prob1:.1%} - {interp1}")
    print(f"Multiple Author Test:  {prob2:.1%} - {interp2}")
    print()
    print("If working correctly:")
    print("- Single author test should show HIGH probability (>60%)")
    print("- Multiple author test should show LOW probability (<40%)")
    
    # Usage example for Pauline letters
    print("\n" + "="*60)
    print("TO ANALYZE PAULINE LETTERS:")
    print("="*60)
    print("1. Create a directory with the 14 Pauline letters as .txt files")
    print("2. Run: python single_authorship_analyzer.py /path/to/pauline_letters/")
    print("3. The output will show probability of single authorship")
    print()
    print("Expected filename structure:")
    print("  pauline_letters/")
    print("    ├── Romans.txt")
    print("    ├── 1_Corinthians.txt") 
    print("    ├── 2_Corinthians.txt")
    print("    ├── Galatians.txt")
    print("    ├── Ephesians.txt")
    print("    ├── Philippians.txt")
    print("    ├── Colossians.txt")
    print("    ├── 1_Thessalonians.txt")
    print("    ├── 2_Thessalonians.txt")
    print("    ├── 1_Timothy.txt")
    print("    ├── 2_Timothy.txt")
    print("    ├── Titus.txt")
    print("    ├── Philemon.txt")
    print("    └── Hebrews.txt")

if __name__ == "__main__":
    main()
