#!/usr/bin/env python3
"""
Paul vs Single Authors Analysis
===============================

This script compares Paul's internal linguistic variation to the internal variation
of each individual ancient Greek author who has multiple texts. This tells us whether
Paul's diversity is normal for a single author or abnormally high.

Method:
1. For each of the 60 discriminative features:
   - Calculate Paul's variance across his 14 letters
   - Calculate each single author's variance across their multiple texts
   - Compare Paul's variance to each author's variance
2. Generate probability scores showing how Paul compares to each individual author
3. Determine if Paul's variation is within normal single-author ranges

This is more sophisticated than comparing to aggregate inter-author variance.
"""

import os
import json
import re
import statistics
import numpy as np
from pathlib import Path
from collections import Counter
import pickle

class PaulVsSingleAuthorsAnalyzer:
    def __init__(self):
        self.load_reference_data()
        self.find_multi_text_authors()
        
    def load_reference_data(self):
        """Load discriminative features from the ancient Greek analysis."""
        print("Loading discriminative features...")
        
        with open('../results/best_discriminative_features.json', 'r', encoding='utf-8') as f:
            reference_features = json.load(f)
        
        # Use all perfect discriminators
        self.discriminative_features = []
        for item in reference_features:
            if item['separation_score'] > 0:
                self.discriminative_features.append(item)
        
        print(f"Using {len(self.discriminative_features)} discriminative features")
    
    def find_multi_text_authors(self):
        """Find authors with multiple text files by checking directories."""
        print("Finding authors with multiple texts...")
        
        self.multi_text_authors = {}
        
        # Check each directory for multiple .txt files
        for item in Path('..').iterdir():
            if (item.is_dir() and 
                not item.name.startswith('.') and 
                not item.name.startswith('__') and 
                item.name not in ['results', 'results documentation', 'test_corpora', 
                                'Paul versus all authors', 'Paul versus single authors']):
                
                txt_files = list(item.glob('*.txt'))
                if len(txt_files) > 1:
                    # Calculate total words
                    total_words = 0
                    for txt_file in txt_files:
                        try:
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                total_words += len(content.split())
                        except:
                            pass
                    
                    if total_words >= 1000:  # Only include authors with sufficient text
                        self.multi_text_authors[item.name] = {
                            'num_texts': len(txt_files),
                            'total_words': total_words,
                            'text_files': [f.name for f in txt_files]
                        }
        
        print(f"Found {len(self.multi_text_authors)} authors with multiple texts for comparison:")
        for author, info in self.multi_text_authors.items():
            print(f"  - {author}: {info['num_texts']} texts, {info['total_words']} words")
    
    def clean_text(self, text):
        """Clean and normalize Greek text."""
        import unicodedata
        
        # CRITICAL: Apply Unicode normalization FIRST
        text = unicodedata.normalize('NFD', text)
        
        # Remove non-Greek characters (include combining diacriticals)
        greek_pattern = r'[^\u0370-\u03FF\u1F00-\u1FFF\u0300-\u036F\s]'
        text = re.sub(greek_pattern, '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        
        # Apply NFC normalization at the end
        text = unicodedata.normalize('NFC', text)
        return text
    
    def calculate_mattr(self, words, window_size=100):
        """Calculate Moving Average Type-Token Ratio (MATTR)."""
        if len(words) < 50:
            return len(set(words)) / len(words) if words else 0
        
        window_size = min(window_size, len(words) // 2)
        if window_size < 10:
            return len(set(words)) / len(words)
        
        ttr_values = []
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            window_types = len(set(window))
            window_ttr = window_types / window_size
            ttr_values.append(window_ttr)
        
        return statistics.mean(ttr_values) if ttr_values else 0
    
    def extract_all_features(self, text):
        """Extract ALL discriminative features from text."""
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        total_words = len(words)
        total_chars = len(cleaned_text.replace(' ', ''))
        
        if total_words == 0:
            return {}
            
        features = {}
        
        # MORPHOLOGICAL FEATURES
        # Comprehensive Greek particles
        particles = {
            'particle_δε_freq': ['δε', 'δέ', 'δὲ'],
            'particle_τε_freq': ['τε', 'τέ', 'τὲ'],
            'particle_μεν_freq': ['μεν', 'μέν', 'μὲν'],
            'particle_γαρ_freq': ['γαρ', 'γάρ', 'γὰρ'],
            'particle_ουν_freq': ['ουν', 'οῦν', 'οὖν'],
            'particle_αν_freq': ['αν', 'ἄν', 'ἂν'],
            'particle_ει_freq': ['ει', 'εἰ', 'εἲ'],
            'particle_αλλα_freq': ['αλλα', 'ἀλλά', 'ἀλλὰ', 'αλλά'],
            'particle_ετι_freq': ['ετι', 'ἔτι', 'ἐτι', 'ἒτι'],
            'particle_μη_freq': ['μη', 'μή', 'μὴ'],
            'particle_ουδε_freq': ['ουδε', 'οὐδέ', 'οὐδὲ'],
            'particle_ουτε_freq': ['ουτε', 'οὔτε', 'οὒτε'],
            'particle_μηδε_freq': ['μηδε', 'μηδέ', 'μηδὲ'],
            'particle_αρα_freq': ['αρα', 'ἄρα', 'ἆρα', 'ἀρα'],
            'particle_δη_freq': ['δη', 'δή', 'δὴ'],
            'particle_γε_freq': ['γε', 'γέ', 'γὲ'],
            'particle_περ_freq': ['περ', 'πέρ', 'πὲρ'],
            'particle_τοι_freq': ['τοι', 'τοί', 'τοὶ'],
            'particle_που_freq': ['που', 'πού', 'ποὺ'],
        }
        
        for feature_name, variants in particles.items():
            count = sum(1 for word in words if word in variants)
            features[feature_name] = count / total_words
        
        # Case endings
        case_endings = {
            'genitive_sg_masc_freq': ['ου', 'οῦ'],
            'genitive_sg_fem_freq': ['ης', 'ῆς', 'ας', 'ᾶς'],
            'dative_sg_freq': ['ῳ', 'ῷ', 'ι', 'ί'],
            'accusative_sg_masc_freq': ['ον', 'όν'],
            'accusative_sg_fem_freq': ['ην', 'ήν'],
            'nominative_pl_freq': ['οι', 'αι'],
            'genitive_pl_freq': ['ων', 'ῶν'],
            'dative_pl_freq': ['οις', 'αις', 'οῖς', 'αῖς'],
            'accusative_pl_freq': ['ους', 'ας', 'οῦς', 'άς']
        }
        
        for case, endings in case_endings.items():
            count = 0
            for word in words:
                for ending in endings:
                    if word.endswith(ending):
                        count += 1
                        break
            features[f'{case}'] = count / total_words
        
        # Verb forms
        verb_endings = {
            'present_3sg_freq': ['ει', 'εῖ'],
            'aorist_3sg_freq': ['ε', 'έ', 'εν', 'έν'],
            'infinitive_freq': ['ειν', 'εῖν', 'αι', 'αῖ'],
            'participle_nom_sg_freq': ['ων', 'ών', 'ουσα', 'οῦσα']
        }
        
        for verb_form, endings in verb_endings.items():
            count = 0
            for word in words:
                for ending in endings:
                    if word.endswith(ending):
                        count += 1
                        break
            features[f'{verb_form}'] = count / total_words
        
        # PHONETIC FEATURES
        if total_chars > 0:
            vowels = ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω']
            for vowel in vowels:
                features[f'vowel_{vowel}_freq'] = cleaned_text.count(vowel) / total_chars
            
            clusters = ['στ', 'σκ', 'σπ', 'σφ', 'σχ', 'κτ', 'πτ', 'φθ', 'χθ', 'νθ', 'μπ', 'ντ']
            for cluster in clusters:
                features[f'cluster_{cluster}_freq'] = cleaned_text.count(cluster) / total_chars
            
            diphthongs = ['αι', 'ει', 'οι', 'υι', 'αυ', 'ευ', 'ου', 'ηυ']
            for diphthong in diphthongs:
                features[f'diphthong_{diphthong}_freq'] = cleaned_text.count(diphthong) / total_chars
        
        # VOCABULARY FEATURES
        if total_words > 0:
            unique_words = len(set(words))
            features['ttr_simple'] = unique_words / total_words
            
            # MATTR - better measure
            features['mattr_50'] = self.calculate_mattr(words, window_size=50)
            features['mattr_100'] = self.calculate_mattr(words, window_size=100)
            features['mattr_200'] = self.calculate_mattr(words, window_size=200)
            
            word_freq = Counter(words)
            hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
            features['hapax_ratio'] = hapax_count / total_words
            
            dis_count = sum(1 for freq in word_freq.values() if freq == 2)
            features['dis_ratio'] = dis_count / total_words
            
            features['avg_word_freq'] = statistics.mean(word_freq.values())
            
            most_frequent_10 = sum(sorted(word_freq.values(), reverse=True)[:10])
            features['top10_ratio'] = most_frequent_10 / total_words
        
        # WORD LENGTH FEATURES
        if words:
            word_lengths = [len(word) for word in words]
            features['avg_length'] = statistics.mean(word_lengths)
            features['median_length'] = statistics.median(word_lengths)
            features['std_length'] = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
            features['max_length'] = max(word_lengths)
            features['min_length'] = min(word_lengths)
            
            length_freq = Counter(word_lengths)
            for length in range(1, 21):
                features[f'length_{length}_freq'] = length_freq.get(length, 0) / total_words
        
        return features
    
    def extract_paul_features(self):
        """Extract features from all Paul's letters."""
        print("Extracting features from Paul's letters...")
        
        paul_letters_dir = Path("../Paul versus all authors/pauline_letters")
        if not paul_letters_dir.exists():
            print("Error: Paul's letters directory not found!")
            return {}
        
        paul_features = {}
        letter_files = list(paul_letters_dir.glob("*.txt"))
        
        for letter_file in letter_files:
            print(f"  Processing: {letter_file.name}")
            
            try:
                with open(letter_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                features = self.extract_all_features(text)
                letter_name = letter_file.stem
                paul_features[letter_name] = features
                
            except Exception as e:
                print(f"  Error reading {letter_file.name}: {e}")
        
        print(f"Extracted features from {len(paul_features)} Paul letters")
        return paul_features
    
    def extract_author_features(self, author_name):
        """Extract features from an individual author's multiple texts."""
        print(f"  Extracting features for {author_name}...")
        
        author_dir = Path(f"../{author_name}")
        if not author_dir.exists():
            return {}
        
        author_features = {}
        text_files = list(author_dir.glob("*.txt"))
        
        for text_file in text_files:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                features = self.extract_all_features(text)
                text_name = text_file.stem
                author_features[text_name] = features
                
            except Exception as e:
                print(f"    Error reading {text_file.name}: {e}")
        
        return author_features
    
    def calculate_author_variance(self, author_features):
        """Calculate variance for each feature across an author's texts."""
        feature_variances = {}
        
        # Get all feature names from first text
        if not author_features:
            return {}
        
        first_text = list(author_features.values())[0]
        
        for feature_name in first_text.keys():
            values = []
            for text_features in author_features.values():
                if feature_name in text_features:
                    values.append(text_features[feature_name])
            
            if len(values) > 1:
                feature_variances[feature_name] = statistics.variance(values)
            else:
                feature_variances[feature_name] = 0.0
        
        return feature_variances
    
    def compare_paul_to_authors(self):
        """Compare Paul's variance to each individual author's variance."""
        print("\n=== Paul vs Single Authors Analysis ===")
        
        # Get Paul's features
        paul_features = self.extract_paul_features()
        if not paul_features:
            print("Could not extract Paul's features!")
            return
        
        # Calculate Paul's variance for each feature
        paul_variances = self.calculate_author_variance(paul_features)
        print(f"Calculated variances for {len(paul_variances)} features from Paul")
        
        # Compare to each individual author
        author_comparisons = {}
        
        print("\nComparing Paul to individual authors:")
        for author_name in self.multi_text_authors.keys():
            print(f"\nAnalyzing {author_name}...")
            
            # Extract features for this author
            author_features = self.extract_author_features(author_name)
            if len(author_features) < 2:
                print(f"  Insufficient texts for {author_name}")
                continue
            
            # Calculate author's variance
            author_variances = self.calculate_author_variance(author_features)
            
            # Compare Paul vs this author
            comparison = self.compare_variances(paul_variances, author_variances, author_name)
            author_comparisons[author_name] = comparison
        
        # Generate final analysis
        self.generate_comparative_analysis(paul_features, author_comparisons, paul_variances)
        
        return author_comparisons
    
    def compare_variances(self, paul_variances, author_variances, author_name):
        """Compare Paul's variances to another author's variances."""
        feature_comparisons = {}
        paul_higher_count = 0
        total_comparisons = 0
        
        for feature_name in paul_variances.keys():
            if feature_name in author_variances:
                paul_var = paul_variances[feature_name]
                author_var = author_variances[feature_name]
                
                if author_var > 0:
                    ratio = paul_var / author_var
                    paul_higher = paul_var > author_var
                    
                    feature_comparisons[feature_name] = {
                        'paul_variance': paul_var,
                        'author_variance': author_var,
                        'ratio': ratio,
                        'paul_higher': paul_higher
                    }
                    
                    if paul_higher:
                        paul_higher_count += 1
                    total_comparisons += 1
        
        consistency_score = 1.0 - (paul_higher_count / total_comparisons) if total_comparisons > 0 else 0.5
        
        return {
            'author_name': author_name,
            'total_comparisons': total_comparisons,
            'paul_higher_count': paul_higher_count,
            'paul_higher_percentage': (paul_higher_count / total_comparisons * 100) if total_comparisons > 0 else 0,
            'consistency_score': consistency_score,
            'feature_comparisons': feature_comparisons
        }
    
    def generate_comparative_analysis(self, paul_features, author_comparisons, paul_variances):
        """Generate comprehensive comparative analysis."""
        print(f"\n{'='*60}")
        print(f"PAUL VS INDIVIDUAL AUTHORS ANALYSIS")
        print(f"{'='*60}")
        
        # Overall statistics
        total_authors = len(author_comparisons)
        consistency_scores = [comp['consistency_score'] for comp in author_comparisons.values()]
        
        if consistency_scores:
            avg_consistency = statistics.mean(consistency_scores)
            print(f"")
            print(f"Paul's letters analyzed: {len(paul_features)}")
            print(f"Authors compared against: {total_authors}")
            print(f"Average consistency with single authors: {avg_consistency:.1%}")
            print(f"")
        
        # Interpretation
        if avg_consistency >= 0.7:
            interpretation = "Paul's variation is NORMAL for a single author"
        elif avg_consistency >= 0.5:
            interpretation = "Paul's variation is MODERATE for a single author"
        elif avg_consistency >= 0.3:
            interpretation = "Paul's variation is HIGH for a single author"
        else:
            interpretation = "Paul's variation is ABNORMALLY HIGH for a single author"
        
        print(f"INTERPRETATION: {interpretation}")
        print(f"")
        
        # Rank authors by how similar they are to Paul
        sorted_authors = sorted(author_comparisons.items(), 
                               key=lambda x: x[1]['consistency_score'], 
                               reverse=True)
        
        print(f"AUTHORS MOST SIMILAR TO PAUL (lowest variance differences):")
        for i, (author_name, comp) in enumerate(sorted_authors[:5]):
            print(f"  {i+1}. {author_name}")
            print(f"     Consistency: {comp['consistency_score']:.1%}")
            print(f"     Paul higher variance: {comp['paul_higher_percentage']:.1f}% of features")
            print(f"     Comparisons: {comp['total_comparisons']} features")
        
        print(f"")
        print(f"AUTHORS MOST DIFFERENT FROM PAUL (highest variance differences):")
        for i, (author_name, comp) in enumerate(sorted_authors[-5:]):
            print(f"  {len(sorted_authors)-i}. {author_name}")
            print(f"     Consistency: {comp['consistency_score']:.1%}")
            print(f"     Paul higher variance: {comp['paul_higher_percentage']:.1f}% of features")
            print(f"     Comparisons: {comp['total_comparisons']} features")
        
        # Feature analysis
        print(f"")
        print(f"FEATURES WHERE PAUL SHOWS HIGHEST VARIANCE:")
        
        # Find features where Paul consistently shows high variance
        feature_paul_higher_counts = {}
        for comp in author_comparisons.values():
            for feature_name, feature_comp in comp['feature_comparisons'].items():
                if feature_comp['paul_higher']:
                    if feature_name not in feature_paul_higher_counts:
                        feature_paul_higher_counts[feature_name] = 0
                    feature_paul_higher_counts[feature_name] += 1
        
        sorted_features = sorted(feature_paul_higher_counts.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        for feature_name, count in sorted_features[:10]:
            percentage = count / total_authors * 100
            paul_var = paul_variances.get(feature_name, 0)
            print(f"  • {feature_name}: Paul higher than {count}/{total_authors} authors ({percentage:.0f}%)")
            print(f"    Paul's variance: {paul_var:.8f}")
        
        # Save detailed results
        results = {
            'paul_letters_count': len(paul_features),
            'paul_letters': list(paul_features.keys()),
            'authors_compared': total_authors,
            'average_consistency': avg_consistency,
            'interpretation': interpretation,
            'author_comparisons': author_comparisons,
            'paul_variances': paul_variances,
            'ranked_authors': [(name, comp['consistency_score']) for name, comp in sorted_authors],
            'problematic_features': dict(sorted_features[:20])
        }
        
        output_file = "paul_vs_authors_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"")
        print(f"{'='*60}")
        print(f"Detailed results saved to: {output_file}")

def main():
    """Main execution function."""
    analyzer = PaulVsSingleAuthorsAnalyzer()
    analyzer.compare_paul_to_authors()

if __name__ == "__main__":
    main()
