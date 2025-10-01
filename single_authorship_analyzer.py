#!/usr/bin/env python3
"""
Single Authorship Probability Analyzer
======================================

This script uses the discriminative features identified from ancient Greek authors
to analyze whether a corpus (like the 14 Pauline letters) was written by a single
author or multiple authors.

Method:
1. Extract the same 60 discriminative features from each text in the corpus
2. Calculate variance/consistency within the corpus for each feature
3. Compare this variance to the expected variance for single authors
4. Generate probability score for single authorship

Example usage:
python single_authorship_analyzer.py /path/to/pauline_letters/
"""

import os
import json
import re
import statistics
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class SingleAuthorshipAnalyzer:
    def __init__(self):
        self.load_reference_data()
        
    def load_reference_data(self):
        """Load the reference data from our 33-author analysis."""
        print("Loading reference data from ancient Greek analysis...")
        
        # Load discriminative features
        with open('results/best_discriminative_features.json', 'r', encoding='utf-8') as f:
            self.reference_features = json.load(f)
        
        # Extract the top discriminative features for analysis
        self.top_features = []
        for item in self.reference_features[:30]:  # Use top 30 features
            if item['separation_score'] > 0.3:  # Only reliable features
                self.top_features.append({
                    'name': f"{item['feature_type']}:{item['feature']}",
                    'type': item['feature_type'],
                    'feature': item['feature'],
                    'separation_score': item['separation_score'],
                    'reference_values': item['author_values']
                })
        
        print(f"Loaded {len(self.top_features)} discriminative features for analysis")
        
        # Calculate expected variance for single authors
        self.calculate_single_author_variance()
    
    def calculate_single_author_variance(self):
        """Calculate expected variance for single authors from reference data."""
        print("Calculating baseline variance thresholds...")
        
        # Calculate baseline variance from the spread of our 33 authors
        # Single author texts should have MUCH lower variance than the spread between different authors
        self.single_author_variance_threshold = {}
        
        for feature in self.top_features:
            # Get the range of values across all 33 authors for this feature
            author_values = list(feature['reference_values'].values())
            
            if len(author_values) > 1:
                # Calculate the inter-author variance (how much authors differ from each other)
                inter_author_variance = statistics.variance(author_values)
                inter_author_range = max(author_values) - min(author_values)
                
                # Single author variance should be much smaller than inter-author variance
                # Use a fraction based on the separation score:
                # - High separation score = authors are very different = single author should have tiny variance
                # - Low separation score = authors are similar = single author can have more variance
                
                separation_factor = feature['separation_score']
                
                # Single author threshold: smaller fraction of inter-author variance for better separated features
                single_author_threshold = inter_author_variance * (0.01 + 0.05 * (1 - separation_factor))
                
                self.single_author_variance_threshold[feature['name']] = single_author_threshold
                
                # Also store range-based threshold
                range_threshold = (inter_author_range ** 2) * (0.001 + 0.01 * (1 - separation_factor))
                
                # Use the more restrictive threshold
                self.single_author_variance_threshold[feature['name']] = min(single_author_threshold, range_threshold)
            else:
                self.single_author_variance_threshold[feature['name']] = 0.001
    
    def clean_text(self, text):
        """Clean and normalize Greek text (same as original analysis)."""
        # Remove non-Greek characters including ALL punctuation
        greek_pattern = r'[^\u0370-\u03FF\u1F00-\u1FFF\s]'
        text = re.sub(greek_pattern, '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        
        return text
    
    def extract_morphological_features(self, text):
        """Extract morphological features from text."""
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            return {}
            
        features = {}
        
        # Greek particles
        particles = {
            'δε': ['δε', 'δέ'], 'τε': ['τε', 'τέ'], 'μεν': ['μεν', 'μέν'], 
            'γαρ': ['γαρ', 'γάρ'], 'ουν': ['ουν', 'οῦν'], 'αν': ['αν', 'ἄν'],
            'ει': ['ει', 'εἰ'], 'αλλα': ['αλλα', 'ἀλλά'], 'ετι': ['ετι', 'ἔτι']
        }
        
        for particle_key, variants in particles.items():
            count = 0
            for word in words:
                if word in variants:
                    count += 1
            features[f'particle_{variants[1]}_freq'] = count / total_words
        
        # Case endings
        case_endings = {
            'genitive_sg_masc': ['ου', 'οῦ'],
            'genitive_sg_fem': ['ης', 'ῆς', 'ας', 'ᾶς'],
            'dative_sg': ['ῳ', 'ῷ', 'ι', 'ί'],
            'accusative_sg_masc': ['ον', 'όν'],
            'accusative_sg_fem': ['ην', 'ήν'],
            'nominative_pl': ['οι', 'αι'],
            'genitive_pl': ['ων', 'ῶν'],
            'dative_pl': ['οις', 'αις', 'οῖς', 'αῖς'],
            'accusative_pl': ['ους', 'ας', 'οῦς', 'άς']
        }
        
        for case, endings in case_endings.items():
            count = 0
            for word in words:
                for ending in endings:
                    if word.endswith(ending):
                        count += 1
                        break
            features[f'{case}_freq'] = count / total_words
        
        return features
    
    def extract_phonetic_features(self, text):
        """Extract phonetic features from text."""
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return {}
            
        features = {}
        
        # Vowel frequencies
        vowels = ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω']
        for vowel in vowels:
            features[f'vowel_{vowel}_freq'] = text.count(vowel) / total_chars
        
        # Consonant clusters
        clusters = ['στ', 'σκ', 'πτ', 'κτ', 'φθ', 'χθ']
        for cluster in clusters:
            features[f'cluster_{cluster}_freq'] = text.count(cluster) / total_chars
        
        # Diphthongs
        diphthongs = ['αι', 'ει', 'ου', 'αυ', 'ευ']
        for diphthong in diphthongs:
            features[f'diphthong_{diphthong}_freq'] = text.count(diphthong) / total_chars
        
        return features
    
    def extract_vocabulary_features(self, text):
        """Extract vocabulary richness features."""
        words = text.split()
        total_words = len(words)
        unique_words = len(set(words))
        
        if total_words == 0:
            return {}
        
        features = {}
        
        # Type-Token Ratio
        features['ttr'] = unique_words / total_words
        
        # Hapax legomena ratio
        word_freq = Counter(words)
        hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
        features['hapax_ratio'] = hapax_count / total_words
        
        # Average word frequency
        features['avg_word_freq'] = statistics.mean(word_freq.values())
        
        return features
    
    def extract_word_length_features(self, text):
        """Extract word length distribution features."""
        words = text.split()
        word_lengths = [len(word) for word in words if word.strip()]
        
        if not word_lengths:
            return {}
        
        features = {}
        features['avg_length'] = statistics.mean(word_lengths)
        features['median_length'] = statistics.median(word_lengths)
        features['std_length'] = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0
        
        return features
    
    def extract_all_features(self, text):
        """Extract all discriminative features from text."""
        cleaned_text = self.clean_text(text)
        
        all_features = {}
        all_features.update(self.extract_morphological_features(cleaned_text))
        all_features.update(self.extract_phonetic_features(cleaned_text))
        all_features.update(self.extract_vocabulary_features(cleaned_text))
        all_features.update(self.extract_word_length_features(cleaned_text))
        
        return all_features
    
    def analyze_corpus_consistency(self, corpus_features):
        """Analyze consistency of features across corpus texts."""
        print(f"Analyzing consistency across {len(corpus_features)} texts...")
        
        consistency_scores = {}
        feature_variances = {}
        
        for feature_info in self.top_features:
            feature_name = feature_info['feature']
            feature_type = feature_info['type']
            full_name = f"{feature_type}:{feature_name}"
            
            # Collect values for this feature across all texts
            values = []
            for text_features in corpus_features:
                if feature_name in text_features:
                    values.append(text_features[feature_name])
            
            if len(values) < 2:
                continue
            
            # Calculate variance within this corpus
            corpus_variance = statistics.variance(values)
            corpus_mean = statistics.mean(values)
            
            # Calculate coefficient of variation (normalized variance)
            cv = (statistics.stdev(values) / corpus_mean) if corpus_mean > 0 else float('inf')
            
            feature_variances[full_name] = {
                'variance': corpus_variance,
                'std_dev': statistics.stdev(values),
                'mean': corpus_mean,
                'coefficient_variation': cv,
                'range': max(values) - min(values),
                'values': values,
                'separation_score': feature_info['separation_score']
            }
            
            # Compare to expected single-author variance
            expected_variance = self.single_author_variance_threshold.get(full_name, 0.001)
            
            # Calculate consistency score with more nuanced approach
            if corpus_variance <= expected_variance:
                # Variance is within expected range for single author
                consistency_score = 1.0
            else:
                # Variance exceeds single author expectation
                # Use exponential decay so high variance gets penalized heavily
                ratio = corpus_variance / expected_variance
                consistency_score = max(0, np.exp(-ratio + 1))
            
            consistency_scores[full_name] = consistency_score
        
        return consistency_scores, feature_variances
    
    def calculate_single_authorship_probability(self, consistency_scores, feature_variances):
        """Calculate overall probability of single authorship."""
        print("Calculating single authorship probability...")
        
        # Weight scores by feature reliability (separation score)
        weighted_scores = []
        weights = []
        
        for feature_info in self.top_features:
            full_name = f"{feature_info['type']}:{feature_info['feature']}"
            if full_name in consistency_scores:
                score = consistency_scores[full_name]
                weight = feature_info['separation_score']
                weighted_scores.append(score * weight)
                weights.append(weight)
        
        if not weighted_scores:
            return 0.5, "Insufficient data"
        
        # Calculate weighted average
        total_weighted_score = sum(weighted_scores)
        total_weight = sum(weights)
        overall_consistency = total_weighted_score / total_weight
        
        # Convert to probability (0.0 = definitely multiple authors, 1.0 = definitely single author)
        probability = max(0.0, min(1.0, overall_consistency))
        
        # Generate interpretation
        if probability >= 0.8:
            interpretation = "Very likely single author"
        elif probability >= 0.6:
            interpretation = "Probably single author"
        elif probability >= 0.4:
            interpretation = "Uncertain - mixed evidence"
        elif probability >= 0.2:
            interpretation = "Probably multiple authors"
        else:
            interpretation = "Very likely multiple authors"
        
        return probability, interpretation
    
    def analyze_corpus(self, corpus_directory):
        """Analyze a corpus for single authorship probability."""
        print(f"\n=== Single Authorship Analysis ===")
        print(f"Analyzing corpus: {corpus_directory}")
        
        corpus_path = Path(corpus_directory)
        if not corpus_path.exists():
            print(f"Error: Directory {corpus_directory} does not exist")
            return
        
        # Load all text files
        text_files = list(corpus_path.glob("*.txt"))
        if not text_files:
            print(f"Error: No .txt files found in {corpus_directory}")
            return
        
        print(f"Found {len(text_files)} text files")
        
        # Extract features from each text
        corpus_features = []
        text_names = []
        
        for text_file in text_files:
            print(f"Processing: {text_file.name}")
            
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Check minimum length
                word_count = len(text.split())
                if word_count < 100:
                    print(f"  Warning: {text_file.name} has only {word_count} words (very short)")
                
                features = self.extract_all_features(text)
                corpus_features.append(features)
                text_names.append(text_file.stem)
                
            except Exception as e:
                print(f"  Error reading {text_file.name}: {e}")
        
        if len(corpus_features) < 2:
            print("Error: Need at least 2 texts for comparison")
            return
        
        # Analyze consistency
        consistency_scores, feature_variances = self.analyze_corpus_consistency(corpus_features)
        
        # Calculate probability
        probability, interpretation = self.calculate_single_authorship_probability(consistency_scores, feature_variances)
        
        # Generate report
        self.generate_report(probability, interpretation, consistency_scores, feature_variances, text_names)
        
        return probability, interpretation
    
    def generate_report(self, probability, interpretation, consistency_scores, feature_variances, text_names):
        """Generate detailed analysis report."""
        print(f"\n{'='*50}")
        print(f"SINGLE AUTHORSHIP ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"")
        print(f"Overall Probability of Single Authorship: {probability:.1%}")
        print(f"Interpretation: {interpretation}")
        print(f"")
        print(f"Analyzed {len(text_names)} texts:")
        for name in text_names:
            print(f"  - {name}")
        print(f"")
        
        # Top consistent features (evidence FOR single authorship)
        print(f"TOP EVIDENCE FOR SINGLE AUTHORSHIP:")
        sorted_consistent = sorted(consistency_scores.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_consistent[:5]:
            feature_type, feature_name = feature.split(':', 1)
            variance_info = feature_variances[feature]
            print(f"  • {feature_name} ({feature_type})")
            print(f"    Consistency: {score:.1%}")
            print(f"    Variance: {variance_info['variance']:.6f}")
            print(f"    Range: {variance_info['range']:.6f}")
        
        print(f"")
        
        # Top inconsistent features (evidence AGAINST single authorship)  
        print(f"TOP EVIDENCE AGAINST SINGLE AUTHORSHIP:")
        sorted_inconsistent = sorted(consistency_scores.items(), key=lambda x: x[1])
        for feature, score in sorted_inconsistent[:5]:
            feature_type, feature_name = feature.split(':', 1)
            variance_info = feature_variances[feature]
            print(f"  • {feature_name} ({feature_type})")
            print(f"    Consistency: {score:.1%}")
            print(f"    Variance: {variance_info['variance']:.6f}")
            print(f"    Range: {variance_info['range']:.6f}")
        
        print(f"")
        print(f"{'='*50}")
        
        # Save detailed results
        results = {
            'probability': probability,
            'interpretation': interpretation,
            'text_count': len(text_names),
            'text_names': text_names,
            'consistency_scores': consistency_scores,
            'feature_variances': feature_variances,
            'summary': {
                'most_consistent_features': dict(sorted_consistent[:10]),
                'least_consistent_features': dict(sorted_inconsistent[:10])
            }
        }
        
        output_file = "single_authorship_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to: {output_file}")

def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python single_authorship_analyzer.py <corpus_directory>")
        print("Example: python single_authorship_analyzer.py /path/to/pauline_letters/")
        return
    
    corpus_directory = sys.argv[1]
    
    analyzer = SingleAuthorshipAnalyzer()
    analyzer.analyze_corpus(corpus_directory)

if __name__ == "__main__":
    main()
