#!/usr/bin/env python3

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
        print("Loading reference data from ancient Greek analysis...")
        
        with open('results/best_discriminative_features.json', 'r', encoding='utf-8') as f:
            self.reference_features = json.load(f)
        
        self.top_features = []
        for item in self.reference_features[:30]:
            if item['separation_score'] > 0.3:
                self.top_features.append({
                    'name': f"{item['feature_type']}:{item['feature']}",
                    'type': item['feature_type'],
                    'feature': item['feature'],
                    'separation_score': item['separation_score'],
                    'reference_values': item['author_values']
                })
        
        print(f"Loaded {len(self.top_features)} discriminative features for analysis")
        
        self.calculate_single_author_variance()
    
    def calculate_single_author_variance(self):
        print("Calculating baseline variance thresholds...")
        
        self.single_author_variance_threshold = {}
        
        for feature in self.top_features:
            author_values = list(feature['reference_values'].values())
            
            if len(author_values) > 1:
                inter_author_variance = statistics.variance(author_values)
                inter_author_range = max(author_values) - min(author_values)
                
                separation_factor = feature['separation_score']
                
                single_author_threshold = inter_author_variance * (0.01 + 0.05 * (1 - separation_factor))
                
                self.single_author_variance_threshold[feature['name']] = single_author_threshold
                
                range_threshold = (inter_author_range ** 2) * (0.001 + 0.01 * (1 - separation_factor))
                
                self.single_author_variance_threshold[feature['name']] = min(single_author_threshold, range_threshold)
            else:
                self.single_author_variance_threshold[feature['name']] = 0.001
    
    def clean_text(self, text):
        greek_pattern = r'[^\u0370-\u03FF\u1F00-\u1FFF\s]'
        text = re.sub(greek_pattern, '', text)
        
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        
        return text
    
    def extract_morphological_features(self, text):
        words = text.split()
        total_words = len(words)
        
        if total_words == 0:
            return {}
            
        features = {}
        
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
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return {}
            
        features = {}
        
        vowels = ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω']
        for vowel in vowels:
            features[f'vowel_{vowel}_freq'] = text.count(vowel) / total_chars
        
        clusters = ['στ', 'σκ', 'πτ', 'κτ', 'φθ', 'χθ']
        for cluster in clusters:
            features[f'cluster_{cluster}_freq'] = text.count(cluster) / total_chars
        
        diphthongs = ['αι', 'ει', 'ου', 'αυ', 'ευ']
        for diphthong in diphthongs:
            features[f'diphthong_{diphthong}_freq'] = text.count(diphthong) / total_chars
        
        return features
    
    def extract_vocabulary_features(self, text):
        words = text.split()
        total_words = len(words)
        unique_words = len(set(words))
        
        if total_words == 0:
            return {}
        
        features = {}
        
        features['ttr'] = unique_words / total_words
        
        word_freq = Counter(words)
        hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
        features['hapax_ratio'] = hapax_count / total_words
        
        features['avg_word_freq'] = statistics.mean(word_freq.values())
        
        return features
    
    def extract_word_length_features(self, text):
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
        cleaned_text = self.clean_text(text)
        
        all_features = {}
        all_features.update(self.extract_morphological_features(cleaned_text))
        all_features.update(self.extract_phonetic_features(cleaned_text))
        all_features.update(self.extract_vocabulary_features(cleaned_text))
        all_features.update(self.extract_word_length_features(cleaned_text))
        
        return all_features
    
    def analyze_corpus_consistency(self, corpus_features):
        print(f"Analyzing consistency across {len(corpus_features)} texts...")
        
        consistency_scores = {}
        feature_variances = {}
        
        for feature_info in self.top_features:
            feature_name = feature_info['feature']
            feature_type = feature_info['type']
            full_name = f"{feature_type}:{feature_name}"
            
            values = []
            for text_features in corpus_features:
                if feature_name in text_features:
                    values.append(text_features[feature_name])
            
            if len(values) < 2:
                continue
            
            corpus_variance = statistics.variance(values)
            corpus_mean = statistics.mean(values)
            
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
            
            expected_variance = self.single_author_variance_threshold.get(full_name, 0.001)
            
            if corpus_variance <= expected_variance:
                consistency_score = 1.0
            else:
                ratio = corpus_variance / expected_variance
                consistency_score = max(0, np.exp(-ratio + 1))
            
            consistency_scores[full_name] = consistency_score
        
        return consistency_scores, feature_variances
    
    def calculate_single_authorship_probability(self, consistency_scores, feature_variances):
        print("Calculating single authorship probability...")
        
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
        
        total_weighted_score = sum(weighted_scores)
        total_weight = sum(weights)
        overall_consistency = total_weighted_score / total_weight
        
        probability = max(0.0, min(1.0, overall_consistency))
        
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
        print(f"\n=== Single Authorship Analysis ===")
        print(f"Analyzing corpus: {corpus_directory}")
        
        corpus_path = Path(corpus_directory)
        if not corpus_path.exists():
            print(f"Error: Directory {corpus_directory} does not exist")
            return
        
        text_files = list(corpus_path.glob("*.txt"))
        if not text_files:
            print(f"Error: No .txt files found in {corpus_directory}")
            return
        
        print(f"Found {len(text_files)} text files")
        
        corpus_features = []
        text_names = []
        
        for text_file in text_files:
            print(f"Processing: {text_file.name}")
            
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
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
        
        consistency_scores, feature_variances = self.analyze_corpus_consistency(corpus_features)
        
        probability, interpretation = self.calculate_single_authorship_probability(consistency_scores, feature_variances)
        
        self.generate_report(probability, interpretation, consistency_scores, feature_variances, text_names)
        
        return probability, interpretation
    
    def generate_report(self, probability, interpretation, consistency_scores, feature_variances, text_names):
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

