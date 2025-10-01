#!/usr/bin/env python3
"""
Greek Authorship Attribution Analysis (Authentic Features Only)
===============================================================

This script analyzes ancient Greek texts (100BC-100AD) to identify AUTHENTIC linguistic 
features that can successfully differentiate between all authors. Only authors with >1000 
words total are included in the analysis.

IMPORTANT: Modern punctuation is excluded as it was not present in original manuscripts.

Features analyzed:
1. Character n-grams (2-char, 3-char, 4-char)
2. Word n-grams (2-word, 3-word)
3. Word frequency patterns
4. Morphological patterns (case endings, particles, verb forms)
5. Word length distributions
6. Phonetic patterns (vowels, consonant clusters, diphthongs)
7. Vocabulary richness (TTR, hapax legomena, lexical diversity)

The goal is to find authentic linguistic features that can separate ALL authors.
"""

import os
import re
import json
import pickle
from collections import defaultdict, Counter
from pathlib import Path
import unicodedata
import statistics
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class GreekTextAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.authors = {}
        self.author_texts = {}
        self.word_count_threshold = 1000
        self.results = {}
        
    def load_texts(self):
        """Load all texts and organize by author."""
        print("Loading texts...")
        
        for author_dir in self.base_path.iterdir():
            if author_dir.is_dir() and author_dir.name not in ['results', 'results documentation', '__pycache__']:
                author_name = author_dir.name
                author_texts = []
                total_words = 0
                
                for text_file in author_dir.glob("*.txt"):
                    try:
                        with open(text_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Clean and normalize the text
                            content = self.clean_text(content)
                            if content:
                                words = len(content.split())
                                total_words += words
                                author_texts.append({
                                    'filename': text_file.name,
                                    'content': content,
                                    'word_count': words
                                })
                    except Exception as e:
                        print(f"Error reading {text_file}: {e}")
                        continue
                
                if total_words >= self.word_count_threshold:
                    self.authors[author_name] = {
                        'texts': author_texts,
                        'total_words': total_words,
                        'num_texts': len(author_texts)
                    }
                    # Combine all texts for this author
                    combined_text = ' '.join([text['content'] for text in author_texts])
                    self.author_texts[author_name] = combined_text
                    print(f"✓ {author_name}: {total_words:,} words in {len(author_texts)} texts")
                else:
                    print(f"✗ {author_name}: Only {total_words} words (< {self.word_count_threshold})")
        
        print(f"\nIncluded {len(self.authors)} authors with sufficient text volume.")
        return self.authors
    
    def clean_text(self, text):
        """Clean and normalize Greek text - removing modern punctuation."""
        # Remove non-Greek characters including ALL punctuation (modern editorial additions)
        # Greek Unicode ranges: 0370-03FF (Greek and Coptic), 1F00-1FFF (Greek Extended)
        # Only keep Greek letters and spaces
        greek_pattern = r'[^\u0370-\u03FF\u1F00-\u1FFF\s]'
        text = re.sub(greek_pattern, '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        return text
    
    def extract_character_ngrams(self, text, n):
        """Extract character n-grams from text."""
        # Remove spaces for character n-grams
        text_no_spaces = re.sub(r'\s+', '', text)
        ngrams = []
        for i in range(len(text_no_spaces) - n + 1):
            ngrams.append(text_no_spaces[i:i+n])
        return ngrams
    
    def extract_word_ngrams(self, text, n):
        """Extract word n-grams from text."""
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    def analyze_character_ngrams(self):
        """Analyze character n-grams (2, 3, 4 characters)."""
        print("\nAnalyzing character n-grams...")
        
        for n in [2, 3, 4]:
            print(f"  Processing {n}-character n-grams...")
            author_ngrams = {}
            
            for author, text in self.author_texts.items():
                ngrams = self.extract_character_ngrams(text, n)
                author_ngrams[author] = Counter(ngrams)
            
            # Find the most distinctive n-grams
            distinctive_ngrams = self.find_distinctive_features(author_ngrams, f"{n}-char-ngrams")
            self.results[f'char_{n}grams'] = {
                'author_features': author_ngrams,
                'distinctive_features': distinctive_ngrams
            }
    
    def analyze_word_ngrams(self):
        """Analyze word n-grams (2, 3 words)."""
        print("\nAnalyzing word n-grams...")
        
        for n in [2, 3]:
            print(f"  Processing {n}-word n-grams...")
            author_ngrams = {}
            
            for author, text in self.author_texts.items():
                ngrams = self.extract_word_ngrams(text, n)
                # Filter out very rare n-grams (appear less than 3 times)
                ngram_counts = Counter(ngrams)
                filtered_ngrams = {k: v for k, v in ngram_counts.items() if v >= 3}
                author_ngrams[author] = filtered_ngrams
            
            distinctive_ngrams = self.find_distinctive_features(author_ngrams, f"{n}-word-ngrams")
            self.results[f'word_{n}grams'] = {
                'author_features': author_ngrams,
                'distinctive_features': distinctive_ngrams
            }
    
    def analyze_word_frequencies(self):
        """Analyze individual word frequencies."""
        print("\nAnalyzing word frequencies...")
        
        author_word_freqs = {}
        all_words = set()
        
        for author, text in self.author_texts.items():
            words = text.split()
            word_freq = Counter(words)
            # Normalize by total words
            total_words = sum(word_freq.values())
            normalized_freq = {word: count/total_words for word, count in word_freq.items()}
            author_word_freqs[author] = normalized_freq
            all_words.update(word_freq.keys())
        
        # Find words that appear in at least half of the authors
        min_authors = len(self.authors) // 2
        common_words = []
        for word in all_words:
            author_count = sum(1 for freq_dict in author_word_freqs.values() if word in freq_dict)
            if author_count >= min_authors:
                common_words.append(word)
        
        print(f"  Found {len(common_words)} common words across authors")
        
        distinctive_words = self.find_distinctive_features(author_word_freqs, "word-frequencies")
        self.results['word_frequencies'] = {
            'author_features': author_word_freqs,
            'distinctive_features': distinctive_words,
            'common_words': common_words
        }
    
    def analyze_morphological_patterns(self):
        """Analyze morphological and syntactic patterns in ancient Greek."""
        print("\nAnalyzing morphological and syntactic patterns...")
        
        author_morph_patterns = {}
        
        for author, text in self.author_texts.items():
            morph_counts = {}
            words = text.split()
            total_words = len(words)
            
            if total_words == 0:
                continue
                
            # Common Greek case endings and particles
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
            
            # Count case ending frequencies
            for case, endings in case_endings.items():
                count = 0
                for word in words:
                    for ending in endings:
                        if word.endswith(ending):
                            count += 1
                            break  # Don't double count
                morph_counts[f'{case}_freq'] = count / total_words
            
            # Common particles and function words
            particles = ['δε', 'δέ', 'γαρ', 'γάρ', 'μεν', 'μέν', 'δη', 'δή', 'τε', 'τέ', 
                        'ουν', 'οῦν', 'αν', 'ἄν', 'ει', 'εἰ', 'αλλα', 'ἀλλά', 'ετι', 'ἔτι']
            
            for particle in particles:
                count = sum(1 for word in words if word == particle)
                morph_counts[f'particle_{particle}_freq'] = count / total_words
            
            # Verb forms (common endings)
            verb_endings = {
                'present_3sg': ['ει', 'εῖ'],
                'aorist_3sg': ['ε', 'έ', 'εν', 'έν'],
                'infinitive': ['ειν', 'εῖν', 'αι', 'αῖ'],
                'participle_nom_sg': ['ων', 'ών', 'ουσα', 'οῦσα']
            }
            
            for verb_form, endings in verb_endings.items():
                count = 0
                for word in words:
                    for ending in endings:
                        if word.endswith(ending):
                            count += 1
                            break
                morph_counts[f'{verb_form}_freq'] = count / total_words
            
            author_morph_patterns[author] = morph_counts
        
        distinctive_morph = self.find_distinctive_features(author_morph_patterns, "morphological-patterns")
        self.results['morphological'] = {
            'author_features': author_morph_patterns,
            'distinctive_features': distinctive_morph
        }
    
    def analyze_word_length_distribution(self):
        """Analyze distribution of word lengths (authentic linguistic feature)."""
        print("\nAnalyzing word length distributions...")
        
        author_word_lengths = {}
        
        for author, text in self.author_texts.items():
            words = text.split()
            word_lengths = [len(word) for word in words if word.strip()]
            
            if not word_lengths:
                continue
                
            # Calculate distribution statistics
            length_dist = {
                'avg_length': statistics.mean(word_lengths),
                'median_length': statistics.median(word_lengths),
                'std_length': statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0,
                'max_length': max(word_lengths),
                'min_length': min(word_lengths)
            }
            
            # Length frequency distribution
            length_freq = Counter(word_lengths)
            total_words = len(word_lengths)
            for length in range(1, 21):  # 1-20 character words
                length_dist[f'length_{length}_freq'] = length_freq.get(length, 0) / total_words
            
            author_word_lengths[author] = length_dist
        
        distinctive_lengths = self.find_distinctive_features(author_word_lengths, "word-length-distributions")
        self.results['word_lengths'] = {
            'author_features': author_word_lengths,
            'distinctive_features': distinctive_lengths
        }
    
    def analyze_phonetic_patterns(self):
        """Analyze phonetic and sound patterns in Greek text."""
        print("\nAnalyzing phonetic patterns...")
        
        author_phonetic_patterns = {}
        
        for author, text in self.author_texts.items():
            words = text.split()
            total_words = len(words)
            
            if total_words == 0:
                continue
                
            phonetic_counts = {}
            
            # Vowel patterns
            vowels = ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω', 'ά', 'έ', 'ή', 'ί', 'ό', 'ύ', 'ώ']
            vowel_count = sum(text.count(vowel) for vowel in vowels)
            total_chars = len(text.replace(' ', ''))
            phonetic_counts['vowel_ratio'] = vowel_count / total_chars if total_chars > 0 else 0
            
            # Specific vowel frequencies
            for vowel in ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω']:
                phonetic_counts[f'vowel_{vowel}_freq'] = text.count(vowel) / total_chars if total_chars > 0 else 0
            
            # Consonant clusters
            clusters = ['στ', 'σκ', 'σπ', 'σφ', 'σχ', 'κτ', 'πτ', 'φθ', 'χθ', 'νθ', 'μπ', 'ντ']
            for cluster in clusters:
                phonetic_counts[f'cluster_{cluster}_freq'] = text.count(cluster) / total_chars if total_chars > 0 else 0
            
            # Diphthongs
            diphthongs = ['αι', 'ει', 'οι', 'υι', 'αυ', 'ευ', 'ου', 'ηυ']
            for diphthong in diphthongs:
                phonetic_counts[f'diphthong_{diphthong}_freq'] = text.count(diphthong) / total_chars if total_chars > 0 else 0
            
            author_phonetic_patterns[author] = phonetic_counts
        
        distinctive_phonetic = self.find_distinctive_features(author_phonetic_patterns, "phonetic-patterns")
        self.results['phonetic'] = {
            'author_features': author_phonetic_patterns,
            'distinctive_features': distinctive_phonetic
        }
    
    def analyze_vocabulary_richness(self):
        """Analyze vocabulary richness and lexical diversity."""
        print("\nAnalyzing vocabulary richness...")
        
        author_vocab_stats = {}
        
        for author, text in self.author_texts.items():
            words = text.split()
            total_words = len(words)
            unique_words = len(set(words))
            
            if total_words == 0:
                continue
                
            vocab_stats = {}
            
            # Type-Token Ratio (TTR)
            vocab_stats['ttr'] = unique_words / total_words
            
            # Hapax legomena (words appearing once)
            word_freq = Counter(words)
            hapax_count = sum(1 for freq in word_freq.values() if freq == 1)
            vocab_stats['hapax_ratio'] = hapax_count / total_words
            
            # Dis legomena (words appearing twice)
            dis_count = sum(1 for freq in word_freq.values() if freq == 2)
            vocab_stats['dis_ratio'] = dis_count / total_words
            
            # Average word frequency
            vocab_stats['avg_word_freq'] = statistics.mean(word_freq.values())
            
            # Most frequent words ratio
            most_frequent_10 = sum(sorted(word_freq.values(), reverse=True)[:10])
            vocab_stats['top10_ratio'] = most_frequent_10 / total_words
            
            author_vocab_stats[author] = vocab_stats
        
        distinctive_vocab = self.find_distinctive_features(author_vocab_stats, "vocabulary-richness")
        self.results['vocabulary'] = {
            'author_features': author_vocab_stats,
            'distinctive_features': distinctive_vocab
        }
    
    def find_distinctive_features(self, author_features, feature_type):
        """Find features that can distinguish between authors."""
        print(f"    Finding distinctive features for {feature_type}...")
        
        distinctive_features = []
        authors = list(author_features.keys())
        
        # For each feature, check if it can separate authors
        all_features = set()
        for features in author_features.values():
            all_features.update(features.keys())
        
        for feature in all_features:
            # Get feature values for each author
            feature_values = {}
            for author in authors:
                value = author_features[author].get(feature, 0)
                feature_values[author] = value
            
            # Check if this feature can separate authors
            separation_score = self.calculate_separation_score(feature_values)
            if separation_score > 0:
                distinctive_features.append({
                    'feature': feature,
                    'separation_score': separation_score,
                    'author_values': feature_values,
                    'feature_type': feature_type
                })
        
        # Sort by separation score
        distinctive_features.sort(key=lambda x: x['separation_score'], reverse=True)
        return distinctive_features[:50]  # Top 50 most distinctive
    
    def calculate_separation_score(self, feature_values):
        """Calculate how well a feature separates authors."""
        values = list(feature_values.values())
        
        if len(set(values)) <= 1:
            return 0  # No variation
        
        # Calculate coefficient of variation (std/mean)
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0
        
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        cv = std_val / abs(mean_val)
        
        # Bonus for having distinct ranges
        sorted_values = sorted(values)
        range_gaps = []
        for i in range(1, len(sorted_values)):
            gap = sorted_values[i] - sorted_values[i-1]
            range_gaps.append(gap)
        
        avg_gap = statistics.mean(range_gaps) if range_gaps else 0
        gap_bonus = avg_gap / (max(values) - min(values) + 1e-10)
        
        return cv + gap_bonus
    
    def test_perfect_discrimination(self):
        """Test if any single feature can perfectly discriminate all authors."""
        print("\nTesting for perfect discrimination...")
        
        perfect_features = []
        
        for result_type, result_data in self.results.items():
            if 'distinctive_features' in result_data:
                for feature_info in result_data['distinctive_features']:
                    author_values = feature_info['author_values']
                    
                    # Check if all authors have unique values for this feature
                    values = list(author_values.values())
                    if len(set(values)) == len(values):  # All unique
                        perfect_features.append({
                            'feature_type': result_type,
                            'feature': feature_info['feature'],
                            'author_values': author_values,
                            'separation_score': feature_info['separation_score']
                        })
        
        if perfect_features:
            print(f"Found {len(perfect_features)} features with perfect discrimination!")
            return perfect_features
        else:
            print("No single feature provides perfect discrimination.")
            # Find the best combinations
            return self.find_best_feature_combinations()
    
    def find_best_feature_combinations(self):
        """Find combinations of features that can discriminate all authors."""
        print("Looking for feature combinations...")
        
        # Get top features from each category
        top_features = []
        for result_type, result_data in self.results.items():
            if 'distinctive_features' in result_data and result_data['distinctive_features']:
                # Take top 5 from each category
                for feature_info in result_data['distinctive_features'][:5]:
                    top_features.append({
                        'type': result_type,
                        'feature': feature_info['feature'],
                        'values': feature_info['author_values'],
                        'score': feature_info['separation_score']
                    })
        
        # Test combinations of 2 features
        best_combinations = []
        for feat1, feat2 in combinations(top_features, 2):
            discrimination_rate = self.test_feature_combination([feat1, feat2])
            if discrimination_rate > 0.8:  # 80% or better discrimination
                best_combinations.append({
                    'features': [feat1, feat2],
                    'discrimination_rate': discrimination_rate
                })
        
        best_combinations.sort(key=lambda x: x['discrimination_rate'], reverse=True)
        return best_combinations[:10]  # Top 10 combinations
    
    def test_feature_combination(self, features):
        """Test how well a combination of features discriminates authors."""
        authors = list(self.author_texts.keys())
        n_authors = len(authors)
        
        # Create feature vectors for each author
        author_vectors = {}
        for author in authors:
            vector = []
            for feature in features:
                value = feature['values'].get(author, 0)
                vector.append(value)
            author_vectors[author] = vector
        
        # Count how many author pairs can be distinguished
        distinguishable_pairs = 0
        total_pairs = 0
        
        for i, author1 in enumerate(authors):
            for j, author2 in enumerate(authors[i+1:], i+1):
                total_pairs += 1
                vec1 = author_vectors[author1]
                vec2 = author_vectors[author2]
                
                # Check if vectors are different
                if vec1 != vec2:
                    distinguishable_pairs += 1
        
        return distinguishable_pairs / total_pairs if total_pairs > 0 else 0
    
    def generate_visualizations(self):
        """Generate visualizations of the analysis results."""
        print("\nGenerating visualizations...")
        
        results_dir = self.base_path / "results"
        
        # 1. Author word count distribution
        plt.figure(figsize=(12, 6))
        authors = list(self.authors.keys())
        word_counts = [self.authors[author]['total_words'] for author in authors]
        
        plt.bar(range(len(authors)), word_counts)
        plt.xticks(range(len(authors)), authors, rotation=45, ha='right')
        plt.ylabel('Total Word Count')
        plt.title('Word Count Distribution by Author')
        plt.tight_layout()
        plt.savefig(results_dir / 'word_count_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top distinctive features heatmap
        if self.results:
            self.create_feature_heatmap(results_dir)
        
        # 3. Word length distribution comparison
        self.create_word_length_comparison(results_dir)
        
        print(f"Visualizations saved to {results_dir}")
    
    def create_feature_heatmap(self, results_dir):
        """Create a heatmap of top distinctive features."""
        # Collect top features from each category
        all_top_features = []
        
        for result_type, result_data in self.results.items():
            if 'distinctive_features' in result_data and result_data['distinctive_features']:
                # Take top 3 from each category
                for i, feature_info in enumerate(result_data['distinctive_features'][:3]):
                    all_top_features.append({
                        'name': f"{result_type}_{feature_info['feature']}"[:50],  # Truncate long names
                        'values': feature_info['author_values']
                    })
        
        if not all_top_features:
            return
        
        # Create matrix
        authors = list(self.author_texts.keys())
        feature_names = [f['name'] for f in all_top_features]
        
        matrix = []
        for feature in all_top_features:
            row = [feature['values'].get(author, 0) for author in authors]
            matrix.append(row)
        
        # Normalize each row (feature) to 0-1 scale
        normalized_matrix = []
        for row in matrix:
            min_val, max_val = min(row), max(row)
            if max_val > min_val:
                normalized_row = [(x - min_val) / (max_val - min_val) for x in row]
            else:
                normalized_row = [0] * len(row)
            normalized_matrix.append(normalized_row)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(normalized_matrix, 
                   xticklabels=authors, 
                   yticklabels=feature_names,
                   cmap='viridis', 
                   annot=False)
        plt.title('Top Distinctive Features by Author (Normalized)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(results_dir / 'distinctive_features_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_word_length_comparison(self, results_dir):
        """Create word length distribution comparison."""
        if 'word_lengths' not in self.results:
            return
        
        authors = list(self.author_texts.keys())
        
        # Extract average word lengths
        avg_lengths = []
        for author in authors:
            if author in self.results['word_lengths']['author_features']:
                avg_len = self.results['word_lengths']['author_features'][author].get('avg_length', 0)
                avg_lengths.append(avg_len)
            else:
                avg_lengths.append(0)
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(authors)), avg_lengths)
        plt.xticks(range(len(authors)), authors, rotation=45, ha='right')
        plt.ylabel('Average Word Length (characters)')
        plt.title('Average Word Length by Author')
        plt.tight_layout()
        plt.savefig(results_dir / 'word_length_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save detailed results to files."""
        print("\nSaving results...")
        
        results_dir = self.base_path / "results"
        docs_dir = self.base_path / "results documentation"
        
        # Save raw results as pickle for further analysis
        with open(results_dir / 'raw_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save author information
        with open(results_dir / 'author_info.json', 'w', encoding='utf-8') as f:
            json.dump(self.authors, f, indent=2, ensure_ascii=False, default=str)
        
        # Test for perfect discrimination
        perfect_or_best = self.test_perfect_discrimination()
        
        # Save the best discriminative features
        with open(results_dir / 'best_discriminative_features.json', 'w', encoding='utf-8') as f:
            json.dump(perfect_or_best, f, indent=2, ensure_ascii=False, default=str)
        
        # Create summary report
        self.create_summary_report(docs_dir, perfect_or_best)
        
        print(f"Results saved to {results_dir}")
        print(f"Documentation saved to {docs_dir}")
    
    def create_summary_report(self, docs_dir, discriminative_features):
        """Create a comprehensive summary report."""
        report_path = docs_dir / 'analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Greek Authorship Attribution Analysis Report\n\n")
            f.write("## Overview\n\n")
            f.write(f"This analysis examined ancient Greek texts from {len(self.authors)} authors ")
            f.write(f"who had more than {self.word_count_threshold} words in their collected works.\n\n")
            
            # Author summary
            f.write("## Included Authors\n\n")
            for author, info in sorted(self.authors.items()):
                f.write(f"- **{author}**: {info['total_words']:,} words across {info['num_texts']} texts\n")
            
            # Analysis methods
            f.write("\n## Analysis Methods\n\n")
            f.write("The following authentic linguistic features were analyzed (excluding modern punctuation):\n\n")
            f.write("1. **Character n-grams**: 2, 3, and 4-character sequences\n")
            f.write("2. **Word n-grams**: 2 and 3-word sequences\n")
            f.write("3. **Word frequencies**: Individual word usage patterns\n")
            f.write("4. **Morphological patterns**: Greek case endings, particles, verb forms\n")
            f.write("5. **Word length distributions**: Statistics about word lengths\n")
            f.write("6. **Phonetic patterns**: Vowel frequencies, consonant clusters, diphthongs\n")
            f.write("7. **Vocabulary richness**: Type-token ratios, hapax legomena, lexical diversity\n\n")
            
            # Results
            f.write("## Key Findings\n\n")
            
            if discriminative_features and len(discriminative_features) > 0:
                if isinstance(discriminative_features[0], dict) and 'feature_type' in discriminative_features[0]:
                    # Perfect discrimination found
                    f.write("### Perfect Discrimination Found! 🎉\n\n")
                    f.write("The following features can perfectly distinguish between ALL authors:\n\n")
                    
                    for i, feature in enumerate(discriminative_features[:5], 1):
                        f.write(f"#### {i}. {feature['feature_type']}: {feature['feature']}\n\n")
                        f.write("Author values:\n")
                        for author, value in sorted(feature['author_values'].items()):
                            f.write(f"- {author}: {value:.6f}\n")
                        f.write(f"\nSeparation score: {feature.get('separation_score', 'N/A'):.4f}\n\n")
                
                else:
                    # Feature combinations
                    f.write("### Best Feature Combinations\n\n")
                    f.write("No single feature provides perfect discrimination, but these combinations work well:\n\n")
                    
                    for i, combo in enumerate(discriminative_features[:3], 1):
                        f.write(f"#### Combination {i} (Discrimination rate: {combo['discrimination_rate']:.2%})\n\n")
                        for feature in combo['features']:
                            f.write(f"- **{feature['type']}**: {feature['feature']} (score: {feature['score']:.4f})\n")
                        f.write("\n")
            
            # Feature statistics
            f.write("## Feature Analysis Summary\n\n")
            for feature_type, data in self.results.items():
                if 'distinctive_features' in data:
                    top_features = data['distinctive_features'][:3]
                    f.write(f"### {feature_type.replace('_', ' ').title()}\n\n")
                    f.write(f"Top 3 most distinctive features:\n")
                    for feature in top_features:
                        f.write(f"1. **{feature['feature']}** (separation score: {feature['separation_score']:.4f})\n")
                    f.write("\n")
            
            # Methodology
            f.write("## Methodology Notes\n\n")
            f.write("- **Text preprocessing**: Greek text was normalized, non-Greek characters removed\n")
            f.write("- **Minimum threshold**: Only authors with >1000 words included\n")
            f.write("- **Separation scoring**: Based on coefficient of variation and value distribution gaps\n")
            f.write("- **Perfect discrimination**: A feature where every author has a unique value\n")
            f.write("- **Feature combinations**: Tested pairs of top features for discrimination capability\n\n")
            
            f.write("## Files Generated\n\n")
            f.write("- `raw_results.pkl`: Complete analysis results for further processing\n")
            f.write("- `author_info.json`: Author statistics and text information\n")
            f.write("- `best_discriminative_features.json`: Top discriminative features or combinations\n")
            f.write("- `word_count_distribution.png`: Visualization of word counts by author\n")
            f.write("- `distinctive_features_heatmap.png`: Heatmap of top features across authors\n")
            f.write("- `word_length_comparison.png`: Average word length comparison\n")
    
    def run_complete_analysis(self):
        """Run the complete authorship attribution analysis."""
        print("=== Greek Authorship Attribution Analysis ===")
        print("Goal: Find features that can differentiate ALL authors")
        print("=" * 50)
        
        # Load and filter texts
        self.load_texts()
        
        if len(self.authors) < 2:
            print("Need at least 2 authors for analysis!")
            return
        
        # Run all analysis methods (excluding modern punctuation)
        self.analyze_character_ngrams()
        self.analyze_word_ngrams()
        self.analyze_word_frequencies()
        self.analyze_morphological_patterns()
        self.analyze_word_length_distribution()
        self.analyze_phonetic_patterns()
        self.analyze_vocabulary_richness()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save all results
        self.save_results()
        
        print("\n" + "=" * 50)
        print("Analysis complete! Check the 'results' and 'results documentation' folders.")
        print("=" * 50)

def main():
    # Set up the analyzer
    base_path = "/home/user/Downloads/100BC to 100AD"
    analyzer = GreekTextAnalyzer(base_path)
    
    # Run the complete analysis
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
