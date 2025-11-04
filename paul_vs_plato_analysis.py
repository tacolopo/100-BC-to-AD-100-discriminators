#!/usr/bin/env python3

import os
import re
import json
import unicodedata
import statistics
from collections import Counter
from pathlib import Path

print("Initializing CLTK for Greek lemmatization...")
os.environ['CLTK_INTERACTIVE'] = 'FALSE'
from cltk.nlp import NLP
CLTK_NLP = NLP("grc", suppress_banner=False)
try:
    import spacy
    for name, proc in CLTK_NLP.pipeline.processes.items():
        if hasattr(proc, 'nlp') and hasattr(proc.nlp, 'max_length'):
            proc.nlp.max_length = 10000000
except Exception as e:
    print(f"Warning: Could not set max_length: {e}")
print("CLTK initialized.\n")

class PaulPlatoComparison:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.discriminative_features = []
        self.paul_corpus = ""
        self.plato_corpus = ""
        
    def load_discriminative_features(self):
        print("Loading 66 perfect discriminators...")
        with open(self.base_path / 'results' / 'best_discriminative_features.json', 'r', encoding='utf-8') as f:
            self.discriminative_features = json.load(f)
        print(f"Loaded {len(self.discriminative_features)} discriminative features\n")
        
    def lemmatize_text(self, text, filename=""):
        word_count = len(text.split())
        char_count = len(text)
        
        if char_count > 900000:
            print(f"    Lemmatizing {word_count} words from {filename} (chunking large text: {char_count} chars)...")
            words = text.split()
            chunk_size = 100000
            lemmatized_words = []
            
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i+chunk_size]
                chunk_text = ' '.join(chunk_words)
                try:
                    doc = CLTK_NLP.analyze(text=chunk_text)
                    chunk_lemmas = [word.lemma if word.lemma else word.string for word in doc.words]
                    lemmatized_words.extend(chunk_lemmas)
                except Exception as e:
                    lemmatized_words.extend(chunk_words)
            
            print(f"    Lemmatization complete for {filename}")
            return ' '.join(lemmatized_words)
        else:
            print(f"    Lemmatizing {word_count} words from {filename}...")
            try:
                doc = CLTK_NLP.analyze(text=text)
                lemmas = [word.lemma if word.lemma else word.string for word in doc.words]
                print(f"    Lemmatization complete for {filename}")
                return ' '.join(lemmas)
            except Exception as e:
                print(f"    Lemmatization failed for {filename}: {e}, using original text")
                return text
    
    def clean_text(self, text, filename=""):
        text = unicodedata.normalize('NFD', text)
        greek_pattern = r'[^\u0370-\u03FF\u1F00-\u1FFF\u0300-\u036F\s]'
        text = re.sub(greek_pattern, '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.lower()
        text = unicodedata.normalize('NFC', text)
        text = self.lemmatize_text(text, filename)
        return text
    
    def load_paul_corpus(self):
        print("Loading Paul's letters as aggregate corpus...")
        paul_dir = self.base_path / 'Paul versus all authors' / 'pauline_letters'
        
        if not paul_dir.exists():
            raise FileNotFoundError(f"Paul letters directory not found at {paul_dir}")
        
        paul_texts = []
        total_words = 0
        
        for text_file in sorted(paul_dir.glob("*.txt")):
            print(f"  Processing {text_file.name}...")
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content = self.clean_text(content, f"Paul/{text_file.name}")
                    if content:
                        words = len(content.split())
                        total_words += words
                        paul_texts.append(content)
                        print(f"  Completed {text_file.name}: {words} words\n")
            except Exception as e:
                print(f"  Error reading {text_file}: {e}\n")
                continue
        
        self.paul_corpus = ' '.join(paul_texts)
        print(f"✓ Paul aggregate corpus: {total_words:,} words from {len(paul_texts)} letters\n")
    
    def load_plato_corpus(self):
        print("Loading Plato's works as aggregate corpus...")
        plato_dir = self.base_path / 'Plato'
        
        if not plato_dir.exists():
            raise FileNotFoundError(f"Plato directory not found at {plato_dir}")
        
        plato_texts = []
        total_words = 0
        
        for text_file in sorted(plato_dir.glob("*.txt")):
            print(f"  Processing {text_file.name}...")
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content = self.clean_text(content, f"Plato/{text_file.name}")
                    if content:
                        words = len(content.split())
                        total_words += words
                        plato_texts.append(content)
                        print(f"  Completed {text_file.name}: {words} words\n")
            except Exception as e:
                print(f"  Error reading {text_file}: {e}\n")
                continue
        
        self.plato_corpus = ' '.join(plato_texts)
        print(f"✓ Plato aggregate corpus: {total_words:,} words from {len(plato_texts)} works\n")
    
    def extract_character_ngrams(self, text, n):
        text_no_spaces = re.sub(r'\s+', '', text)
        ngrams = []
        for i in range(len(text_no_spaces) - n + 1):
            ngrams.append(text_no_spaces[i:i+n])
        return ngrams
    
    def extract_word_ngrams(self, text, n):
        words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    def calculate_mattr(self, text, window_size):
        words = text.split()
        if len(words) < window_size:
            return 0.0
        
        ttr_values = []
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            unique_words = len(set(window))
            ttr = unique_words / window_size
            ttr_values.append(ttr)
        
        return statistics.mean(ttr_values) if ttr_values else 0.0
    
    def extract_all_features(self, text, author_name):
        print(f"Extracting all discriminative features for {author_name}...")
        
        words = text.split()
        total_words = len(words)
        cleaned_text = text
        total_chars = len(re.sub(r'\s+', '', cleaned_text))
        
        if total_words == 0:
            return {}
        
        features = {}
        feature_names = [f['feature'] for f in self.discriminative_features]
        
        particles = {
            'particle_δε_freq': ['δε', 'δέ', 'δὲ'],
            'particle_τε_freq': ['τε', 'τέ', 'τὲ'],
            'particle_μεν_freq': ['μεν', 'μέν', 'μὲν'],
            'particle_γαρ_freq': ['γαρ', 'γάρ', 'γὰρ'],
            'particle_ουν_freq': ['ουν', 'οῦν', 'οὖν'],
            'particle_αν_freq': ['αν', 'ἄν', 'ἂν'],
            'particle_δη_freq': ['δη', 'δή', 'δὴ'],
            'particle_γε_freq': ['γε', 'γέ', 'γὲ'],
        }
        
        for feature_name in feature_names:
            if feature_name.startswith('char_'):
                n = int(feature_name.split('_')[1][0])
                ngrams = self.extract_character_ngrams(cleaned_text, n)
                if ngrams:
                    ngram_counts = Counter(ngrams)
                    target_ngram = feature_name.split('_', 2)[2]
                    features[feature_name] = ngram_counts.get(target_ngram, 0) / len(ngrams)
                else:
                    features[feature_name] = 0.0
                
            elif feature_name.startswith('word_'):
                if feature_name.startswith('word_1gram_'):
                    word = feature_name.split('_', 2)[2]
                    features[feature_name] = words.count(word) / total_words
                elif feature_name.startswith('word_2gram_'):
                    bigram = feature_name.split('_', 2)[2]
                    bigrams = self.extract_word_ngrams(cleaned_text, 2)
                    features[feature_name] = bigrams.count(bigram) / len(bigrams) if bigrams else 0.0
                elif feature_name.startswith('word_3gram_'):
                    trigram = feature_name.split('_', 2)[2]
                    trigrams = self.extract_word_ngrams(cleaned_text, 3)
                    features[feature_name] = trigrams.count(trigram) / len(trigrams) if trigrams else 0.0
                    
            elif feature_name in particles:
                variants = particles[feature_name]
                count = sum(1 for word in words if word in variants)
                features[feature_name] = count / total_words
                    
            elif feature_name.startswith('mattr_'):
                window_size = int(feature_name.split('_')[1])
                features[feature_name] = self.calculate_mattr(cleaned_text, window_size)
                
            elif feature_name == 'avg_word_length' or feature_name == 'avg_length':
                word_lengths = [len(word) for word in words]
                features[feature_name] = statistics.mean(word_lengths) if word_lengths else 0.0
            
            elif feature_name == 'std_length':
                word_lengths = [len(word) for word in words]
                features[feature_name] = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0.0
                
            elif feature_name.startswith('word_length_') or feature_name.startswith('length_'):
                if 'word_length_' in feature_name:
                    length = int(feature_name.split('_')[2])
                else:
                    length = int(feature_name.split('_')[1])
                count = sum(1 for word in words if len(word) == length)
                features[feature_name] = count / total_words
                
            elif feature_name.startswith('vowel_'):
                if feature_name == 'vowel_ratio':
                    vowels = ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω', 'ά', 'έ', 'ή', 'ί', 'ό', 'ύ', 'ώ']
                    vowel_count = sum(cleaned_text.count(vowel) for vowel in vowels)
                    features[feature_name] = vowel_count / total_chars if total_chars > 0 else 0.0
                else:
                    vowel = feature_name.split('_')[1]
                    if vowel in ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω']:
                        features[feature_name] = cleaned_text.count(vowel) / total_chars if total_chars > 0 else 0.0
            
            elif feature_name.startswith('diphthong_'):
                diphthong = feature_name.split('_')[1]
                features[feature_name] = cleaned_text.count(diphthong) / total_chars if total_chars > 0 else 0.0
            
            elif feature_name.startswith('cluster_'):
                cluster = feature_name.split('_')[1]
                features[feature_name] = cleaned_text.count(cluster) / total_chars if total_chars > 0 else 0.0
            
            elif feature_name in ['ttr_simple', 'hapax_ratio', 'dis_ratio', 'top10_ratio', 'avg_word_freq']:
                word_counts = Counter(words)
                unique_words = len(word_counts)
                if feature_name == 'ttr_simple':
                    features[feature_name] = unique_words / total_words if total_words > 0 else 0.0
                elif feature_name == 'hapax_ratio':
                    hapax = sum(1 for count in word_counts.values() if count == 1)
                    features[feature_name] = hapax / unique_words if unique_words > 0 else 0.0
                elif feature_name == 'dis_ratio':
                    dis = sum(1 for count in word_counts.values() if count == 2)
                    features[feature_name] = dis / unique_words if unique_words > 0 else 0.0
                elif feature_name == 'top10_ratio':
                    top10_count = sum(count for word, count in word_counts.most_common(10))
                    features[feature_name] = top10_count / total_words if total_words > 0 else 0.0
                elif feature_name == 'avg_word_freq':
                    features[feature_name] = total_words / unique_words if unique_words > 0 else 0.0
        
        print(f"✓ Extracted {len(features)} features for {author_name}\n")
        return features
    
    def compare_authors(self, paul_features, plato_features):
        print("Comparing Paul vs Plato on discriminative features...")
        print("=" * 80)
        
        separating_features = []
        overlapping_features = []
        
        for feature_data in self.discriminative_features:
            feature_name = feature_data['feature']
            
            if feature_name not in paul_features or feature_name not in plato_features:
                continue
            
            paul_val = paul_features[feature_name]
            plato_val = plato_features[feature_name]
            
            diff = abs(paul_val - plato_val)
            relative_diff = (diff / max(paul_val, plato_val, 0.0001)) * 100
            
            comparison = {
                'feature': feature_name,
                'feature_type': feature_data['feature_type'],
                'paul_value': paul_val,
                'plato_value': plato_val,
                'absolute_diff': diff,
                'relative_diff_percent': relative_diff
            }
            
            if relative_diff > 20:
                separating_features.append(comparison)
            else:
                overlapping_features.append(comparison)
        
        separating_features.sort(key=lambda x: x['relative_diff_percent'], reverse=True)
        overlapping_features.sort(key=lambda x: x['relative_diff_percent'])
        
        print(f"\n{'STRONGLY SEPARATED FEATURES (>20% difference)':^80}")
        print("=" * 80)
        print(f"Found {len(separating_features)} features showing strong separation\n")
        
        for i, comp in enumerate(separating_features[:20], 1):
            print(f"{i}. {comp['feature']} ({comp['feature_type']})")
            print(f"   Paul:  {comp['paul_value']:.6f}")
            print(f"   Plato: {comp['plato_value']:.6f}")
            print(f"   Difference: {comp['relative_diff_percent']:.1f}%\n")
        
        if len(separating_features) > 20:
            print(f"... and {len(separating_features) - 20} more separated features\n")
        
        print(f"\n{'OVERLAPPING FEATURES (<20% difference)':^80}")
        print("=" * 80)
        print(f"Found {len(overlapping_features)} features showing overlap\n")
        
        for i, comp in enumerate(overlapping_features[:10], 1):
            print(f"{i}. {comp['feature']} ({comp['feature_type']})")
            print(f"   Paul:  {comp['paul_value']:.6f}")
            print(f"   Plato: {comp['plato_value']:.6f}")
            print(f"   Difference: {comp['relative_diff_percent']:.1f}%\n")
        
        total_features = len(separating_features) + len(overlapping_features)
        separation_rate = (len(separating_features) / total_features * 100) if total_features > 0 else 0
        
        print("\n" + "=" * 80)
        print(f"{'SUMMARY':^80}")
        print("=" * 80)
        print(f"Total discriminative features compared: {total_features}")
        print(f"Features showing strong separation (>20%): {len(separating_features)}")
        print(f"Features showing overlap (<20%): {len(overlapping_features)}")
        print(f"Separation rate: {separation_rate:.1f}%")
        print("\n" + "=" * 80)
        
        if separation_rate >= 80:
            print("CONCLUSION: Paul and Plato are CLEARLY SEPARATED by these discriminators ✓")
        elif separation_rate >= 60:
            print("CONCLUSION: Paul and Plato are MODERATELY SEPARATED by these discriminators ~")
        else:
            print("CONCLUSION: Paul and Plato show SIGNIFICANT OVERLAP in these discriminators ✗")
        print("=" * 80)
        
        results = {
            'total_features': total_features,
            'separated_count': len(separating_features),
            'overlapping_count': len(overlapping_features),
            'separation_rate_percent': separation_rate,
            'separated_features': separating_features,
            'overlapping_features': overlapping_features
        }
        
        output_file = self.base_path / 'paul_vs_plato_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        return results

def main():
    import sys
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])
    else:
        base_path = Path(__file__).parent
    
    print(f"Using base path: {base_path}\n")
    
    analyzer = PaulPlatoComparison(base_path)
    analyzer.load_discriminative_features()
    analyzer.load_paul_corpus()
    analyzer.load_plato_corpus()
    
    paul_features = analyzer.extract_all_features(analyzer.paul_corpus, "Paul")
    plato_features = analyzer.extract_all_features(analyzer.plato_corpus, "Plato")
    
    analyzer.compare_authors(paul_features, plato_features)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
