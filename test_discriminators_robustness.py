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

class DiscriminatorRobustnessTest:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.original_discriminators = []
        self.all_author_features = {}
        
    def load_saved_features(self):
        print("Loading saved features for 32 original authors...")
        features_file = self.base_path / 'all_author_features.json'
        
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found. Run extract_all_author_features.py first!")
        
        with open(features_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for author_name, author_data in data.items():
            self.all_author_features[author_name] = author_data['features']
        
        print(f"✓ Loaded features for {len(self.all_author_features)} authors")
        print(f"  Authors: {', '.join(sorted(self.all_author_features.keys()))}\n")
        
    def load_discriminators(self):
        print("Loading 66 perfect discriminators...")
        with open(self.base_path / 'results' / 'best_discriminative_features.json', 'r', encoding='utf-8') as f:
            self.original_discriminators = json.load(f)
        print(f"✓ Loaded {len(self.original_discriminators)} discriminators\n")
        
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
    
    def calculate_mattr(self, words, window_size=100):
        if len(words) < window_size:
            return 0
        
        ttr_values = []
        for i in range(len(words) - window_size + 1):
            window = words[i:i + window_size]
            unique_words = len(set(window))
            ttr = unique_words / window_size
            ttr_values.append(ttr)
        
        return statistics.mean(ttr_values) if ttr_values else 0
    
    def extract_all_features(self, text):
        words = text.split()
        total_words = len(words)
        total_chars = len(text.replace(' ', ''))
        
        if total_words == 0:
            return {}
            
        features = {}
        
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
        
        if total_chars > 0:
            vowels = ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω', 'ά', 'έ', 'ή', 'ί', 'ό', 'ύ', 'ώ']
            vowel_count = sum(text.count(vowel) for vowel in vowels)
            features['vowel_ratio'] = vowel_count / total_chars
            
            for vowel in ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω']:
                features[f'vowel_{vowel}_freq'] = text.count(vowel) / total_chars
            
            clusters = ['στ', 'σκ', 'σπ', 'σφ', 'σχ', 'κτ', 'πτ', 'φθ', 'χθ', 'νθ', 'μπ', 'ντ']
            for cluster in clusters:
                features[f'cluster_{cluster}_freq'] = text.count(cluster) / total_chars
            
            diphthongs = ['αι', 'ει', 'οι', 'υι', 'αυ', 'ευ', 'ου', 'ηυ']
            for diphthong in diphthongs:
                features[f'diphthong_{diphthong}_freq'] = text.count(diphthong) / total_chars
        
        if total_words > 0:
            unique_words = len(set(words))
            features['ttr_simple'] = unique_words / total_words
            
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
    
    def load_and_process_plato(self):
        print("Loading and processing Plato's complete works...")
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
                        print(f"  ✓ {text_file.name}: {words} words\n")
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                continue
        
        plato_corpus = ' '.join(plato_texts)
        print(f"✓ Plato aggregate corpus: {total_words:,} words from {len(plato_texts)} works\n")
        
        print("Extracting all features for Plato...\n")
        plato_features = self.extract_all_features(plato_corpus)
        print(f"✓ Extracted {len(plato_features)} features for Plato\n")
        
        self.all_author_features['Plato'] = plato_features
        
    def test_discriminators(self):
        print("="*80)
        print("TESTING DISCRIMINATORS WITH PLATO AS 33RD AUTHOR")
        print("="*80)
        print()
        
        surviving_discriminators = []
        broken_discriminators = []
        
        for disc in self.original_discriminators:
            feature_name = disc['feature']
            feature_type = disc['feature_type']
            
            values = []
            for author_name, features in self.all_author_features.items():
                if feature_name in features:
                    values.append((author_name, features[feature_name]))
            
            if len(values) != 33:
                print(f"⚠ Warning: {feature_name} missing for some authors, skipping")
                continue
            
            unique_values = set(val for _, val in values)
            is_still_perfect = (len(unique_values) == 33)
            
            result = {
                'feature': feature_name,
                'feature_type': feature_type,
                'unique_values_count': len(unique_values),
                'still_perfect_discriminator': is_still_perfect,
                'plato_value': next(val for author, val in values if author == 'Plato')
            }
            
            if is_still_perfect:
                surviving_discriminators.append(result)
            else:
                duplicate_values = [val for val in unique_values 
                                   if sum(1 for _, v in values if v == val) > 1]
                result['duplicate_values'] = duplicate_values[:5]
                broken_discriminators.append(result)
        
        total_tested = len(surviving_discriminators) + len(broken_discriminators)
        survival_rate = (len(surviving_discriminators) / total_tested * 100) if total_tested > 0 else 0
        
        print(f"\n{'SURVIVING DISCRIMINATORS (all 33 authors have unique values)':^80}")
        print("="*80)
        print(f"{len(surviving_discriminators)} discriminators SURVIVE\n")
        
        for i, result in enumerate(surviving_discriminators[:10], 1):
            print(f"{i}. {result['feature']} ({result['feature_type']})")
            print(f"   All 33 values are unique ✓")
            print(f"   Plato's value: {result['plato_value']:.8f}\n")
        
        if len(surviving_discriminators) > 10:
            print(f"... and {len(surviving_discriminators) - 10} more surviving discriminators\n")
        
        print(f"\n{'BROKEN DISCRIMINATORS (some authors share values)':^80}")
        print("="*80)
        print(f"{len(broken_discriminators)} discriminators BREAK\n")
        
        for i, result in enumerate(broken_discriminators[:10], 1):
            print(f"{i}. {result['feature']} ({result['feature_type']})")
            print(f"   Unique values: {result['unique_values_count']}/33 (collision detected)")
            print(f"   Plato's value: {result['plato_value']:.8f}\n")
        
        if len(broken_discriminators) > 10:
            print(f"... and {len(broken_discriminators) - 10} more broken discriminators\n")
        
        print("\n" + "="*80)
        print(f"{'ROBUSTNESS TEST RESULTS':^80}")
        print("="*80)
        print(f"Original authors: 32")
        print(f"With Plato added: 33 authors")
        print(f"Original perfect discriminators: 66")
        print(f"Discriminators tested: {total_tested}")
        print(f"Discriminators that SURVIVE: {len(surviving_discriminators)}")
        print(f"Discriminators that BREAK: {len(broken_discriminators)}")
        print(f"Survival rate: {survival_rate:.1f}%")
        print("\n" + "="*80)
        
        if survival_rate >= 90:
            print("CONCLUSION: The discriminators are HIGHLY ROBUST ✓✓✓")
        elif survival_rate >= 70:
            print("CONCLUSION: The discriminators are MODERATELY ROBUST ✓✓")
        elif survival_rate >= 50:
            print("CONCLUSION: The discriminators show SOME ROBUSTNESS ✓")
        else:
            print("CONCLUSION: The discriminators were OVERFIT to the original 32 authors ✗")
        print("="*80)
        
        results = {
            'original_authors_count': 32,
            'with_plato_count': 33,
            'original_discriminators': 66,
            'tested_discriminators': total_tested,
            'surviving_count': len(surviving_discriminators),
            'broken_count': len(broken_discriminators),
            'survival_rate_percent': survival_rate,
            'surviving_discriminators': surviving_discriminators,
            'broken_discriminators': broken_discriminators
        }
        
        output_file = self.base_path / 'discriminators_robustness_test.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

def main():
    import sys
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])
    else:
        base_path = Path(__file__).parent
    
    print(f"Using base path: {base_path}\n")
    
    tester = DiscriminatorRobustnessTest(base_path)
    tester.load_saved_features()
    tester.load_discriminators()
    tester.load_and_process_plato()
    tester.test_discriminators()
    
    print("\n✓ Robustness test complete!")

if __name__ == "__main__":
    main()

