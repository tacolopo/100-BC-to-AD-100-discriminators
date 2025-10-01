import os
import json
import re
import statistics
import numpy as np
from pathlib import Path
from collections import Counter
class PaulineAuthorshipAnalyzer:
    def __init__(self):
        self.load_reference_data()
    def load_reference_data(self):
        print("Loading reference data...")
        with open('results/best_discriminative_features.json', 'r', encoding='utf-8') as f:
            reference_features = json.load(f)
        self.key_features = []
        for item in reference_features:
            if item['separation_score'] > 0:
                self.key_features.append(item)
        print(f"Using {len(self.key_features)} discriminative features (all perfect discriminators)")
        self.inter_author_stats = {}
        for feature in self.key_features:
            values = list(feature['author_values'].values())
            self.inter_author_stats[feature['feature']] = {
                'mean': statistics.mean(values),
                'variance': statistics.variance(values),
                'std_dev': statistics.stdev(values),
                'range': max(values) - min(values),
                'separation_score': feature['separation_score']
            }
    def clean_text(self, text):
        greek_pattern = r'[^\u0370-\u03FF\u1F00-\u1FFF\s]'
        text = re.sub(greek_pattern, '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    def extract_all_features(self, text):
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        total_words = len(words)
        total_chars = len(cleaned_text.replace(' ', ''))
        if total_words == 0:
            return {}
        features = {}
        particles = {
            'particle_δέ_freq': ['δε', 'δέ'],
            'particle_τε_freq': ['τε', 'τέ'], 
            'particle_μέν_freq': ['μεν', 'μέν'],
            'particle_γάρ_freq': ['γαρ', 'γάρ'],
            'particle_οῦν_freq': ['ουν', 'οῦν'],
            'particle_ἄν_freq': ['αν', 'ἄν'],
            'particle_εἰ_freq': ['ει', 'εἰ'],
            'particle_ἀλλά_freq': ['αλλα', 'ἀλλά'],
            'particle_ἔτι_freq': ['ετι', 'ἔτι']
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
            vowels = ['α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω']
            for vowel in vowels:
                features[f'vowel_{vowel}_freq'] = cleaned_text.count(vowel) / total_chars
            clusters = ['στ', 'σκ', 'σπ', 'σφ', 'σχ', 'κτ', 'πτ', 'φθ', 'χθ', 'νθ', 'μπ', 'ντ']
            for cluster in clusters:
                features[f'cluster_{cluster}_freq'] = cleaned_text.count(cluster) / total_chars
            diphthongs = ['αι', 'ει', 'οι', 'υι', 'αυ', 'ευ', 'ου', 'ηυ']
            for diphthong in diphthongs:
                features[f'diphthong_{diphthong}_freq'] = cleaned_text.count(diphthong) / total_chars
        if total_words > 0:
            unique_words = len(set(words))
            features['ttr'] = unique_words / total_words
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
        text_no_spaces = cleaned_text.replace(' ', '')
        if len(text_no_spaces) >= 2:
            bigrams = [text_no_spaces[i:i+2] for i in range(len(text_no_spaces)-1)]
            bigram_freq = Counter(bigrams)
            total_bigrams = len(bigrams)
            for bigram, count in bigram_freq.most_common(50):
                features[f'2gram_{bigram}'] = count / total_bigrams
        if len(text_no_spaces) >= 3:
            trigrams = [text_no_spaces[i:i+3] for i in range(len(text_no_spaces)-2)]
            trigram_freq = Counter(trigrams)
            total_trigrams = len(trigrams)
            for trigram, count in trigram_freq.most_common(50):
                features[f'3gram_{trigram}'] = count / total_trigrams
        if len(text_no_spaces) >= 4:
            fourgrams = [text_no_spaces[i:i+4] for i in range(len(text_no_spaces)-3)]
            fourgram_freq = Counter(fourgrams)
            total_fourgrams = len(fourgrams)
            for fourgram, count in fourgram_freq.most_common(30):
                features[f'4gram_{fourgram}'] = count / total_fourgrams
        if len(words) >= 2:
            word_bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            word_bigram_freq = Counter(word_bigrams)
            total_word_bigrams = len(word_bigrams)
            for bigram, count in word_bigram_freq.most_common(30):
                features[f'word_2gram_{bigram}'] = count / total_word_bigrams
        if len(words) >= 3:
            word_trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
            word_trigram_freq = Counter(word_trigrams)
            total_word_trigrams = len(word_trigrams)
            for trigram, count in word_trigram_freq.most_common(20):
                features[f'word_3gram_{trigram}'] = count / total_word_trigrams
        return features
    def analyze_pauline_corpus(self, corpus_directory):
        print(f"\n=== Pauline Letters Authorship Analysis ===")
        print(f"Analyzing: {corpus_directory}")
        corpus_path = Path(corpus_directory)
        if not corpus_path.exists():
            print(f"Error: Directory {corpus_directory} does not exist")
            return
        letter_files = list(corpus_path.glob("*.txt"))
        if not letter_files:
            print(f"Error: No .txt files found in {corpus_directory}")
            return
        print(f"Found {len(letter_files)} letters")
        letter_features = {}
        letter_names = []
        for letter_file in letter_files:
            print(f"Processing: {letter_file.name}")
            try:
                with open(letter_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                word_count = len(text.split())
                print(f"  Word count: {word_count}")
                if word_count < 50:
                    print(f"  Warning: Very short letter ({word_count} words)")
                features = self.extract_all_features(text)
                letter_name = letter_file.stem
                letter_features[letter_name] = features
                letter_names.append(letter_name)
            except Exception as e:
                print(f"  Error reading {letter_file.name}: {e}")
        if len(letter_features) < 2:
            print("Error: Need at least 2 letters for comparison")
            return
        return self.calculate_authorship_probability(letter_features, letter_names)
    def calculate_authorship_probability(self, letter_features, letter_names):
        print(f"\nAnalyzing consistency across {len(letter_names)} letters...")
        feature_analysis = {}
        consistency_scores = []
        for feature_info in self.key_features:
            feature_name = feature_info['feature']
            feature_type = feature_info['feature_type']
            if feature_name not in self.inter_author_stats:
                continue
            values = []
            for letter_name, features in letter_features.items():
                if feature_name in features:
                    values.append(features[feature_name])
                elif feature_name.replace('_freq', '') in features:
                    values.append(features[feature_name.replace('_freq', '')])
                elif f'{feature_name}_freq' in features:
                    values.append(features[f'{feature_name}_freq'])
                else:
                    values.append(0.0)
            if len(values) < 2:
                continue
            pauline_variance = statistics.variance(values) if len(values) > 1 else 0
            pauline_mean = statistics.mean(values)
            pauline_range = max(values) - min(values)
            inter_author_variance = self.inter_author_stats[feature_name]['variance']
            inter_author_range = self.inter_author_stats[feature_name]['range']
            separation_score = self.inter_author_stats[feature_name]['separation_score']
            if inter_author_variance > 0:
                variance_ratio = pauline_variance / inter_author_variance
                range_ratio = pauline_range / inter_author_range
                combined_ratio = (variance_ratio + range_ratio) / 2
                consistency = max(0, 1 - combined_ratio)
                weighted_consistency = consistency * separation_score
                consistency_scores.append(weighted_consistency)
                feature_analysis[feature_name] = {
                    'pauline_values': values,
                    'pauline_variance': pauline_variance,
                    'pauline_mean': pauline_mean,
                    'pauline_range': pauline_range,
                    'inter_author_variance': inter_author_variance,
                    'inter_author_range': inter_author_range,
                    'variance_ratio': variance_ratio,
                    'range_ratio': range_ratio,
                    'consistency_score': consistency,
                    'weighted_consistency': weighted_consistency,
                    'separation_score': separation_score
                }
        if consistency_scores:
            overall_consistency = statistics.mean(consistency_scores)
            probability = max(0.0, min(1.0, overall_consistency))
        else:
            probability = 0.5
        if probability >= 0.8:
            interpretation = "Strong evidence for single authorship"
        elif probability >= 0.6:
            interpretation = "Moderate evidence for single authorship" 
        elif probability >= 0.4:
            interpretation = "Mixed evidence - uncertain"
        elif probability >= 0.2:
            interpretation = "Moderate evidence for multiple authorship"
        else:
            interpretation = "Strong evidence for multiple authorship"
        self.generate_report(probability, interpretation, feature_analysis, letter_names)
        return probability, interpretation
    def generate_report(self, probability, interpretation, feature_analysis, letter_names):
        print(f"\n{'='*60}")
        print(f"PAULINE LETTERS AUTHORSHIP ANALYSIS")
        print(f"{'='*60}")
        print(f"")
        print(f"Single Authorship Probability: {probability:.1%}")
        print(f"Interpretation: {interpretation}")
        print(f"")
        print(f"Letters analyzed: {len(letter_names)}")
        for name in letter_names:
            print(f"  • {name}")
        print(f"")
        sorted_features = sorted(feature_analysis.items(), 
                               key=lambda x: x[1]['weighted_consistency'], 
                               reverse=True)
        print(f"EVIDENCE FOR SINGLE AUTHORSHIP:")
        for feature_name, analysis in sorted_features[:3]:
            print(f"  • {feature_name}")
            print(f"    Consistency: {analysis['consistency_score']:.1%}")
            print(f"    Pauline variance: {analysis['pauline_variance']:.6f}")
            print(f"    Inter-author variance: {analysis['inter_author_variance']:.6f}")
            print(f"    Variance ratio: {analysis['variance_ratio']:.2f} (lower = more consistent)")
        print(f"")
        print(f"EVIDENCE AGAINST SINGLE AUTHORSHIP:")
        for feature_name, analysis in sorted_features[-3:]:
            print(f"  • {feature_name}")
            print(f"    Consistency: {analysis['consistency_score']:.1%}")
            print(f"    Pauline variance: {analysis['pauline_variance']:.6f}")
            print(f"    Inter-author variance: {analysis['inter_author_variance']:.6f}")
            print(f"    Variance ratio: {analysis['variance_ratio']:.2f} (higher = less consistent)")
        print(f"")
        print(f"METHODOLOGY:")
        print(f"- Used top 10 discriminative morphological features from ancient Greek analysis")
        print(f"- Compared variance within Pauline corpus to variance between different authors")
        print(f"- Lower variance within Pauline corpus = evidence for single authorship")
        print(f"- Higher variance within Pauline corpus = evidence for multiple authorship")
        print(f"")
        print(f"{'='*60}")
        results = {
            'probability': probability,
            'interpretation': interpretation,
            'letter_count': len(letter_names),
            'letters': letter_names,
            'feature_analysis': feature_analysis,
            'methodology': {
                'features_used': len(feature_analysis),
                'comparison_basis': 'Variance within Pauline corpus vs. variance between 33 ancient Greek authors',
                'key_principle': 'Single author should show lower variance than multiple authors'
            }
        }
        output_file = "pauline_authorship_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed results saved to: {output_file}")
def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pauline_authorship_analyzer.py <pauline_letters_directory>")
        print("")
        print("The directory should contain the 14 Pauline letters as .txt files:")
        print("  Romans.txt, 1_Corinthians.txt, 2_Corinthians.txt, etc.")
        return
    corpus_directory = sys.argv[1]
    analyzer = PaulineAuthorshipAnalyzer()
    analyzer.analyze_pauline_corpus(corpus_directory)
if __name__ == "__main__":
    main()