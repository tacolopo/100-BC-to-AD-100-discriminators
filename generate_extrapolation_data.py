#!/usr/bin/env python3

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import statistics

def load_results():
    print("Loading analysis results...")
    
    with open('results/raw_results.pkl', 'rb') as f:
        raw_results = pickle.load(f)
    
    with open('results/best_discriminative_features.json', 'r', encoding='utf-8') as f:
        discriminative_features = json.load(f)
    
    return raw_results, discriminative_features

def calculate_feature_statistics(feature_data):
    values = list(feature_data.values())
    
    stats = {
        'count': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
        'variance': statistics.variance(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'range': max(values) - min(values),
        'quartiles': {
            'q1': np.percentile(values, 25),
            'q2': np.percentile(values, 50),
            'q3': np.percentile(values, 75)
        },
        'percentiles': {f'p{p}': np.percentile(values, p) for p in [5, 10, 25, 50, 75, 90, 95]},
        'outlier_bounds': {
            'lower': np.percentile(values, 25) - 1.5 * (np.percentile(values, 75) - np.percentile(values, 25)),
            'upper': np.percentile(values, 75) + 1.5 * (np.percentile(values, 75) - np.percentile(values, 25))
        }
    }
    
    return stats

def calculate_separation_score_detailed(feature_data):
    values = sorted(feature_data.values())
    authors = list(feature_data.keys())
    
    gaps = []
    for i in range(len(values) - 1):
        gap = values[i + 1] - values[i]
        gaps.append(gap)
    
    min_gap = min(gaps) if gaps else 0
    max_gap = max(gaps) if gaps else 0
    feature_range = values[-1] - values[0] if len(values) > 1 else 0
    
    separation_score = min_gap / feature_range if feature_range > 0 else 0
    
    return {
        'separation_score': separation_score,
        'min_gap': min_gap,
        'max_gap': max_gap,
        'avg_gap': statistics.mean(gaps) if gaps else 0,
        'feature_range': feature_range,
        'gap_consistency': statistics.stdev(gaps) if len(gaps) > 1 else 0,
        'total_unique_values': len(set(values))
    }

def generate_author_profiles(raw_results, discriminative_features):
    print("Generating author profiles...")
    
    author_profiles = {}
    
    all_features = {}
    for feature_item in discriminative_features:
        feature_key = f"{feature_item['feature_type']}:{feature_item['feature']}"
        all_features[feature_key] = feature_item['author_values']
    
    feature_categories = ['character_ngrams', 'word_ngrams', 'word_frequencies', 
                         'morphological', 'word_lengths', 'phonetic', 'vocabulary']
    
    for category in feature_categories:
        if category in raw_results:
            category_data = raw_results[category]
            if 'author_features' in category_data:
                for author, features in category_data['author_features'].items():
                    if author not in author_profiles:
                        author_profiles[author] = {}
                    for feature_name, value in features.items():
                        feature_key = f"{category}:{feature_name}"
                        if feature_key not in all_features:
                            all_features[feature_key] = {}
                        all_features[feature_key][author] = value
    
    for author in raw_results.get('authors', {}):
        profile = {
            'metadata': raw_results['authors'][author],
            'features': {},
            'rankings': {},
            'percentiles': {},
            'confidence_indicators': {}
        }
        
        for feature_key, feature_data in all_features.items():
            if author in feature_data:
                value = feature_data[author]
                
                sorted_values = sorted(feature_data.values())
                rank = sorted_values.index(value) + 1
                
                percentile = (rank - 1) / (len(sorted_values) - 1) * 100 if len(sorted_values) > 1 else 50
                
                profile['features'][feature_key] = value
                profile['rankings'][feature_key] = rank
                profile['percentiles'][feature_key] = percentile
        
        author_profiles[author] = profile
    
    return author_profiles, all_features

def generate_feature_metadata(all_features, discriminative_features):
    print("Generating feature metadata...")
    
    feature_metadata = {}
    
    discriminative_lookup = {}
    for item in discriminative_features:
        key = f"{item['feature_type']}:{item['feature']}"
        discriminative_lookup[key] = item
    
    for feature_key, feature_data in all_features.items():
        if len(feature_data) < 2:
            continue
            
        stats = calculate_feature_statistics(feature_data)
        
        separation = calculate_separation_score_detailed(feature_data)
        
        feature_type = feature_key.split(':')[0]
        feature_name = feature_key.split(':', 1)[1]
        
        reliability_score = min(1.0, separation['separation_score'] * 1.2)
        
        is_perfect = separation['total_unique_values'] == len(feature_data)
        
        metadata = {
            'feature_type': feature_type,
            'feature_name': feature_name,
            'full_key': feature_key,
            'statistics': stats,
            'separation_metrics': separation,
            'reliability_score': reliability_score,
            'is_perfect_discriminator': is_perfect,
            'recommendation': classify_feature_utility(separation['separation_score'], is_perfect),
            'author_count': len(feature_data),
            'description': generate_feature_description(feature_type, feature_name)
        }
        
        feature_metadata[feature_key] = metadata
    
    return feature_metadata

def classify_feature_utility(separation_score, is_perfect):
    if not is_perfect:
        return "not_recommended"
    elif separation_score >= 0.8:
        return "primary_discriminator"
    elif separation_score >= 0.6:
        return "strong_discriminator"
    elif separation_score >= 0.4:
        return "moderate_discriminator"
    elif separation_score >= 0.2:
        return "weak_discriminator"
    else:
        return "poor_discriminator"

def generate_feature_description(feature_type, feature_name):
    descriptions = {
        'morphological': {
            'particle_δέ_freq': 'Frequency of δέ particle (connecting/adversative particle)',
            'particle_τε_freq': 'Frequency of τε particle (connecting particle meaning "and")',
            'dative_pl_freq': 'Frequency of dative plural case endings (-οις, -αις, etc.)',
            'genitive_sg_masc_freq': 'Frequency of masculine genitive singular endings (-ου)',
            'accusative_sg_fem_freq': 'Frequency of feminine accusative singular endings (-ην)',
        },
        'phonetic': {
            'vowel_α_freq': 'Frequency of alpha (α) vowel',
            'vowel_ε_freq': 'Frequency of epsilon (ε) vowel',
            'cluster_στ_freq': 'Frequency of sigma-tau (στ) consonant cluster',
            'diphthong_αι_freq': 'Frequency of ai (αι) diphthong',
        },
        'vocabulary': {
            'ttr': 'Type-Token Ratio (lexical diversity measure)',
            'hapax_ratio': 'Hapax legomena ratio (unique words)',
            'avg_word_freq': 'Average word frequency',
        }
    }
    
    if feature_type in descriptions and feature_name in descriptions[feature_type]:
        return descriptions[feature_type][feature_name]
    else:
        return f"{feature_type.title()} feature: {feature_name}"

def create_classification_template():
    template = {
        'classification_workflow': {
            'step_1': 'Extract all 60+ features from unknown text',
            'step_2': 'Calculate distance to each known author profile',
            'step_3': 'Weight distances by feature reliability scores',
            'step_4': 'Compute overall similarity scores',
            'step_5': 'Rank candidates and assess confidence'
        },
        'minimum_requirements': {
            'text_length': 1000,
            'recommended_length': 5000,
            'high_confidence_length': 10000
        },
        'confidence_thresholds': {
            'high_confidence': 0.85,
            'moderate_confidence': 0.70,
            'low_confidence': 0.55,
            'inconclusive': 0.55
        },
        'feature_weighting': {
            'primary_discriminators': 1.0,
            'strong_discriminators': 0.8,
            'moderate_discriminators': 0.6,
            'weak_discriminators': 0.4,
            'poor_discriminators': 0.2
        }
    }
    return template

def main():
    print("=== Extrapolation Data Generator ===")
    
    raw_results, discriminative_features = load_results()
    
    author_profiles, all_features = generate_author_profiles(raw_results, discriminative_features)
    feature_metadata = generate_feature_metadata(all_features, discriminative_features)
    classification_template = create_classification_template()
    
    output_dir = Path('results/extrapolation_data')
    output_dir.mkdir(exist_ok=True)
    
    print("Saving author profiles...")
    with open(output_dir / 'author_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(author_profiles, f, indent=2, ensure_ascii=False)
    
    print("Saving feature metadata...")
    with open(output_dir / 'feature_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(feature_metadata, f, indent=2, ensure_ascii=False)
    
    print("Saving classification template...")
    with open(output_dir / 'classification_template.json', 'w', encoding='utf-8') as f:
        json.dump(classification_template, f, indent=2, ensure_ascii=False)
    
    print("Generating summary report...")
    summary = {
        'total_authors': len(author_profiles),
        'total_features': len(feature_metadata),
        'perfect_discriminators': sum(1 for f in feature_metadata.values() if f['is_perfect_discriminator']),
        'primary_discriminators': sum(1 for f in feature_metadata.values() if f['recommendation'] == 'primary_discriminator'),
        'feature_categories': {},
        'top_features': []
    }
    
    for feature_key, metadata in feature_metadata.items():
        category = metadata['feature_type']
        if category not in summary['feature_categories']:
            summary['feature_categories'][category] = 0
        summary['feature_categories'][category] += 1
    
    sorted_features = sorted(feature_metadata.items(), 
                           key=lambda x: x[1]['separation_metrics']['separation_score'], 
                           reverse=True)
    
    for feature_key, metadata in sorted_features[:20]:
        summary['top_features'].append({
            'feature': feature_key,
            'separation_score': metadata['separation_metrics']['separation_score'],
            'recommendation': metadata['recommendation'],
            'description': metadata['description']
        })
    
    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtrapolation data generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - author_profiles.json ({len(author_profiles)} authors)")
    print(f"  - feature_metadata.json ({len(feature_metadata)} features)")
    print(f"  - classification_template.json")
    print(f"  - summary.json")
    
    print(f"\nSummary statistics:")
    print(f"  - Total features analyzed: {summary['total_features']}")
    print(f"  - Perfect discriminators: {summary['perfect_discriminators']}")
    print(f"  - Primary discriminators: {summary['primary_discriminators']}")
    print(f"  - Feature categories: {list(summary['feature_categories'].keys())}")

if __name__ == "__main__":
    main()

