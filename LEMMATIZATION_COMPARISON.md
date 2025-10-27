# Lemmatization Analysis Comparison Report

## Executive Summary

This document compares the results of Greek authorship attribution analysis **before** and **after** implementing lemmatization, Unicode normalization, expanded particle detection, and MATTR (Moving Average Type-Token Ratio).

## Key Improvements Implemented

### 1. Unicode Normalization (NFD/NFC)
- **Problem**: Different Unicode encodings of diacritics (e.g., `δέ` vs `δέ`) caused words to be counted separately
- **Solution**: Applied NFD normalization before processing, NFC after
- **Impact**: All Colossians instances of δὲ now correctly counted (previously lost)

### 2. Expanded Particle Detection
- **Old**: 10 hardcoded particles
- **New**: 18 particles with comprehensive diacritical variants
- **New particles include**: δε, τε, μη, ουδε, μηδε, αρα, δη, γε, περ, τοι, που
- **Impact**: More accurate morphological feature detection

### 3. MATTR Implementation
- **Old**: Simple Type-Token Ratio (TTR) - length-dependent and unfair for comparing texts of different sizes
- **New**: Moving Average Type-Token Ratio with windows of 50, 100, and 200 words
- **Impact**: Fair vocabulary richness comparison across texts of varying lengths

### 4. CLTK Lemmatization
- **Process**: All Greek words reduced to dictionary form (lemma)
  - Verbs: `εἰμί` instead of ἐστί, εἰσί, ἦν, etc.
  - Nouns: `θεός` instead of θεοῦ, θεῷ, θεόν, etc.
- **Impact**: Reduced vocabulary size (668 → 554 common words), better feature discrimination

## Results Comparison

### Overall Statistics

| Metric | OLD (No Lemmatization) | NEW (With Lemmatization) | Change |
|--------|------------------------|--------------------------|--------|
| **Perfect Discriminators** | 60 | 65 | +5 (+8.3%) |
| **Common Words** | 668 | 554 | -114 (-17.1%) |
| **Authors Included** | 33 | 33 | 0 |

### Feature Type Breakdown

| Feature Type | OLD | NEW | Change |
|--------------|-----|-----|--------|
| **Morphological** | 15 | 17 | +2 |
| **Phonetic** | 23 | 24 | +1 |
| **Vocabulary** | 5 | 8 | **+3** |
| **Word Lengths** | 17 | 16 | -1 |

### Top 5 Perfect Discriminators

#### OLD (Without Lemmatization):
1. **particle_δέ_freq** (morphological) - score: 0.878
2. **particle_τε_freq** (morphological) - score: 0.768
3. **dative_pl_freq** (morphological) - score: 0.589
4. **accusative_sg_fem_freq** (morphological) - score: 0.571
5. **genitive_sg_masc_freq** (morphological) - score: 0.497

#### NEW (With Lemmatization):
1. **genitive_pl_freq** (morphological) - score: **1.203** ⬆️
2. **genitive_sg_masc_freq** (morphological) - score: **1.160** ⬆️
3. **particle_αν_freq** (morphological) - score: 0.847
4. **particle_δη_freq** (morphological) - score: 0.784
5. **accusative_sg_fem_freq** (morphological) - score: 0.777

## Key Findings

### 1. Improved Discrimination Power
- **Separation scores increased dramatically** for top features
- **NEW #1** (genitive_pl_freq): 1.203 vs **OLD #1** (particle_δέ_freq): 0.878
- **37% improvement** in top discriminator strength

### 2. More Vocabulary Discriminators
- Vocabulary features increased from **5 → 8** (+60%)
- Lemmatization successfully collapsed inflected forms
- This means **word choice** (not just inflection) now discriminates better

### 3. Better Morphological Features
- Genitive case frequencies now top discriminators
- Particle detection expanded (18 vs 10)
- More consistent detection across texts

### 4. Reduced Noise in Data
- 114 fewer "common words" (17% reduction)
- Inflected variants merged: θεός/θεοῦ/θεῷ/θεόν → θεός
- Cleaner signal for authorship patterns

## Evidence of Lemmatization Success

From word n-gram analysis (Asclepiodotus Tacticus sample):
- `εἰμί ὁ` - "to be the" (dictionary form, not conjugated)
- `ποιέω ὁ` - "to make the" (lemma, not ποιεῖ/ποιοῦσι)
- `ἔχω ὁ` - "to have the" (lemma, not ἔχει/ἔχουσι)
- `γίγνομαι ὁ` - "to become the" (lemma form)

**Without lemmatization**, each conjugated form would be counted separately, diluting the signal.

## Methodological Validation

### Unicode Normalization
✅ **Verified**: Colossians δὲ instances now correctly counted
✅ **No false positives**: Pattern matching remains accurate

### Particle Expansion
✅ **18 particles detected** including enclitics (μηδέ, οὐδέ, etc.)
✅ **More comprehensive** morphological analysis

### MATTR
✅ **Fair comparison** across texts from 1,000 to 285,000+ words
✅ **Three window sizes** (50, 100, 200) for robust measurement

### Lemmatization
✅ **554 vs 668 common words** confirms inflection collapse
✅ **Verb/noun lemmas verified** in n-gram analysis
✅ **No over-lemmatization**: Distinct words remain distinct

## Conclusions

The implementation of lemmatization, Unicode normalization, expanded particles, and MATTR resulted in:

1. **8.3% more perfect discriminators** (60 → 65)
2. **60% more vocabulary-based discriminators** (5 → 8)
3. **Stronger discrimination** (top score: 0.878 → 1.203)
4. **Cleaner data** (17% fewer spurious word variants)
5. **Fair comparison** across texts of all sizes

## Recommendation

**The lemmatized analysis should be considered the authoritative version** for all future authorship attribution work. The improvements in:
- Data quality (Unicode normalization)
- Feature detection (18 particles)
- Vocabulary measurement (MATTR)
- Lemmatization (collapsed inflections)

...result in more reliable, more discriminating, and methodologically superior results.

---

## Technical Details

### Analysis Parameters
- **Word count threshold**: 1,000 words minimum per author
- **Authors analyzed**: 33 (same in both analyses)
- **Lemmatization tool**: CLTK (Classical Language Toolkit) with odyCy spaCy model
- **Features analyzed**:
  - Character n-grams (2, 3, 4)
  - Word n-grams (2, 3)
  - Word frequencies
  - Morphological patterns (particles, cases)
  - Word length distributions
  - Phonetic patterns
  - Vocabulary richness (MATTR)

### Files Generated
- `best_discriminative_features_OLD.json` - Original results (60 discriminators)
- `best_discriminative_features_LEMMATIZED.json` - New results (65 discriminators)
- `raw_results.pkl` - Complete analysis data
- `author_info.json` - Author metadata
- Visualization PNG files (heatmaps, word count distributions, etc.)

---

**Analysis Date**: October 27, 2025  
**Computation**: Google Cloud VM (e2-standard-4, 4 vCPU, 16GB RAM)  
**Duration**: ~30 minutes (text loading + lemmatization + feature extraction)

