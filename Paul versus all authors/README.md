# Pauline Letters Authorship Analysis

This folder contains the complete analysis of the 14 Pauline letters for single authorship probability using authentic ancient Greek linguistic features.

## 📁 Contents

### **Corpus Data**
- `pauline_letters/` - Directory containing all 14 Pauline letters as individual .txt files
  - Romans.txt (7,055 words)
  - 1 Corinthians.txt (6,812 words) 
  - 2 Corinthians.txt (4,473 words)
  - Galatians.txt (2,226 words)
  - Ephesians.txt (2,417 words)
  - Philippians.txt (1,626 words)
  - Colossians.txt (1,581 words)
  - 1 Thessalonians.txt (1,473 words)
  - 2 Thessalonians.txt (820 words)
  - 1 Timothy.txt (1,591 words)
  - 2 Timothy.txt (1,236 words)
  - Titus.txt (659 words)
  - Philemon.txt (334 words)
  - Hebrews.txt (4,935 words)

### **Analysis Tools**
- `pauline_authorship_analyzer.py` - Python script that performs the authorship analysis
  - Uses 60 discriminative features identified from 33 ancient Greek authors
  - Extracts morphological, phonetic, vocabulary, and stylistic features
  - Compares variance within Pauline corpus to variance between different authors

### **Results**
- `pauline_authorship_analysis.json` - Complete analysis results
  - Single authorship probability: **12.3%**
  - Interpretation: **Strong evidence for multiple authorship**
  - Detailed breakdown of all 60 features
  - Individual consistency scores and variance ratios

## 🎯 **Key Findings**

**Single Authorship Probability: 12.3%** - Strong evidence for multiple authorship

### **Methodology**
The analysis uses 60 authentic linguistic features that perfectly distinguish between 33 ancient Greek authors (100BC-100AD). For each feature, we:

1. Calculate variance within the Pauline corpus
2. Compare to variance between different ancient authors  
3. Generate consistency scores (lower variance = higher consistency = more likely single author)
4. Weight by feature reliability (separation scores)
5. Average across all features

### **What 12.3% Means**
The Pauline letters show 87.7% as much linguistic diversity as a random collection of texts by completely different ancient Greek authors. This suggests the corpus contains multiple authorship rather than stylistic variation within a single author.

### **Supporting Evidence**
- **Phonetic features** (vowel frequencies, diphthongs) vary more within Pauline corpus than between different authors
- **Morphological patterns** show mixed consistency across letters
- **Vocabulary richness** shows some consistency but with notable outliers

### **Scholarly Alignment**
This computational analysis aligns with modern biblical scholarship consensus:
- **Undisputed Pauline**: Romans, 1-2 Corinthians, Galatians, Philippians, 1 Thessalonians, Philemon
- **Disputed**: Ephesians, Colossians, 2 Thessalonians
- **Pastoral Letters**: 1-2 Timothy, Titus (widely considered non-Pauline)
- **Hebrews**: Almost universally recognized as non-Pauline

## 🚀 **Usage**

To run the analysis:
```bash
cd "Paul all authors"
python pauline_authorship_analyzer.py pauline_letters/
```

## 📊 **Technical Details**

- **Features analyzed**: 60 (morphological, phonetic, vocabulary, character n-grams, word n-grams)
- **Comparison baseline**: 33 ancient Greek authors with >1000 words each
- **Total Pauline corpus**: ~34,300 words across 14 letters
- **Analysis method**: Variance ratio comparison with weighted consistency scoring

This analysis demonstrates the power of computational stylometry using authentic linguistic features for historical authorship attribution.
