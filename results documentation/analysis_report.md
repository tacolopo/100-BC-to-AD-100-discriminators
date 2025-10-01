# Greek Authorship Attribution Analysis Report

## Overview

This analysis examined ancient Greek texts from 33 authors who had more than 1000 words in their collected works.

## Included Authors

- **Adamantius**: 11,602 words across 2 texts
- **Aelius Theon**: 11,514 words across 1 texts
- **Aristonicus of Alexandria**: 16,961 words across 1 texts
- **Arius Didymus**: 20,855 words across 1 texts
- **Asclepiodotus Tacticus**: 6,655 words across 1 texts
- **Bion of Phlossa**: 1,814 words across 3 texts
- **Brutus**: 3,871 words across 1 texts
- **Cebes**: 4,719 words across 1 texts
- **Chariton**: 47,525 words across 2 texts
- **Chion of Heraclea**: 4,797 words across 1 texts
- **Clement of Rome**: 15,841 words across 4 texts
- **Comarius**: 2,312 words across 1 texts
- **Didymus Chalcenterus**: 1,311 words across 1 texts
- **Dionysius of Halicarnassus**: 403,938 words across 14 texts
- **Flavius Josephus**: 467,376 words across 4 texts
- **Geminus**: 20,603 words across 1 texts
- **Heron of Alexandria**: 110,674 words across 12 texts
- **Homer**: 2,149 words across 1 texts
- **Isidorus of Charax**: 1,221 words across 1 texts
- **Lesbonax**: 2,127 words across 2 texts
- **Longinus**: 12,682 words across 1 texts
- **Musonius Rufus**: 18,007 words across 3 texts
- **Onasander**: 12,125 words across 1 texts
- **Parthenius of Nicaea**: 6,550 words across 1 texts
- **Pedianus Discorides**: 125,739 words across 2 texts
- **Philo of Alexandria**: 417,486 words across 31 texts
- **Philodemus**: 22,324 words across 2 texts
- **Polemon of Laodicea**: 6,060 words across 1 texts
- **Scymnus**: 5,610 words across 1 texts
- **Strabo**: 285,570 words across 1 texts
- **Theodosius of Bithynia**: 27,490 words across 1 texts
- **Tryphon**: 2,945 words across 1 texts
- **Xenophon of Ephesus**: 16,476 words across 1 texts

## Analysis Methods

The following authentic linguistic features were analyzed (excluding modern punctuation):

1. **Character n-grams**: 2, 3, and 4-character sequences
2. **Word n-grams**: 2 and 3-word sequences
3. **Word frequencies**: Individual word usage patterns
4. **Morphological patterns**: Greek case endings, particles, verb forms
5. **Word length distributions**: Statistics about word lengths
6. **Phonetic patterns**: Vowel frequencies, consonant clusters, diphthongs
7. **Vocabulary richness**: Type-token ratios, hapax legomena, lexical diversity

## Key Findings

### Perfect Discrimination Found! 🎉

The following features can perfectly distinguish between ALL authors:

#### 1. morphological: particle_δέ_freq

Author values:
- Adamantius: 0.000948
- Aelius Theon: 0.001476
- Aristonicus of Alexandria: 0.003832
- Arius Didymus: 0.006090
- Asclepiodotus Tacticus: 0.002705
- Bion of Phlossa: 0.007166
- Brutus: 0.000258
- Cebes: 0.001271
- Chariton: 0.002272
- Chion of Heraclea: 0.003335
- Clement of Rome: 0.000884
- Comarius: 0.000433
- Didymus Chalcenterus: 0.000000
- Dionysius of Halicarnassus: 0.001017
- Flavius Josephus: 0.000815
- Geminus: 0.002621
- Heron of Alexandria: 0.000660
- Homer: 0.004653
- Isidorus of Charax: 0.000819
- Lesbonax: 0.000940
- Longinus: 0.001656
- Musonius Rufus: 0.002277
- Onasander: 0.001485
- Parthenius of Nicaea: 0.002443
- Pedianus Discorides: 0.002052
- Philo of Alexandria: 0.001174
- Philodemus: 0.000582
- Polemon of Laodicea: 0.002475
- Scymnus: 0.001248
- Strabo: 0.001492
- Theodosius of Bithynia: 0.000837
- Tryphon: 0.005433
- Xenophon of Ephesus: 0.001396

Separation score: 0.8776

#### 2. morphological: particle_τε_freq

Author values:
- Adamantius: 0.006895
- Aelius Theon: 0.006427
- Aristonicus of Alexandria: 0.006544
- Arius Didymus: 0.005898
- Asclepiodotus Tacticus: 0.009016
- Bion of Phlossa: 0.002205
- Brutus: 0.002842
- Cebes: 0.002331
- Chariton: 0.003051
- Chion of Heraclea: 0.005420
- Clement of Rome: 0.003093
- Comarius: 0.001298
- Didymus Chalcenterus: 0.002288
- Dionysius of Halicarnassus: 0.011014
- Flavius Josephus: 0.009635
- Geminus: 0.001505
- Heron of Alexandria: 0.001934
- Homer: 0.006980
- Isidorus of Charax: 0.000000
- Lesbonax: 0.014575
- Longinus: 0.005914
- Musonius Rufus: 0.006386
- Onasander: 0.005691
- Parthenius of Nicaea: 0.013130
- Pedianus Discorides: 0.006975
- Philo of Alexandria: 0.004637
- Philodemus: 0.004793
- Polemon of Laodicea: 0.000825
- Scymnus: 0.020677
- Strabo: 0.007896
- Theodosius of Bithynia: 0.002510
- Tryphon: 0.006452
- Xenophon of Ephesus: 0.005341

Separation score: 0.7676

#### 3. morphological: dative_pl_freq

Author values:
- Adamantius: 0.029305
- Aelius Theon: 0.024839
- Aristonicus of Alexandria: 0.007252
- Arius Didymus: 0.018557
- Asclepiodotus Tacticus: 0.015026
- Bion of Phlossa: 0.004961
- Brutus: 0.036425
- Cebes: 0.013350
- Chariton: 0.017254
- Chion of Heraclea: 0.017511
- Clement of Rome: 0.023799
- Comarius: 0.016003
- Didymus Chalcenterus: 0.001526
- Dionysius of Halicarnassus: 0.032857
- Flavius Josephus: 0.029950
- Geminus: 0.021647
- Heron of Alexandria: 0.007003
- Homer: 0.014891
- Isidorus of Charax: 0.003276
- Lesbonax: 0.000000
- Longinus: 0.024208
- Musonius Rufus: 0.019215
- Onasander: 0.032990
- Parthenius of Nicaea: 0.015725
- Pedianus Discorides: 0.022077
- Philo of Alexandria: 0.024777
- Philodemus: 0.008377
- Polemon of Laodicea: 0.021287
- Scymnus: 0.019608
- Strabo: 0.021270
- Theodosius of Bithynia: 0.006548
- Tryphon: 0.004414
- Xenophon of Ephesus: 0.013231

Separation score: 0.5886

#### 4. morphological: accusative_sg_fem_freq

Author values:
- Adamantius: 0.004482
- Aelius Theon: 0.006340
- Aristonicus of Alexandria: 0.010966
- Arius Didymus: 0.015680
- Asclepiodotus Tacticus: 0.009617
- Bion of Phlossa: 0.002205
- Brutus: 0.005425
- Cebes: 0.013986
- Chariton: 0.016328
- Chion of Heraclea: 0.013967
- Clement of Rome: 0.007702
- Comarius: 0.007353
- Didymus Chalcenterus: 0.003814
- Dionysius of Halicarnassus: 0.009598
- Flavius Josephus: 0.010963
- Geminus: 0.010193
- Heron of Alexandria: 0.003967
- Homer: 0.020009
- Isidorus of Charax: 0.011466
- Lesbonax: 0.005642
- Longinus: 0.005677
- Musonius Rufus: 0.006609
- Onasander: 0.005691
- Parthenius of Nicaea: 0.021069
- Pedianus Discorides: 0.006148
- Philo of Alexandria: 0.007306
- Philodemus: 0.015051
- Polemon of Laodicea: 0.008086
- Scymnus: 0.017825
- Strabo: 0.009994
- Theodosius of Bithynia: 0.001237
- Tryphon: 0.006791
- Xenophon of Ephesus: 0.017966

Separation score: 0.5706

#### 5. morphological: genitive_sg_masc_freq

Author values:
- Adamantius: 0.017669
- Aelius Theon: 0.034567
- Aristonicus of Alexandria: 0.033253
- Arius Didymus: 0.023783
- Asclepiodotus Tacticus: 0.028099
- Bion of Phlossa: 0.001103
- Brutus: 0.017050
- Cebes: 0.021191
- Chariton: 0.028848
- Chion of Heraclea: 0.027309
- Clement of Rome: 0.063254
- Comarius: 0.053201
- Didymus Chalcenterus: 0.025172
- Dionysius of Halicarnassus: 0.031547
- Flavius Josephus: 0.041356
- Geminus: 0.057079
- Heron of Alexandria: 0.055225
- Homer: 0.013960
- Isidorus of Charax: 0.031941
- Lesbonax: 0.013164
- Longinus: 0.032960
- Musonius Rufus: 0.033598
- Onasander: 0.024495
- Parthenius of Nicaea: 0.043053
- Pedianus Discorides: 0.051321
- Philo of Alexandria: 0.034047
- Philodemus: 0.019665
- Polemon of Laodicea: 0.040594
- Scymnus: 0.024599
- Strabo: 0.038432
- Theodosius of Bithynia: 0.076391
- Tryphon: 0.038031
- Xenophon of Ephesus: 0.033989

Separation score: 0.4965

## Feature Analysis Summary

### Char 2Grams

Top 3 most distinctive features:
1. **ὸ΄** (separation score: 5.7758)
1. **΄ι** (separation score: 5.7758)
1. **΄ϲ** (separation score: 5.7758)

### Char 3Grams

Top 3 most distinctive features:
1. **αυί** (separation score: 5.7758)
1. **φέψ** (separation score: 5.7758)
1. **ἰωά** (separation score: 5.7758)

### Char 4Grams

Top 3 most distinctive features:
1. **νὄξε** (separation score: 5.7758)
1. **αυίδ** (separation score: 5.7758)
1. **δαυί** (separation score: 5.7758)

### Word 2Grams

Top 3 most distinctive features:
1. **γίνονται πόδες** (separation score: 5.7758)
1. **σὺν οἴνῳ** (separation score: 5.7758)
1. **τὰ φύλλα** (separation score: 5.7758)

### Word 3Grams

Top 3 most distinctive features:
1. **δύναμιν δὲ ἔχει** (separation score: 5.7758)
1. **ἡ διπλῆ ὅτι** (separation score: 5.7758)
1. **ἐφ ἑαυτά γίνονται** (separation score: 5.7758)

### Word Frequencies

Top 3 most distinctive features:
1. **σχοῖνοι** (separation score: 5.7758)
1. **ἄδωνιν** (separation score: 5.7758)
1. **στερεοὺς** (separation score: 5.7758)

### Morphological

Top 3 most distinctive features:
1. **particle_οῦν_freq** (separation score: 5.7758)
1. **particle_ετι_freq** (separation score: 5.7758)
1. **particle_γαρ_freq** (separation score: 5.5491)

### Word Lengths

Top 3 most distinctive features:
1. **length_20_freq** (separation score: 4.1256)
1. **length_19_freq** (separation score: 2.9482)
1. **length_18_freq** (separation score: 1.3165)

### Phonetic

Top 3 most distinctive features:
1. **diphthong_ηυ_freq** (separation score: 2.8183)
1. **diphthong_υι_freq** (separation score: 2.6387)
1. **cluster_σχ_freq** (separation score: 2.1321)

### Vocabulary

Top 3 most distinctive features:
1. **avg_word_freq** (separation score: 0.7813)
1. **hapax_ratio** (separation score: 0.5617)
1. **ttr** (separation score: 0.4698)

## Methodology Notes

- **Text preprocessing**: Greek text was normalized, non-Greek characters removed
- **Minimum threshold**: Only authors with >1000 words included
- **Separation scoring**: Based on coefficient of variation and value distribution gaps
- **Perfect discrimination**: A feature where every author has a unique value
- **Feature combinations**: Tested pairs of top features for discrimination capability

## Files Generated

- `raw_results.pkl`: Complete analysis results for further processing
- `author_info.json`: Author statistics and text information
- `best_discriminative_features.json`: Top discriminative features or combinations
- `word_count_distribution.png`: Visualization of word counts by author
- `distinctive_features_heatmap.png`: Heatmap of top features across authors
- `word_length_comparison.png`: Average word length comparison
