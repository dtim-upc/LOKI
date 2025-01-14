# Dataset Splitting and Oversampling Methodology Summary

## Overview
We implemented a topic-aware dataset splitting strategy with selective oversampling to handle class imbalance while maintaining data integrity. The approach consists of two main components: (1) a topic-stratified split and (2) targeted oversampling of minority topics.

## Dataset Statistics
- Total dataset size: 3,157 examples
- Number of unique topics: 63
- Topic size range: 10-117 examples per topic
- Split ratios: 70% train, 15% test, 15% validation

## Methodology

### 1. Topic-Stratified Splitting
- Implemented a stratified split that maintains topic distributions across train/test/validation sets
- Each topic's examples are split proportionally:
  - Training: 70% of each topic's examples
  - Testing: 15% of each topic's examples
  - Validation: 15% of each topic's examples
- This ensures representative sampling across all topics in each split

### 2. Selective Oversampling Strategy
- Applied only to the training set to avoid contaminating test and validation sets
- Target size for minority topics: 50% of median topic size
- Oversampling process:
  - Identifies topics below the target size threshold
  - Creates exact copies of examples from underrepresented topics
  - Assigns unique oversample IDs to track duplicated examples
  - Maintains original example IDs for traceability

## Results

### Simple Split (Without Oversampling)
- Training set: 2,179 examples (69.0%)
- Test set: 446 examples (14.1%)
- Validation set: 532 examples (16.9%)

### Oversampled Split
- Training set: 2,277 examples
  - 2,179 unique examples
  - 98 oversampled examples
  - Oversampling factor: 1.04x
- Test set: 446 examples (unchanged)
- Validation set: 532 examples (unchanged)

### Impact on Minority Topics
Examples of balanced improvements:
- Tennis News: 15 → 30 samples
- Sports Teams: 13 → 26 samples
- Education: 13 → 26 samples
- Ireland Geography: 7 → 21 samples
- African Colleges: 9 → 18 samples
- Greyhound Racing: 9 → 18 samples
- Collegiate Sports: 12 → 24 samples

## Key Benefits
1. Maintains topic distribution integrity in test and validation sets
2. Moderately increases minority topic representation without aggressive oversampling
3. Provides clear traceability of oversampled examples
4. Preserves original data characteristics while reducing class imbalance
5. Enables fair evaluation through untouched test and validation sets
