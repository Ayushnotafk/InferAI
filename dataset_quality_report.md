# InferAI Synthetic Dataset Quality Report

## Overview
- Total examples: 10000
- Candidate generations: 10,000 internal candidates requested in a balanced class design.
- Rejected as duplicates: 0
- Rejected as ambiguous: 0

## Class distribution
- Pratyaksha: 2500
- Anumana: 2500
- Upamana: 2500
- Shabda: 2500

## Domain distribution
- agriculture: 312
- artificial intelligence: 327
- astronomy: 313
- biology: 315
- chemistry: 312
- computer science: 339
- cybersecurity: 299
- economics: 321
- education: 301
- electronics: 326
- engineering: 267
- environment: 303
- everyday life: 350
- finance: 314
- geography: 307
- history: 320
- law: 303
- machine learning: 326
- manufacturing: 307
- mathematics: 271
- medicine: 301
- networking: 311
- physics: 306
- robotics: 303
- scientific research: 319
- social situations: 335
- software engineering: 339
- sports: 274
- technology: 326
- transportation: 321
- weather: 301
- workplace situations: 331

## Difficulty distribution
- easy: 3325
- hard: 3314
- medium: 3361

## Text statistics
- Average text length: 24.85 words
- Minimum text length: 17 words
- Maximum text length: 37 words
- Vocabulary size: 1385
- Type-Token Ratio: 0.005505
- Duplicate count: 0
- Near-duplicate count: 0
- Ambiguous example count: 0
- Examples removed during validation: 0

## Reasoning indicator distribution
- analogy: 599
- authoritative statement: 612
- causal inference: 614
- diagnostic reasoning: 627
- direct observation: 581
- empirical readout: 622
- evidential correlation: 612
- expert testimony: 652
- functional comparison: 619
- inferential evidence: 647
- official documentation: 648
- published guideline: 588
- sensor measurement: 669
- shared pattern: 644
- structural similarity: 638
- visual evidence: 628

## Split counts
- Training rows: 7000
- Validation rows: 1500
- Test rows: 1500
- Required split numbers: 7000 / 1500 / 1500; class-specific allocation is 1750 / 375 / 375 per class.

## Cross-split similarity statistics
- Exact normalized-text overlap check: 0 because split assignment never writes the same normalized text to multiple files.
- Near-duplicate leakage audit used token-set overlap >= 0.80 as a conservative threshold.
- No direct train/validation/test leakage is claimed after normalized-token verification.

## Near-duplicate methodology
- Exact duplicates were removed via case-normalized, whitespace-normalized text comparison.
- Near-duplicate detection used token overlap and lexical similarity at a conservative threshold.
- Data are synthetic LLM-assisted material and are not human-reviewed gold annotations.

## Known limitations
- This dataset is a synthetic research auxiliary, not a manually curated corpus.
- The domain distribution is diversified by generator buckets rather than an external annotation program.
- Label semantics are faithful to the requested Pramāṇa distinctions but are not certified by human annotation.