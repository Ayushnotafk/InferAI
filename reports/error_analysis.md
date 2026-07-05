# InferAI — Error Analysis (Retrained Model)

Diagnostic evaluation on the held-out curated set `dataset/test_set.csv` (n=200). **No retraining performed.**

## 1. Overall performance

- **Accuracy:** 0.6850 (68.5%)
- **Misclassified:** 63

## 2. Per-class metrics

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Anumana | 0.5238 | 1.0 | 0.6875 | 44 |
| Pratyaksha | 0.66 | 0.7174 | 0.6875 | 46 |
| Shabda | 0.8929 | 0.4386 | 0.5882 | 57 |
| Upamana | 0.9211 | 0.6604 | 0.7692 | 53 |

## 3. Key findings

- **Lowest recall class:** Shabda (0.4386)
- **Lowest precision class:** Anumana (0.5238)

### Most common confusion pairs

| True class | Predicted class | Count | Rate (of true class) |
| --- | --- | ---: | ---: |
| Upamana | Anumana | 18 | 34.0% |
| Shabda | Pratyaksha | 17 | 29.8% |
| Shabda | Anumana | 12 | 21.1% |
| Pratyaksha | Anumana | 10 | 21.7% |
| Pratyaksha | Shabda | 3 | 6.5% |
| Shabda | Upamana | 3 | 5.3% |

## 4. Confusion matrix

![Confusion matrix](figures/error_analysis_confusion_matrix.png)

| True \ Pred | Anumana | Pratyaksha | Shabda | Upamana |
| --- | --- | --- | --- | --- |
| Anumana | 44 | 0 | 0 | 0 |
| Pratyaksha | 10 | 33 | 3 | 0 |
| Shabda | 12 | 17 | 25 | 3 |
| Upamana | 18 | 0 | 0 | 35 |

## 5. Classification report

```text
precision    recall  f1-score   support

     Anumana     0.5238    1.0000    0.6875        44
  Pratyaksha     0.6600    0.7174    0.6875        46
      Shabda     0.8929    0.4386    0.5882        57
     Upamana     0.9211    0.6604    0.7692        53

    accuracy                         0.6850       200
   macro avg     0.7494    0.7041    0.6831       200
weighted avg     0.7656    0.6850    0.6809       200
```

## 6. Misclassified examples (up to 50)

### 1. True: **Pratyaksha** → Predicted: **Anumana** (confidence 52.4%)

**Text:** "Bench notes (science): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.523, Pratyaksha=0.166, Shabda=0.213, Upamana=0.097

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 2. True: **Pratyaksha** → Predicted: **Anumana** (confidence 46.1%)

**Text:** "Bench notes (technology): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.461, Pratyaksha=0.182, Shabda=0.255, Upamana=0.102

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: log, fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 3. True: **Pratyaksha** → Predicted: **Anumana** (confidence 50.2%)

**Text:** "Bench notes (education): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.502, Pratyaksha=0.172, Shabda=0.214, Upamana=0.111

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 4. True: **Pratyaksha** → Predicted: **Anumana** (confidence 49.1%)

**Text:** "Bench notes (medicine): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.491, Pratyaksha=0.200, Shabda=0.201, Upamana=0.108

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 5. True: **Pratyaksha** → Predicted: **Anumana** (confidence 47.8%)

**Text:** "Bench notes (politics): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.478, Pratyaksha=0.177, Shabda=0.251, Upamana=0.094

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 6. True: **Pratyaksha** → Predicted: **Anumana** (confidence 60.6%)

**Text:** "Bench notes (philosophy): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.606, Pratyaksha=0.156, Shabda=0.154, Upamana=0.084

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 7. True: **Pratyaksha** → Predicted: **Anumana** (confidence 55.4%)

**Text:** "Bench notes (legal reasoning): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.554, Pratyaksha=0.144, Shabda=0.253, Upamana=0.049

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 8. True: **Pratyaksha** → Predicted: **Anumana** (confidence 57.6%)

**Text:** "Bench notes (public debate): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.576, Pratyaksha=0.147, Shabda=0.213, Upamana=0.064

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 9. True: **Pratyaksha** → Predicted: **Shabda** (confidence 40.9%)

**Text:** "Bench notes (social media discourse): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.317, Pratyaksha=0.149, Shabda=0.409, Upamana=0.125

**Likely confusion:** Weak perceptual cues without strong sensory verbs; testimonial or authority framing may be absent but Anumana/Shabda overlap in short curated spans.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 10. True: **Pratyaksha** → Predicted: **Anumana** (confidence 41.2%)

**Text:** "Bench notes (news analysis): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.412, Pratyaksha=0.176, Shabda=0.314, Upamana=0.097

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 11. True: **Pratyaksha** → Predicted: **Anumana** (confidence 39.8%)

**Text:** "Bench notes (daily life): the sample fluoresced under UV immediately after staining."

**Probability distribution:** Anumana=0.397, Pratyaksha=0.235, Shabda=0.257, Upamana=0.111

**Likely confusion:** Perceptual or measurement language co-occurs with inferential templates (therefore/suggests), so the linear head follows dominant Anumana training patterns.

**Contributing factors:** class imbalance (only 69 Pratyaksha training examples, 4.7% of corpus); missing or weak heuristics for true class (found Pratyaksha cues: fluoresced, bench notes); insufficient training examples for Pratyaksha (69 rows)

### 12. True: **Upamana** → Predicted: **Anumana** (confidence 53.6%)

**Text:** "Teaching education with metaphors is like using training wheels: helpful until intuition stabilizes."

**Probability distribution:** Anumana=0.536, Pratyaksha=0.035, Shabda=0.022, Upamana=0.406

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: like)

### 13. True: **Upamana** → Predicted: **Anumana** (confidence 46.4%)

**Text:** "Teaching medicine with metaphors is like using training wheels: helpful until intuition stabilizes."

**Probability distribution:** Anumana=0.464, Pratyaksha=0.049, Shabda=0.023, Upamana=0.464

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: like)

### 14. True: **Upamana** → Predicted: **Anumana** (confidence 45.0%)

**Text:** "Teaching politics with metaphors is like using training wheels: helpful until intuition stabilizes."

**Probability distribution:** Anumana=0.450, Pratyaksha=0.032, Shabda=0.073, Upamana=0.445

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: like)

### 15. True: **Upamana** → Predicted: **Anumana** (confidence 57.9%)

**Text:** "Teaching philosophy with metaphors is like using training wheels: helpful until intuition stabilizes."

**Probability distribution:** Anumana=0.579, Pratyaksha=0.042, Shabda=0.031, Upamana=0.348

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: like)

### 16. True: **Upamana** → Predicted: **Anumana** (confidence 71.1%)

**Text:** "Teaching legal reasoning with metaphors is like using training wheels: helpful until intuition stabilizes."

**Probability distribution:** Anumana=0.711, Pratyaksha=0.041, Shabda=0.066, Upamana=0.182

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: like)

### 17. True: **Upamana** → Predicted: **Anumana** (confidence 44.1%)

**Text:** "Teaching public debate with metaphors is like using training wheels: helpful until intuition stabilizes."

**Probability distribution:** Anumana=0.441, Pratyaksha=0.053, Shabda=0.123, Upamana=0.383

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: like)

### 18. True: **Upamana** → Predicted: **Anumana** (confidence 40.9%)

**Text:** "Teaching social media discourse with metaphors is like using training wheels: helpful until intuition stabilizes."

**Probability distribution:** Anumana=0.409, Pratyaksha=0.052, Shabda=0.133, Upamana=0.406

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: like)

### 19. True: **Upamana** → Predicted: **Anumana** (confidence 74.1%)

**Text:** "A causal DAG in science is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.741, Pratyaksha=0.029, Shabda=0.019, Upamana=0.211

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 20. True: **Upamana** → Predicted: **Anumana** (confidence 71.3%)

**Text:** "A causal DAG in technology is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.713, Pratyaksha=0.031, Shabda=0.018, Upamana=0.238

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 21. True: **Upamana** → Predicted: **Anumana** (confidence 78.7%)

**Text:** "A causal DAG in education is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.787, Pratyaksha=0.018, Shabda=0.010, Upamana=0.185

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 22. True: **Upamana** → Predicted: **Anumana** (confidence 74.1%)

**Text:** "A causal DAG in medicine is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.741, Pratyaksha=0.042, Shabda=0.016, Upamana=0.200

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 23. True: **Upamana** → Predicted: **Anumana** (confidence 81.5%)

**Text:** "A causal DAG in politics is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.815, Pratyaksha=0.023, Shabda=0.020, Upamana=0.141

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 24. True: **Upamana** → Predicted: **Anumana** (confidence 85.2%)

**Text:** "A causal DAG in philosophy is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.852, Pratyaksha=0.023, Shabda=0.011, Upamana=0.115

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 25. True: **Upamana** → Predicted: **Anumana** (confidence 87.1%)

**Text:** "A causal DAG in legal reasoning is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.871, Pratyaksha=0.028, Shabda=0.021, Upamana=0.080

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 26. True: **Upamana** → Predicted: **Anumana** (confidence 75.2%)

**Text:** "A causal DAG in public debate is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.752, Pratyaksha=0.038, Shabda=0.048, Upamana=0.162

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 27. True: **Upamana** → Predicted: **Anumana** (confidence 69.1%)

**Text:** "A causal DAG in social media discourse is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.691, Pratyaksha=0.052, Shabda=0.060, Upamana=0.197

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 28. True: **Upamana** → Predicted: **Anumana** (confidence 67.9%)

**Text:** "A causal DAG in news analysis is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.679, Pratyaksha=0.059, Shabda=0.059, Upamana=0.203

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 29. True: **Upamana** → Predicted: **Anumana** (confidence 77.8%)

**Text:** "A causal DAG in daily life is analogous to a map of one-way streets between variables."

**Probability distribution:** Anumana=0.778, Pratyaksha=0.029, Shabda=0.009, Upamana=0.184

**Likely confusion:** Comparative structure is subtle or missing explicit 'like/similar to'; argument functions as general inference.

**Contributing factors:** class imbalance (only 104 Upamana training examples, 7.1% of corpus); missing or weak heuristics for true class (found Upamana cues: analogous); ambiguous wording (multiple pramana cue families present in the same span)

### 30. True: **Shabda** → Predicted: **Anumana** (confidence 66.6%)

**Text:** "Court filings in science quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.666, Pratyaksha=0.080, Shabda=0.176, Upamana=0.078

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 31. True: **Shabda** → Predicted: **Anumana** (confidence 60.3%)

**Text:** "Court filings in technology quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.603, Pratyaksha=0.079, Shabda=0.208, Upamana=0.110

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 32. True: **Shabda** → Predicted: **Anumana** (confidence 81.4%)

**Text:** "Court filings in education quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.814, Pratyaksha=0.035, Shabda=0.082, Upamana=0.069

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 33. True: **Shabda** → Predicted: **Anumana** (confidence 78.4%)

**Text:** "Court filings in medicine quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.784, Pratyaksha=0.064, Shabda=0.087, Upamana=0.065

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 34. True: **Shabda** → Predicted: **Anumana** (confidence 70.8%)

**Text:** "Court filings in politics quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.708, Pratyaksha=0.055, Shabda=0.153, Upamana=0.084

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 35. True: **Shabda** → Predicted: **Anumana** (confidence 80.6%)

**Text:** "Court filings in philosophy quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.806, Pratyaksha=0.046, Shabda=0.097, Upamana=0.050

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 36. True: **Shabda** → Predicted: **Anumana** (confidence 82.9%)

**Text:** "Court filings in legal reasoning quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.829, Pratyaksha=0.041, Shabda=0.071, Upamana=0.059

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 37. True: **Shabda** → Predicted: **Anumana** (confidence 73.5%)

**Text:** "Court filings in public debate quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.736, Pratyaksha=0.049, Shabda=0.156, Upamana=0.059

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 38. True: **Shabda** → Predicted: **Anumana** (confidence 62.2%)

**Text:** "Court filings in social media discourse quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.622, Pratyaksha=0.055, Shabda=0.243, Upamana=0.080

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 39. True: **Shabda** → Predicted: **Anumana** (confidence 65.4%)

**Text:** "Court filings in news analysis quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.654, Pratyaksha=0.061, Shabda=0.221, Upamana=0.064

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 40. True: **Shabda** → Predicted: **Anumana** (confidence 74.3%)

**Text:** "Court filings in daily life quote the statute verbatim when arguing mens rea."

**Probability distribution:** Anumana=0.743, Pratyaksha=0.054, Shabda=0.134, Upamana=0.069

**Likely confusion:** Authority or source cues are implicit; the span reads as causal or inferential reason-giving rather than explicit testimony.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 41. True: **Shabda** → Predicted: **Pratyaksha** (confidence 31.2%)

**Text:** "Test augmentation (science): synthetic vignette uid test-pad-1; intended label Shabda at moderate strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.171, Pratyaksha=0.312, Shabda=0.264, Upamana=0.253

**Likely confusion:** Testimonial claim embeds direct observation verbs, blurring Shabda and Pratyaksha.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 42. True: **Shabda** → Predicted: **Upamana** (confidence 28.6%)

**Text:** "Test augmentation (education): synthetic vignette uid test-pad-2; intended label Shabda at moderate strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.243, Pratyaksha=0.253, Shabda=0.218, Upamana=0.286

**Likely confusion:** the model confuses rhetorical patterns associated with Shabda and Upamana.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 43. True: **Shabda** → Predicted: **Pratyaksha** (confidence 30.2%)

**Text:** "Test augmentation (technology): synthetic vignette uid test-pad-9; intended label Shabda at strong strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.185, Pratyaksha=0.303, Shabda=0.264, Upamana=0.249

**Likely confusion:** Testimonial claim embeds direct observation verbs, blurring Shabda and Pratyaksha.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 44. True: **Shabda** → Predicted: **Pratyaksha** (confidence 33.3%)

**Text:** "Test augmentation (daily life): synthetic vignette uid test-pad-21; intended label Shabda at weak strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.183, Pratyaksha=0.333, Shabda=0.269, Upamana=0.215

**Likely confusion:** Testimonial claim embeds direct observation verbs, blurring Shabda and Pratyaksha.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 45. True: **Shabda** → Predicted: **Pratyaksha** (confidence 30.9%)

**Text:** "Test augmentation (technology): synthetic vignette uid test-pad-22; intended label Shabda at weak strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.175, Pratyaksha=0.310, Shabda=0.291, Upamana=0.225

**Likely confusion:** Testimonial claim embeds direct observation verbs, blurring Shabda and Pratyaksha.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 46. True: **Shabda** → Predicted: **Pratyaksha** (confidence 33.3%)

**Text:** "Test augmentation (daily life): synthetic vignette uid test-pad-25; intended label Shabda at strong strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.173, Pratyaksha=0.333, Shabda=0.263, Upamana=0.231

**Likely confusion:** Testimonial claim embeds direct observation verbs, blurring Shabda and Pratyaksha.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 47. True: **Shabda** → Predicted: **Pratyaksha** (confidence 29.8%)

**Text:** "Test augmentation (science): synthetic vignette uid test-pad-29; intended label Shabda at strong strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.178, Pratyaksha=0.298, Shabda=0.271, Upamana=0.252

**Likely confusion:** Testimonial claim embeds direct observation verbs, blurring Shabda and Pratyaksha.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 48. True: **Shabda** → Predicted: **Pratyaksha** (confidence 32.3%)

**Text:** "Test augmentation (daily life): synthetic vignette uid test-pad-32; intended label Shabda at weak strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.172, Pratyaksha=0.323, Shabda=0.280, Upamana=0.225

**Likely confusion:** Testimonial claim embeds direct observation verbs, blurring Shabda and Pratyaksha.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 49. True: **Shabda** → Predicted: **Upamana** (confidence 29.0%)

**Text:** "Test augmentation (education): synthetic vignette uid test-pad-34; intended label Shabda at weak strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.242, Pratyaksha=0.237, Shabda=0.231, Upamana=0.290

**Likely confusion:** the model confuses rhetorical patterns associated with Shabda and Upamana.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

### 50. True: **Shabda** → Predicted: **Pratyaksha** (confidence 31.8%)

**Text:** "Test augmentation (medicine): synthetic vignette uid test-pad-40; intended label Shabda at strong strength for evaluation bookkeeping."

**Probability distribution:** Anumana=0.221, Pratyaksha=0.318, Shabda=0.235, Upamana=0.226

**Likely confusion:** Testimonial claim embeds direct observation verbs, blurring Shabda and Pratyaksha.

**Contributing factors:** class imbalance (only 125 Shabda training examples, 8.6% of corpus); missing heuristics for true class (no strong Shabda lexical cues in span)

## 7. Summary diagnosis

### Training corpus skew (merged retraining set)

- Pratyaksha: 69 (4.7%)
- Anumana: 1163 (79.6%)
- Upamana: 104 (7.1%)
- Shabda: 125 (8.6%)

### Root causes

1. **Class imbalance:** Anumana dominates the merged training corpus (~80%), biasing the logistic head toward inferential labels even when test spans carry perception, testimony, or analogy cues.
2. **Missing heuristics:** The hybrid rule engine is not used at training time; embedding-only classification misses explicit Nyaya cue words unless they align with Sentence-BERT geometry learned from skewed data.
3. **Ambiguous wording:** Curated test seeds reuse template frames across domains (e.g. *Bench notes*, *Test augmentation*) where multiple pramana readings are plausible.
4. **Insufficient training examples:** Pratyaksha, Upamana, and Shabda each have fewer than 150 merged rows vs. >1,100 Anumana rows, limiting recall on minority classes.

The single most frequent error is **Upamana → Anumana** (18 cases), consistent with the imbalance and template overlap described above.
