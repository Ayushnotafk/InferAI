# Oversampling vs Original vs Class-Weight — Comparison

Held-out evaluation on `dataset/test_set.csv` (n=200). Training strategies share the same merged deduplicated corpus before resampling.

## Class counts (oversampling)

### Before oversampling

| Class | Count |
| --- | ---: |
| Pratyaksha | 69 |
| Anumana | 1163 |
| Upamana | 104 |
| Shabda | 125 |

### After oversampling

| Class | Count |
| --- | ---: |
| Pratyaksha | 500 |
| Anumana | 1163 |
| Upamana | 500 |
| Shabda | 500 |

Anumana rows kept unchanged. Minority classes upsampled to ~500 with replacement (`random_state=42`).

## Aggregate metrics (test_set.csv)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.6850 | 0.7494 | 0.7041 | 0.6831 |
| Class-weight (`balanced`) | 0.6600 | 0.7061 | 0.6628 | 0.6556 |
| Oversampled minority | 0.6000 | 0.6483 | 0.6054 | 0.6007 |

## Original model

### Per-class metrics

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Pratyaksha | 0.6600 | 0.7174 | 0.6875 |
| Anumana | 0.5238 | 1.0000 | 0.6875 |
| Upamana | 0.9211 | 0.6604 | 0.7692 |
| Shabda | 0.8929 | 0.4386 | 0.5882 |

### Confusion matrix

| True \ Pred | Anumana | Pratyaksha | Shabda | Upamana |
| --- | --- | --- | --- | --- |
| Anumana | 44 | 0 | 0 | 0 |
| Pratyaksha | 10 | 33 | 3 | 0 |
| Shabda | 12 | 17 | 25 | 3 |
| Upamana | 18 | 0 | 0 | 35 |

## Class-weight model

### Per-class metrics

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Pratyaksha | 0.4353 | 0.8043 | 0.5649 |
| Anumana | 0.6471 | 0.5000 | 0.5641 |
| Upamana | 1.0000 | 0.9434 | 0.9709 |
| Shabda | 0.7419 | 0.4035 | 0.5227 |

### Confusion matrix

| True \ Pred | Anumana | Pratyaksha | Shabda | Upamana |
| --- | --- | --- | --- | --- |
| Anumana | 22 | 22 | 0 | 0 |
| Pratyaksha | 1 | 37 | 8 | 0 |
| Shabda | 8 | 26 | 23 | 0 |
| Upamana | 3 | 0 | 0 | 50 |

## Oversampled model

### Per-class metrics

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Pratyaksha | 0.4268 | 0.7609 | 0.5469 |
| Anumana | 0.4889 | 0.5000 | 0.4944 |
| Upamana | 1.0000 | 0.7925 | 0.8842 |
| Shabda | 0.6774 | 0.3684 | 0.4773 |

### Confusion matrix

| True \ Pred | Anumana | Pratyaksha | Shabda | Upamana |
| --- | --- | --- | --- | --- |
| Anumana | 22 | 22 | 0 | 0 |
| Pratyaksha | 1 | 35 | 10 | 0 |
| Shabda | 11 | 25 | 21 | 0 |
| Upamana | 11 | 0 | 0 | 42 |

## Recommendation

Based on held-out **macro F1** (primary) and **accuracy** (secondary), the **Original** configuration performs best on `dataset/test_set.csv`.

- Original training without `class_weight` or oversampling achieves the strongest balance on the curated test set despite Anumana skew in merged data.

### Summary comparison

| Criterion | Original | Class-weight | Oversampled |
| --- | --- | --- | --- |
| Accuracy | 0.6850 | 0.6600 | 0.6000 |
| Macro F1 | 0.6831 | 0.6556 | 0.6007 |
| Shabda recall | 0.4386 | 0.4035 | 0.3684 |
| Upamana recall | 0.6604 | 0.9434 | 0.7925 |
| Pratyaksha recall | 0.7174 | 0.8043 | 0.7609 |
