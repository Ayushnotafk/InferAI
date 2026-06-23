# InferAI — Data Leakage and Duplicate Audit

Cross-split duplicate detection between training (80%), validation (20%), and held-out test data. **No rows were removed** — this is an audit-only report.

## Methodology

- **Exact duplicates:** normalized lowercase stripped text equality.
- **Near duplicates:** RapidFuzz `ratio` ≥ 90% **or** cosine similarity ≥ 0.9 on `all-MiniLM-L6-v2` embeddings.
- **Splits:** merged training table partitioned with `train_test_split(test_size=0.2, random_state=42)`; `test_set.csv` is fully held out during merge.

## Summary

| Pair Type                                |   Count |
|:-----------------------------------------|--------:|
| Exact duplicates (cross-split)           |       0 |
| Near duplicates (fuzz≥90% or cosine≥0.9) |     506 |

## Split sizes

| Split         |   Rows |
|:--------------|-------:|
| train         |    648 |
| validation    |    163 |
| held_out_test |    200 |

## Exact duplicates

No exact text duplicates were found across train, validation, and held-out test splits. The merge pipeline's test-set exclusion guard appears effective.


## Near-duplicate examples (showing up to 25)

1. **train ↔ held_out_test** (fuzz=95.4%, cosine=0.867)  
   - A: "Master augmentation (philosophy): synthetic vignette uid master-pad-4; intended label Upamana at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (philosophy): synthetic vignette uid test-pad-74; intended label Upamana at moderate strength for evaluation bookkeeping.…"
2. **train ↔ held_out_test** (fuzz=95.3%, cosine=0.865)  
   - A: "Master augmentation (news analysis): synthetic vignette uid master-pad-1; intended label Anumana at weak strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (news analysis): synthetic vignette uid test-pad-15; intended label Anumana at weak strength for evaluation bookkeeping.…"
3. **train ↔ held_out_test** (fuzz=95.3%, cosine=0.867)  
   - A: "Master augmentation (social media discourse): synthetic vignette uid master-pad-14; intended label Shabda at weak strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (social media discourse): synthetic vignette uid test-pad-71; intended label Shabda at weak strength for evaluation bookkeeping.…"
4. **train ↔ held_out_test** (fuzz=95.3%, cosine=0.851)  
   - A: "Master augmentation (education): synthetic vignette uid master-pad-2; intended label Anumana at strong strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (education): synthetic vignette uid test-pad-62; intended label Anumana at strong strength for evaluation bookkeeping.…"
5. **train ↔ held_out_test** (fuzz=95.1%, cosine=0.888)  
   - A: "Master augmentation (public debate): synthetic vignette uid master-pad-31; intended label Upamana at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (public debate): synthetic vignette uid test-pad-35; intended label Upamana at moderate strength for evaluation bookkeeping.…"
6. **train ↔ held_out_test** (fuzz=95.1%, cosine=0.866)  
   - A: "Master augmentation (legal reasoning): synthetic vignette uid master-pad-34; intended label Anumana at strong strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (legal reasoning): synthetic vignette uid test-pad-54; intended label Anumana at strong strength for evaluation bookkeeping.…"
7. **train ↔ held_out_test** (fuzz=95.1%, cosine=0.850)  
   - A: "Master augmentation (social media discourse): synthetic vignette uid master-pad-22; intended label Anumana at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (social media discourse): synthetic vignette uid test-pad-5; intended label Anumana at moderate strength for evaluation bookkeeping.…"
8. **train ↔ held_out_test** (fuzz=95.0%, cosine=0.864)  
   - A: "Master augmentation (social media discourse): synthetic vignette uid master-pad-9; intended label Shabda at strong strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (social media discourse): synthetic vignette uid test-pad-38; intended label Shabda at strong strength for evaluation bookkeeping.…"
9. **train ↔ held_out_test** (fuzz=94.9%, cosine=0.893)  
   - A: "Master augmentation (science): synthetic vignette uid master-pad-37; intended label Upamana at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (science): synthetic vignette uid test-pad-76; intended label Upamana at moderate strength for evaluation bookkeeping.…"
10. **train ↔ held_out_test** (fuzz=94.8%, cosine=0.871)  
   - A: "Master augmentation (science): synthetic vignette uid master-pad-43; intended label Anumana at strong strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (science): synthetic vignette uid test-pad-31; intended label Anumana at strong strength for evaluation bookkeeping.…"
11. **train ↔ held_out_test** (fuzz=94.7%, cosine=0.866)  
   - A: "Master augmentation (public debate): synthetic vignette uid master-pad-28; intended label Pratyaksha at weak strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (public debate): synthetic vignette uid test-pad-7; intended label Pratyaksha at weak strength for evaluation bookkeeping.…"
12. **train ↔ held_out_test** (fuzz=94.7%, cosine=0.872)  
   - A: "Master augmentation (social media discourse): synthetic vignette uid master-pad-25; intended label Upamana at strong strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (social media discourse): synthetic vignette uid test-pad-81; intended label Upamana at strong strength for evaluation bookkeeping.…"
13. **train ↔ held_out_test** (fuzz=94.7%, cosine=0.840)  
   - A: "Master augmentation (politics): synthetic vignette uid master-pad-3; intended label Pratyaksha at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (politics): synthetic vignette uid test-pad-87; intended label Pratyaksha at moderate strength for evaluation bookkeeping.…"
14. **train ↔ held_out_test** (fuzz=94.7%, cosine=0.886)  
   - A: "Master augmentation (daily life): synthetic vignette uid master-pad-21; intended label Upamana at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (daily life): synthetic vignette uid test-pad-4; intended label Upamana at moderate strength for evaluation bookkeeping.…"
15. **train ↔ held_out_test** (fuzz=94.4%, cosine=0.869)  
   - A: "Master augmentation (medicine): synthetic vignette uid master-pad-0; intended label Anumana at weak strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (medicine): synthetic vignette uid test-pad-89; intended label Anumana at weak strength for evaluation bookkeeping.…"
16. **train ↔ held_out_test** (fuzz=94.4%, cosine=0.857)  
   - A: "Master augmentation (politics): synthetic vignette uid master-pad-23; intended label Pratyaksha at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (politics): synthetic vignette uid test-pad-87; intended label Pratyaksha at moderate strength for evaluation bookkeeping.…"
17. **train ↔ held_out_test** (fuzz=94.2%, cosine=0.864)  
   - A: "Master augmentation (philosophy): synthetic vignette uid master-pad-16; intended label Upamana at strong strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (philosophy): synthetic vignette uid test-pad-52; intended label Upamana at strong strength for evaluation bookkeeping.…"
18. **train ↔ held_out_test** (fuzz=94.2%, cosine=0.863)  
   - A: "Master augmentation (politics): synthetic vignette uid master-pad-10; intended label Upamana at strong strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (politics): synthetic vignette uid test-pad-47; intended label Upamana at strong strength for evaluation bookkeeping.…"
19. **train ↔ held_out_test** (fuzz=93.9%, cosine=0.870)  
   - A: "Master augmentation (science): synthetic vignette uid master-pad-27; intended label Anumana at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (science): synthetic vignette uid test-pad-100; intended label Anumana at moderate strength for evaluation bookkeeping.…"
20. **train ↔ held_out_test** (fuzz=93.4%, cosine=0.773)  
   - A: "Master augmentation (social media discourse): synthetic vignette uid master-pad-12; intended label Anumana at strong strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (social media discourse): synthetic vignette uid test-pad-81; intended label Upamana at strong strength for evaluation bookkeeping.…"
21. **train ↔ held_out_test** (fuzz=93.2%, cosine=0.750)  
   - A: "Master augmentation (legal reasoning): synthetic vignette uid master-pad-42; intended label Anumana at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (legal reasoning): synthetic vignette uid test-pad-82; intended label Upamana at moderate strength for evaluation bookkeeping.…"
22. **train ↔ held_out_test** (fuzz=92.9%, cosine=0.755)  
   - A: "Master augmentation (news analysis): synthetic vignette uid master-pad-18; intended label Upamana at weak strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (news analysis): synthetic vignette uid test-pad-15; intended label Anumana at weak strength for evaluation bookkeeping.…"
23. **train ↔ held_out_test** (fuzz=92.7%, cosine=0.743)  
   - A: "Master augmentation (daily life): synthetic vignette uid master-pad-13; intended label Anumana at weak strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (daily life): synthetic vignette uid test-pad-12; intended label Upamana at weak strength for evaluation bookkeeping.…"
24. **train ↔ held_out_test** (fuzz=92.4%, cosine=0.740)  
   - A: "Master augmentation (legal reasoning): synthetic vignette uid master-pad-8; intended label Shabda at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (legal reasoning): synthetic vignette uid test-pad-82; intended label Upamana at moderate strength for evaluation bookkeeping.…"
25. **train ↔ held_out_test** (fuzz=92.4%, cosine=0.736)  
   - A: "Master augmentation (public debate): synthetic vignette uid master-pad-41; intended label Anumana at moderate strength for evaluation bookkeeping.…"  
   - B: "Test augmentation (public debate): synthetic vignette uid test-pad-35; intended label Upamana at moderate strength for evaluation bookkeeping.…"

## Interpretation

Some cross-split similarity was detected. Review listed examples manually; near duplicates may reflect paraphrase rather than true leakage but can still inflate validation optimism if semantically identical arguments appear in both training and test corpora.
