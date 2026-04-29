# xa25 Gating Analysis: Can HKVD Deviation Predict When Repair Helps?

## Data source

WP-B first-k2694 grid: `/data/knlp-key-results/wpb-firstk2694-grid-20260427/summaries/`
60 JSONL files, 200 questions each, 3 probes x 10 structural configs x {noxa, xa25}.

## Answer: NO -- deviation magnitudes are not logged

The per-question JSONL traces log three HKVD fields:

| Field | Type | Content |
|---|---|---|
| `hkvd_indices` | `list[int]` | Which 64 token positions were refreshed (top-25% deviated) |
| `hkvd_ratio` | `float` | Always 0.25 |
| `hkvd_compute_s` | `float` | Wall-clock time for HKVD computation |

**Missing (required for gating):** the per-token deviation magnitudes
(`per_token_diff` in `eval_xattn_hybrid.py` line 395). The function
`compute_hkvd_indices()` computes `(cache_K - fresh_K)^2` summed over
heads and head_dim, producing a per-token scalar, then calls `topk()`
and returns only the indices. The magnitudes are discarded.

Without magnitudes, we cannot answer:
- Is overall deviation high (cartridge badly drifted) or low (cartridge
  nearly matches fresh)?
- Is the deviation sharply concentrated (a few very wrong positions) or
  diffuse (everything slightly off)?
- Does the magnitude of deviation at refreshed positions correlate with
  whether refresh helps?

## What CAN be analyzed from existing traces

### 1. HKVD indices are question-dependent (useful for future work)

The HKVD indices vary substantially per question: consecutive questions
share only ~45% Jaccard similarity. This is because the reference
forward includes the query tokens, so different queries produce
different fresh K/V and thus different deviation patterns. This means
per-question gating is structurally possible if magnitudes are logged.

### 2. Index spatial distribution does NOT separate helped from hurt

Spatial features of the 64 HKVD indices (quartile fractions, max gap,
span) show negligible effect sizes between questions where xa25 helped
vs. hurt:

| Feature | Helped (n=25) | Hurt (n=17) | Cohen's d |
|---|---|---|---|
| mean_idx | 196.9 | 197.0 | -0.02 |
| q4_frac | 0.605 | 0.621 | -0.40 |
| max_gap | 133.2 | 132.3 | +0.20 |

All HKVD indices cluster in the last quarter of the compacted cache
(~60% in Q4) regardless of outcome. Index placement alone carries no
gating signal.

### 3. Margin-based gating shows modest promise (proxy signal)

Using the noxa run's logprob margin as a gate ("apply xa25 only when
margin < threshold") yields:

| Strategy | Accuracy |
|---|---|
| Always noxa | 34.5% |
| Always xa25 | 38.5% |
| Best margin gate (t=1.7) | 40.0% |
| Oracle (best per question) | 47.0% |

The margin gate recovers +1.5pp over always-xa25 on the diagnosis
probe. However:
- The optimal threshold (1.7) applies xa25 to 167/200 questions, so
  it is very permissive and only skips 33 high-confidence questions.
- The gain does NOT generalize cleanly to the R=4 configs where xa25
  is harmful (margin gating on A1R4K11 yields 33.0% vs. 32.5% xa25).
- The 7pp gap to oracle (40% vs 47%) suggests much better gating is
  possible with the right signal.

### 4. R=4 xa25 kill pattern is structural, not per-question

The xa25 lift depends deterministically on the structural config:

| Recent blocks | Configs | Mean xa25 lift (diagnosis) |
|---|---|---|
| R=0 | pure_topK, A2R0K14 | +4.0pp |
| R=2 | A1R2K13, A2R2K12 | +4.5pp |
| R=4 | A1R4K11, A2R4K10, A0R4K12, A4R4K8 | +0.25pp |
| R=6+ | A2R6K8, A1R8K7 | +3.5pp |

R=4 consistently kills xa25. R=6 and R=8 recover it. This suggests
the effect is mediated by WHICH blocks enter the cache (R=4 pins
blocks 124-127 which may overlap with positions that xa25 damages),
not by per-question deviation statistics.

## Required changes to eval_xattn_hybrid.py for gating analysis

The fix is minimal -- 6 lines in `compute_hkvd_indices()` and 4 lines
in the logging section of `main()`:

### In compute_hkvd_indices() (around line 393-397):

Return `per_token_diff` alongside the indices:

```python
# Current: returns only indices
top = torch.topk(per_token_diff, n_refresh).indices
return top.sort().values

# Needed: return (indices, per_token_diff)
top = torch.topk(per_token_diff, n_refresh).indices
return top.sort().values, per_token_diff
```

### In main() JSONL row construction (around line 782-786):

Log deviation summary statistics:

```python
# Add after "hkvd_indices":
"hkvd_dev_mean": float(per_token_diff.mean().item()) if per_token_diff is not None else 0.0,
"hkvd_dev_max": float(per_token_diff.max().item()) if per_token_diff is not None else 0.0,
"hkvd_dev_top25_mean": float(per_token_diff[hkvd_indices].mean().item()) if ... else 0.0,
"hkvd_dev_bottom75_mean": float(per_token_diff[~top_mask].mean().item()) if ... else 0.0,
```

Full `per_token_diff` vector (256 floats) could also be logged for
richer analysis but the summary stats above are sufficient for gating
threshold search.

### Priority

This is the **cheapest unblocking step** for xa25 gating. A re-run of
the pure_topK_K16 xa25 config (322 seconds on the WP-B pod) with
deviation logging would immediately answer whether total deviation
magnitude predicts per-question xa25 benefit. The R=4 structural
pattern suggests there may also be a block-level signal (deviation
concentrated in pinned recent blocks vs. routed middle blocks).
