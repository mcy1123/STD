# CSV-0 certificate probe

Config: 128 frames, 128 tokens, gamma=9, K+text=1024, 0 sample(s): 

## A. Correctness ceiling (paired sparse vs dense)

- verified positions: **424** across **48** rounds
- position-level sparse/dense agreement: **95.52%**
- rounds where every position agreed (the certificate ceiling): **68.75%**
- sparse/draft agreement at drafted positions: **100.00%**
- rounds where the mask variants disagreed on logits: **0** (max |delta| = 0.000e+00; the cached mask is an optimisation, not an approximation, so this must stay 0)

Operating points per certificate rule (thresholds chosen in sample, then held out on
a deterministic even/odd split of rounds so the strict coverage is not self-selected):

| rule | strict threshold | in-sample coverage | held-out coverage | held-out precision | held-out wrong skips |
|---|---:|---:|---:|---:|---:|
| margin | 0.685 | 14.6% | 27.1% | 88.9% | 2 |
| margin_and_draft | 0.685 | 14.6% | 27.1% | 88.9% | 2 |

- margin-only strict operating point (precision 1.00): margin >= 0.685 certifies **14.58%** of rounds with **0** wrong skips

Selected curve rows (margin-only rule):

| margin >= | certified rounds | coverage | precision | wrong skips |
|---:|---:|---:|---:|---:|
| 0.016 | 48 | 100.0% | 68.8% | 15 |
| 0.078 | 42 | 87.5% | 73.8% | 11 |
| 0.120 | 33 | 68.8% | 75.8% | 8 |
| 0.163 | 27 | 56.2% | 81.5% | 5 |
| 0.281 | 22 | 45.8% | 86.4% | 3 |
| 0.391 | 16 | 33.3% | 87.5% | 2 |
| 0.547 | 10 | 20.8% | 90.0% | 1 |
| 1.668 | 3 | 6.2% | 100.0% | 0 |

Selected curve rows (margin AND sparse/draft agreement rule):

| margin >= | certified rounds | coverage | precision | wrong skips |
|---:|---:|---:|---:|---:|
| 0.016 | 48 | 100.0% | 68.8% | 15 |
| 0.078 | 42 | 87.5% | 73.8% | 11 |
| 0.120 | 33 | 68.8% | 75.8% | 8 |
| 0.163 | 27 | 56.2% | 81.5% | 5 |
| 0.281 | 22 | 45.8% | 86.4% | 3 |
| 0.391 | 16 | 33.3% | 87.5% | 2 |
| 0.547 | 10 | 20.8% | 90.0% | 1 |
| 1.668 | 3 | 6.2% | 100.0% | 0 |

## B. Middle-pass cost (ms per round)

| stage | ms/round | note |
|---|---:|---|
| sparse draft (9 sequential passes) | 223.2 | needed only to obtain the block |
| **sparse batched pass (mask rebuilt per layer)** | **42.6** | existing implementation; all-inclusive |
| sparse batched pass (mask built once) | 28.5 | forward only |
| + one mask build per round | 0.3 | 9x1 mask |
| **= mask-cached variant, all-inclusive** | **28.8** | this probe's variant |
| dense pass (what a certificate would skip) | 61.0 | authoritative |
| bonus step, dense share | 55.3 | a skipped round avoids this; the sparse bonus still runs (combined 78.5) |

- mask builds: per-layer variant 1344 (3.2 ms of CPU launch time), cached variant 1344 hits / 0 misses (0 misses means the cached mask was really reused)
- standalone mask-build microbenchmark: 0.241 ms (CPU launch time, no sync; diagnostic only)

- existing middle pass / dense pass = **0.70x**
- mask-cached middle pass / dense pass = **0.47x**

## C. Gate (from `hsd_feasibility.py`)

- saving when a round skips dense: **116.3 ms/round**
- existing middle pass: NECESSARY: certificates must hold in >37% of rounds
- cached-mask middle pass: NECESSARY: certificates must hold in >25% of rounds

Compare the required rate against the ceiling in section A. A requirement above the ceiling
cannot be met by any certificate, because no rule computable from the sparse pass alone can
certify a round whose sparse and dense argmax differ.

## D. Is the mechanism viable?

- required certificate rate (mask-cached middle pass): **25%**
- round-level ceiling: **68.8%**
- best precision-1.0 coverage, in sample: **14.6%**
- best precision-1.0 coverage, held out: **27.1%** (wrong skips on held-out folds: **4**)
- best achievable speedup vs static STD (certificate fires on every round): **1.164x**
- **UNRESOLVED (sample too small).** In-sample and held-out strict coverage disagree by more than ten points, and the held-out threshold emitted 4 wrong skip(s); at this round count that is threshold-selection noise rather than evidence, so more rounds are needed before either number is trusted.

## E. Projected end-to-end effect

Every round still pays the sparse draft, so the projection is bounded by how small the
verifiable part of the round is:

| certificate rate | projected round (ms) | static round (ms) | speedup vs static STD |
|---:|---:|---:|---:|
| 68.8% (ceiling) | 311.6 | 362.7 | **1.164x** |
| 27.1% (margin_and_draft_heldout) | 360.0 | 362.7 | **1.007x** |
| 14.6% (margin_and_draft_strict) | 374.6 | 362.7 | **0.968x** |
| 27.1% (margin_heldout) | 360.0 | 362.7 | **1.007x** |
| 14.6% (margin_strict) | 374.6 | 362.7 | **0.968x** |

