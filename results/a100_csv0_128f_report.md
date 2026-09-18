# CSV-0 certificate probe

Config: 128 frames, 128 tokens, gamma=9, K+text=1024, 3 sample(s): 050-1, 717-1, 496-3

## A. Correctness ceiling (paired sparse vs dense)

- verified positions: **424** across **48** rounds
- position-level sparse/dense agreement: **95.52%**
- rounds where every position agreed (the certificate ceiling): **68.75%**
- rounds where the mask variants disagreed on logits: **0** (max |delta| = 0.000e+00; the cached mask is an optimisation, not an approximation, so this must stay 0)

- strict operating point (precision 1.00): margin >= 0.685 certifies **14.58%** of rounds with **0** wrong skips
- 0.95-precision operating point: margin >= 0.685 certifies **14.58%** of rounds (0 wrong skips)

Selected curve rows (min-margin threshold -> coverage / precision):

| margin >= | certified rounds | coverage | precision | wrong skips |
|---:|---:|---:|---:|---:|
| 0.016 | 48 | 100.0% | 68.8% | 15 |
| 0.062 | 44 | 91.7% | 72.7% | 12 |
| 0.094 | 40 | 83.3% | 72.5% | 11 |
| 0.120 | 33 | 68.8% | 75.8% | 8 |
| 0.147 | 29 | 60.4% | 79.3% | 6 |
| 0.183 | 25 | 52.1% | 84.0% | 4 |
| 0.281 | 22 | 45.8% | 86.4% | 3 |
| 0.338 | 17 | 35.4% | 82.4% | 3 |
| 0.418 | 13 | 27.1% | 84.6% | 2 |
| 0.547 | 10 | 20.8% | 90.0% | 1 |
| 0.807 | 5 | 10.4% | 100.0% | 0 |
| 2.453 | 1 | 2.1% | 100.0% | 0 |

## B. Middle-pass cost (ms per round)

| stage | ms/round | note |
|---|---:|---|
| sparse draft (9 sequential passes) | 229.2 | needed only to obtain the block |
| **sparse batched pass (mask rebuilt per layer)** | **47.1** | existing implementation; all-inclusive |
| sparse batched pass (mask built once) | 31.3 | forward only |
| + one mask build per round | 0.5 | 9x1 mask |
| **= mask-cached variant, all-inclusive** | **31.7** | this probe's variant |
| dense pass (what a certificate would skip) | 63.4 | authoritative |
| bonus step, dense share | 56.3 | a skipped round avoids this; the sparse bonus still runs (combined 81.3) |

- mask builds: per-layer variant 1344 (3.5 ms of CPU launch time), cached variant 1344 hits / 0 misses (0 misses means the cached mask was really reused)
- standalone mask-build microbenchmark: 0.085 ms (CPU launch time, no sync; diagnostic only)

- existing middle pass / dense pass = **0.74x**
- mask-cached middle pass / dense pass = **0.50x**

## C. Gate (from `hsd_feasibility.py`)

- saving when a round skips dense: **119.6 ms/round**
- existing middle pass: NECESSARY: certificates must hold in >39% of rounds
- cached-mask middle pass: NECESSARY: certificates must hold in >27% of rounds

Compare the required rate against the ceiling in section A. A requirement above the ceiling
cannot be met by any certificate, because no rule computable from the sparse pass alone can
certify a round whose sparse and dense argmax differ.

