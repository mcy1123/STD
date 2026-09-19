# Visual-anchor protection: premise check on existing verification traces

Source: `results/routing_traces_videomme` (20 samples, 20 traces on disk, 1768 (round, head) observations).
visual_len V = **15876**, K kept per head = **967** (retention **6.09%**).

## Q1 - can a cumulative-attention (top-p) rule hold ~10% retention?

| cumulative mass target | tokens needed | share of V |
|---|---:|---:|
| 50% | 1316 | 8.29% |
| 90% | 7696 | 48.48% |
| 99% | 13585 | 85.57% |
| the current K | 967 | 6.09% |

The selector's top-K captures **44.0%** of the visual attention mass.

## Q2 - is per-token max attention a different signal from summed attention?

- Jaccard(top-K by sum, top-K by max) = **0.734**
- tokens the anchor criterion keeps but the selector evicts: **146.9** of K=967

## Q3 - how much mass can anchor protection actually rescue?

- aggregate attention mass those evicted anchors carry: **2.3089%**

## Reading

1. A top-p rule priced at 90% is unreachable at this sparsity: it needs roughly half of the visual tokens, ~8x the current K. Per the quoted proposal's own table that would move retention from ~10% to ~50%, which is a different (and far more expensive) operating point.
2. The anchor criterion is mostly the selector it is meant to correct (Jaccard ~0.73), and the set it would additionally protect carries a low single-digit percentage of the attention mass.
3. These are the numbers the dense verifier's own attention provides -- the friendliest possible oracle. An online rule estimated from 2 queries can only be noisier.
