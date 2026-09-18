A100 dynamic STD comparison — R1_incr_three_att

Video-MME: 10 available-subset samples, seed-42 order; 128 requested frames; 128 fixed output tokens; 1 measured repetitions per method/sample after 16-token warmups.
Physical GPU 0; FP16; sparse backend gqa_sdpa; collector=v2; refresh=incremental; update_interval=1; min_change=0.0; gamma=9; query_mode=three; bootstrap=attention; coverage=0.25; value_weight=0.25; K+text=1024; fallback=none.

Times below sum each sample's median across repetitions. Speedup = baseline / method; greater than 1 is faster.

| Method | Prefill (s) | Decode (s) | vs AR | vs static | Inference (s) | Inf. vs AR | Inf. vs static | Exact | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ar | 308.619 | 85.620 | 1.000x | 0.644x | 394.316 | 1.000x | 0.924x | 10/10 | 33.22 |
| static | 308.664 | 55.107 | 1.554x | 1.000x | 364.167 | 1.083x | 1.000x | 8/10 | 33.22 |
| dynamic_v2 | 309.595 | 67.315 | 1.272x | 0.819x | 377.309 | 1.045x | 0.965x | 8/10 | 33.22 |

Component totals (when --profile-components is enabled; medians per sample):
| Method | Cache init | Selection prefill | Selection/top-k | Dense prefill | Sparse cache | Draft | Verify | Bonus | Cache adjust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ar | 0.076 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| static | 0.083 | 308.664 | 0.098 | 0.000 | 0.212 | 39.283 | 11.472 | 4.331 | 0.004 |
| dynamic_v2 | 0.080 | 309.595 | 0.096 | 0.000 | 0.212 | 39.680 | 12.205 | 4.351 | 0.003 |

Per-sample paired acceptance (H_headroom; the aggregate mean is not trusted):
| Method | Sample | Static accept | Dynamic accept | delta |
|---|---|---:|---:|---:|
| dynamic_v2 | 050-1 | 0.663 | 0.712 | +0.049 |
| dynamic_v2 | 717-1 | 0.983 | 0.778 | -0.205 |
| dynamic_v2 | 496-3 | 0.796 | 0.796 | +0.000 |
| dynamic_v2 | 754-1 | 0.507 | 0.698 | +0.191 |
| dynamic_v2 | 154-3 | 0.541 | 0.647 | +0.106 |
| dynamic_v2 | 445-2 | 0.831 | 0.647 | -0.184 |
| dynamic_v2 | 102-2 | 0.850 | 0.688 | -0.162 |
| dynamic_v2 | 647-1 | 0.966 | 0.943 | -0.024 |
| dynamic_v2 | 504-2 | 0.966 | 0.891 | -0.076 |
| dynamic_v2 | 599-1 | 1.000 | 1.000 | +0.000 |

Stratified by static headroom (equal-count bins, LOW headroom first):
| Method | Bin | n | static range | mean static accept | mean delta accept |
|---|---:|---:|---|---:|---:|
| dynamic_v2 | 0 | 3 | 0.507-0.663 | 0.570 | +0.115 |
| dynamic_v2 | 1 | 3 | 0.796-0.850 | 0.825 | -0.115 |
| dynamic_v2 | 2 | 4 | 0.966-1.000 | 0.979 | -0.076 |

Measured generation continues after EOS to equalize work. Inference includes prefill and decoding; both timing measures exclude model loading and video processing. Exact-trial counts compare complete token sequences against AR. Unequal outputs do not demonstrate lossless speedup.

This small available-video subset is exploratory, not a full Video-MME accuracy evaluation. The synchronized total decode times include all dynamic selection/collection/refresh overhead.

Raw trials: R1_incr_three_att.jsonl
Git base: 3781fa7f28859f69ea163eeb9a7163b2697e9dca; worktree modified=False; exact source hashes are authoritative in the raw manifest.
