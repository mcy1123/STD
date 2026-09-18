A100 dynamic STD comparison — R0_full_three_att

Video-MME: 10 available-subset samples, seed-42 order; 128 requested frames; 128 fixed output tokens; 1 measured repetitions per method/sample after 16-token warmups.
Physical GPU 0; FP16; sparse backend gqa_sdpa; collector=v2; refresh=full; update_interval=1; min_change=0.0; gamma=9; query_mode=three; bootstrap=attention; coverage=0.25; value_weight=0.25; K+text=1024; fallback=none.

Times below sum each sample's median across repetitions. Speedup = baseline / method; greater than 1 is faster.

| Method | Prefill (s) | Decode (s) | vs AR | vs static | Inference (s) | Inf. vs AR | Inf. vs static | Exact | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ar | 326.279 | 87.857 | 1.000x | 0.715x | 414.300 | 1.000x | 1.062x | 10/10 | 34.31 |
| static | 376.355 | 62.791 | 1.399x | 1.000x | 440.010 | 0.942x | 1.000x | 8/10 | 34.31 |
| dynamic_v2 | 377.811 | 100.970 | 0.870x | 0.622x | 480.648 | 0.862x | 0.915x | 8/10 | 34.31 |

Component totals (when --profile-components is enabled; medians per sample):
| Method | Cache init | Selection prefill | Selection/top-k | Dense prefill | Sparse cache | Draft | Verify | Bonus | Cache adjust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ar | 0.163 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| static | 0.171 | 376.355 | 0.152 | 0.000 | 0.535 | 45.047 | 12.734 | 4.987 | 0.006 |
| dynamic_v2 | 0.797 | 377.811 | 0.215 | 0.000 | 0.835 | 48.589 | 16.253 | 5.165 | 0.004 |

Per-sample paired acceptance (H_headroom; the aggregate mean is not trusted):
| Method | Sample | Static accept | Dynamic accept | delta |
|---|---|---:|---:|---:|
| dynamic_v2 | 050-1 | 0.663 | 0.712 | +0.049 |
| dynamic_v2 | 717-1 | 0.983 | 0.796 | -0.187 |
| dynamic_v2 | 496-3 | 0.796 | 0.796 | +0.000 |
| dynamic_v2 | 754-1 | 0.507 | 0.698 | +0.191 |
| dynamic_v2 | 154-3 | 0.541 | 0.647 | +0.106 |
| dynamic_v2 | 445-2 | 0.831 | 0.647 | -0.184 |
| dynamic_v2 | 102-2 | 0.850 | 0.688 | -0.162 |
| dynamic_v2 | 647-1 | 0.966 | 0.935 | -0.031 |
| dynamic_v2 | 504-2 | 0.966 | 0.877 | -0.089 |
| dynamic_v2 | 599-1 | 1.000 | 1.000 | +0.000 |

Stratified by static headroom (equal-count bins, LOW headroom first):
| Method | Bin | n | static range | mean static accept | mean delta accept |
|---|---:|---:|---|---:|---:|
| dynamic_v2 | 0 | 3 | 0.507-0.663 | 0.570 | +0.115 |
| dynamic_v2 | 1 | 3 | 0.796-0.850 | 0.825 | -0.115 |
| dynamic_v2 | 2 | 4 | 0.966-1.000 | 0.979 | -0.077 |

Measured generation continues after EOS to equalize work. Inference includes prefill and decoding; both timing measures exclude model loading and video processing. Exact-trial counts compare complete token sequences against AR. Unequal outputs do not demonstrate lossless speedup.

This small available-video subset is exploratory, not a full Video-MME accuracy evaluation. The synchronized total decode times include all dynamic selection/collection/refresh overhead.

Raw trials: R0_full_three_att.jsonl
Git base: d6c4feae6461722dabf4b537500ac080e8ed2ebd; worktree modified=True; exact source hashes are authoritative in the raw manifest.
