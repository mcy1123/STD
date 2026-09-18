# A100 GPU1 HSD feasibility result

Configuration: Qwen2.5-VL-7B target, Qwen2.5-VL-3B draft, Video-MME, 32 sampled frames, 64 generated tokens, 3 samples (`050-1`, `496-3`, `717-1`), 2 measured repeats. Timings exclude model/video loading and preprocessing.

| method | decode (s, mean) | inference (s, mean) | exact vs AR |
|---|---:|---:|---:|
| AR (7B dense) | 1.640 | 3.508 | reference |
| static STD (7B sparse→dense) | 1.897 | 3.822 | 6/6 |
| small dense (3B→7B dense) | 5.967 | 9.509 | 6/6 |
| HSD spike (3B→7B sparse→7B dense) | 5.227 | 8.874 | 6/6 |

HSD was therefore **2.76× slower than static STD in decode** and **2.32× slower in end-to-end generation** on this setup. It was also slower than plain AR (3.19× decode, 2.53× inference). All measured HSD outputs exactly matched the dense AR reference, so the nested verification path is functionally correct.

The bottleneck is structural: the 3B draft plus the intermediate sparse verifier adds two model passes, while the current sparse verifier still has per-block launch/cache-management overhead. The experiment validates correctness but does not show a useful speedup. A faster result would require a genuinely cheaper intermediate verifier (e.g. a separate slim model/subnetwork) and/or overlapped scheduling; merely inserting sparse attention between draft and dense verification is not sufficient.

Raw reproducible records: `a100_hsd_spike_20260909_l3.jsonl`.
