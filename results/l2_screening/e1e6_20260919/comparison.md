# dynamic-routing configuration comparison

| run | host | gpu | collector | query | refresh | timing_valid | samples | rounds | exact T1 |
|---|---|---:|---|---|---|---|---:|---:|---|
| A_base_v2_i1 | gpu23 | 1 | v2 | three | incremental | False | 10 | 169 | 30/30 |
| B_E1_v2_i4 | gpu23 | 1 | v2 | three | incremental | False | 10 | 164 | 30/30 |
| C_E6_v3_i1 | gpu23 | 1 | v3 | three | incremental | False | 10 | 169 | 30/30 |
| D_E1E6_v3_i4 | gpu23 | 1 | v3 | three | incremental | False | 10 | 161 | 30/30 |

## Paired acceptance vs the run's own static baseline

| run | mean static | mean d accept | mean d accept_len | corr(static, d) | low-headroom d |
|---|---:|---:|---:|---:|---:|
| A_base_v2_i1 | 0.8161 | -0.0429 | -0.3748 | -0.670 | +0.0767 |
| B_E1_v2_i4 | 0.8161 | -0.0121 | -0.0533 | -0.391 | +0.0366 |
| C_E6_v3_i1 | 0.8161 | -0.0454 | -0.3926 | -0.715 | +0.1012 |
| D_E1E6_v3_i4 | 0.8161 | -0.0008 | +0.0473 | -0.609 | +0.1170 |

## Per-sample d accept (only samples present in every run)

| sample | static | A_base_v2_i1 | B_E1_v2_i4 | C_E6_v3_i1 | D_E1E6_v3_i4 |
|---|---:|---:|---:|---:|---:|
| 050-1 | 0.623 | +0.044 | -0.074 | +0.124 | -0.048 |
| 102-2 | 0.891 | -0.203 | -0.170 | -0.242 | -0.199 |
| 154-3 | 0.692 | +0.011 | +0.086 | +0.004 | +0.178 |
| 445-2 | 0.790 | -0.143 | -0.053 | -0.101 | -0.053 |
| 496-3 | 0.864 | -0.102 | -0.006 | -0.152 | -0.006 |
| 504-2 | 0.857 | -0.061 | +0.070 | -0.061 | +0.000 |
| 599-1 | 1.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 647-1 | 0.966 | -0.024 | -0.031 | -0.031 | +0.025 |
| 717-1 | 0.983 | -0.126 | -0.040 | -0.170 | -0.126 |
| 754-1 | 0.495 | +0.175 | +0.098 | +0.175 | +0.221 |

## Headroom strata (equal-count bins, LOW headroom first)

**A_base_v2_i1**

| bin | n | static range | mean d accept |
|---:|---:|---|---:|
| 0 | 3 | 0.495-0.692 | +0.0767 |
| 1 | 3 | 0.790-0.864 | -0.1021 |
| 2 | 4 | 0.891-1.000 | -0.0882 |

**B_E1_v2_i4**

| bin | n | static range | mean d accept |
|---:|---:|---|---:|
| 0 | 3 | 0.495-0.692 | +0.0366 |
| 1 | 3 | 0.790-0.864 | +0.0035 |
| 2 | 4 | 0.891-1.000 | -0.0604 |

**C_E6_v3_i1**

| bin | n | static range | mean d accept |
|---:|---:|---|---:|
| 0 | 3 | 0.495-0.692 | +0.1012 |
| 1 | 3 | 0.790-0.864 | -0.1047 |
| 2 | 4 | 0.891-1.000 | -0.1108 |

**D_E1E6_v3_i4**

| bin | n | static range | mean d accept |
|---:|---:|---|---:|
| 0 | 3 | 0.495-0.692 | +0.1170 |
| 1 | 3 | 0.790-0.864 | -0.0200 |
| 2 | 4 | 0.891-1.000 | -0.0749 |

## Per-round cost (wall-clock is only comparable when timing_valid)

| run | static decode ms/round | dynamic decode ms/round | toll ms/round | refresh ms/round | refresh p99 | selection ms/round | mean changed ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_base_v2_i1 | 287.8 | 531.2 | +243.4 | 86.5 | 416.9 | 9.6 | 0.467 |
| B_E1_v2_i4 | 338.0 | 430.2 | +92.2 | 15.7 | 85.0 | 5.1 | 0.136 |
| C_E6_v3_i1 | 296.0 | 552.0 | +256.0 | 70.6 | 416.0 | 11.3 | 0.436 |
| D_E1E6_v3_i4 | 283.7 | 446.6 | +162.9 | 18.6 | 110.7 | 5.1 | 0.125 |
