# Phase 1: tuning BM25 on development queries

Each grid is nDCG@10 on development queries only. The best setting is then compared with the default (k1 = 0.9, b = 0.4) on test queries, once. Intervals are 95% paired bootstrap over queries; p-values are two-sided paired randomization tests.

## scifact

Development: 809 queries from the train split. Test: 300 queries.

| k1 \ b | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.75 | 0.9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.6866 | 0.6887 | 0.6897 | 0.6904 | 0.6903 | 0.6894 | 0.6894 |
| 0.7 | 0.6932 | 0.6944 | 0.6951 | 0.6970 | 0.6950 | 0.6947 | 0.6927 |
| 0.9 | 0.6947 | 0.6960 | 0.6965 | 0.6974 | 0.6977 | 0.6956 | 0.6956 |
| 1.2 | 0.6960 | 0.6981 | 0.6981 | 0.6984 | 0.6980 | **0.6990** | 0.6965 |
| 1.5 | 0.6967 | 0.6968 | 0.6978 | 0.6970 | 0.6959 | 0.6959 | 0.6958 |
| 2.0 | 0.6889 | 0.6928 | 0.6930 | 0.6921 | 0.6921 | 0.6907 | 0.6923 |

Best on development: k1 = 1.2, b = 0.75 (0.6990 against 0.6965 for the default).

On test, tuned against default: 0.6830 against 0.6776, difference +0.0053 (95% interval -0.0034 to +0.0140, p = 0.237).

| Analyzer (development queries, default k1 and b) | nDCG@10 | Difference from default | 95% interval | p |
| --- | --- | --- | --- | --- |
| default analyzer | 0.6965 | | | |
| no stemming | 0.6665 | -0.0301 | -0.0425 to -0.0177 | 0.0001 |
| no stopword removal | 0.6944 | -0.0022 | -0.0061 to +0.0013 | 0.267 |

## nfcorpus

Development: 324 queries from the dev split. Test: 323 queries.

| k1 \ b | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.75 | 0.9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.2841 | 0.2858 | 0.2865 | 0.2859 | 0.2859 | 0.2852 | 0.2842 |
| 0.7 | 0.2836 | 0.2860 | 0.2872 | 0.2879 | 0.2875 | 0.2859 | 0.2861 |
| 0.9 | 0.2851 | 0.2867 | 0.2878 | 0.2890 | 0.2887 | 0.2875 | 0.2860 |
| 1.2 | 0.2886 | 0.2900 | 0.2895 | 0.2909 | 0.2903 | 0.2888 | 0.2894 |
| 1.5 | 0.2895 | 0.2906 | 0.2917 | **0.2920** | 0.2918 | 0.2897 | 0.2893 |
| 2.0 | 0.2901 | 0.2902 | 0.2907 | 0.2909 | 0.2915 | 0.2906 | 0.2892 |

Best on development: k1 = 1.5, b = 0.5 (0.2920 against 0.2878 for the default).

On test, tuned against default: 0.3259 against 0.3205, difference +0.0054 (95% interval +0.0018 to +0.0090, p = 0.0034).

| Analyzer (development queries, default k1 and b) | nDCG@10 | Difference from default | 95% interval | p |
| --- | --- | --- | --- | --- |
| default analyzer | 0.2878 | | | |
| no stemming | 0.2661 | -0.0217 | -0.0344 to -0.0094 | 0.0008 |
| no stopword removal | 0.2872 | -0.0006 | -0.0037 to +0.0022 | 0.726 |

## fiqa

Development: 500 queries from the dev split. Test: 648 queries.

| k1 \ b | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.75 | 0.9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.2245 | 0.2424 | 0.2467 | 0.2546 | 0.2599 | 0.2665 | 0.2635 |
| 0.7 | 0.2327 | 0.2453 | 0.2559 | 0.2625 | 0.2657 | 0.2697 | 0.2678 |
| 0.9 | 0.2353 | 0.2509 | 0.2572 | 0.2658 | 0.2672 | **0.2708** | 0.2661 |
| 1.2 | 0.2357 | 0.2501 | 0.2613 | 0.2651 | 0.2673 | 0.2670 | 0.2589 |
| 1.5 | 0.2359 | 0.2469 | 0.2570 | 0.2622 | 0.2677 | 0.2592 | 0.2534 |
| 2.0 | 0.2289 | 0.2423 | 0.2521 | 0.2569 | 0.2563 | 0.2526 | 0.2473 |

Best on development: k1 = 0.9, b = 0.75 (0.2708 against 0.2572 for the default).

On test, tuned against default: 0.2499 against 0.2354, difference +0.0145 (95% interval +0.0051 to +0.0242, p = 0.0027).

| Analyzer (development queries, default k1 and b) | nDCG@10 | Difference from default | 95% interval | p |
| --- | --- | --- | --- | --- |
| default analyzer | 0.2572 | | | |
| no stemming | 0.2371 | -0.0201 | -0.0343 to -0.0065 | 0.0064 |
| no stopword removal | 0.2579 | +0.0007 | -0.0045 to +0.0060 | 0.805 |
