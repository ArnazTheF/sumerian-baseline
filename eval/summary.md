Сводка
======

| task | baseline | variant | n | bleu | chrf | exact | len_ratio | empty_pred | degenerate | cer_corpus | cer_mean | cer_median | line_count_match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nogpu_mt | baseline_tm | raw | 1010 | 24.53 | 44.42 | 1.49 | 1.08 | 0.00 | 0.69 |  |  |  |  |
| nogpu_mt | baseline_tm | no_placeholder | 788 | 26.64 | 46.12 | 1.90 | 0.90 | 0.00 | 0.89 |  |  |  |  |
| nogpu_mt | baseline_tm | normalized | 788 | 26.64 | 46.12 | 1.90 | 0.90 | 0.00 | 0.89 |  |  |  |  |
| nogpu_mt | baseline_tm | normalized|seen | 0 |  |  |  |  |  |  |  |  |  |  |
| nogpu_mt | baseline_tm | normalized|unseen | 788 | 26.64 | 46.12 | 1.90 | 0.90 | 0.00 | 0.89 |  |  |  |  |
| nogpu_mt | baseline_zeroshot | raw | 1010 | 12.12 | 48.40 | 1.09 | 1.42 | 0.00 | 0.59 |  |  |  |  |
| nogpu_mt | baseline_zeroshot | no_placeholder | 788 | 15.31 | 50.94 | 1.40 | 1.13 | 0.00 | 0.63 |  |  |  |  |
| nogpu_mt | baseline_zeroshot | normalized | 788 | 39.22 | 59.21 | 18.91 | 1.02 | 0.38 | 0.63 |  |  |  |  |
| nogpu_mt | baseline_zeroshot | normalized|seen | 0 |  |  |  |  |  |  |  |  |  |  |
| nogpu_mt | baseline_zeroshot | normalized|unseen | 788 | 39.22 | 59.21 | 18.91 | 1.02 | 0.38 | 0.63 |  |  |  |  |
| nogpu_ocr_lineart | baseline_zeroshot | raw | 200 | 54.90 | 57.28 | 3.00 | 1.03 | 0.00 | 2.00 | 39.22 | 36.76 | 30.61 | 36.00 |
| nogpu_ocr_lineart | baseline_zeroshot | signs | 200 |  | 38.31 | 4.50 | 1.01 | 0.00 | 1.50 | 57.80 | 53.16 | 46.50 |  |
| nogpu_ocr_lineart | baseline_zeroshot | signs|seen | 0 |  |  |  |  |  |  |  |  |  |  |
| nogpu_ocr_lineart | baseline_zeroshot | signs|unseen | 200 |  | 38.31 | 4.50 | 1.01 | 0.00 | 1.50 | 57.80 | 53.16 | 46.50 |  |
| nogpu_translit | baseline_tm | raw | 657 | 32.82 | 40.69 | 8.98 | 0.66 | 0.00 | 3.35 | 69.90 | 67.19 | 63.64 |  |
| nogpu_translit | baseline_tm | raw|seen | 0 |  |  |  |  |  |  |  |  |  |  |
| nogpu_translit | baseline_tm | raw|unseen | 657 | 32.82 | 40.69 | 8.98 | 0.66 | 0.00 | 3.35 | 69.90 | 67.19 | 63.64 |  |
| nogpu_translit | baseline_zeroshot | raw | 657 | 91.16 | 92.22 | 61.95 | 0.99 | 0.30 | 0.76 | 9.93 | 3.78 | 0.00 |  |
| nogpu_translit | baseline_zeroshot | raw|seen | 0 |  |  |  |  |  |  |  |  |  |  |
| nogpu_translit | baseline_zeroshot | raw|unseen | 657 | 91.16 | 92.22 | 61.95 | 0.99 | 0.30 | 0.76 | 9.93 | 3.78 | 0.00 |  |

По длине эталона (основной вариант)
===================================

| task | baseline | variant | length_bin | n | bleu | chrf | exact | len_ratio | empty_pred | degenerate | cer_corpus | cer_mean | cer_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nogpu_mt | baseline_tm | normalized | (0.999, 3.0] | 296 | 3.44 | 38.66 | 3.72 | 1.25 | 0.00 | 0.00 |  |  |  |
| nogpu_mt | baseline_tm | normalized | (3.0, 4.0] | 129 | 12.95 | 44.87 | 1.55 | 0.91 | 0.00 | 1.55 |  |  |  |
| nogpu_mt | baseline_tm | normalized | (4.0, 6.0] | 174 | 21.80 | 44.43 | 1.15 | 0.84 | 0.00 | 2.30 |  |  |  |
| nogpu_mt | baseline_tm | normalized | (6.0, 37.0] | 189 | 32.16 | 50.37 | 0.00 | 0.79 | 0.00 | 0.53 |  |  |  |
| nogpu_mt | baseline_zeroshot | normalized | (0.999, 3.0] | 296 | 6.85 | 55.36 | 21.28 | 1.36 | 0.34 | 0.68 |  |  |  |
| nogpu_mt | baseline_zeroshot | normalized | (3.0, 4.0] | 129 | 38.37 | 61.72 | 26.36 | 1.04 | 0.78 | 2.33 |  |  |  |
| nogpu_mt | baseline_zeroshot | normalized | (4.0, 6.0] | 174 | 39.17 | 60.78 | 17.24 | 0.96 | 0.00 | 0.00 |  |  |  |
| nogpu_mt | baseline_zeroshot | normalized | (6.0, 37.0] | 189 | 40.85 | 59.14 | 11.64 | 0.92 | 0.53 | 0.00 |  |  |  |
| nogpu_ocr_lineart | baseline_zeroshot | signs | (6.999, 34.0] | 52 |  | 49.12 | 13.46 | 1.10 | 0.00 | 3.85 | 53.75 | 55.69 | 40.04 |
| nogpu_ocr_lineart | baseline_zeroshot | signs | (34.0, 48.5] | 48 |  | 55.16 | 4.17 | 1.03 | 0.00 | 2.08 | 37.28 | 37.47 | 25.22 |
| nogpu_ocr_lineart | baseline_zeroshot | signs | (48.5, 70.25] | 50 |  | 44.66 | 0.00 | 1.02 | 0.00 | 0.00 | 51.38 | 51.05 | 46.70 |
| nogpu_ocr_lineart | baseline_zeroshot | signs | (70.25, 196.0] | 50 |  | 25.60 | 0.00 | 0.97 | 0.00 | 0.00 | 70.62 | 67.69 | 72.82 |
| nogpu_translit | baseline_tm | raw | (2.999, 49.0] | 169 | 53.16 | 67.40 | 30.18 | 1.16 | 0.00 | 5.33 | 55.68 | 64.39 | 33.33 |
| nogpu_translit | baseline_tm | raw | (49.0, 89.0] | 162 | 45.15 | 50.54 | 2.47 | 0.89 | 0.00 | 4.32 | 63.61 | 63.28 | 62.16 |
| nogpu_translit | baseline_tm | raw | (89.0, 156.0] | 163 | 35.30 | 42.26 | 1.84 | 0.73 | 0.00 | 1.84 | 66.77 | 66.07 | 66.67 |
| nogpu_translit | baseline_tm | raw | (156.0, 893.0] | 163 | 24.86 | 34.42 | 0.61 | 0.53 | 0.00 | 1.84 | 74.15 | 75.10 | 74.58 |
| nogpu_translit | baseline_zeroshot | raw | (2.999, 49.0] | 169 | 96.57 | 97.64 | 82.84 | 1.00 | 1.18 | 0.00 | 2.21 | 3.12 | 0.00 |
| nogpu_translit | baseline_zeroshot | raw | (49.0, 89.0] | 162 | 97.58 | 98.35 | 71.60 | 1.01 | 0.00 | 0.00 | 1.51 | 1.51 | 0.00 |
| nogpu_translit | baseline_zeroshot | raw | (89.0, 156.0] | 163 | 97.49 | 98.22 | 66.26 | 1.00 | 0.00 | 0.00 | 1.58 | 1.57 | 0.00 |
| nogpu_translit | baseline_zeroshot | raw | (156.0, 893.0] | 163 | 87.14 | 87.88 | 26.38 | 0.98 | 0.00 | 3.07 | 16.04 | 8.91 | 1.85 |
