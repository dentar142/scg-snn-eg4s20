# Dia class error analysis

Hold-out subjects: ['sub003', 'sub006', 'sub009', 'sub013', 'sub020', 'sub021', 'sub024', 'sub026']  (40575 total windows, 6573 are Dia ground truth)

## Per-subject Dia recall

| Subject | n_dia | Dia recall | -> BG | -> Sys |
|---------|------:|-----------:|------:|-------:|
| sub021 | 995 | 62.61% | 11.2% | 26.2% |
| sub013 | 925 | 69.51% | 20.0% | 10.5% |
| sub006 | 800 | 80.25% | 1.8% | 18.0% |
| sub026 | 396 | 82.83% | 13.4% | 3.8% |
| sub024 | 826 | 83.54% | 14.8% | 1.7% |
| sub020 | 822 | 88.44% | 6.8% | 4.7% |
| sub003 | 889 | 93.25% | 0.2% | 6.5% |
| sub009 | 920 | 98.15% | 1.6% | 0.2% |

## Aggregate Dia-class confusion

| Predicted | Count | Fraction |
|-----------|------:|---------:|
| BG | 558 | 8.49% |
| Sys | 630 | 9.58% |
| Dia | 5385 | 81.93% |

Dia recall overall = 81.93%

## Per-modality signal energy (RMS)

| Modality | Correct Dia | Error Dia | Δ (err-corr) |
|----------|-------------:|----------:|------:|
| PVDF | 0.2488 ± 0.0002 | 0.2488 ± 0.0002 | -0.0000 |
| PZT | 0.2488 ± 0.0002 | 0.2487 ± 0.0002 | -0.0000 |
| ACC | 0.2488 ± 0.0003 | 0.2488 ± 0.0003 | +0.0000 |
| PCG | 0.2495 ± 0.0005 | 0.2495 ± 0.0005 | -0.0000 |
| ERB | 0.2478 ± 0.0112 | 0.2476 ± 0.0127 | -0.0001 |

## Findings

- Of all Dia errors, 47.0% go to BG and 53.0% go to Sys -- almost balanced.
- Worst subject for Dia recall: sub021 (62.61%)
- Best subject for Dia recall: sub009 (98.15%)
- Largest energy gap modality: ERB (|delta_RMS| = 0.0001)
- Total Dia recall = 81.93% (consistent with reported per-class acc 81.13% from full bench)
