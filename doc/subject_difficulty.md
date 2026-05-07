# Per-subject difficulty profile (hold-out)

| Subject | Board acc | HR mean | HR std | R-peak SNR | Cycle align | PVDF | PZT | ACC | PCG | ERB |
|---------|----------:|--------:|-------:|-----------:|------------:|-----:|----:|----:|----:|----:|
| sub013 | 91.04% | 75.1 | 6.7 | 5.64 | 0.951 | 57.18 | 66.97 | 8.95 | 4.73 | 0.57 |
| sub021 | 91.06% | 67.5 | 3.9 | 5.52 | 0.858 | 22.83 | 17.11 | 3.67 | 6.02 | 2.30 |
| sub026 | 92.87% | 82.8 | 15.3 | 5.13 | 0.834 | 19.52 | 31.97 | 3.86 | 6.54 | 0.53 |
| sub006 | 93.13% | 76.4 | 9.5 | 5.45 | 0.776 | 31.30 | 18.60 | 4.62 | 5.68 | 0.40 |
| sub024 | 93.28% | 73.6 | 12.0 | 5.73 | 0.773 | 43.83 | 51.10 | 4.64 | 8.06 | 1.17 |
| sub020 | 95.91% | 71.3 | 5.6 | 5.50 | 0.919 | 7.44 | 7.67 | 2.94 | 4.42 | 0.53 |
| sub003 | 97.86% | 71.2 | 4.8 | 5.65 | 0.742 | 31.47 | 35.77 | 4.21 | 5.27 | 0.61 |
| sub009 | 98.76% | 69.5 | 7.1 | 5.43 | 0.955 | 62.48 | 48.26 | 3.47 | 8.48 | 0.36 |

## Correlations with board accuracy

| Feature | Pearson r |
|---------|----------:|
| Mean HR (bpm) | -0.327 |
| HR std (bpm) | -0.225 |
| R-peak SNR (median |amp|/std baseline) | +0.017 |
| Inter-cycle cos similarity (PVDF) | +0.018 |
| SNR (PVDF) | +0.128 |
| SNR (PZT) | -0.052 |
| SNR (ACC) | -0.501 |
| SNR (PCG) | +0.249 |
| SNR (ERB) | -0.498 |

## Findings

- Hardest subject: **sub013** (91.04%) — HR 75 bpm, alignment 0.951
- Easiest subject: **sub009** (98.76%) — HR 70 bpm, alignment 0.955
