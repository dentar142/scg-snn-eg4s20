# Pareto sweep + mechanism analysis

Hold-out subjects: ['sub003', 'sub006', 'sub009', 'sub013', 'sub020', 'sub021', 'sub024', 'sub026']

## Per-config result table

| H | T | Params | Val acc | Mean L1 spikes/inf | Sparsity | LUT4 used (%) | BRAM9K | DSP | Synth status |
|---|---|-------:|-------:|-------------------:|---------:|--------------:|-------:|----:|---|
| 16 | 32 | 20,528 | 93.68% | 128.0 / 512 | 75.0% | 1,377 (7.03%) | 21/64 | 1/29 | ok |
| 32 | 8 | 41,056 | 94.20% | 76.5 / 256 | 70.1% | 2,100 (10.71%) | 39/64 | 1/29 | ok |
| 32 | 16 | 41,056 | 94.43% | 151.1 / 512 | 70.5% | 2,098 (10.70%) | 38/64 | 1/29 | ok |
| 32 | 32 | 41,056 | 94.26% | 323.8 / 1024 | 68.4% | 2,098 (10.70%) | 39/64 | 1/29 | ok |
| 32 | 48 | 41,056 | 94.38% | 471.9 / 1536 | 69.3% | 2,095 (10.69%) | 39/64 | 1/29 | ok |
| 64 | 32 | 82,112 | 94.33% | 725.8 / 2048 | 64.6% | - | - | - | FAILED (PHY-9009, MSlice 16131>4900) |

## Key observations

1. **Temporal-depth saturation**: at H=32, accuracy peaks at T=16 (94.43%) and stays within +/- 0.2 pp for T in {8, 16, 32, 48} -- temporal integration above T=8 is largely redundant. Latency drops 4x from T=32 to T=8 with 0.06 pp acc loss.
2. **Hidden-size diminishing return**: H=64 only gains +0.07 pp vs H=32 (94.33 vs 94.26%) but **fails synth on EG4S20** -- overshooting MSlice budget. The chip caps viable multimodal SNN at H<=32 with channel-bank.
3. **Spike sparsity**: all configs >= 64% sparse; H=16 reaches 75% sparsity, supporting the SNN energy-efficiency thesis.
4. **Amplitude robustness**: all configs hold acc within +/- 2 pp under input scaling 0.7-1.3x; H=64 most robust at scale=0.5 (+5.6 pp vs H=16).
