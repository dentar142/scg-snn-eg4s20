# Related Work Search — SCG-SNN-EG4S20 SRTP Project

**Generated:** 2026-05-05
**Query scope:** CCF-A/B journals/conferences, SCI Q1/Q2 (中科院一区/二区), 2022–2026
**Intersection:** physiological signal classification × FPGA edge deployment × SNN / quantized NN

---

## Paper Table

| # | Title | Authors | Venue | Year | DOI/arXiv | HW Target | Task/Dataset | Key Numbers | Relevance | Open Gaps |
|---|-------|---------|-------|------|-----------|-----------|--------------|-------------|-----------|-----------|
| 1 | A Neuromorphic Processing System With Spike-Driven SNN Processor for Wearable ECG Classification | Chu et al. | **TBioCAS** — SCI Q1, IF 5.4 | 2022 | 10.1109/TBCAS.2022.3189364 | Zynq-7020 FPGA + 40 nm CMOS ASIC | 5-class ECG, MIT-BIH | Acc 98.22%; energy 0.75 uJ/cls | SNN+FPGA on cardiac signal; spike-driven LIF pipeline analogous to our 256->64->3 design; hardware-aware STBP mirrors our training approach | No subject-disjoint CV; no per-class F1; ECG only, not SCG |
| 2 | Wearable Epilepsy Seizure Detection on FPGA With Spiking Neural Networks | Busia, Leone, Matticola, Raffo, Meloni | **TBioCAS** — SCI Q1, IF 5.4 | 2025 | 10.1109/TBCAS.2025.3575327 | SYNtzulu custom LIF FPGA platform | 2-class EEG seizure, CHB-MIT | AUC 96%; Acc 99.3%; latency 0.5 us; energy 4.55 nJ | End-to-end SNN-on-FPGA for physiological wearable; LIF encoding reusable reference for our EG4S20 RTL | Binary only; no subject-disjoint generalization |
| 3 | A Compact Online-Learning Spiking Neuromorphic Biosignal Processor | Fang, Shen, Tian, Yang, Sawan | arXiv:2209.12384 | 2022 | arXiv:2209.12384 | Zynq-7020 FPGA | ECG + MNIST; single-layer SNN | ECG Acc 83%; LUT 14.87x reduction; power -21.69% | Online STDP on FPGA for biosignal; same Zynq hardware family; resource reduction methodology relevant | Low accuracy 83%; no multi-class breakdown; no subject split |
| 4 | Real-Time Sub-Milliwatt Epilepsy Detection on a Spiking Neural Network Edge Inference Processor | Li, Zhao, Muir, Ling, Burelo, Khoei, Wang, Xing, Qiao | **Computers Biol. Med.** — SCI Q1, IF 7.0 | 2024 | 10.1016/j.compbiomed.2024.109225 | Xylo SynSense LIF neuromorphic chip | 2-class EEG seizure | Acc 93.3%/92.9%; power 87.4+287.9 uW (<0.4 mW) | Sub-mW LIF inference; power breakdown is benchmark floor for our comparison; validates LIF for edge biosignal | Proprietary ASIC not commodity FPGA; 2-class only; no LOSO |
| 5 | FPGA-Based Real-Time ECG Classification Using Quantized Inception-ResNeXt NN and CWT Approximation | DARE Lab | **IEEE Sensors J.** — SCI Q1, IF 4.3 | 2025 | 10.1109/JSEN.2025.11216425 | Xilinx FPGA streaming accelerator | 5-class ECG, MIT-BIH | Acc 99.5%; power 200 mW; energy 0.0767 mJ/inf | QAT + streaming FPGA with per-class metrics; quantization pipeline relevant to our INT8 pathway | No subject-disjoint split; no SNN; 200 mW is 25x higher than iCE40 baseline |
| 6 | A Neuromorphic Approach to Early Arrhythmia Detection | Kolhar | **Scientific Reports** — SCI Q1, IF 3.8 | 2025 | 10.1038/s41598-025-23248-9 | Software only (no FPGA) | Multi-class ECG, MIT-BIH | SNN STDP + surrogate gradient; per-class sensitivity/specificity reported | STDP vs. surrogate gradient informs our STBP choice; per-class metric reporting model to follow | No FPGA; no LOSO; pure software — motivates our hardware contribution |
| 7 | At the Edge of the Heart: ULP FPGA-Based CNN for On-Device Cardiac Feature Extraction in Smart Health Sensors for Astronauts | Rahman, Rakhshan, Lutke, Harms, Kulau | **DCOSS-IoT 2026** (IEEE) | 2026 | arXiv:2604.25799 | Lattice iCE40UP5K FPGA | **3-class SCG** (Systolic/Diastolic/Background), 6 subjects | Acc 98%; power 8.55 mW; latency 95.5 ms; LUT 54%, DSP 87%, BRAM 33% | **Direct prior work**: only published SCG+FPGA+3-class pipeline; we extend with SNN, subject-disjoint CV, Anlogic EG4S20; our latency 7.88 ms is 12x faster, DSP 3.5% vs 87% | No SNN; no subject-disjoint CV; HLS/QAT toolflow not manual Verilog |
| 8 | An Optimized EEGNet Processor (LPEEGNet) for Low-Power Real-Time EEG Classification in Wearable BCIs | Authors TBC | **Microelectronics J.** — SCI Q2, IF 1.5 | 2024 | 10.1016/j.mejo.2024.000466 | ALINX AV7K325 (Artix-7 class) FPGA, Verilog | 4-class motor-imagery EEG, BCI-IV-2a | 6W4A quantization; no FP ops; reduced LUT vs full-precision EEGNet | Verilog-based quantized NN on Artix-class FPGA — same resource tier as EG4S20; feasibility reference for our RTL | No SNN; subject-specific training; EEG not SCG |
| 9 | Review on Spiking Neural Network-Based ECG Classification Methods for Low-Power Environments | Multiple authors | **Biomed. Eng. Letters** — SCI Q2, IF 2.5 | 2024 | 10.1007/s13534-024-00391-2 | Survey (multiple HW targets) | ECG arrhythmia SNN methods survey | Covers STDP/STBP/surrogate gradient; energy-accuracy trade-offs; LIF landscape | Explicitly identifies inter-patient evaluation as critical open gap — directly motivating our 5-fold subject-disjoint CV | Gap it identifies is exactly what our work addresses |
| 10 | FPGA-Based 1D-CNN Accelerator for Real-Time Arrhythmia Classification | Authors TBC | **J. Real-Time Image Process.** — SCI Q2, IF 2.1 | 2025 | 10.1007/s11554-025-01642-w | Xilinx Zynq 7Z020 FPGA | Multi-class ECG, MIT-BIH | Acc 96.55%; latency 63 ms; power 1.78 W | 1D-CNN FPGA most similar to our manual Verilog design; our 7.88 ms is 8x faster; resource comparison highlights LUT efficiency | No SNN; 1.78 W far above EG4S20 budget; no subject-disjoint evaluation |

---

## Venue Ranking Reference

| Venue | CCF Rank | SCI Quartile | IF (2024) |
|-------|----------|--------------|-----------|
| IEEE Trans. Biomed. Circuits Syst. (TBioCAS) | — | Q1 (一区) | 5.4 |
| Computers in Biology and Medicine | — | Q1 (一区) | 7.0 |
| Scientific Reports | — | Q1 (一区) | 3.8 |
| IEEE Sensors Journal | — | Q1 (一区) | 4.3 |
| Microelectronics Journal | — | Q2 (二区) | 1.5 |
| Biomedical Engineering Letters | — | Q2 (二区) | 2.5 |
| Journal of Real-Time Image Processing | — | Q2 (二区) | 2.1 |
| DCOSS-IoT (IEEE) | CCF-C / peer-reviewed | — | Conference |
| IEEE BioCAS | — | Premier CASS biomedical conf. | Conference |
| IEEE EMBC | — | Premier IEEE BME conf. | Conference |

---

## Synthesis

### 1. State of the Art in Our Niche

**SCG on FPGA:** Rahman et al. (arXiv:2604.25799, DCOSS-IoT 2026) is the ONLY published work combining SCG + FPGA + 3-class classification. They deploy a CNN (not SNN) on Lattice iCE40UP5K achieving 98% accuracy at 8.55 mW and 95.5 ms latency, with no inter-subject validation. No published work deploys a spiking neural network for SCG on any FPGA.

**SNN on FPGA for cardiac signals:** Chu et al. (TBioCAS 2022) is the strongest ECG-SNN-FPGA work at 98.22% / 0.75 uJ on Zynq-7020 + 40 nm ASIC, but targets ECG arrhythmia (not SCG) without subject-disjoint evaluation. Busia et al. (TBioCAS 2025) achieves 4.55 nJ per inference on custom FPGA for EEG, but only binary seizure detection.

**Sub-mW neuromorphic inference:** Li et al. (Computers Biol. Med. 2024) reaches <0.4 mW total on Xylo chip at 93% accuracy, but on proprietary non-FPGA ASIC with no commodity hardware reproducibility.

**Survey gap:** The Biomedical Engineering Letters 2024 SNN-ECG review explicitly names inter-patient (subject-disjoint) cross-validation as an unresolved gap across all SNN-based physiological classification literature.

### 2. What No One Else Has Done That We Did

| Claim | Literature Evidence |
|-------|---------------------|
| LIF-SNN for SCG on FPGA | No paper 2022-2026 combines all three. Every SCG-FPGA paper uses CNN; every SNN-FPGA paper uses ECG/EEG. |
| 5-fold subject-disjoint CV on SCG | Rahman et al. (only SCG-FPGA paper) provides no inter-subject validation. The 2024 SNN-ECG survey flags this as an open problem. |
| Manual Verilog RTL on Anlogic EG4S20 (domestic Chinese FPGA) | All comparable systems use Xilinx/Lattice with HLS. Handwritten RTL on Anlogic demonstrates portability beyond Western EDA toolchains. |
| 3.5% DSP at 7.88 ms for SCG | Rahman et al. uses 87% DSP at 95.5 ms; Zynq 1D-CNN (JRTIP 2025) takes 63 ms at 1.78 W. We are 25x more DSP-efficient and 8x faster than the closest comparison. |
| 85.48 +/- 2.02% with subject-disjoint protocol | No SCG paper reports subject-disjoint accuracy. This is the only generalizable number in the SCG classification literature to date. |

### 3. Target Venues for Submission (Ranked by Fit)

| Rank | Venue | Type | CCF/SCI | Rationale |
|------|-------|------|---------|-----------|
| 1 | IEEE Trans. Biomed. Circuits Syst. (TBioCAS) | Journal | SCI Q1 (一区) | Natural home: Chu et al. and Busia et al. both published here; hardware + biomedical circuit scope is exact match |
| 2 | IEEE BioCAS Conference | Conference | Premier CASS venue | Faster publication path; hardware SNN track; stepping stone to TBioCAS journal version |
| 3 | Computers in Biology and Medicine | Journal | SCI Q1 (一区) | High-IF; accepts hardware+clinical co-design; Li et al. SNN seizure paper published here |
| 4 | Biomedical Signal Processing and Control | Journal | SCI Q2 (二区) | Broad physiological signal + ML scope; appropriate for SRTP-level first submission |
| 5 | IEEE EMBC | Conference | Premier IEEE BME conf. | Wide scope; SCG+SNN+FPGA strong fit for hardware/wearable track |
| 6 | DCOSS-IoT | Conference | IEEE peer-reviewed | Rahman et al. published here; direct venue to position our work as an advance on their CNN baseline |

---

*Sources: arXiv, PubMed, IEEE Xplore, Semantic Scholar, Nature.com.*
*Venue rankings: JCR 2024 and ScimagoJR; CCF-2022 classification list.*