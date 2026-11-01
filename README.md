# IndicVoices TTS Preprocessing Pipeline

Automated preprocessing pipeline that turns the [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) dataset into a TTS-training-ready corpus across **22 Indian languages**.

| Final corpus | Duration | Avg DNSMOS (OVRL) | Avg SNR |
|:---:|:---:|:---:|:---:|
| **8,315 samples** | **16.97 hours** | **3.176** | **18.16 dB** |

---

## Table of Contents

- [Highlights](#highlights)
- [Pipeline Architecture](#pipeline-architecture)
- [Quality Results](#quality-results)
- [Languages Covered](#languages-covered)
- [Processing Details](#processing-details)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Output Files](#output-files)
- [Infrastructure](#infrastructure)
- [Limitations](#limitations)
- [Dataset Source](#dataset-source)

---

## Highlights

- **Two-pass design:** a fast CPU screening pass followed by a GPU deep-processing pass.
- **Multi-signal quality filtering:** duration, clipping, RMS, VAD, DNSMOS, SNR and a composite score.
- **Balanced sampling:** stratified by duration and gender, with a per-speaker cap for diversity.
- **Clean audio:** DeepFilterNet3 denoising and LUFS loudness normalization.
- **Text hygiene:** Unicode normalization, script-purity checks and deduplication.

---

## Pipeline Architecture

```text
INPUT: 1000 samples per language (22,000 total)
        ↓
PASS 1 — Fast Screening (CPU)
  • Hard filters: duration (1.5–15 s), clipping, RMS
  • Text filters: unintelligible tags, script purity, dedup
  • Silero VAD: voiced ratio > 0.40
  • DNSMOS raw: loose gate ≥ 1.5  →  ~700 candidates per language
        ↓
PRE-SELECTION
  • Top 600 by DNSMOS score
  • Stratified: duration buckets + gender balance
  • Speaker cap: max 15 samples per speaker
        ↓
PASS 2 — Deep Processing (GPU)
  • DNSMOS strict gate ≥ 2.7
  • DeepFilterNet3 noise removal
  • LUFS normalization: −23 LUFS (EBU R128)
  • Composite quality score ≥ 50  →  263–520 final samples kept per language
```

---

## Quality Results

| Metric          | All Input | Final Kept |
|-----------------|:---------:|:----------:|
| Samples         | 22,000    | 8,315      |
| DNSMOS OVRL     | ~2.3      | 3.176      |
| SNR             | ~10.9 dB  | 18.16 dB   |
| Voiced ratio    | ~0.57     | ~0.91      |
| Composite score | ~24       | 71.68      |

Plots visualizing these results are in [`plots/`](plots/).

---

## Languages Covered

22 languages:

Assamese · Bengali · Bodo · Dogri · Gujarati · Hindi · Kannada · Kashmiri · Konkani · Maithili · Malayalam · Manipuri · Marathi · Nepali · Odia · Punjabi · Sanskrit · Santali · Sindhi · Tamil · Telugu · Urdu

---

## Processing Details

### Audio

1. **Resampling:** all audio converted to 16 kHz mono
2. **Silence trimming:** librosa VAD (`top_db=35`)
3. **Peak normalization:** applied before DNSMOS scoring
4. **Noise removal:** DeepFilterNet3 on pre-selected candidates
5. **Loudness normalization:** −23 LUFS (EBU R128 broadcast standard)

### Text

1. **NFC normalization:** Unicode canonical form
2. **Unintelligible rejection:** removes samples tagged `<unintelligible>`, `<noise>`, etc.
3. **Script purity:** more than 85% native-script characters required
4. **Speaking-rate filter:** 0.3–7 words per second
5. **Deduplication:** audio hash and text hash

### Stratified Sampling

| Dimension | Rule |
|-----------|------|
| Duration strata | 30% (1.5–4 s), 45% (4–8 s), 25% (8–15 s) |
| Gender balance | 50/50 female/male within each stratum |
| Speaker diversity | Max 15 samples per speaker |
| Quality ranking | Samples ranked by DNSMOS within each stratum |

---

## Repository Structure

```text
indicvoices-tts-preprocessing/
├── config.yaml               # Pipeline configuration and thresholds
├── audit_raw.py              # Audit of the raw input data
├── filter_pipeline.py        # Core filtering / processing logic
├── run_v3.py                 # Main entry point for the pipeline
├── monitor.py                # Progress / run monitoring
├── visualize.py              # Generates the plots
│
├── kept_samples.csv          # Final 8,315 samples with all metrics
├── kept_samples_sample.csv   # Small preview of kept_samples.csv
├── filter_metrics.csv        # Metrics for all 19,783 processed samples
├── manifest.jsonl            # JSONL manifest for TTS training
├── filter_summary.json       # Pipeline summary statistics
│
├── plots/                    # Visualization outputs
├── .gitignore
└── README.md
```

---

## Getting Started

### Requirements

- Python 3.10
- NVIDIA GPU with CUDA 12.1 for Pass 2 (PyTorch 2.1.0)
- Models: Silero VAD, DeepFilterNet3, DNSMOS P.835
- Python packages listed in `requirements.txt`

```bash
git clone https://github.com/vivekyaadav/indicvoices-tts-preprocessing.git
cd indicvoices-tts-preprocessing
pip install -r requirements.txt
```

### Configuration

- Thresholds and filter settings: `config.yaml`
- Language list and output directory (`./output_filtered_v5`): top of `run_v3.py`

> **Note:** `run_v3.py` currently calls `os.chdir('/root/indic_filter')`, the working directory used on the vast.ai instance. Change this line to your own project path before running elsewhere.

### Run

```bash
python audit_raw.py
python run_v3.py
python monitor.py
python visualize.py
```

| Script | Purpose |
|--------|---------|
| `audit_raw.py` | Audit the raw input data |
| `run_v3.py` | Run the full two-pass pipeline |
| `monitor.py` | Monitor a running job |
| `visualize.py` | Generate the plots |

---

## Output Files

| File | Description |
|------|-------------|
| [`kept_samples.csv`](kept_samples.csv) | Final 8,315 samples with all quality metrics |
| [`kept_samples_sample.csv`](kept_samples_sample.csv) | Small preview of the kept samples |
| [`filter_metrics.csv`](filter_metrics.csv) | Metrics for all 19,783 processed samples |
| [`manifest.jsonl`](manifest.jsonl) | Manifest in JSONL format for TTS training |
| [`filter_summary.json`](filter_summary.json) | Pipeline-level summary statistics |
| [`plots/`](plots/) | Visualization plots (listed below) |

### Plots

| Plot | Shows |
|------|-------|
| `1_pipeline_funnel.png` | Samples remaining after each stage |
| `2_duration_comparison.png` | Duration distribution before vs. after |
| `3_gender_balance.png` | Gender balance per language |
| `4_dnsmos_distribution.png` | DNSMOS score distribution |
| `5_snr_improvement.png` | SNR before vs. after processing |
| `6_keep_rate.png` | Keep rate per language |
| `7_speaker_diversity.png` | Speaker diversity per language |
| `8_quality_dashboard.png` | Combined quality overview |

---

## Infrastructure

| Item | Details |
|------|---------|
| GPU | NVIDIA RTX 4060 Ti 16 GB (vast.ai) |
| Models | Silero VAD, DeepFilterNet3, DNSMOS P.835 |
| Framework | PyTorch 2.1.0, CUDA 12.1 |
| Runtime | ~25 minutes for all 22 languages |

---

## Limitations

- **Small scale:** ~0.77 hours per language (research scale, not production).
- **Gender imbalance in some languages**, reflecting the raw data: Manipuri 74% male, Sanskrit 72% male, Tamil 70% female.
- **No human MOS evaluation.**
- **No Whisper transcript verification**, due to Indic-script mismatch.

---

## Dataset Source

[IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) by AI4Bharat.

Please follow the dataset's license and terms of use when using this corpus.
