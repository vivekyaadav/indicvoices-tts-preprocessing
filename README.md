# IndicVoices TTS Preprocessing Pipeline

## Overview
Automated preprocessing pipeline for building TTS-training-ready corpus from 
IndicVoices dataset across 22 Indian languages.

**Final Corpus:** 8,315 samples | 16.97 hours | Avg DNSMOS 3.176

---

## Pipeline Architecture

### Two-Pass Design
INPUT: 1000 samples per language (22,000 total)
↓
PASS 1 — Fast Screening (CPU)
• Hard filters: duration (1.5-15s), clipping, RMS
• Text filters: unintelligible tags, script purity, dedup
• Silero VAD: voiced ratio > 0.40
• DNSMOS raw: loose gate ≥ 1.5
→ ~700 candidates per language
↓
PRE-SELECTION
• Top 600 by DNSMOS score
• Stratified: duration buckets + gender balance
• Speaker cap: max 15 samples per speaker
↓
PASS 2 — Deep Processing (GPU)
• DNSMOS strict gate ≥ 2.7
• DeepFilterNet3: noise removal
• LUFS normalization: -23 LUFS (EBU R128)
• Composite quality score ≥ 50
→ 263-520 final kept per language

---

## Quality Metrics

| Metric | All Input | Final Kept |
|--------|-----------|------------|
| Samples | 22,000 | 8,315 |
| DNSMOS OVRL | ~2.3 | 3.176 |
| SNR | ~10.9 dB | 18.16 dB |
| Voiced ratio | ~0.57 | ~0.91 |
| Composite score | ~24 | 71.68 |

---

## Languages Covered (22)

Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri,
Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia,
Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu

---

## Audio Processing

1. **Resampling**: All audio → 16kHz mono
2. **Silence trimming**: librosa VAD (top_db=35)
3. **Peak normalization**: before DNSMOS scoring
4. **DeepFilterNet3**: noise removal on pre-selected candidates
5. **LUFS normalization**: -23 LUFS broadcast standard

---

## Text Processing

1. **NFC normalization**: Unicode canonical form
2. **Unintelligible rejection**: removes `<unintelligible>`, `<noise>`, etc.
3. **Script purity**: >85% native script characters
4. **Speaking rate filter**: 0.3-7 words/second
5. **Deduplication**: audio hash + text hash

---

## Stratified Sampling

- **Duration strata**: 30% (1.5-4s), 45% (4-8s), 25% (8-15s)
- **Gender balance**: 50/50 female/male within each stratum
- **Speaker diversity**: max 15 samples per speaker
- **Quality sorting**: samples ranked by DNSMOS score within strata

---

## Output Files
output_filtered_v5/
├── kept_samples.csv          # 8,315 final samples with all metrics
├── filter_metrics.csv        # All 19,783 processed samples
├── manifest.jsonl            # JSONL manifest for TTS training
├── filter_summary.json       # Pipeline summary statistics
├── README.md                 # This file
└── plots/                    # Visualization plots
├── 1_pipeline_funnel.png
├── 2_duration_comparison.png
├── 3_gender_balance.png
├── 4_dnsmos_distribution.png
├── 5_snr_improvement.png
├── 6_keep_rate.png
├── 7_speaker_diversity.png
└── 8_quality_dashboard.png

---

## Infrastructure

- **GPU**: NVIDIA RTX 4060 Ti 16GB (vast.ai)
- **Models**: Silero VAD, DeepFilterNet3, DNSMOS P.835
- **Processing time**: ~25 minutes for 22 languages
- **Framework**: PyTorch 2.1.0, CUDA 12.1

---

## Limitations

- ~0.77 hours per language (research scale, not production)
- Gender imbalance in some languages reflects raw data
  (Manipuri 74% male, Sanskrit 72% male, Tamil 70% female)
- No human MOS evaluation
- Whisper transcript verification not applied (Indic script mismatch)

---

## Dataset Source

IndicVoices: [ai4bharat/IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices)
