"""
filter_pipeline.py
==================
Full TTS filter pipeline for IndicVoices.

Stages:
    0. Hard filters     — duration, clipping, rms, silence
    1. Text filters     — NFC, script purity, speaking rate, dedup
    2. Silero VAD       — voiced_ratio, speech_quality (GPU)
    3. DNSMOS           — perceptual quality score (GPU)
    4. DeepFilterNet    — noise removal on flagged samples (GPU)
    5. Whisper CER      — transcript match check (GPU)
    6. Composite score  — weighted 0-100
    7. Stratified sample— 1000 per language

Config driven — all thresholds in config.yaml.
"""

from __future__ import annotations
import os
import csv
import json
import time
import hashlib
import unicodedata
import logging
import numpy as np
import soundfile as sf
import librosa
import yaml
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Thread

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# GPU lock — serializes GPU ops across threads
# CPU stages run in parallel, GPU stages are serialized
import threading
_GPU_LOCK = threading.Lock()

SR = 16000

# ── Script ranges for purity check ───────────────────────────────────────────
SCRIPT_RANGES = {
    'assamese' : [(0x0980, 0x09FF)],
    'bengali'  : [(0x0980, 0x09FF)],
    'bodo'     : [(0x0900, 0x097F)],
    'dogri'    : [(0x0900, 0x097F)],
    'gujarati' : [(0x0A80, 0x0AFF)],
    'hindi'    : [(0x0900, 0x097F)],
    'kannada'  : [(0x0C80, 0x0CFF)],
    'kashmiri' : [(0x0600, 0x06FF)],
    'konkani'  : [(0x0900, 0x097F)],
    'maithili' : [(0x0900, 0x097F)],
    'malayalam': [(0x0D00, 0x0D7F)],
    'manipuri' : [(0xABC0, 0xABFF), (0x0900, 0x097F)],
    'marathi'  : [(0x0900, 0x097F)],
    'nepali'   : [(0x0900, 0x097F)],
    'odia'     : [(0x0B00, 0x0B7F)],
    'punjabi'  : [(0x0A00, 0x0A7F)],
    'sanskrit' : [(0x0900, 0x097F)],
    'santali'  : [(0x1C50, 0x1C7F)],
    'sindhi'   : [(0x0900, 0x097F)],  # Devanagari Sindhi
    'tamil'    : [(0x0B80, 0x0BFF)],
    'telugu'   : [(0x0C00, 0x0C7F)],
    'urdu'     : [(0x0600, 0x06FF)],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    language      : str
    audio         : np.ndarray
    sr            : int
    transcript    : str
    speaker_id    : str
    gender        : str
    age_group     : str
    area          : str
    state         : str
    duration_sec  : float
    verification_report: str = ''

    # Computed metrics
    word_count    : int   = 0
    words_per_sec : float = 0.0
    nfc_ok        : bool  = True
    script_purity : float = 1.0
    audio_hash    : str   = ''
    text_hash     : str   = ''
    rms_energy    : float = 0.0
    clipping_ratio: float = 0.0
    snr_db        : float = 0.0
    silence_ratio : float = 0.0
    voiced_ratio  : float = 0.0
    spectral_flatness: float = 0.0
    speech_quality: float = 0.0
    dnsmos_sig    : float = 0.0
    dnsmos_bak    : float = 0.0
    dnsmos_ovrl   : float = 0.0
    cer           : float = 0.0
    composite_score: float = 0.0
    deepfilter_applied: bool = False

    # Decision
    keep          : bool  = False
    reject_reason : str   = ''
    filter_stage  : str   = ''


# ─────────────────────────────────────────────────────────────────────────────
# Audio preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_audio(raw: np.ndarray, sr_in: int,
                  top_db: float = 35.0) -> tuple[np.ndarray, int]:
    import pyloudnorm as pyln
    if raw.ndim > 1:
        raw = raw.mean(axis=0)
    raw = raw.flatten()
    audio = raw.astype(np.float32)
    if sr_in != SR:
        audio = librosa.resample(audio, orig_sr=sr_in, target_sr=SR)
    # Trim silence
    trimmed, _ = librosa.effects.trim(
        audio, top_db=top_db, frame_length=512, hop_length=128)
    if len(trimmed) >= int(0.3 * SR):
        audio = trimmed
    # Peak normalize only (LUFS done after DNSMOS scoring)
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio, SR


def lufs_normalize(audio: np.ndarray, sr: int = 16000,
                   target_lufs: float = -23.0) -> np.ndarray:
    """Apply LUFS normalization after DNSMOS scoring."""
    import pyloudnorm as pyln
    import warnings
    try:
        if len(audio) >= int(0.5 * sr):
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            if loudness > -70 and not np.isinf(loudness):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    audio = pyln.normalize.loudness(audio, loudness, target_lufs)
                # Hard clip prevention
                peak = np.abs(audio).max()
                if peak > 0.95:
                    audio = audio / peak * 0.95
    except Exception:
        pass
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — Hard filters
# ─────────────────────────────────────────────────────────────────────────────

def stage0_hard_filters(s: Sample, cfg: dict) -> bool:
    hf = cfg['hard_filters']

    if s.duration_sec < hf['duration_min_sec']:
        s.reject_reason = f"too_short:{s.duration_sec:.2f}s"
        s.filter_stage  = 'stage0'
        return False

    if s.duration_sec > hf['duration_max_sec']:
        s.reject_reason = f"too_long:{s.duration_sec:.2f}s"
        s.filter_stage  = 'stage0'
        return False

    # RMS                                  
    s.rms_energy = float(np.sqrt(np.mean(s.audio ** 2)))
    if s.rms_energy < hf['rms_min']:
        s.reject_reason = f"too_quiet:{s.rms_energy:.5f}"
        s.filter_stage  = 'stage0'
        return False

    # Clipping
    s.clipping_ratio = float(np.mean(np.abs(s.audio) >= 0.99))
    if s.clipping_ratio > hf['clipping_max_ratio']:
        s.reject_reason = f"clipping:{s.clipping_ratio:.3f}"
        s.filter_stage  = 'stage0'
        return False

    # SNR (fast energy-based estimate)
    frame_len = int(0.025 * SR)
    hop_len   = int(0.010 * SR)
    frames    = librosa.util.frame(
        s.audio, frame_length=frame_len, hop_length=hop_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=0))
    rms = rms[rms > 0]
    if len(rms) >= 4:
        noise  = np.percentile(rms, 20)
        signal = np.percentile(rms, 60)
        s.snr_db = float(10 * np.log10(
            signal ** 2 / noise ** 2)) if noise > 0 else 40.0
    if s.snr_db < hf['snr_min_db']:
        s.reject_reason = f"too_noisy:{s.snr_db:.1f}dB"
        s.filter_stage  = 'stage0'
        return False

    # Silence ratio (energy-based, fast)
    rms_frames = librosa.feature.rms(
        y=s.audio, frame_length=512, hop_length=128)[0]
    rms_db = librosa.amplitude_to_db(rms_frames, ref=np.max)
    voiced = (rms_db > -30).sum()
    s.silence_ratio = 1.0 - voiced / max(len(rms_db), 1)
    s.voiced_ratio  = 1.0 - s.silence_ratio
    if s.silence_ratio > hf['silence_max_ratio']:
        s.reject_reason = f"too_silent:{s.silence_ratio:.2f}"
        s.filter_stage  = 'stage0'
        return False

    # Spectral flatness
    S    = np.abs(librosa.stft(s.audio, n_fft=512, hop_length=128))
    s.spectral_flatness = float(
        np.mean(librosa.feature.spectral_flatness(S=S)))

    # Verification flag
    if hf.get('drop_verification_flags', True):
        if s.verification_report and 'reject' in s.verification_report.lower():
            s.reject_reason = 'human_rejected'
            s.filter_stage  = 'stage0'
            return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Text filters
# ─────────────────────────────────────────────────────────────────────────────

def stage1_text_filters(s: Sample, cfg: dict,
                        audio_hashes: set,
                        text_hashes: set) -> bool:
    hf = cfg['hard_filters']

    # NFC normalization
    text_nfc = unicodedata.normalize('NFC', s.transcript or '')
    s.nfc_ok = (text_nfc == s.transcript)
    s.transcript = text_nfc

    # Script purity
    ranges = SCRIPT_RANGES.get(s.language, [])
    if ranges and text_nfc:
        total = sum(1 for c in text_nfc
                    if c.strip() and not c.isascii())
        if total > 0:
            in_script = sum(
                1 for c in text_nfc
                if any(lo <= ord(c) <= hi for lo, hi in ranges))
            s.script_purity = round(in_script / total, 4)
        else:
            s.script_purity = 1.0
    else:
        s.script_purity = 1.0

    if s.script_purity < hf.get('script_purity_min', 0.85):
        s.reject_reason = f"wrong_script:{s.script_purity:.2f}"
        s.filter_stage  = 'stage1'
        return False

    # Speaking rate
    s.word_count    = len(text_nfc.split())
    s.words_per_sec = round(
        s.word_count / max(s.duration_sec, 1), 3)

    if s.words_per_sec < hf.get('words_per_sec_min', 0.3):
        s.reject_reason = f"too_slow:{s.words_per_sec:.2f}wps"
        s.filter_stage  = 'stage1'
        return False

    if s.words_per_sec > hf.get('words_per_sec_max', 7.0):
        s.reject_reason = f"too_fast:{s.words_per_sec:.2f}wps"
        s.filter_stage  = 'stage1'
        return False

    # Audio deduplication
    if hf.get('drop_audio_duplicates', True):
        s.audio_hash = hashlib.md5(s.audio.tobytes()).hexdigest()
        if s.audio_hash in audio_hashes:
            s.reject_reason = 'duplicate_audio'
            s.filter_stage  = 'stage1'
            return False
        audio_hashes.add(s.audio_hash)

    # Text deduplication
    if hf.get('drop_text_duplicates', False):
        s.text_hash = hashlib.md5(text_nfc.encode()).hexdigest()
        if s.text_hash in text_hashes:
            s.reject_reason = 'duplicate_text'
            s.filter_stage  = 'stage1'
            return False
        text_hashes.add(s.text_hash)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Silero VAD (GPU)
# ─────────────────────────────────────────────────────────────────────────────

_silero_model = None

def _get_silero():
    global _silero_model
    if _silero_model is None:
        from silero_vad import load_silero_vad
        _silero_model = load_silero_vad()
    return _silero_model


def stage2_silero_vad(s: Sample, cfg: dict) -> bool:
    import torch
    from silero_vad import get_speech_timestamps

    model    = _get_silero()
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model    = model.to(device)
    waveform = torch.tensor(s.audio).float().to(device)

    with _GPU_LOCK:
        ts = get_speech_timestamps(
            waveform, model,
            sampling_rate=SR,
            threshold=0.5,
            return_seconds=False
        )

    voiced_samples = sum(t['end'] - t['start'] for t in ts)
    s.voiced_ratio  = voiced_samples / max(len(s.audio), 1)
    s.silence_ratio = 1.0 - s.voiced_ratio

    # Per-chunk probability → speech quality
    window, probs = 512, []
    for i in range(0, len(s.audio) - window, window):
        chunk = waveform[i:i + window].unsqueeze(0)
        with torch.no_grad():
            probs.append(model(chunk, SR).item())
    s.speech_quality = float(np.mean(probs)) if probs else 0.0

    hf = cfg['hard_filters']
    if s.voiced_ratio < hf.get('voiced_min_ratio', 0.40):
        s.reject_reason = f"low_voiced:{s.voiced_ratio:.2f}"
        s.filter_stage  = 'stage2'
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — DNSMOS P.835 (GPU/CPU)
# ─────────────────────────────────────────────────────────────────────────────

_dnsmos_session = None

def _get_dnsmos_session():
    global _dnsmos_session
    if _dnsmos_session is None:
        import onnxruntime as ort
        # Download DNSMOS model if not present
        model_path = '/root/indic_filter/dnsmos_p835.onnx'
        if not os.path.exists(model_path):
            import urllib.request
            url = ('https://github.com/microsoft/DNS-Challenge/'
                   'raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx')
            print("Downloading DNSMOS model...")
            urllib.request.urlretrieve(url, model_path)
            print("DNSMOS model downloaded")
        _dnsmos_session = ort.InferenceSession(model_path)
    return _dnsmos_session


def stage3_dnsmos(s: Sample, cfg: dict) -> bool:
    if not cfg.get('dnsmos', {}).get('enabled', False):
        return True

    try:
        import numpy as np

        # DNSMOS expects 16kHz mono, normalized
        audio = s.audio.copy()

        # Compute features for DNSMOS
        # Use 10s window if longer, else pad
        target_len = 144160  # DNSMOS P.835 model fixed input size
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        sess   = _get_dnsmos_session()
        inp    = audio.reshape(1, -1).astype(np.float32)
        result = sess.run(None, {'input_1': inp})[0][0]

        # result = [SIG, BAK, OVRL]
        s.dnsmos_sig  = float(result[0])
        s.dnsmos_bak  = float(result[1])
        s.dnsmos_ovrl = float(result[2])

        dc = cfg['dnsmos']
        if s.dnsmos_ovrl < dc.get('min_ovrl', 3.0):
            # Don't reject yet — flag for DeepFilterNet
            s.filter_stage = 'stage3_flagged'
            return True  # still passes

    except Exception as e:
        log.warning(f"DNSMOS failed: {e}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — DeepFilterNet (on flagged samples)
# ─────────────────────────────────────────────────────────────────────────────

_dfn_model = None

def _get_dfn():
    global _dfn_model
    if _dfn_model is None:
        from df.enhance import init_df, enhance
        _dfn_model = init_df()
    return _dfn_model


def stage4_deepfilter(s: Sample, cfg: dict) -> bool:
    if not cfg.get('deepfilternet', {}).get('enabled', False):
        return True

    # Only apply if DNSMOS flagged OR SNR below threshold
    dfn_cfg   = cfg['deepfilternet']
    snr_thresh = dfn_cfg.get('snr_flag_threshold', 20.0)
    dnsmos_thresh = cfg.get('dnsmos', {}).get('min_ovrl', 3.0)

    should_enhance = (
        s.snr_db < snr_thresh or
        (s.dnsmos_ovrl > 0 and s.dnsmos_ovrl < dnsmos_thresh)
    )

    if not should_enhance:
        return True

    try:
        import torch
        from df.enhance import enhance

        model, df_state, _ = _get_dfn()
        audio_tensor = torch.tensor(s.audio).unsqueeze(0).float()
        enhanced = enhance(model, df_state, audio_tensor)
        s.audio  = enhanced.squeeze().numpy()
        s.deepfilter_applied = True
        log.debug(f"DeepFilterNet applied to {s.language} sample")
    except Exception as e:
        log.warning(f"DeepFilterNet failed: {e}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Whisper CER (GPU)
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model = None

def _get_whisper(size: str = 'tiny', device: str = 'cpu'):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(size, device=device)
    return _whisper_model


def _cer(ref: str, hyp: str) -> float:
    """
    Character error rate with script normalization.
    If scripts differ completely (e.g. Devanagari vs Latin),
    fall back to word-level overlap score instead.
    """
    import unicodedata
    ref = unicodedata.normalize('NFC', ref.strip())
    hyp = unicodedata.normalize('NFC', hyp.strip())
    if not ref:
        return 0.0
    if not hyp:
        return 1.0

    # Check if scripts are compatible
    def is_ascii_latin(s):
        return all(ord(c) < 256 for c in s if c.strip())

    ref_latin = is_ascii_latin(ref)
    hyp_latin = is_ascii_latin(hyp)

    # Script mismatch — use word overlap instead
    if ref_latin != hyp_latin:
        ref_words = set(ref.lower().split())
        hyp_words = set(hyp.lower().split())
        if not ref_words:
            return 1.0
        overlap = len(ref_words & hyp_words) / len(ref_words)
        return round(1.0 - overlap, 4)

    # Same script — use character edit distance
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return round(dp[n] / max(len(ref), 1), 4)


def stage5_whisper_cer(s: Sample, cfg: dict) -> bool:
    wcfg = cfg.get('whisper', {})
    if not wcfg.get('enabled', False):
        return True

    try:
        import torch as _torch
        _device = 'cuda' if _torch.cuda.is_available() else 'cpu'
        model  = _get_whisper(wcfg.get('model_size', 'tiny'), device=_device)
        with _GPU_LOCK:
            result = model.transcribe(s.audio,
                                      fp16=(_device == 'cuda'))
        hyp    = result.get('text', '').strip()
        s.cer  = _cer(s.transcript, hyp)

        if s.cer > wcfg.get('max_cer', 0.35):
            s.reject_reason = f"high_cer:{s.cer:.2f}"
            s.filter_stage  = 'stage5'
            return False

    except Exception as e:
        log.warning(f"Whisper failed: {e}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Composite scoring
# ─────────────────────────────────────────────────────────────────────────────

def stage6_composite_score(s: Sample, cfg: dict) -> bool:
    sm = cfg['soft_metrics']
    sc = cfg['scoring']

    def norm(val, lo, hi):
        if val != val:
            return 0.0
        return float(max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9))))

    scores = {}

    # SNR
    c = sm['snr']
    scores['snr'] = norm(s.snr_db, c['min_db'], c['good_db'])

    # Speech quality (Silero)
    c = sm['speech_quality']
    scores['speech_quality'] = norm(
        s.speech_quality, c['min_quality'], c['good_quality'])

    # Spectral flatness (inverted)
    c = sm['spectral_flatness']
    scores['spectral_flatness'] = norm(
        c['max_flat'] - s.spectral_flatness, 0.0, c['max_flat'])

    # Voiced ratio
    c = sm['voiced_ratio']
    scores['voiced_ratio'] = norm(
        s.voiced_ratio, c['min_ratio'], c['good_ratio'])

    # RMS energy
    c = sm['rms_energy']
    scores['rms_energy'] = norm(
        s.rms_energy, c['min_rms'], c['good_rms'])

    # DNSMOS (if available)
    if s.dnsmos_ovrl > 0 and sm.get('dnsmos', {}).get('enabled', False):
        c = sm['dnsmos']
        scores['dnsmos'] = norm(
            s.dnsmos_ovrl, c['min_ovrl'], c['good_ovrl'])

    # Weighted sum
    weight_map = {
        'snr'              : sm['snr']['weight'],
        'speech_quality'   : sm['speech_quality']['weight'],
        'spectral_flatness': sm['spectral_flatness']['weight'],
        'voiced_ratio'     : sm['voiced_ratio']['weight'],
        'rms_energy'       : sm['rms_energy']['weight'],
    }
    if s.dnsmos_ovrl > 0 and sm.get('dnsmos', {}).get('enabled', False):
        weight_map['dnsmos'] = sm['dnsmos']['weight']

    total_w = sum(weight_map.values())
    weighted = sum(scores.get(k, 0) * w for k, w in weight_map.items())
    s.composite_score = round(weighted / total_w * 100, 2)

    threshold = sc.get('composite_threshold', 50.0)
    if s.composite_score < threshold:
        s.reject_reason = f"low_score:{s.composite_score:.1f}"
        s.filter_stage  = 'stage6'
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Stratified sampling
# ─────────────────────────────────────────────────────────────────────────────

def stage7_stratified_sample(
        kept: list[dict], cfg: dict,
        target: int = 1000) -> list[dict]:
    """
    Select target samples from kept pool using stratified sampling.
    Stratification by: duration bucket, gender, speaking rate.
    """
    import random
    import pandas as pd

    if len(kept) <= target:
        log.info(f"Pool size {len(kept)} <= target {target}, returning all")
        return kept

    sp_cfg = cfg.get('sampling', {})
    strata_cfg = sp_cfg.get('duration_strata', {
        '2.0-4.0': 0.25,
        '4.0-8.0': 0.45,
        '8.0-12.0': 0.30
    })

    df = pd.DataFrame(kept)
    selected = []
    must_keep = []

    # Phoneme coverage — must keep rare characters
    all_text = ' '.join(str(r.get('transcript', '')) for r in kept)
    char_freq = {}
    for c in all_text:
        if not c.isascii() and not c.isspace():
            char_freq[c] = char_freq.get(c, 0) + 1

    rare_chars = {c for c, f in char_freq.items() if f < 50}
    if rare_chars:
        for r in kept:
            txt = str(r.get('transcript', ''))
            if any(c in txt for c in rare_chars):
                must_keep.append(r)

    must_keep_ids = {id(r) for r in must_keep}
    selected.extend(must_keep)
    remaining = [r for r in kept if id(r) not in must_keep_ids]
    remaining_target = target - len(selected)

    if remaining_target <= 0:
        return selected[:target]

    # Duration strata sampling
    def get_bucket(dur):
        for bucket, _ in strata_cfg.items():
            lo, hi = map(float, bucket.split('-'))
            if lo <= dur < hi:
                return bucket
        return None

    # Gender balance within each stratum
    max_per_speaker = sp_cfg.get('speaker_diversity', {}).get(
        'max_samples_per_speaker', 50)

    speaker_counts = {}
    pool = pd.DataFrame(remaining)
    if pool.empty:
        return selected[:target]

    pool['bucket'] = pool['duration_sec'].apply(get_bucket)

    for bucket, pct in strata_cfg.items():
        n_bucket = int(remaining_target * pct)
        sub = pool[pool['bucket'] == bucket]
        if sub.empty:
            continue

        # Gender balance
        n_male   = n_bucket // 2
        n_female = n_bucket - n_male

        for gender, n_target in [('Male', n_male), ('Female', n_female)]:
            gsub = sub[sub.get('gender', 'Unknown') == gender] \
                if 'gender' in sub.columns else sub
            if gsub.empty:
                gsub = sub  # fallback to any gender
            # Sort by DNSMOS score descending — pick cleanest audio first
            if 'dnsmos_ovrl' in gsub.columns:
                gsub = gsub.sort_values('dnsmos_ovrl', ascending=False)

            # Sort by composite score descending
            if 'composite_score' in gsub.columns:
                gsub = gsub.sort_values('composite_score', ascending=False)

            count = 0
            for _, row in gsub.iterrows():
                if count >= n_target:
                    break
                spk = str(row.get('speaker_id', 'unknown'))
                if speaker_counts.get(spk, 0) >= max_per_speaker:
                    continue
                selected.append(row.to_dict())
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
                count += 1

    # Fill remaining quota with best scores
    selected_ids = {id(r) for r in selected}
    if len(selected) < target:
        pool_dicts = pool.to_dict('records')
        extras = [r for r in pool_dicts if id(r) not in selected_ids]
        extras.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        selected.extend(extras[:target - len(selected)])

    log.info(f"Stratified sample: {len(selected)} from pool of {len(kept)}")
    return selected[:target]


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

FIELDS = [
    'language','speaker_id','gender','age_group','area','state',
    'duration_sec','word_count','words_per_sec','transcript',
    'nfc_ok','script_purity','audio_hash',
    'rms_energy','clipping_ratio','snr_db',
    'silence_ratio','voiced_ratio','spectral_flatness',
    'speech_quality','dnsmos_sig','dnsmos_bak','dnsmos_ovrl',
    'cer','deepfilter_applied','composite_score',
    'keep','reject_reason','filter_stage',
]


class ResultWriter:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self._csv_path  = os.path.join(output_dir, 'filter_metrics.csv')
        self._kept_path = os.path.join(output_dir, 'kept.jsonl')
        self._rej_path  = os.path.join(output_dir, 'rejected.jsonl')
        self._csv_fh    = open(self._csv_path, 'w', newline='', encoding='utf-8')
        self._kept_fh   = open(self._kept_path, 'w', encoding='utf-8')
        self._rej_fh    = open(self._rej_path,  'w', encoding='utf-8')
        self._writer    = csv.DictWriter(
            self._csv_fh, fieldnames=FIELDS,
            extrasaction='ignore', restval='')
        self._writer.writeheader()

    def write(self, s: Sample):
        row = {f: getattr(s, f, '') for f in FIELDS}
        self._writer.writerow(row)
        target = self._kept_fh if s.keep else self._rej_fh
        meta = {f: row[f] for f in [
            'language','speaker_id','gender','duration_sec',
            'transcript','composite_score','reject_reason']}
        target.write(json.dumps(meta, ensure_ascii=False) + '\n')

    def flush(self):
        self._csv_fh.flush()
        self._kept_fh.flush()
        self._rej_fh.flush()

    def close(self):
        self._csv_fh.close()
        self._kept_fh.close()
        self._rej_fh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def run_language(lang: str, cfg: dict,
                 hf_token: str,
                 output_dir: str) -> dict:
    """Process one language through all stages."""
    from datasets import load_dataset

    per_lang = cfg['dataset'].get('per_language_samples', {})
    n_stream = per_lang.get(lang, cfg['dataset']['samples_per_language'])
    hf = cfg['hard_filters']

    audio_hashes = set()
    text_hashes  = set()

    kept_rows = []
    all_rows  = []

    stats = {
        'language': lang, 'total': 0, 'kept': 0,
        'stage0': 0, 'stage1': 0, 'stage2': 0,
        'stage3': 0, 'stage4': 0, 'stage5': 0, 'stage6': 0,
    }

    ds = load_dataset(
        cfg['dataset']['name'], lang,
        split=cfg['dataset']['split'],
        streaming=True, token=hf_token
    )

    for raw_sample in ds:
        if stats['total'] >= n_stream:
            break
        stats['total'] += 1

        try:
            audio_field = raw_sample.get('audio') or raw_sample.get('audio_filepath')
            if isinstance(audio_field, dict):
                raw   = np.array(audio_field['array'], dtype=np.float32)
                sr_in = audio_field['sampling_rate']
            elif hasattr(audio_field, 'get_all_samples'):
                frames = audio_field.get_all_samples()
                raw    = frames.data.numpy()
                sr_in  = frames.sample_rate
            else:
                raise ValueError(f"Unknown audio format: {type(audio_field)}")
            audio, sr = prepare_audio(
                raw, sr_in,
                top_db=cfg['audio'].get('top_db', 35.0)
            )
            dur = len(audio) / SR

            s = Sample(
                language           = lang,
                audio              = audio,
                sr                 = sr,
                transcript         = raw_sample.get('text', '') or '',
                speaker_id         = raw_sample.get('speaker_id', ''),
                gender             = raw_sample.get('gender', ''),
                age_group          = raw_sample.get('age_group', ''),
                area               = raw_sample.get('area', ''),
                state              = raw_sample.get('state', ''),
                duration_sec       = round(dur, 3),
                verification_report= raw_sample.get(
                    'verification_report', '') or '',
            )

        except Exception as e:
            log.warning(f"[{lang}] decode error: {e}")
            continue

        # Run pipeline stages
        passed = (
            stage0_hard_filters(s, cfg) and
            stage1_text_filters(s, cfg, audio_hashes, text_hashes) and
            stage2_silero_vad(s, cfg) and
            stage3_dnsmos(s, cfg) and
            stage4_deepfilter(s, cfg) and
            stage5_whisper_cer(s, cfg) and
            stage6_composite_score(s, cfg)
        )

        s.keep = passed
        if not passed:
            stage = s.filter_stage.replace('stage', '') if s.filter_stage else '?'
            stats[s.filter_stage] = stats.get(s.filter_stage, 0) + 1

        all_rows.append(s)
        if passed:
            kept_rows.append({f: getattr(s, f, '') for f in FIELDS})
            stats['kept'] += 1

        if stats['total'] % 200 == 0:
            log.info(f"[{lang}] {stats['total']}/{n_stream} "
                     f"kept={stats['kept']}")

    # Stage 7: stratified sampling
    target = cfg.get('sampling', {}).get('target_per_language', 1000)
    sampled = stage7_stratified_sample(kept_rows, cfg, target)

    sampled_ids = {r.get('audio_hash', '') for r in sampled}

    # Mark final keep
    for s in all_rows:
        if s.keep and s.audio_hash in sampled_ids:
            s.keep = True
        elif s.keep:
            s.keep = False
            s.reject_reason = 'not_selected_sampling'
            s.filter_stage  = 'stage7'

    stats['final_kept'] = len(sampled)
    log.info(f"[{lang}] DONE — {stats['final_kept']}/{stats['total']} "
             f"({stats['final_kept']/max(stats['total'],1)*100:.1f}%)")

    return {'rows': all_rows, 'stats': stats}


def run_pipeline(config_path: str,
                 hf_token: str,
                 output_dir: str):
    cfg = yaml.safe_load(open(config_path))
    os.makedirs(output_dir, exist_ok=True)

    writer  = ResultWriter(output_dir)
    summary = {'languages': {}, 'total': 0, 'kept': 0}

    t0 = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    write_lock = threading.Lock()
    n_workers = min(len(cfg['languages']), 6)
    log.info(f"Running {len(cfg['languages'])} languages with {n_workers} parallel workers")

    def process_lang(lang):
        log.info(f"\n── {lang} ──────────────────────────────")
        return lang, run_language(lang, cfg, hf_token, output_dir)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_lang, lang): lang
                   for lang in cfg['languages']}
        for future in as_completed(futures):
            lang, result = future.result()
            with write_lock:
                for s in result['rows']:
                    writer.write(s)
                writer.flush()
            stats = result['stats']
            summary['languages'][lang] = stats
            summary['total'] += stats['total']
            summary['kept']  += stats.get('final_kept', 0)
            log.info(f"[{lang}] complete — kept={stats.get('final_kept',0)}")

    writer.close()

    elapsed = time.time() - t0
    summary['elapsed_sec']    = round(elapsed, 1)
    summary['samples_per_sec'] = round(summary['total'] / max(elapsed, 1), 2)

    with open(os.path.join(output_dir, 'filter_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    log.info(f"\n{'='*55}")
    log.info(f"  Pipeline complete in {elapsed/60:.1f} minutes")
    log.info(f"  Total: {summary['total']:,}  Kept: {summary['kept']:,} "
             f"({summary['kept']/max(summary['total'],1)*100:.1f}%)")
    log.info(f"{'='*55}")

    return summary


if __name__ == '__main__':
    import os
    run_pipeline(
        config_path='config.yaml',
        hf_token=os.environ.get('HF_TOKEN', ''),
        output_dir='./output_filtered'
    )
