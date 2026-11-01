"""
run_v3.py — Two-pass pipeline
==============================
Pass 1 (CPU/fast): Hard filters → Silero VAD → DNSMOS raw → Pre-select 600
Pass 2 (GPU/deep): DNSMOS strict gate (raw audio) → DeepFilterNet → LUFS normalize → Composite score → Final keep

Goals:
  - Clean audio (DNSMOS >= 2.7, scored on raw audio before DeepFilter)
  - 1000 input samples per language
  - Better stratification (DNSMOS-sorted)
  - Text normalization
"""
import os, sys, time, yaml, json, logging, numpy as np
import multiprocessing as mp

os.chdir('/root/indic_filter')
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['LD_LIBRARY_PATH'] = (
    '/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:'
    + os.environ.get('LD_LIBRARY_PATH',''))

OUTPUT_DIR = './output_filtered_v5'
LANGUAGES  = [
    'assamese','bengali','bodo','dogri','gujarati',
    'hindi','kannada','kashmiri','konkani','maithili',
    'malayalam','manipuri','marathi','nepali','odia',
    'punjabi','sanskrit','santali','sindhi','tamil',
    'telugu','urdu'
]
BAD_TASKS = ['Keywords Spotting', 'Ola/Uber Prompts']
WHISPER_DISABLED = {
    # Tribal/rare languages — Whisper has no training data
    'assamese','bodo','santali','manipuri','dogri',
    # Borderline — Whisper unreliable
    'konkani','gujarati','punjabi',
    # Very poor Whisper support
    'kashmiri','sindhi','maithili','sanskrit',
}

N_INPUT          = 1000   # samples to read per language
N_CANDIDATES     = 600    # pre-select for deep processing
DNSMOS_RAW_MIN   = 1.5    # pass 1 gate (loose)
DNSMOS_CLEAN_MIN = 2.7    # pass 2 gate (strict)
MAX_PER_SPEAKER  = 15     # diversity cap


def normalize_text(text: str) -> str:
    """
    Normalize text for TTS.
    Returns empty string if text is unusable (unintelligible, too noisy).
    """
    import re, unicodedata
    if not text: return ''

    # Reject samples with unintelligible markers
    if re.search(r'<unintelligible>|<noise>|<laugh>|<breath>|<cough>',
                 text, re.IGNORECASE):
        return ''  # empty = will be rejected by text filter

    # NFC normalize
    text = unicodedata.normalize('NFC', text)

    # Remove other noise tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove leading/trailing punctuation
    text = re.sub(r'^[,;:\-\(\)\[\]]+', '', text)
    text = re.sub(r'[,;:\-\(\)\[\]]+$', '', text)

    # Reject if too short after cleaning (less than 2 words)
    if len(text.split()) < 2:
        return ''

    return text.strip()


def process_language(lang):
    import soundfile as sf, io, pandas as pd, csv, glob, warnings
    import torch, onnxruntime as ort, urllib.request, yaml
    from silero_vad import load_silero_vad, get_speech_timestamps
    from df.enhance import init_df, enhance
    from filter_pipeline import (prepare_audio, stage0_hard_filters,
        stage1_text_filters, stage6_composite_score,
        Sample, FIELDS, lufs_normalize)

    logging.basicConfig(level=logging.INFO,
        format=f'%(asctime)s [{lang}] %(message)s', datefmt='%H:%M:%S')
    log = logging.getLogger(lang)

    cfg = yaml.safe_load(open('config.yaml'))
    cfg['hard_filters']['duration_min_sec'] = 1.5
    cfg['hard_filters']['duration_max_sec'] = 15.0
    cfg['dnsmos']['enabled']                = True
    cfg['dnsmos']['min_ovrl']               = DNSMOS_CLEAN_MIN
    cfg['whisper']['enabled']               = False  # disabled for all — DNSMOS handles quality
    cfg['whisper']['model_size']            = 'small'
    cfg['whisper']['max_cer']               = 0.40

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hf = cfg['hard_filters']

    # ── Load models ───────────────────────────────────────────
    silero = load_silero_vad().to(device)

    model_path = '/root/indic_filter/dnsmos_p835.onnx'
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(
            'https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx',
            model_path)
    dnsmos_sess = ort.InferenceSession(model_path,
        providers=['CUDAExecutionProvider','CPUExecutionProvider'])

    dfn_model, df_state, _ = init_df()

    if cfg['whisper']['enabled']:
        import whisper as _whisper
        whisper_model = _whisper.load_model('small', device=device)
        log.info('Whisper loaded')

    log.info(f'Models ready | DNSMOS:{dnsmos_sess.get_providers()[0]}')

    # ── Load data ─────────────────────────────────────────────
    files = sorted(glob.glob(f'/root/data/indicvoices/{lang}/**/*.parquet',
                              recursive=True))
    if not files:
        log.error('No parquet files found')
        return lang, {'total':0,'kept':0,'final_kept':0}

    df_raw = pd.read_parquet(files[0])
    if 'task_name' in df_raw.columns:
        df_raw = df_raw[~df_raw['task_name'].isin(BAD_TASKS)]
    df_raw = df_raw.head(N_INPUT).reset_index(drop=True)
    log.info(f'Loaded {len(df_raw)} rows')

    # ── PASS 1: Fast screening ────────────────────────────────
    log.info('Pass 1: Hard filters + Silero + DNSMOS raw...')
    audio_hashes, text_hashes = set(), set()
    pass1_candidates = []
    all_rows = []
    stats = {'language':lang,'total':0,'kept':0,
             'stage0':0,'stage1':0,'stage2':0,
             'stage3_raw':0,'stage3_clean':0,
             'stage5':0,'stage6':0,'final_kept':0}

    for idx, row in df_raw.iterrows():
        stats['total'] += 1
        try:
            af = row.get('audio_filepath') or row.get('audio')
            if isinstance(af, dict):
                raw, sr_in = sf.read(io.BytesIO(af['bytes']))
            else:
                continue
            raw = np.array(raw, dtype=np.float32)
            audio, sr = prepare_audio(raw, sr_in)
            transcript = normalize_text(str(row.get('text','') or ''))

            # Reject unintelligible/empty transcripts
            if not transcript:
                stats['stage1'] += 1
                continue

            s = Sample(
                language=lang, audio=audio, sr=sr,
                transcript=transcript,
                speaker_id=str(row.get('speaker_id','')),
                gender=str(row.get('gender','')),
                age_group=str(row.get('age_group','')),
                area=str(row.get('area','')),
                state=str(row.get('state','')),
                duration_sec=round(len(audio)/sr, 3),
                verification_report=str(row.get('verification_report','') or ''),
            )

            # Stage 0: Hard filters
            if not stage0_hard_filters(s, cfg):
                stats['stage0'] += 1; all_rows.append(s); continue

            # Stage 1: Text filters
            if not stage1_text_filters(s, cfg, audio_hashes, text_hashes):
                stats['stage1'] += 1; all_rows.append(s); continue

            # Stage 2: Silero VAD
            import torch as _t
            waveform = _t.tensor(audio).float().to(device)
            ts = get_speech_timestamps(waveform, silero,
                sampling_rate=16000, threshold=0.5, return_seconds=False)
            voiced = sum(t['end']-t['start'] for t in ts)
            s.voiced_ratio  = voiced / max(len(audio), 1)
            s.silence_ratio = 1.0 - s.voiced_ratio
            s.speech_quality = float(s.voiced_ratio)
            if s.voiced_ratio < hf.get('voiced_min_ratio', 0.40):
                s.reject_reason = f"low_voiced:{s.voiced_ratio:.2f}"
                s.filter_stage  = 'stage2'
                stats['stage2'] += 1; all_rows.append(s); continue

            # DNSMOS on raw audio (loose gate)
            a = np.pad(audio,(0,max(0,144160-len(audio))))[:144160]
            dres = dnsmos_sess.run(None,
                {'input_1':a.reshape(1,-1).astype(np.float32)})[0][0]
            s.dnsmos_sig  = float(dres[0])
            s.dnsmos_bak  = float(dres[1])
            s.dnsmos_ovrl = float(dres[2])

            if s.dnsmos_ovrl < DNSMOS_RAW_MIN:
                s.reject_reason = f"low_dnsmos_raw:{s.dnsmos_ovrl:.2f}"
                s.filter_stage  = 'stage3_raw'
                stats['stage3_raw'] += 1; all_rows.append(s); continue

            pass1_candidates.append(s)

        except Exception as e:
            log.warning(f'Pass1 row {idx}: {e}')

    log.info(f'Pass 1 done: {len(pass1_candidates)} candidates from {stats["total"]}')

    # ── Pre-select top N_CANDIDATES by stratification ─────────
    log.info(f'Pre-selecting top {N_CANDIDATES} candidates...')

    def pre_select(candidates, n_select):
        if len(candidates) <= n_select:
            return candidates
        import pandas as pd
        df_c = pd.DataFrame([{
            'idx': i,
            'duration': c.duration_sec,
            'gender': c.gender,
            'speaker_id': c.speaker_id,
            'dnsmos': c.dnsmos_ovrl,
        } for i, c in enumerate(candidates)])

        strata = [(1.5,4.0,0.30),(4.0,8.0,0.45),(8.0,15.0,0.25)]
        selected_idx = set()
        spk_counts = {}

        for lo, hi, pct in strata:
            n_t = int(n_select * pct)
            sub = df_c[(df_c['duration']>=lo)&(df_c['duration']<hi)]
            sub = sub.sort_values('dnsmos', ascending=False)
            n_f = n_t // 2
            n_m = n_t - n_f
            for gender, n_g in [('Female',n_f),('Male',n_m)]:
                gsub = sub[sub['gender']==gender]
                if gsub.empty: gsub = sub
                added = 0
                for _, r in gsub.iterrows():
                    if r['idx'] in selected_idx: continue
                    spk = str(r['speaker_id'])
                    if spk_counts.get(spk,0) >= MAX_PER_SPEAKER: continue
                    selected_idx.add(r['idx'])
                    spk_counts[spk] = spk_counts.get(spk,0) + 1
                    added += 1
                    if added >= n_g: break

        # Fill remaining
        remaining = df_c.sort_values('dnsmos', ascending=False)
        for _, r in remaining.iterrows():
            if len(selected_idx) >= n_select: break
            if r['idx'] in selected_idx: continue
            spk = str(r['speaker_id'])
            if spk_counts.get(spk,0) >= MAX_PER_SPEAKER: continue
            selected_idx.add(r['idx'])
            spk_counts[spk] = spk_counts.get(spk,0) + 1

        return [candidates[i] for i in selected_idx]

    selected_candidates = pre_select(pass1_candidates, N_CANDIDATES)
    selected_ids = {id(s) for s in selected_candidates}
    not_selected = [s for s in pass1_candidates
                    if id(s) not in selected_ids]

    # Mark not selected from pass1
    for s in not_selected:
        s.keep = False
        s.reject_reason = 'not_selected_pass1'
        s.filter_stage  = 'stage_preselect'
        all_rows.append(s)

    log.info(f'Pre-selected: {len(selected_candidates)} for deep processing')

    # ── PASS 2: Deep processing on candidates ─────────────────
    log.info('Pass 2: DNSMOS strict gate + DeepFilterNet + LUFS...')
    kept_rows = []

    for s in selected_candidates:
        try:
            audio = s.audio
            import torch as _t

            # DNSMOS on raw audio first (before DeepFilterNet)
            # DeepFilterNet hurts DNSMOS scores so score on raw
            a = np.pad(audio,(0,max(0,144160-len(audio))))[:144160]
            dres = dnsmos_sess.run(None,
                {'input_1':a.reshape(1,-1).astype(np.float32)})[0][0]
            s.dnsmos_sig  = float(dres[0])
            s.dnsmos_bak  = float(dres[1])
            s.dnsmos_ovrl = float(dres[2])

            if s.dnsmos_ovrl < DNSMOS_CLEAN_MIN:
                s.reject_reason = f"low_dnsmos:{s.dnsmos_ovrl:.2f}"
                s.filter_stage  = 'stage3_clean'
                stats['stage3_clean'] += 1
                all_rows.append(s)
                continue

            # DeepFilterNet — enhance AFTER DNSMOS scoring
            at = _t.tensor(audio).unsqueeze(0).float()
            enhanced = enhance(dfn_model, df_state, at)
            audio = enhanced.squeeze().numpy().astype(np.float32)
            s.audio = audio
            s.deepfilter_applied = True

            # LUFS normalize
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                audio = lufs_normalize(audio)
            s.audio = audio

            # Whisper CER
            if cfg['whisper']['enabled']:
                with _t.no_grad():
                    res = whisper_model.transcribe(audio, fp16=(device=='cuda'))
                hyp = res.get('text','').strip()
                ref_words = set(s.transcript.split())
                hyp_words = set(hyp.split())
                overlap = len(ref_words & hyp_words) / max(len(ref_words),1)
                s.cer = round(1.0 - overlap, 4)
                if s.cer > cfg['whisper'].get('max_cer', 0.40):
                    s.reject_reason = f"high_cer:{s.cer:.2f}"
                    s.filter_stage  = 'stage5'
                    stats['stage5'] += 1
                    all_rows.append(s)
                    continue

            # Composite score
            if not stage6_composite_score(s, cfg):
                stats['stage6'] += 1
                all_rows.append(s)
                continue

            s.keep = True
            kept_rows.append({f: getattr(s,f,'') for f in FIELDS})
            all_rows.append(s)
            stats['kept'] += 1

        except Exception as e:
            log.warning(f'Pass2 error: {e}')
            all_rows.append(s)

    stats['final_kept'] = len(kept_rows)
    dnsmos_mean = np.mean([float(r.get("dnsmos_ovrl",0)) for r in kept_rows]) if kept_rows else 0
    log.info(f'[{lang}] DONE — {stats["final_kept"]} final kept | DNSMOS={dnsmos_mean:.3f} | Pass1={stats["stage0"]}rejected Pass2={stats["stage3_clean"]}rejected')

    # Write CSV
    os.makedirs(f'{OUTPUT_DIR}/per_language', exist_ok=True)
    with open(f"{OUTPUT_DIR}/per_language/{lang}_metrics.csv",
              'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS,
                               extrasaction='ignore', restval='')
        writer.writeheader()
        for s in all_rows:
            writer.writerow({field: getattr(s,field,'') for field in FIELDS})

    return lang, stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/per_language', exist_ok=True)
    t0 = time.time()

    ctx = mp.get_context('spawn')
    n_workers = 4
    log.info(f"run_v3 | {len(LANGUAGES)} langs | {n_workers} workers | "
             f"input={N_INPUT} candidates={N_CANDIDATES} dnsmos_clean={DNSMOS_CLEAN_MIN}")

    import glob
    completed = [f.replace('_metrics.csv','').split('/')[-1]
                 for f in glob.glob(f'{OUTPUT_DIR}/per_language/*.csv')]
    remaining = [l for l in LANGUAGES if l not in completed]
    log.info(f"Completed: {completed} | Remaining: {len(remaining)}")

    import csv
    summary = {'languages':{}, 'total':0, 'kept':0}

    with ctx.Pool(n_workers) as pool:
        results = pool.map(process_language, remaining)

    for lang, stats in (results or []):
        summary['languages'][lang] = stats
        summary['total'] += stats.get('total',0)
        summary['kept']  += stats.get('final_kept',0)

    # Merge CSVs
    all_rows, fnames = [], None
    for f in sorted(glob.glob(f'{OUTPUT_DIR}/per_language/*.csv')):
        with open(f, encoding='utf-8') as fh:
            r = csv.DictReader(fh)
            if not fnames: fnames = r.fieldnames
            all_rows.extend(list(r))

    if all_rows and fnames:
        with open(f'{OUTPUT_DIR}/filter_metrics.csv','w',
                  newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fnames,
                               extrasaction='ignore', restval='')
            w.writeheader(); w.writerows(all_rows)

    elapsed = time.time()-t0
    summary['elapsed_sec'] = round(elapsed,1)
    with open(f'{OUTPUT_DIR}/filter_summary.json','w') as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nDone in {elapsed/60:.1f} min | "
             f"Total:{summary['total']:,} Kept:{summary['kept']:,}")


if __name__ == '__main__':
    main()
