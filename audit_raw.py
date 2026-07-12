import os, glob, pandas as pd, numpy as np, soundfile as sf, io

DATA_DIR = '/root/data/indicvoices'
LANGUAGES = [
    'assamese','bengali','bodo','dogri','gujarati',
    'hindi','kannada','kashmiri','konkani','maithili',
    'malayalam','manipuri','marathi','nepali','odia',
    'punjabi','sanskrit','santali','sindhi','tamil',
    'telugu','urdu'
]

print(f"{'='*65}")
print(f"  IndicVoices Raw Data Audit")
print(f"{'='*65}\n")

summary = []

for lang in LANGUAGES:
    files = sorted(glob.glob(f'{DATA_DIR}/{lang}/**/*.parquet', recursive=True))
    if not files:
        print(f"[{lang}] NO DATA FOUND")
        continue

    df = pd.read_parquet(files[0])

    # Duration stats
    dur = df['duration'].dropna()

    # Gender
    gender = df['gender'].value_counts().to_dict() if 'gender' in df.columns else {}

    # Task distribution
    tasks = df['task_name'].value_counts().head(3).to_dict() if 'task_name' in df.columns else {}

    # Audio quality — sample 50 random rows
    snrs = []
    sample_rows = df.sample(min(50, len(df)), random_state=42)
    for _, row in sample_rows.iterrows():
        try:
            af = row.get('audio_filepath') or row.get('audio')
            if isinstance(af, dict) and 'bytes' in af:
                audio, sr = sf.read(io.BytesIO(af['bytes']))
                audio = np.array(audio, dtype=np.float32)
                if audio.ndim > 1: audio = audio.mean(axis=0)
                # SNR estimate
                rms = np.sqrt(np.mean(audio**2))
                if rms > 1e-6:
                    noise = np.percentile(np.abs(audio), 10)
                    snr = 20 * np.log10(rms / max(noise, 1e-6))
                    snrs.append(snr)
        except: pass

    # Short sample count
    short = (dur < 2.0).sum()
    long  = (dur > 12.0).sum()
    good  = ((dur >= 2.0) & (dur <= 12.0)).sum()

    print(f"── {lang} {'─'*(40-len(lang))}")
    print(f"  Total samples : {len(df):,}")
    print(f"  Duration      : mean={dur.mean():.1f}s  min={dur.min():.1f}s  max={dur.max():.1f}s")
    print(f"  Good (2-12s)  : {good:,} ({good/len(df)*100:.1f}%)")
    print(f"  Too short(<2s): {short:,} ({short/len(df)*100:.1f}%)")
    print(f"  Too long(>12s): {long:,} ({long/len(df)*100:.1f}%)")
    if snrs:
        print(f"  Est. SNR      : mean={np.mean(snrs):.1f}dB  min={np.min(snrs):.1f}dB")
    if gender:
        g_str = '  '.join([f"{k}:{v}" for k,v in gender.items()])
        print(f"  Gender        : {g_str}")
    if tasks:
        top_task = list(tasks.items())[0]
        print(f"  Top task      : {top_task[0]} ({top_task[1]} samples)")
    print()

    summary.append({
        'lang': lang, 'total': len(df),
        'good_pct': round(good/len(df)*100, 1),
        'mean_dur': round(dur.mean(), 1),
        'mean_snr': round(np.mean(snrs), 1) if snrs else 0,
    })

print(f"{'='*65}")
print(f"  SUMMARY")
print(f"{'='*65}")
print(f"  {'Lang':<12} {'Total':>7} {'Good%':>6} {'AvgDur':>7} {'AvgSNR':>7}")
print(f"  {'-'*45}")
for s in summary:
    print(f"  {s['lang']:<12} {s['total']:>7,} {s['good_pct']:>5.1f}% {s['mean_dur']:>6.1f}s {s['mean_snr']:>6.1f}dB")
