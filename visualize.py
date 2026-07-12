import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR  = '/root/data/indicvoices'
FINAL_CSV = '/root/indic_filter/output_filtered_v5/kept_samples.csv'
FILT_CSV  = '/root/indic_filter/output_filtered_v5/filter_metrics.csv'
PLOTS_DIR = '/root/indic_filter/output_filtered_v5/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 130
plt.rcParams['font.size']  = 9
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

LANGUAGES = [
    'assamese','bengali','bodo','dogri','gujarati',
    'hindi','kannada','kashmiri','konkani','maithili',
    'malayalam','manipuri','marathi','nepali','odia',
    'punjabi','sanskrit','santali','sindhi','tamil',
    'telugu','urdu'
]
BAD_TASKS = ['Keywords Spotting', 'Ola/Uber Prompts']

def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {name}")

# Load raw (1000 per language = pipeline input)
print("Loading raw input (1000/lang)...")
raw_dfs = []
for lang in LANGUAGES:
    files = sorted(glob.glob(f'{DATA_DIR}/{lang}/**/*.parquet', recursive=True))
    if files:
        df = pd.read_parquet(files[0])
        df['language'] = lang
        if 'task_name' in df.columns:
            df = df[~df['task_name'].isin(BAD_TASKS)]
        raw_dfs.append(df.head(1000))
raw = pd.concat(raw_dfs, ignore_index=True)

final = pd.read_csv(FINAL_CSV)
filt  = pd.read_csv(FILT_CSV)
filt['keep'] = filt['keep'].astype(str).str.lower().isin(['true','1'])

print(f"Raw: {len(raw):,} | Final: {len(final):,}\n")
w = 0.35

# ── Chart 1: Pipeline funnel ──────────────────────────────────
print("Chart 1: Pipeline funnel...")
fig, ax = plt.subplots(figsize=(12, 5))
stage_map = {
    'stage0'       : 'Hard Filters\n(dur/SNR)',
    'stage1'       : 'Text Filters\n(unintelligible)',
    'stage2'       : 'Silero VAD\n(voiced ratio)',
    'stage3_raw'   : 'DNSMOS Raw\n(pass1 gate)',
    'stage_preselect': 'Pre-selection\n(top 600)',
    'stage3_clean' : 'DNSMOS Clean\n(≥2.7 gate)',
    'stage6'       : 'Composite\nScore',
}
rej = filt[filt['keep']==False]
stage_counts = rej['filter_stage'].value_counts()
labels = list(stage_map.values())
values = [stage_counts.get(k, 0) for k in stage_map.keys()]
colors = ['#e74c3c','#e67e22','#f39c12','#9b59b6','#3498db','#1abc9c','#95a5a6']

bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85, width=0.6)
ax.bar(len(labels), len(final), color='#2ecc71', alpha=0.85, width=0.6)
all_labels = labels + ['Final\nKept']
ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels(all_labels, fontsize=8)
ax.set_ylabel('Samples')
ax.set_title(f'Pipeline Funnel — {len(raw):,} input → {len(final):,} kept ({len(final)/len(raw)*100:.1f}%)',
             fontweight='bold')
for i, v in enumerate(values + [len(final)]):
    if v > 0:
        ax.text(i, v+30, f'{v:,}', ha='center', fontsize=8, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save(fig, '1_pipeline_funnel.png')

# ── Chart 2: Duration before vs after ────────────────────────
print("Chart 2: Duration...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
bins = np.linspace(0, 15, 45)
ax.hist(raw['duration'].clip(0,15), bins=bins, color='#e74c3c',
        alpha=0.55, label=f'Input (n={len(raw):,})', density=True)
ax.hist(final['duration_sec'].clip(0,15), bins=bins, color='#2ecc71',
        alpha=0.75, label=f'Final (n={len(final):,})', density=True)
ax.axvline(1.5,  color='black', linestyle='--', lw=1.5, label='Min 1.5s')
ax.axvline(15.0, color='black', linestyle=':', lw=1.5, label='Max 15s')
ax.set_xlabel('Duration (s)'); ax.set_ylabel('Density')
ax.set_title('Duration: Input vs Final', fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

ax = axes[1]
bins_d = [0,2,4,8,12,15]
labels_d = ['<2s','2-4s','4-8s','8-12s','12-15s']
raw_dd  = pd.cut(raw['duration'].clip(0,15), bins=bins_d, labels=labels_d).value_counts().reindex(labels_d)
fin_dd  = pd.cut(final['duration_sec'].clip(0,15), bins=bins_d, labels=labels_d).value_counts().reindex(labels_d)
x2 = np.arange(len(labels_d))
ax.bar(x2-w/2, raw_dd/len(raw)*100,   w, color='#e74c3c', alpha=0.8, label='Input')
ax.bar(x2+w/2, fin_dd/len(final)*100, w, color='#2ecc71', alpha=0.8, label='Final')
ax.set_xticks(x2); ax.set_xticklabels(labels_d)
ax.set_ylabel('%'); ax.set_title('Duration Strata: Input vs Final', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save(fig, '2_duration_comparison.png')

# ── Chart 3: Gender balance ───────────────────────────────────
print("Chart 3: Gender...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, df, title, gcol in [
    (axes[0], raw,   'Input (1000/lang)', 'gender'),
    (axes[1], final, 'Final (kept)',       'gender')]:
    gdf = df.groupby(['language', gcol]).size().unstack(fill_value=0)
    cols = [c for c in ['Female','Male'] if c in gdf.columns]
    pct = gdf[cols].div(gdf[cols].sum(axis=1), axis=0)*100
    pct.plot(kind='bar', ax=ax, color=['#e91e8c','#3498db'], alpha=0.85, width=0.7)
    ax.axhline(50, color='gray', linestyle='--', lw=1.5, alpha=0.7)
    ax.fill_between([-0.5, len(pct)-0.5], 40, 60, alpha=0.08, color='green')
    ax.set_title(f'Gender Balance — {title}', fontweight='bold')
    ax.tick_params(axis='x', rotation=35)
    ax.set_ylabel('%'); ax.set_ylim(0,100)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save(fig, '3_gender_balance.png')

# ── Chart 4: DNSMOS distribution ─────────────────────────────
print("Chart 4: DNSMOS...")
if 'dnsmos_ovrl' in filt.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    all_dns = filt['dnsmos_ovrl'].dropna()
    fin_dns = final['dnsmos_ovrl'].dropna() if 'dnsmos_ovrl' in final.columns else pd.Series()
    bins_d2 = np.linspace(1, 4, 40)
    ax.hist(all_dns, bins=bins_d2, color='#e74c3c', alpha=0.55,
            label=f'All processed (mean={all_dns.mean():.2f})', density=True)
    if len(fin_dns):
        ax.hist(fin_dns, bins=bins_d2, color='#2ecc71', alpha=0.8,
                label=f'Final kept (mean={fin_dns.mean():.2f})', density=True)
    ax.axvline(2.7, color='black', linestyle='--', lw=2, label='Threshold 2.7')
    ax.set_xlabel('DNSMOS OVRL Score'); ax.set_ylabel('Density')
    ax.set_title('DNSMOS Distribution: All vs Final', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    if len(fin_dns) and 'language' in final.columns:
        dns_lang = final.groupby('language')['dnsmos_ovrl'].mean().sort_values()
        bars6 = ax.barh(dns_lang.index, dns_lang.values, color='#3498db', alpha=0.85)
        ax.axvline(dns_lang.mean(), color='gray', linestyle='--', lw=1.5,
                   label=f'Mean ({dns_lang.mean():.3f})')
        for bar, val in zip(bars6, dns_lang.values):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)
        ax.set_xlabel('Mean DNSMOS OVRL')
        ax.set_title('DNSMOS per Language (Final)', fontweight='bold')
        ax.legend(); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    save(fig, '4_dnsmos_distribution.png')

# ── Chart 5: SNR improvement ──────────────────────────────────
print("Chart 5: SNR...")
if 'snr_db' in filt.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    all_snr = filt['snr_db'].dropna()
    fin_snr = final['snr_db'].dropna() if 'snr_db' in final.columns else pd.Series()
    bins_s  = np.linspace(0, 50, 40)
    ax.hist(all_snr, bins=bins_s, color='#e74c3c', alpha=0.55,
            label=f'All processed (mean={all_snr.mean():.1f}dB)', density=True)
    if len(fin_snr):
        ax.hist(fin_snr, bins=bins_s, color='#2ecc71', alpha=0.8,
                label=f'Final kept (mean={fin_snr.mean():.1f}dB)', density=True)
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Density')
    ax.set_title('SNR: All Processed vs Final', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    if len(fin_snr):
        snr_lang = final.groupby('language')['snr_db'].mean().sort_values()
        ax.barh(snr_lang.index, snr_lang.values, color='#e67e22', alpha=0.85)
        ax.axvline(snr_lang.mean(), color='gray', linestyle='--',
                   label=f'Mean ({snr_lang.mean():.1f}dB)')
        ax.set_xlabel('Mean SNR (dB)')
        ax.set_title('SNR per Language (Final)', fontweight='bold')
        ax.legend(); ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    save(fig, '5_snr_improvement.png')

# ── Chart 6: Keep rate per language ──────────────────────────
print("Chart 6: Keep rate...")
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(LANGUAGES))
raw_counts   = [len(raw[raw['language']==l]) for l in LANGUAGES]
final_counts = [len(final[final['language']==l]) for l in LANGUAGES]
keep_pcts    = [f/r*100 if r>0 else 0 for f,r in zip(final_counts, raw_counts)]
colors_k = ['#2ecc71' if p>=25 else '#e74c3c' for p in keep_pcts]
bars = ax.bar(x, keep_pcts, color=colors_k, alpha=0.85, width=0.7)
ax.set_xticks(x); ax.set_xticklabels(LANGUAGES, rotation=35, ha='right', fontsize=8)
ax.set_ylabel('% Kept')
ax.set_title(f'Keep Rate per Language (avg {np.mean(keep_pcts):.1f}%)', fontweight='bold')
ax.axhline(np.mean(keep_pcts), color='gray', linestyle='--',
           label=f'Avg {np.mean(keep_pcts):.1f}%')
for i, (p, k) in enumerate(zip(keep_pcts, final_counts)):
    ax.text(i, p+0.5, f'{k}', ha='center', fontsize=7, fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
save(fig, '6_keep_rate.png')

# ── Chart 7: Speaker diversity ────────────────────────────────
print("Chart 7: Speakers...")
if 'speaker_id' in final.columns:
    fig, ax = plt.subplots(figsize=(13, 5))
    spk = final.groupby('language')['speaker_id'].nunique().reindex(LANGUAGES)
    colors_s = ['#2ecc71' if v>=50 else '#e67e22' if v>=30 else '#e74c3c'
                for v in spk.values]
    ax.bar(range(len(LANGUAGES)), spk.values, color=colors_s, alpha=0.85, width=0.7)
    ax.set_xticks(range(len(LANGUAGES)))
    ax.set_xticklabels(LANGUAGES, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Unique Speakers')
    ax.set_title('Speaker Diversity per Language (Final)', fontweight='bold')
    ax.axhline(50, color='gray', linestyle='--', label='50 speakers')
    for i, v in enumerate(spk.values):
        ax.text(i, v+1, str(v), ha='center', fontsize=7)
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save(fig, '7_speaker_diversity.png')

# ── Chart 8: Quality dashboard ───────────────────────────────
print("Chart 8: Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax = axes[0,0]
ax.bar(range(len(LANGUAGES)), keep_pcts, color=colors_k, alpha=0.85, width=0.7)
ax.set_xticks(range(len(LANGUAGES)))
ax.set_xticklabels(LANGUAGES, rotation=35, ha='right', fontsize=7)
ax.set_ylabel('% Kept'); ax.set_title('Keep Rate', fontweight='bold')
ax.axhline(np.mean(keep_pcts), color='gray', linestyle='--')
ax.grid(axis='y', alpha=0.3)

ax = axes[0,1]
hrs = final.groupby('language')['duration_sec'].sum()/3600 if 'duration_sec' in final.columns else pd.Series()
if len(hrs):
    hrs = hrs.reindex(LANGUAGES)
    ax.bar(range(len(LANGUAGES)), hrs.values, color='#3498db', alpha=0.85, width=0.7)
    ax.set_xticks(range(len(LANGUAGES)))
    ax.set_xticklabels(LANGUAGES, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Hours'); ax.set_title('Audio Hours per Language', fontweight='bold')
    ax.axhline(hrs.mean(), color='gray', linestyle='--', label=f'Avg {hrs.mean():.2f}h')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

ax = axes[1,0]
if 'dnsmos_ovrl' in final.columns:
    dns = final.groupby('language')['dnsmos_ovrl'].mean().reindex(LANGUAGES)
    colors_d = ['#2ecc71' if v>=3.1 else '#e67e22' for v in dns.values]
    ax.bar(range(len(LANGUAGES)), dns.values, color=colors_d, alpha=0.85, width=0.7)
    ax.set_xticks(range(len(LANGUAGES)))
    ax.set_xticklabels(LANGUAGES, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('DNSMOS OVRL'); ax.set_title('Audio Quality (DNSMOS)', fontweight='bold')
    ax.axhline(dns.mean(), color='gray', linestyle='--', label=f'Mean {dns.mean():.3f}')
    ax.set_ylim(2.8, 3.5); ax.legend(); ax.grid(axis='y', alpha=0.3)

ax = axes[1,1]
if 'speaker_id' in final.columns:
    spk2 = final.groupby('language')['speaker_id'].nunique().reindex(LANGUAGES)
    colors_s2 = ['#2ecc71' if v>=50 else '#e67e22' if v>=30 else '#e74c3c'
                 for v in spk2.values]
    ax.bar(range(len(LANGUAGES)), spk2.values, color=colors_s2, alpha=0.85, width=0.7)
    ax.set_xticks(range(len(LANGUAGES)))
    ax.set_xticklabels(LANGUAGES, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Speakers'); ax.set_title('Speaker Diversity', fontweight='bold')
    ax.axhline(50, color='gray', linestyle='--', label='50 target')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

fig.suptitle('IndicVoices TTS Corpus — Quality Dashboard (v5)', 
             fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
save(fig, '8_quality_dashboard.png')

print(f"\n✅ All plots saved to {PLOTS_DIR}/")
print(f"Input: {len(raw):,} | Final: {len(final):,} | Audio: {final['duration_sec'].sum()/3600:.2f}h")
