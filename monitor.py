import pandas as pd
import os
import time
import sys

csv = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/indic_filter/output/metrics.csv')

while True:
    os.system('clear')
    print(f"Monitoring: {csv}")
    if os.path.exists(csv):
        try:
            df = pd.read_csv(csv)
            print("=== Live Pipeline Monitor ===")
            print(f"Total processed : {len(df)}")
            kept = int(df['keep'].sum())
            print(f"Kept            : {kept} ({round(100*df['keep'].mean(),1)}%)")
            print(f"Rejected        : {len(df) - kept}")
            print()
            print("Per language:")
            for lang, grp in df.groupby('language'):
                k = int(grp['keep'].sum())
                t = len(grp)
                r = round(100*k/max(t,1),1)
                bar = '█' * int(r/5) + '░' * (20 - int(r/5))
                print(f"  {lang:<10} {k:>4}/{t:<6} ({r:>5}%)  {bar}")
            print()
            print("Top reject reasons:")
            rej = df[df['keep']==False]['reject_reason'].value_counts().head(5)
            for reason, count in rej.items():
                key = str(reason).split(':')[0]
                print(f"  {count:>5}x  {key}")
            print()
            scored = df[df['keep']==True]['composite_score'].dropna()
            if len(scored) > 0:
                print(f"Avg score (kept): {round(scored.mean(),2)}")
        except Exception as e:
            print(f"Reading... ({e})")
    else:
        print("Waiting for pipeline to start...")
    print("\n[refreshes every 5s — Ctrl+C to stop]")
    time.sleep(5)
