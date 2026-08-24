"""
usage: python create_data_csvs.py
Stratified multi-label train / val / test split (70 / 15 / 15)
at the SessionID level using iterative stratification.
"""

import os, json
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ══════════════════════════════════════════════════════════════
DATA_DIR     = os.path.join(os.getcwd(), 'data', 'preprocessed_uppsala_data')
METADATA_CSV = os.path.join(DATA_DIR, 'metadata.csv')
SPLITS_DIR   = os.path.join(DATA_DIR, 'split_csvs')
VITALS_COLS  = ['Vital_Resp_First', 'Vital_HR_First', 'Vital_Temp_First']
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42

# ══════════════════════════════════════════════════════════════

_log = []
def L(msg=''):
    print(msg); _log.append(msg)

def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # ── 1. Load metadata ──────────────────────────────────────
    df = pd.read_csv(METADATA_CSV, low_memory=False)
    label_cols = sorted(c for c in df.columns if c.startswith('label_'))
    L(f'Segments  : {len(df):,}')
    L(f'Sessions  : {df["SessionID"].nunique():,}')
    L(f'Labels    : {len(label_cols)}')

    # ── 2. Session-level aggregation ──────────────────────────
    # One row per session; labels are already identical across segments
    ses = df.groupby('SessionID').first().reset_index()

    # ── 3. Drop sessions where ALL 3 vitals are NaN ───────────
    all_nan = ses[VITALS_COLS].isna().all(axis=1)
    drop    = set(ses.loc[all_nan, 'SessionID'])
    if drop:
        L(f'Dropped {len(drop):,} sessions (all vitals NaN)')
        df  = df[~df['SessionID'].isin(drop)].reset_index(drop=True)
        ses = ses[~all_nan].reset_index(drop=True)

    # ── 4. Label matrix ───────────────────────────────────────
    ses[label_cols] = ses[label_cols].fillna(0).astype(int)
    Y           = ses[label_cols].values
    session_ids = ses['SessionID'].values
    n_sessions  = len(session_ids)

    # ── 5. Iterative stratified split ─────────────────────────
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_SEED)
    trainval_idx, test_idx = next(msss1.split(np.zeros(n_sessions), Y))

    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=RANDOM_SEED)
    train_sub, val_sub = next(msss2.split(np.zeros(len(trainval_idx)), Y[trainval_idx]))

    train_idx = trainval_idx[train_sub]
    val_idx   = trainval_idx[val_sub]

    train_ses = set(session_ids[train_idx])
    val_ses   = set(session_ids[val_idx])
    test_ses  = set(session_ids[test_idx])

    assert not (train_ses & val_ses)  and \
           not (train_ses & test_ses) and \
           not (val_ses   & test_ses), 'Overlap detected!'
    assert len(train_ses) + len(val_ses) + len(test_ses) == n_sessions
    L('No overlap between splits ✓')

    # ── 6. Map segments to splits & save ──────────────────────
    split_map = {s: 'train' for s in train_ses}
    split_map.update({s: 'val'   for s in val_ses})
    split_map.update({s: 'test'  for s in test_ses})

    df['split'] = df['SessionID'].map(split_map)
    assert df['split'].isna().sum() == 0

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df   = df[df['split'] == 'val'  ].reset_index(drop=True)
    test_df  = df[df['split'] == 'test' ].reset_index(drop=True)

    train_df.to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
    val_df.to_csv(  os.path.join(SPLITS_DIR, 'val.csv'),   index=False)
    test_df.to_csv( os.path.join(SPLITS_DIR, 'test.csv'),  index=False)

    # ── 7. Vitals medians from train sessions ─────────────────
    train_ses_df   = ses[ses['SessionID'].isin(train_ses)]
    vitals_medians = {vc: round(float(m), 4) if pd.notna(m := train_ses_df[vc].median()) else 0.0
                      for vc in VITALS_COLS}
    with open(os.path.join(SPLITS_DIR, 'vitals_medians.json'), 'w') as f:
        json.dump(vitals_medians, f, indent=2)

    # ── 8. Summary ────────────────────────────────────────────
    L(f'\n{"═"*70}')
    L(f'{"Split":<8} {"Sessions":>10} {"Segments":>10} {"Ses%":>8} {"Seg%":>8}')
    L('─'*50)
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        ns = sdf['SessionID'].nunique()
        L(f'{name:<8} {ns:>10,} {len(sdf):>10,} '
          f'{100*ns/n_sessions:>7.1f}% {100*len(sdf)/len(df):>7.1f}%')
    L('─'*50)
    L(f'{"TOTAL":<8} {n_sessions:>10,} {len(df):>10,}')

    L(f'\nPer-label prevalence (session-level):')
    L(f'{"Label":<20} {"Train":>14} {"Val":>14} {"Test":>14}')
    L('─'*65)
    for i, lc in enumerate(label_cols):
        tr, va, te = Y[train_idx,i].sum(), Y[val_idx,i].sum(), Y[test_idx,i].sum()
        L(f'{lc:<20} {tr:>5}({100*tr/len(train_idx):>5.2f}%) '
          f'{va:>5}({100*va/len(val_idx):>5.2f}%) '
          f'{te:>5}({100*te/len(test_idx):>5.2f}%)')

    with open(os.path.join(SPLITS_DIR, 'split_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_log))

if __name__ == '__main__':
    main()