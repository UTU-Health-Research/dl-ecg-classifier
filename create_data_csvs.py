"""
Stratified multi-label train / val / test split (70 / 15 / 15)
at the CareEpisodeID level using iterative stratification.

Drop rule
    Episodes where ALL three vitals (Resp, HR, Temp) are NaN → removed
    before splitting.

Imputation
    Training-set vitals medians saved to vitals_medians.json so the
    dataloader can median-fill partially-missing vitals without leakage.

Requirements
    pip install iterative-stratification

Input
    data/preprocessed_uppsala_data/metadata.csv

Output
    data/preprocessed_uppsala_data/splits/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    ├── vitals_medians.json
    └── split_summary.txt
"""

import os, json
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
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
#  LOGGING HELPER
# ══════════════════════════════════════════════════════════════

_summary_lines = []

def L(msg=''):
    """Print and accumulate for summary file."""
    print(msg)
    _summary_lines.append(msg)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # ── 1  Load metadata ─────────────────────────────────────
    L(f'Loading {METADATA_CSV}')
    df = pd.read_csv(METADATA_CSV)
    L(f'  Segments loaded : {len(df):,}')
    L(f'  Episodes loaded : {df["CareEpisodeID"].nunique():,}')

    label_cols = sorted(c for c in df.columns if c.startswith('label_'))
    L(f'  Label columns   : {len(label_cols)}')

    # ── 2  Drop episodes where ALL 3 vitals are NaN ──────────
    ep = df.groupby('CareEpisodeID').first().reset_index()
    all_nan_mask = ep[VITALS_COLS].isna().all(axis=1)
    drop_eps = set(ep.loc[all_nan_mask, 'CareEpisodeID'])

    if drop_eps:
        ep_before, seg_before = ep.shape[0], len(df)
        df = df[~df['CareEpisodeID'].isin(drop_eps)].reset_index(drop=True)
        ep = ep[~all_nan_mask].reset_index(drop=True)
        L(f'\n  Dropped {len(drop_eps):,} episodes (all 3 vitals NaN)')
        L(f'    Episodes : {ep_before:,} → {len(ep):,}')
        L(f'    Segments : {seg_before:,} → {len(df):,}')
    else:
        L(f'\n  No episodes dropped (all have ≥ 1 vital)')

    # ── 3  Prepare label matrix ──────────────────────────────
    ep[label_cols] = ep[label_cols].fillna(0).astype(int)
    Y = ep[label_cols].values                       # (n_episodes, 19)
    episode_ids = ep['CareEpisodeID'].values
    n_episodes  = len(episode_ids)

    # Diagnostics
    label_sums = Y.sum(axis=0)
    for i, lc in enumerate(label_cols):
        if label_sums[i] == 0:
            L(f'  ⚠  {lc} has ZERO positives — will not affect stratification')

    no_pos = (Y.sum(axis=1) == 0).sum()
    L(f'\n  Episodes with no positive label : {no_pos:,} / {n_episodes:,} '
      f'({100 * no_pos / n_episodes:.1f}%)')

    # ── 4  Iterative stratified split ────────────────────────
    L(f'\n  Running iterative stratification (seed={RANDOM_SEED}) ...')

    # Step A: trainval 85 % | test 15 %
    msss1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_SEED,
    )
    trainval_idx, test_idx = next(msss1.split(np.zeros(n_episodes), Y))

    # Step B: train 70 % | val 15 %  (= 17.65 % of trainval)
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=RANDOM_SEED,
    )
    train_sub, val_sub = next(
        msss2.split(np.zeros(len(trainval_idx)), Y[trainval_idx])
    )
    train_idx = trainval_idx[train_sub]
    val_idx   = trainval_idx[val_sub]

    # Episode-ID sets
    train_eps = set(episode_ids[train_idx])
    val_eps   = set(episode_ids[val_idx])
    test_eps  = set(episode_ids[test_idx])

    assert not (train_eps & val_eps),  'Train / val overlap!'
    assert not (train_eps & test_eps), 'Train / test overlap!'
    assert not (val_eps & test_eps),   'Val / test overlap!'
    assert len(train_eps) + len(val_eps) + len(test_eps) == n_episodes

    L('  ✓ No overlap between splits')

    # ── 5  Map to segment level & save CSVs ──────────────────
    split_map = {}
    for eid in train_eps: split_map[eid] = 'train'
    for eid in val_eps:   split_map[eid] = 'val'
    for eid in test_eps:  split_map[eid] = 'test'

    df['split'] = df['CareEpisodeID'].map(split_map)
    assert df['split'].isna().sum() == 0, 'Unmapped episodes found!'

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df   = df[df['split'] == 'val'].reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].reset_index(drop=True)

    train_df.to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, 'val.csv'),     index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, 'test.csv'),   index=False)

    # ── 6  Training-set vitals medians (for imputation) ──────
    train_ep = ep[ep['CareEpisodeID'].isin(train_eps)]
    vitals_medians = {}
    for vc in VITALS_COLS:
        med = train_ep[vc].median()                 # NaN-safe
        vitals_medians[vc] = round(float(med), 4) if pd.notna(med) else 0.0

    med_path = os.path.join(SPLITS_DIR, 'vitals_medians.json')
    with open(med_path, 'w') as f:
        json.dump(vitals_medians, f, indent=2)

    L(f'\n  Training-set vitals medians (for imputation):')
    for vc, mv in vitals_medians.items():
        L(f'    {vc:<25}: {mv}')

    # ══════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ══════════════════════════════════════════════════════════

    total_seg = len(df)

    L(f'\n{"═" * 72}')
    L('SPLIT OVERVIEW')
    L('═' * 72)
    L(f'  {"Split":<8} {"Episodes":>10} {"Segments":>10} '
      f'{"Ep %":>8}  {"Seg %":>8}')
    L(f'  {"─" * 52}')

    for name, sdf in [('train', train_df), ('val', val_df), ('test', test_df)]:
        ne = sdf['CareEpisodeID'].nunique()
        ns = len(sdf)
        L(f'  {name:<8} {ne:>10,} {ns:>10,} '
          f'{100 * ne / n_episodes:>7.1f}%  {100 * ns / total_seg:>7.1f}%')

    L(f'  {"─" * 52}')
    L(f'  {"TOTAL":<8} {n_episodes:>10,} {total_seg:>10,} '
      f'{"100.0%":>8}  {"100.0%":>8}')

    # ── Per-label prevalence ─────────────────────────────────
    train_Y = Y[train_idx]
    val_Y   = Y[val_idx]
    test_Y  = Y[test_idx]

    L(f'\n  Per-label prevalence (episode-level):')
    L(f'  {"Label":<20} {"Train":>14}  {"Val":>14}  '
      f'{"Test":>14}  {"Total":>14}')
    L(f'  {"─" * 80}')

    for i, lc in enumerate(label_cols):
        tr = int(train_Y[:, i].sum())
        va = int(val_Y[:, i].sum())
        te = int(test_Y[:, i].sum())
        to = tr + va + te

        tr_p = 100 * tr / len(train_idx)
        va_p = 100 * va / len(val_idx)
        te_p = 100 * te / len(test_idx)
        to_p = 100 * to / n_episodes

        L(f'  {lc:<20} {tr:>5} ({tr_p:>5.2f}%)  '
          f'{va:>5} ({va_p:>5.2f}%)  '
          f'{te:>5} ({te_p:>5.2f}%)  '
          f'{to:>5} ({to_p:>5.2f}%)')

    # ── Vitals NaN rates ─────────────────────────────────────
    L(f'\n  Vitals NaN rate (segment-level):')
    L(f'  {"Vital":<25} {"Train":>16}  {"Val":>16}  {"Test":>16}')
    L(f'  {"─" * 78}')

    for vc in VITALS_COLS:
        parts = []
        for sdf in [train_df, val_df, test_df]:
            nm = sdf[vc].isna().sum()
            pct = 100 * nm / len(sdf) if len(sdf) else 0
            parts.append(f'{nm:>6} ({pct:>5.1f}%)')
        L(f'  {vc:<25} {parts[0]:>16}  {parts[1]:>16}  {parts[2]:>16}')

    # ── Files ────────────────────────────────────────────────
    L(f'\n  Output directory : {SPLITS_DIR}')
    L(f'    train.csv               ({len(train_df):>8,} segments)')
    L(f'    val.csv                 ({len(val_df):>8,} segments)')
    L(f'    test.csv                ({len(test_df):>8,} segments)')
    L(f'    vitals_medians.json')
    L(f'    split_summary.txt')
    L('═' * 72)

    # ── Save summary ─────────────────────────────────────────
    with open(os.path.join(SPLITS_DIR, 'split_summary.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(_summary_lines))


if __name__ == '__main__':
    main()