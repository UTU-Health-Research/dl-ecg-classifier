"""
Usage: python create_yaml_files.py
Generates a single experiment configuration YAML that serves as
the source of truth for training, evaluation, and inference.

Reads metadata.csv and vitals_medians.json to auto-populate
label columns, vitals stats, and data dimensions.

Input
    data/preprocessed_uppsala_data/metadata.csv
    data/preprocessed_uppsala_data/splits/vitals_medians.json

Output
    configs/experiment_xxx.yaml
"""

import os, json, sys
import pandas as pd
from ruamel.yaml import YAML
from datetime import datetime


# ══════════════════════════════════════════════════════════════
#  PATHS  (edit these if your layout differs)
# ══════════════════════════════════════════════════════════════

DATA_DIR        = os.path.join(os.getcwd(), 'data', 'preprocessed_uppsala_data')
SPLITS_DIR      = os.path.join(DATA_DIR, 'split_csvs')
METADATA_CSV    = os.path.join(DATA_DIR, 'metadata.csv')
MEDIANS_JSON    = os.path.join(SPLITS_DIR, 'vitals_medians.json')

CONFIG_DIR      = os.path.join(os.getcwd(), 'configs')
EXPERIMENT_NAME = 'experiment_004'


# ══════════════════════════════════════════════════════════════
#  AUTO-DISCOVER FROM DATA
# ══════════════════════════════════════════════════════════════

def discover_from_data():
    """Read metadata + medians to extract labels, vitals info, etc."""

    info = {}

    # ── Label columns ────────────────────────────────────────
    print(f'Reading {METADATA_CSV} ...')
    df = pd.read_csv(METADATA_CSV, nrows=5)          # only need columns
    label_cols = sorted(c for c in df.columns if c.startswith('label_'))
    info['label_cols']  = label_cols
    info['num_classes'] = len(label_cols)
    print(f'  Found {len(label_cols)} label columns')

    # ── Split file sizes ─────────────────────────────────────
    for split in ['train', 'val', 'test']:
        path = os.path.join(SPLITS_DIR, f'{split}.csv')
        if os.path.exists(path):
            n = sum(1 for _ in open(path)) - 1       # minus header
            info[f'{split}_segments'] = n
            print(f'  {split}.csv : {n:,} segments')

    # ── Vitals medians ───────────────────────────────────────
    if os.path.exists(MEDIANS_JSON):
        with open(MEDIANS_JSON) as f:
            info['vitals_medians'] = json.load(f)
        print(f'  Vitals medians loaded from {MEDIANS_JSON}')
    else:
        print(f'  ⚠ {MEDIANS_JSON} not found — vitals_medians will be empty')
        info['vitals_medians'] = {}

    return info


# ══════════════════════════════════════════════════════════════
#  BUILD CONFIG DICT
# ══════════════════════════════════════════════════════════════

def build_config(info):
    """Construct the full experiment config as an ordered dict."""

    config = {}

    # ── Experiment metadata ──────────────────────────────────
    config['experiment'] = {
        'name':        EXPERIMENT_NAME,
        'description': 'Multi-label ECG classification with vitals fusion',
        'created':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'seed':        42,
    }

    # ── Data ─────────────────────────────────────────────────
    config['data'] = {
        'data_dir':     os.path.relpath(DATA_DIR, os.getcwd()),
        'splits_dir':   os.path.relpath(SPLITS_DIR, os.getcwd()),
        'train_csv':    'train.csv',
        'val_csv':      'val.csv',
        'test_csv':     'test.csv',

        # Signal properties
        'fs':               250,
        'segment_samples':  2500,
        'num_leads':        11,
        'lead_order':       ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                             'V1', 'V2', 'V3', 'V4', 'V5'],

        # Labels
        'label_columns':    info['label_cols'],
        'num_classes':      info['num_classes'],
    }

    # ── Vitals ───────────────────────────────────────────────
    config['vitals'] = {
        'columns':       ['Vital_Resp_First', 'Vital_HR_First', 'Vital_Temp_First'],
        'dim':           3,
        'impute_method': 'median',
        'medians':       info['vitals_medians'],
    }

    # ── Model ────────────────────────────────────────────────
    config['model'] = {
        'architecture':     'MobileNetV2_1D_Vitals',
        'input_channels':   11,
        'num_classes':      info['num_classes'],
        'alpha':            1.0,
        'stride_size':      [2, 2, 2, 2, 2],
        'kernel_size':      9,
        'dropout_rate':     0.3,
        'vitals_dim':       3,
        'vitals_hidden_dim': 16,
    }

    # ── Training ─────────────────────────────────────────────
    config['training'] = {
        'batch_size':     64,
        'num_workers':    4,
        'epochs':         30,
        'lr':             0.001,
        'weight_decay':   0.00001,

        # Scheduler
        'scheduler':      'cosine',
        'warmup_epochs':  3,
        'min_lr':         0.000001,

        # Loss
        'loss':           'BCEWithLogitsLoss',
        'class_weights':  'auto',   

        # Early stopping
        'early_stopping': True,
        'patience':       7,
        'monitor':        'val_auroc_macro',
        'monitor_mode':   'max',

        # Checkpointing
        'save_dir':       'checkpoints',
        'save_best_only': True,
    }

    # ── Evaluation ───────────────────────────────────────────
    config['evaluation'] = {
        'threshold':  0.5,
        'metrics':    ['auroc_macro', 'auroc_per_class',
                       'auprc_macro', 'auprc_per_class',
                       'f1_macro', 'f1_per_class'],
    }

    # ── Device ───────────────────────────────────────────────
    config['device'] = {
        'gpu_count':  1,
        'fp16':       False,
    }

    return config


# ══════════════════════════════════════════════════════════════
#  SAVE
# ══════════════════════════════════════════════════════════════

def save_config(config, config_path):
    """Write config dict to a YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f'\n  ✓ Config saved to {config_path}')


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('CREATE EXPERIMENT CONFIG')
    print('=' * 60)

    # Auto-discover from data artifacts
    info = discover_from_data()

    # Build config
    config = build_config(info)

    # Save
    config_path = os.path.join(CONFIG_DIR, f'{EXPERIMENT_NAME}.yaml')
    save_config(config, config_path)

    # Print what was generated
    print(f'\n{"─" * 60}')
    print('Config contents:')
    print(f'{"─" * 60}')

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.dump(config, sys.stdout)


if __name__ == '__main__':
    main()