"""
preprocess_data.py
Lead order (11): I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5
Code mapping   : 64→I  65→II  70→V1  71→V2  72→V3  73→V4  74→V5
"""

import os, glob, shutil, gc
import numpy as np
import pandas as pd
import h5py
import pyarrow.parquet as pq
from tqdm import tqdm
from collections import defaultdict, Counter
from src.dataloader.transforms import BandPassFilter, Spline_interpolation

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════
PARQUET_DIR        = f'H:/UppsalaData(Parquet)'
OUTPUT_DIR         = os.path.join(os.getcwd(), 'data', 'preprocessed_uppsala_data')
TEMP_DIR           = os.path.join(os.getcwd(), 'data', '_temp_sessions')
VALID_SESSIONS_CSV = f'../UppsalaData-properSignals(19Diseases)-hotenc.csv'

TARGET_FS          = 250
SEGMENT_DURATION_S = 10
SEGMENT_SAMPLES    = TARGET_FS * SEGMENT_DURATION_S   # 2500
MIN_SEGMENT_FRAC   = 0.5

SEGMENTS_PER_BATCH = 10_000
HDF5_COMPRESSION   = 'gzip'
HDF5_COMP_LEVEL    = 4

RECORDED_CODES = [64, 65, 70, 71, 72, 73, 74]
SIGNAL_COLS    = ['SessionID', 'WaveChannelCode', 'StartSamplePosition',
                  'SampleData', 'SamplingFrequency', 'NumBitsPerSample']
CSV_NON_LABEL  = {'SessionID','64','65','66','70','71','72','73','74','75','ParquetFiles'}


# ══════════════════════════════════════════════════════════════
#  BATCHED HDF5 WRITER
# ══════════════════════════════════════════════════════════════
class BatchedHDF5Writer:
    def __init__(self, out_dir, spb, comp, lvl):
        self.out_dir, self.spb, self.comp, self.lvl = out_dir, spb, comp, lvl
        self.idx = self.cnt = self.total = 0
        self.file = None

    def _new_batch(self):
        if self.file: self.file.close()
        self.file = h5py.File(os.path.join(self.out_dir, f'batch_{self.idx:04d}.h5'), 'w')
        self.cnt = 0

    def write(self, seg_id, arr):
        if self.file is None or self.cnt >= self.spb:
            if self.file: self.idx += 1
            self._new_batch()
        self.file.create_dataset(seg_id, data=arr, dtype='float32',
                                 compression=self.comp, compression_opts=self.lvl)
        self.cnt += 1; self.total += 1
        return f'batch_{self.idx:04d}.h5'

    def close(self):
        if self.file: self.file.close(); self.file = None


# ══════════════════════════════════════════════════════════════
#  SIGNAL HELPERS
# ══════════════════════════════════════════════════════════════
def decode_chunk(raw, nbits):
    dtype = {8: np.int8, 16: np.int16, 32: np.int32}.get(int(nbits))
    return np.frombuffer(raw, dtype=dtype).astype(np.float32) if dtype else None

def parse_and_concat_lead(df):
    """Sort 5-second chunks by StartSamplePosition and concatenate → full signal."""
    df = df.sort_values('StartSamplePosition').reset_index(drop=True)
    segs, fs = [], None
    for _, r in df.iterrows():
        if r['SampleData'] is None or pd.isna(r.get('NumBitsPerSample', np.nan)):
            continue
        sig = decode_chunk(r['SampleData'], r['NumBitsPerSample'])
        if sig is None or not len(sig):
            continue
        segs.append(sig)
        if fs is None:
            fs = float(r['SamplingFrequency'])
    if not segs:
        return None, None
    return np.concatenate(segs).astype(np.float32), fs

def compute_leads(rl):
    I, II = rl[64], rl[65]
    return np.stack([I, II, II-I, -(I+II)/2, I-II/2, II-I/2,
                     rl[70], rl[71], rl[72], rl[73], rl[74]])

def apply_transforms(ml, fs_orig):
    bpf = BandPassFilter(fs=fs_orig)
    si  = Spline_interpolation(fs_new=TARGET_FS, fs_old=fs_orig)
    try: return si(bpf(ml))
    except Exception: return np.stack([si(bpf(ml[c])) for c in range(ml.shape[0])])

def make_segments(ml, seg_len, min_frac):
    _, N = ml.shape
    segs = [ml[:, i*seg_len:(i+1)*seg_len] for i in range(N // seg_len)]
    rem  = N % seg_len
    if rem >= int(min_frac * seg_len):
        pad = np.zeros((ml.shape[0], seg_len), dtype=ml.dtype)
        pad[:, :rem] = ml[:, (N // seg_len) * seg_len:]
        segs.append(pad)
    return segs


# ══════════════════════════════════════════════════════════════
#  PROCESS ONE SESSION
# ══════════════════════════════════════════════════════════════
def process_session(ses_id, ses_df, ses_meta, seg_counter, meta_rows, writer):
    lead_sigs, fs_orig = {}, None

    for code in RECORDED_CODES:
        sub = ses_df[ses_df['WaveChannelCode'] == code]
        if sub.empty: return seg_counter, 'missing_lead'
        sig, fs = parse_and_concat_lead(sub)
        if sig is None: return seg_counter, 'decode_fail'
        lead_sigs[code] = sig
        if fs_orig is None: fs_orig = fs

    # Trim all leads to minimum length (handle minor length mismatches)
    min_len      = min(len(s) for s in lead_sigs.values())
    min_required = int(SEGMENT_DURATION_S * MIN_SEGMENT_FRAC * fs_orig)
    if min_len < min_required: return seg_counter, 'too_short'
    lead_sigs = {c: s[:min_len] for c, s in lead_sigs.items()}

    ml = compute_leads(lead_sigs)
    del lead_sigs

    try:    ml = apply_transforms(ml, int(fs_orig))
    except: return seg_counter, 'transform_fail'

    all_segs = make_segments(ml, SEGMENT_SAMPLES, MIN_SEGMENT_FRAC)
    del ml
    if not all_segs: return seg_counter, 'no_segments'

    n_total = len(all_segs)
    for i, arr in enumerate(all_segs):
        seg_id = f'seg_{seg_counter:08d}'
        batch  = writer.write(seg_id, arr)
        meta_rows.append({
            'ECG_ID': seg_id, 'SessionID': ses_id,
            'batch_file': batch, 'fs': TARGET_FS,
            'segment_index': i, 'n_segments_total': n_total,
            **ses_meta,
        })
        seg_counter += 1

    return seg_counter, 'ok'


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR,   exist_ok=True)

    df_valid   = pd.read_csv(VALID_SESSIONS_CSV)
    valid_ids  = set(df_valid['SessionID'])
    label_cols = [c for c in df_valid.columns if c not in CSV_NON_LABEL]
    ses_labels = df_valid.set_index('SessionID')[label_cols].to_dict('index')
    print(f'Valid sessions : {len(valid_ids):,}')
    print(f'Label columns  : {label_cols}')

    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, '*.parquet')))
    assert parquet_files, f'No parquet files in {PARQUET_DIR}'
    print(f'Parquet files  : {len(parquet_files)}')

    # ── PASS 1: index sessions ─────────────────────────────────
    print('\nPASS 1 · Indexing')
    schema     = pq.read_schema(parquet_files[0])
    pass1_cols = [c for c in schema.names
                  if c not in ('SampleData', 'WaveChannelCode', 'StartSamplePosition',
                               'StopSamplePosition', 'SamplingFrequency',
                               'NumBitsPerSample', 'NumSamples', 'WaveDataID')]

    session_files, session_meta = defaultdict(set), {}

    for pf in tqdm(parquet_files, desc='Pass 1'):
        df = pd.read_parquet(pf, columns=[c for c in pass1_cols if c in schema.names])
        for ses_id, grp in df.groupby('SessionID'):
            if ses_id not in valid_ids: continue
            session_files[ses_id].add(pf)
            if ses_id not in session_meta:
                meta = {k: v for k, v in grp.iloc[0].to_dict().items()
                        if k != 'CareEpisodeID'}
                meta.update(ses_labels.get(ses_id, {}))
                session_meta[ses_id] = meta
        del df; gc.collect()

    single_ses = {s for s, f in session_files.items() if len(f) == 1}
    multi_ses  = {s for s, f in session_files.items() if len(f) > 1}
    print(f'Single-file: {len(single_ses):,}  |  Multi-file: {len(multi_ses):,}')

    # ── PASS 2: single-file sessions ───────────────────────────
    print('\nPASS 2 · Processing signals')
    meta_rows, seg_counter, stats = [], 0, Counter()
    processed = set()
    writer = BatchedHDF5Writer(OUTPUT_DIR, SEGMENTS_PER_BATCH, HDF5_COMPRESSION, HDF5_COMP_LEVEL)

    for pf in tqdm(parquet_files, desc='Pass 2'):
        df = pd.read_parquet(pf, columns=SIGNAL_COLS)
        for ses_id, ses_df in df.groupby('SessionID'):
            if ses_id not in valid_ids or ses_id in processed: continue
            if ses_id in single_ses:
                seg_counter, status = process_session(
                    ses_id, ses_df, session_meta.get(ses_id, {}),
                    seg_counter, meta_rows, writer)
                stats[status] += 1; processed.add(ses_id)
            elif ses_id in multi_ses:
                td = os.path.join(TEMP_DIR, str(ses_id))
                os.makedirs(td, exist_ok=True)
                ses_df.to_parquet(
                    os.path.join(td, f'chunk_{os.path.basename(pf)}'), index=False)
        del df; gc.collect()

    # ── PASS 3: multi-file sessions ────────────────────────────
    if multi_ses:
        print(f'\nPASS 3 · {len(multi_ses):,} multi-file sessions')
        for ses_id in tqdm(sorted(multi_ses), desc='Pass 3'):
            td     = os.path.join(TEMP_DIR, str(ses_id))
            chunks = sorted(glob.glob(os.path.join(td, '*.parquet')))
            if not chunks: stats['missing_temp'] += 1; continue
            ses_df = pd.concat([pd.read_parquet(f) for f in chunks], ignore_index=True)
            seg_counter, status = process_session(
                ses_id, ses_df, session_meta.get(ses_id, {}),
                seg_counter, meta_rows, writer)
            stats[status] += 1
            del ses_df; shutil.rmtree(td, ignore_errors=True); gc.collect()

    writer.close()
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    csv_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
    pd.DataFrame(meta_rows).to_csv(csv_path, index=False)

    print(f'\n{"═"*55}')
    print(f'Segments saved  : {seg_counter:,}')
    print(f'Batch files     : {writer.idx + 1}')
    for k, v in sorted(stats.items()):
        print(f'  {k:28s}: {v:,}')
    print(f'Output          : {OUTPUT_DIR}')
    print('═'*55)

if __name__ == '__main__':
    main()