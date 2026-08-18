"""
- MAX_SEGMENTS_PER_EPISODE: randomly sample N segments per episode
    to prevent long recordings from dominating + control disk usage
- Batched HDF5: segments packed into batch files (10K per file)
    instead of one file per segment → massive NTFS improvement
- gzip compression on HDF5 datasets
- Disk estimate printed before processing starts
- Resume support: skips already-processed episodes

Estimated output size (default settings):
    20K episodes × 10 segments × ~60KB compressed ≈ 12 GB

Output:
    preprocessed_parquet_data/
    ├── metadata.csv
    ├── batch_0000.h5    # contains seg_00000000 … seg_00009999
    ├── batch_0001.h5
    └── ...

Each batch .h5 internal structure:
    /seg_00000000   → (11, 2500) float32
    /seg_00000001   → (11, 2500) float32
    ...
"""

import os, glob, sys, shutil, gc, random
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

PARQUET_DIR = r'F:/model_dataset_parquet'
OUTPUT_DIR  = os.path.join(os.getcwd(), 'data', 'preprocessed_uppsala_data')
TEMP_DIR    = os.path.join(os.getcwd(), 'data', '_temp_multifile_episodes')

TARGET_FS          = 250
SEGMENT_DURATION_S = 10
SEGMENT_SAMPLES    = TARGET_FS * SEGMENT_DURATION_S   # 2500
MIN_SEGMENT_FRAC   = 0.5

# ── Segment cap ──
MAX_SEGMENTS_PER_EPISODE = 50   # randomly sample this many; set None for all
RANDOM_SEED              = 42

# ── Batched HDF5 ──
SEGMENTS_PER_BATCH = 10_000     # segments packed per .h5 file
HDF5_COMPRESSION   = 'gzip'
HDF5_COMP_LEVEL    = 4          # 1=fast, 9=small (4 is good balance)

# ── Lead configuration ──
LEAD_I_CODE    = 64
LEAD_II_CODE   = 65
RECORDED_CODES = [64, 65, 70, 71, 72, 73, 74]

FINAL_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                     'V1', 'V2', 'V3', 'V4', 'V5']
NUM_LEADS = 11

# ── Metadata / vitals ──
VITALS_COLS = ['Vital_Resp_First', 'Vital_HR_First', 'Vital_Temp_First']
META_COLS   = [
    'CareEpisodeID', 'SessionID', 'Age_Years', 'Gender',
    'Vital_BPSys_First', 'Vital_BPDia_First', 'Vital_SpO2_First',
] + VITALS_COLS

SIGNAL_COLS = [
    'CareEpisodeID', 'WaveChannelCode', 'StartSamplePosition',
    'SampleData', 'SamplingFrequency', 'NumBitsPerSample', 'NumSamples',
]


# ══════════════════════════════════════════════════════════════
#  BATCHED HDF5 WRITER
# ══════════════════════════════════════════════════════════════

class BatchedHDF5Writer:
    """Packs segments into batch files of fixed size."""

    def __init__(self, output_dir, segments_per_batch,
                 compression, comp_level):
        self.output_dir       = output_dir
        self.segments_per_batch = segments_per_batch
        self.compression      = compression
        self.comp_level       = comp_level

        self.current_batch_idx = 0
        self.current_count     = 0    # count within current batch
        self.current_file      = None
        self.total_written     = 0

    def _open_new_batch(self):
        if self.current_file is not None:
            self.current_file.close()

        batch_name = f'batch_{self.current_batch_idx:04d}.h5'
        batch_path = os.path.join(self.output_dir, batch_name)
        self.current_file = h5py.File(batch_path, 'w')
        self.current_count = 0

    def write_segment(self, seg_id, seg_array):
        """Write one (11, 2500) segment. Returns batch filename."""
        if self.current_file is None or \
           self.current_count >= self.segments_per_batch:
            if self.current_file is not None:
                self.current_file.close()
                self.current_batch_idx += 1
            self._open_new_batch()

        self.current_file.create_dataset(
            seg_id,
            data=seg_array,
            dtype='float32',
            compression=self.compression,
            compression_opts=self.comp_level,
        )
        self.current_count += 1
        self.total_written += 1

        return f'batch_{self.current_batch_idx:04d}.h5'

    def close(self):
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None


# ══════════════════════════════════════════════════════════════
#  SIGNAL PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════

def discover_columns(parquet_files):
    schema     = pq.read_schema(parquet_files[0])
    all_cols   = schema.names
    label_cols = sorted(c for c in all_cols if c.startswith('label_'))
    pass1_cols = [c for c in all_cols if c != 'SampleData']
    return all_cols, label_cols, pass1_cols


def safe_dirname(ep_id):
    return str(ep_id).replace('/', '_').replace('\\', '_').replace(' ', '_')


def decode_chunk(raw_bytes, num_bits=16):
    dtype = {8: np.int8, 16: np.int16, 32: np.int32}.get(int(num_bits))
    if dtype is None:
        return None
    return np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)


def parse_lead_chunks(lead_df):
    lead_df = lead_df.sort_values('StartSamplePosition').reset_index(drop=True)
    chunks, fs = [], None

    for _, row in lead_df.iterrows():
        if row['SampleData'] is None:
            continue
        if pd.isna(row.get('NumBitsPerSample', np.nan)):
            continue
        if pd.isna(row.get('SamplingFrequency', np.nan)):
            continue

        sig = decode_chunk(row['SampleData'], int(row['NumBitsPerSample']))
        if sig is None or len(sig) == 0:
            continue

        chunks.append((int(row['StartSamplePosition']), sig))
        if fs is None:
            fs = float(row['SamplingFrequency'])

    return chunks, fs


def extract_in_window(chunks, win_start, win_stop):
    length = win_stop - win_start
    signal = np.zeros(length, dtype=np.float32)
    valid  = np.zeros(length, dtype=bool)

    for chunk_start, chunk_sig in chunks:
        chunk_end = chunk_start + len(chunk_sig)
        eff_start = max(chunk_start, win_start)
        eff_end   = min(chunk_end, win_stop)
        if eff_start >= eff_end:
            continue

        src_start = eff_start - chunk_start
        src_end   = eff_end   - chunk_start
        dst_start = eff_start - win_start
        dst_end   = eff_end   - win_start

        signal[dst_start:dst_end] = chunk_sig[src_start:src_end]
        valid[dst_start:dst_end]  = True

    return signal, valid


def find_contiguous_regions(mask, min_samples):
    if not mask.any():
        return []

    diff   = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    stops  = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        stops = np.concatenate([stops, [len(mask)]])

    return [(s, e) for s, e in zip(starts, stops) if (e - s) >= min_samples]


def compute_derived_leads(region_leads):
    I  = region_leads[LEAD_I_CODE]
    II = region_leads[LEAD_II_CODE]

    return np.stack([
        I,
        II,
        II - I,                   # III
        -(I + II) / 2.0,          # aVR
        I - II / 2.0,             # aVL
        II - I / 2.0,             # aVF
        region_leads[70],         # V1
        region_leads[71],         # V2
        region_leads[72],         # V3
        region_leads[73],         # V4
        region_leads[74],         # V5
    ])


def apply_transforms(multi_lead, fs_orig, fs_new):
    bpf = BandPassFilter(fs=fs_orig)
    si  = Spline_interpolation(fs_new=fs_new, fs_old=fs_orig)
    try:
        return si(bpf(multi_lead))
    except Exception:
        out = []
        for ch in range(multi_lead.shape[0]):
            out.append(si(bpf(multi_lead[ch])))
        return np.stack(out)


def segment_ecg(multi_lead, seg_len, min_frac):
    _, total  = multi_lead.shape
    n_full    = total // seg_len
    remainder = total % seg_len

    segs = [multi_lead[:, i * seg_len : (i + 1) * seg_len]
            for i in range(n_full)]

    if remainder >= int(min_frac * seg_len):
        padded = np.zeros((multi_lead.shape[0], seg_len),
                          dtype=multi_lead.dtype)
        padded[:, :remainder] = multi_lead[:, n_full * seg_len:]
        segs.append(padded)

    return segs


# ══════════════════════════════════════════════════════════════
#  CORE — process one episode
# ══════════════════════════════════════════════════════════════

def process_episode(ep_id, ep_df, ep_meta, seg_counter,
                    all_meta_rows, carry_cols, writer, rng):
    """
    Returns (seg_counter, status).
    """

    # ── 1. Parse chunks per recorded lead ─────────────────────
    lead_chunks = {}
    fs_orig = None

    for code in RECORDED_CODES:
        ch_df = ep_df[ep_df['WaveChannelCode'] == code]
        if ch_df.empty:
            return seg_counter, 'missing_lead'

        chunks, fs = parse_lead_chunks(ch_df)
        if not chunks:
            return seg_counter, 'decode_fail'

        lead_chunks[code] = chunks
        if fs_orig is None:
            fs_orig = fs

    # ── 2. Find overlap window across ALL recorded leads ──────
    lead_ranges = {}
    for code, chunks in lead_chunks.items():
        earliest = min(s for s, _ in chunks)
        latest   = max(s + len(sig) for s, sig in chunks)
        lead_ranges[code] = (earliest, latest)

    win_start = max(r[0] for r in lead_ranges.values())
    win_stop  = min(r[1] for r in lead_ranges.values())

    if win_stop <= win_start:
        return seg_counter, 'no_overlap'

    # ── 3. Position-aware extraction + validity masks ─────────
    lead_signals = {}
    combined_valid = np.ones(win_stop - win_start, dtype=bool)

    for code in RECORDED_CODES:
        sig, valid = extract_in_window(lead_chunks[code],
                                       win_start, win_stop)
        lead_signals[code] = sig
        combined_valid &= valid

    del lead_chunks

    # ── 4. Find contiguous valid regions ──────────────────────
    min_region = int(SEGMENT_DURATION_S * MIN_SEGMENT_FRAC * fs_orig)
    regions = find_contiguous_regions(combined_valid, min_region)

    if not regions:
        return seg_counter, 'no_valid_region'

    # ── 5. Process each region → collect ALL segments first ───
    all_segments = []

    for region_start, region_stop in regions:
        region_leads = {
            code: lead_signals[code][region_start:region_stop]
            for code in RECORDED_CODES
        }

        multi_lead = compute_derived_leads(region_leads)
        del region_leads

        try:
            multi_lead = apply_transforms(multi_lead, int(fs_orig),
                                          TARGET_FS)
        except Exception:
            continue

        segments = segment_ecg(multi_lead, SEGMENT_SAMPLES,
                               MIN_SEGMENT_FRAC)
        del multi_lead
        all_segments.extend(segments)

    del lead_signals

    if not all_segments:
        return seg_counter, 'transform_fail'

    # ── 6. Cap segments (random subsample) ────────────────────
    if (MAX_SEGMENTS_PER_EPISODE is not None and
            len(all_segments) > MAX_SEGMENTS_PER_EPISODE):
        indices = sorted(rng.sample(range(len(all_segments)),
                                    MAX_SEGMENTS_PER_EPISODE))
        all_segments = [all_segments[i] for i in indices]

    # ── 7. Write to batched HDF5 ─────────────────────────────
    episode_seg_count = 0

    for seg_arr in all_segments:
        seg_id   = f'seg_{seg_counter:08d}'
        batch_fn = writer.write_segment(seg_id, seg_arr)

        row = {col: ep_meta.get(col) for col in carry_cols}
        row['ECG_ID']          = seg_id
        row['batch_file']      = batch_fn
        row['fs']              = TARGET_FS
        row['segment_index']   = episode_seg_count
        row['n_segments_total'] = len(all_segments)
        all_meta_rows.append(row)

        seg_counter += 1
        episode_seg_count += 1

    return seg_counter, 'ok'


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR,   exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, '*.parquet')))
    assert parquet_files, f'No parquet files in {PARQUET_DIR}'
    print(f'Found {len(parquet_files)} parquet files')

    all_columns, label_cols, pass1_cols = discover_columns(parquet_files)
    carry_cols = [c for c in META_COLS + label_cols if c in all_columns]
    print(f'Label columns ({len(label_cols)}): {label_cols}')
    print(f'Final leads   ({NUM_LEADS}): {FINAL_LEAD_ORDER}')

    # ── Estimate output size ──
    est_episodes  = 20_000
    est_segs      = MAX_SEGMENTS_PER_EPISODE or 100
    est_total     = est_episodes * est_segs
    est_size_gb   = est_total * 11 * 2500 * 4 / (1024**3)
    est_comp_gb   = est_size_gb * 0.5   # ~50% compression ratio
    print(f'\n── Disk estimate ──')
    print(f'  Max segments/episode : {MAX_SEGMENTS_PER_EPISODE or "unlimited"}')
    print(f'  Est. total segments  : ~{est_total:,}')
    print(f'  Est. size (raw)      : ~{est_size_gb:.1f} GB')
    print(f'  Est. size (gzip)     : ~{est_comp_gb:.1f} GB')

    rng = random.Random(RANDOM_SEED)

    # ══════════════════════════════════════════════════════════
    # PASS 1 — Index episodes (no SampleData)
    # ══════════════════════════════════════════════════════════
    print(f'\n{"═" * 60}')
    print('PASS 1 · Indexing episodes')
    print('═' * 60)

    episode_files = defaultdict(set)
    episode_meta  = {}

    for fpath in tqdm(parquet_files, desc='Pass 1'):
        df = pd.read_parquet(fpath, columns=pass1_cols)
        for ep_id, grp in df.groupby('CareEpisodeID'):
            episode_files[ep_id].add(fpath)
            if ep_id not in episode_meta:
                first = grp.iloc[0]
                episode_meta[ep_id] = {
                    c: first[c] for c in carry_cols if c in first.index
                }
        del df
        gc.collect()

    single_eps = {ep for ep, fs in episode_files.items() if len(fs) == 1}
    multi_eps  = {ep for ep, fs in episode_files.items() if len(fs) > 1}
    print(f'\nTotal episodes  : {len(episode_files):,}')
    print(f'  single-file   : {len(single_eps):,}')
    print(f'  multi-file    : {len(multi_eps):,}')

    # ══════════════════════════════════════════════════════════
    # PASS 2 — Process single-file; stash multi-file
    # ══════════════════════════════════════════════════════════
    print(f'\n{"═" * 60}')
    print('PASS 2 · Processing signals')
    print('═' * 60)

    all_meta_rows = []
    seg_counter   = 0
    stats         = Counter()
    processed     = set()

    writer = BatchedHDF5Writer(
        OUTPUT_DIR, SEGMENTS_PER_BATCH,
        HDF5_COMPRESSION, HDF5_COMP_LEVEL,
    )

    for fpath in tqdm(parquet_files, desc='Pass 2'):
        df = pd.read_parquet(fpath, columns=SIGNAL_COLS)

        for ep_id, ep_df in df.groupby('CareEpisodeID'):

            if ep_id in single_eps and ep_id not in processed:
                seg_counter, status = process_episode(
                    ep_id, ep_df, episode_meta.get(ep_id, {}),
                    seg_counter, all_meta_rows, carry_cols, writer, rng,
                )
                stats[status] += 1
                processed.add(ep_id)

            elif ep_id in multi_eps and ep_id not in processed:
                temp_ep_dir = os.path.join(TEMP_DIR,
                                           safe_dirname(ep_id))
                os.makedirs(temp_ep_dir, exist_ok=True)
                ep_df.to_parquet(
                    os.path.join(temp_ep_dir,
                                 f'chunk_{os.path.basename(fpath)}'),
                    index=False,
                )

        del df
        gc.collect()

    # ══════════════════════════════════════════════════════════
    # PASS 3 — Multi-file episodes
    # ══════════════════════════════════════════════════════════
    if multi_eps:
        print(f'\n{"═" * 60}')
        print(f'PASS 3 · {len(multi_eps):,} multi-file episodes')
        print('═' * 60)

        for ep_id in tqdm(sorted(multi_eps), desc='Pass 3'):
            temp_ep_dir = os.path.join(TEMP_DIR, safe_dirname(ep_id))
            if not os.path.exists(temp_ep_dir):
                stats['missing_temp'] += 1
                continue

            chunk_files = sorted(glob.glob(
                os.path.join(temp_ep_dir, '*.parquet')))
            ep_df = pd.concat(
                [pd.read_parquet(f) for f in chunk_files],
                ignore_index=True,
            )

            seg_counter, status = process_episode(
                ep_id, ep_df, episode_meta.get(ep_id, {}),
                seg_counter, all_meta_rows, carry_cols, writer, rng,
            )
            stats[status] += 1

            del ep_df
            shutil.rmtree(temp_ep_dir, ignore_errors=True)
            gc.collect()

    writer.close()
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # ══════════════════════════════════════════════════════════
    # Save metadata CSV
    # ══════════════════════════════════════════════════════════
    meta_df  = pd.DataFrame(all_meta_rows)
    csv_path = os.path.join(OUTPUT_DIR, 'metadata.csv')
    meta_df.to_csv(csv_path, index=False)

    # ══════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════
    print(f'\n{"═" * 60}')
    print('DONE')
    print('═' * 60)
    print(f'  Segments saved              : {seg_counter:,}')
    print(f'  Batch files created         : {writer.current_batch_idx + 1}')
    print(f'  Episodes processed (ok)     : {stats.get("ok", 0):,}')
    for reason in sorted(stats):
        if reason != 'ok':
            print(f'  Skipped ({reason:24s}): {stats[reason]:,}')
    n_ok = stats.get('ok', 0)
    if n_ok > 0:
        print(f'  Avg segments/episode        : {seg_counter / n_ok:.1f}')
    print(f'  Output directory            : {OUTPUT_DIR}')
    print(f'  Metadata CSV                : {csv_path}')
    print('═' * 60)


if __name__ == '__main__':
    main()