import os, numpy as np, pandas as pd, torch, h5py
from torch.utils.data import Dataset

class ECGVitalsDataset(Dataset):
    def __init__(self, csv_path, data_dir, label_columns, vitals_columns, vitals_medians, transform=None):
        self.data_dir, self.transform, self._h5 = data_dir, transform, {}
        df = pd.read_csv(csv_path)
        for c in vitals_columns: df[c] = df[c].fillna(vitals_medians.get(c, 0.0))
        df[label_columns] = df[label_columns].fillna(0)
        self.ecg_ids     = df['ECG_ID'].values
        self.batch_files = df['batch_file'].values
        self.vitals      = df[vitals_columns].values.astype(np.float32)
        self.labels      = df[label_columns].values.astype(np.float32)

    def __len__(self): return len(self.ecg_ids)

    def __getitem__(self, idx):
        bf = self.batch_files[idx]
        if bf not in self._h5: self._h5[bf] = h5py.File(os.path.join(self.data_dir, bf), 'r')
        ecg = torch.from_numpy(self._h5[bf][self.ecg_ids[idx]][:].astype(np.float32))
        if self.transform: ecg = self.transform(ecg)
        return ecg, torch.from_numpy(self.vitals[idx].copy()), torch.from_numpy(self.labels[idx].copy())

    def close(self):
        for h in self._h5.values(): h.close()
        self._h5.clear()