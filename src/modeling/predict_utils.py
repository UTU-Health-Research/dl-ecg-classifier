import os, json
import numpy as np, torch, torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataloader.ecg_dataset import ECGVitalsDataset
# from src.modeling.models.mobilenetv2_vitals_11lead import MobileNetV2_1D_Vitals
from src.modeling.models.seresnet18_vitals_11lead import SEResNet18_1D_Vitals



def _worker_init(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class Predicting:
    def __init__(self, config):
        self.cfg = config
        self.dc, self.vc, self.mc, self.tc, self.ec = (
            config['data'], config['vitals'], config['model'],
            config['training'], config['evaluation'])
        self.logger = config.get('logger')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def log(self, msg):
        print(msg)
        if self.logger: self.logger.info(msg)

    def setup(self):
        self.log(f'Device: {self.device}')
        test_csv = os.path.join(self.dc['splits_dir'], self.dc['test_csv'])

        self.test_meta = pd.read_csv(test_csv, usecols=['ECG_ID', 'SessionID'])

        self.test_ds = ECGVitalsDataset(
            csv_path=test_csv, data_dir=self.dc['data_dir'],
            label_columns=self.dc['label_columns'],
            vitals_columns=self.vc['columns'], vitals_medians=self.vc['medians'])
        self.log(f'Test segments: {len(self.test_ds):,}')

        nw = self.tc['num_workers']
        self.test_loader = DataLoader(
            self.test_ds, batch_size=self.tc['batch_size'], shuffle=False,
            num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
            worker_init_fn=_worker_init)

        mc = self.mc
        # self.model = MobileNetV2_1D_Vitals(
        #     input_channels=mc['input_channels'], alpha=mc['alpha'],
        #     num_classes=mc['num_classes'], vitals_dim=mc['vitals_dim'],
        #     vitals_hidden_dim=mc['vitals_hidden_dim'],
        #     stride_size=list(mc['stride_size']),
        #     kernel_size=mc['kernel_size'], dropout_rate=mc['dropout_rate'])
        self.model = SEResNet18_1D_Vitals(
                input_channels=mc['input_channels'],
                num_classes=mc['num_classes'],
                vitals_dim=mc['vitals_dim'],
                vitals_hidden_dim=mc['vitals_hidden_dim'],
                dropout_rate=mc['dropout_rate'])

        ckpt_path = self.cfg['checkpoint_path']
        self.log(f'Loading checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.log(f'Loaded from epoch {ckpt.get("epoch", "?")}')

        self.save_dir = os.path.join(os.getcwd(), 'experiments', self.cfg['experiment']['name'])
        self.roc_dir  = os.path.join(self.save_dir, 'ROC_curves')
        os.makedirs(self.roc_dir, exist_ok=True)

        lc    = self.dc['label_columns']
        tuned = self.cfg.get('tuned_thresholds')
        if tuned:
            self.thresholds = np.array([tuned[n] for n in lc])
            self.log('\nPer-class thresholds:')
            for n, t in zip(lc, self.thresholds): self.log(f'  {n:20s}: {t:.3f}')
        else:
            default = self.ec.get('threshold', 0.5)
            self.thresholds = np.full(len(lc), default)
            self.log(f'\nUsing default threshold: {default}')

    def _run_inference(self):
        self.model.eval()
        total, n, preds, tgts = 0., 0, [], []
        criterion = nn.BCEWithLogitsLoss()
        for ecg, vit, lbl in tqdm(self.test_loader, desc='Test', leave=False):
            ecg, vit, lbl = ecg.to(self.device), vit.to(self.device), lbl.to(self.device)
            with torch.no_grad():
                logits = self.model(ecg, vit)
                total += criterion(logits, lbl).item(); n += 1
                preds.append(torch.sigmoid(logits).cpu().numpy())
            tgts.append(lbl.cpu().numpy())
        return total / max(n, 1), np.concatenate(preds), np.concatenate(tgts)

    def _metrics(self, preds, tgts):
        lc, thr = self.dc['label_columns'], self.thresholds
        m, aurocs, auprcs, per_class = {}, [], [], []
        for i, c in enumerate(lc):
            p, n = tgts[:, i].sum(), len(tgts) - tgts[:, i].sum()
            row = {'label': c, 'n_pos': int(p), 'n_neg': int(n), 'threshold': float(thr[i])}
            if p > 0 and n > 0:
                a = roc_auc_score(tgts[:, i], preds[:, i]); aurocs.append(a); m[f'auroc_{c}'] = a; row['auroc'] = a
            if p > 0:
                a = average_precision_score(tgts[:, i], preds[:, i]); auprcs.append(a); m[f'auprc_{c}'] = a; row['auprc'] = a
            row['f1'] = float(f1_score(tgts[:, i], (preds[:, i] >= thr[i]).astype(int), zero_division=0))
            per_class.append(row)
        binary = (preds >= thr).astype(int)
        m['auroc_macro'] = float(np.mean(aurocs)) if aurocs else 0.
        m['auprc_macro'] = float(np.mean(auprcs)) if auprcs else 0.
        m['f1_macro']    = float(f1_score(tgts, binary, average='macro',  zero_division=0))
        m['auroc_micro'] = float(roc_auc_score(tgts, preds, average='micro'))
        m['auprc_micro'] = float(average_precision_score(tgts, preds, average='micro'))
        m['f1_micro']    = float(f1_score(tgts, binary, average='micro',  zero_division=0))
        return m, per_class

    def _save_predictions_csv(self, preds, tgts):
        lc = self.dc['label_columns']
        df = self.test_meta.copy()
        for i, c in enumerate(lc):
            df[f'{c}_prob'] = preds[:, i].astype(np.float32)
            df[f'{c}_pred'] = (preds[:, i] >= self.thresholds[i]).astype(np.int8)
            df[f'{c}_true'] = tgts[:, i].astype(np.int8)
        df.to_csv(os.path.join(self.save_dir, 'test_segment_predictions.csv'), index=False)
        self.log('Saved → test_segment_predictions.csv')

    def predict(self):
        self.log(f'\n{"═"*74}\nTEST EVALUATION — Segment-Level\n{"═"*74}')
        loss, preds, tgts = self._run_inference()
        metrics, per_class = self._metrics(preds, tgts)

        self.log(f'\nTest Loss : {loss:.4f}')
        self.log(f'\n{"Metric":<16} {"Macro":>10} {"Micro":>10}\n{"─"*38}')
        for name, mk, mik in [("AUROC","auroc_macro","auroc_micro"),
                               ("AUPRC","auprc_macro","auprc_micro"),
                               ("F1",  "f1_macro",   "f1_micro")]:
            self.log(f'{name:<16} {metrics[mk]:>10.4f} {metrics[mik]:>10.4f}')

        self.log(f'\n{"Label":<20} {"AUROC":>8} {"AUPRC":>8} {"F1":>8} {"Thresh":>8} {"Pos":>6} {"Neg":>6}')
        self.log('─' * 68)
        for r in per_class:
            self.log(f'{r["label"]:<20} {r.get("auroc",0):>8.4f} {r.get("auprc",0):>8.4f} '
                     f'{r["f1"]:>8.4f} {r["threshold"]:>8.3f} {r["n_pos"]:>6} {r["n_neg"]:>6}')

        with open(os.path.join(self.save_dir, 'test_results.json'), 'w') as f:
            json.dump({k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
                       for k, v in {'test_loss': loss, **metrics, 'per_class': per_class}.items()}, f, indent=2)

        self._save_predictions_csv(preds, tgts)
        self._roc_curves(preds, tgts)
        self.test_ds.close()
        self.log(f'\nResults saved to {self.save_dir}')

    def _roc_curves(self, preds, tgts):
        lc, nc = self.dc['label_columns'], preds.shape[1]
        cols = 5; rows = (nc + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        for i, ax in enumerate(axes.flatten()):
            if i >= nc: ax.set_visible(False); continue
            p, n = tgts[:, i].sum(), len(tgts) - tgts[:, i].sum()
            if p > 0 and n > 0:
                fpr, tpr, _ = roc_curve(tgts[:, i], preds[:, i])
                ax.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_score(tgts[:,i],preds[:,i]):.3f}')
                ax.plot([0,1],[0,1],'k--',alpha=.3); ax.legend(fontsize=8)
            else: ax.text(.5,.5,'N/A',ha='center',va='center')
            ax.set_title(lc[i], fontsize=9)
        fig.suptitle('ROC — Test Set (Segment-Level)'); plt.tight_layout()
        plt.savefig(os.path.join(self.roc_dir, 'roc_test.png'), dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f'ROC curves saved to {self.roc_dir}')