import os, time, json, copy, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataloader.ecg_dataset import ECGVitalsDataset
from src.modeling.models.mobilenetv2_vitals_11lead import MobileNetV2_1D_Vitals

def _worker_init(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Training:
    def __init__(self, config):
        self.cfg = config
        self.dc, self.vc, self.mc, self.tc, self.ec = (
            config['data'], config['vitals'], config['model'], config['training'], config['evaluation'])
        self.logger = config.get('logger')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def log(self, msg):
        print(msg)
        if self.logger: self.logger.info(msg)

    # ── SETUP ────────────────────────────────────────────────
    def setup(self):
        self.log(f'Device: {self.device}')
        ds_kw = dict(data_dir=self.dc['data_dir'], label_columns=self.dc['label_columns'],
                     vitals_columns=self.vc['columns'], vitals_medians=self.vc['medians'])
        self.train_ds = ECGVitalsDataset(os.path.join(self.dc['splits_dir'], self.dc['train_csv']), **ds_kw)
        self.val_ds   = ECGVitalsDataset(os.path.join(self.dc['splits_dir'], self.dc['val_csv']),   **ds_kw)
        self.log(f'Train: {len(self.train_ds):,}  Val: {len(self.val_ds):,} segments')

        nw, bs = self.tc['num_workers'], self.tc['batch_size']
        ldr_kw = dict(batch_size=bs, num_workers=nw, pin_memory=True,
                      persistent_workers=nw > 0, worker_init_fn=_worker_init)
        self.train_loader = DataLoader(self.train_ds, shuffle=True,  drop_last=True,  **ldr_kw)
        self.val_loader   = DataLoader(self.val_ds,   shuffle=False, drop_last=False, **ldr_kw)

        mc = self.mc
        self.model = MobileNetV2_1D_Vitals(
            input_channels=mc['input_channels'], alpha=mc['alpha'], num_classes=mc['num_classes'],
            vitals_dim=mc['vitals_dim'], vitals_hidden_dim=mc['vitals_hidden_dim'],
            stride_size=list(mc['stride_size']), kernel_size=mc['kernel_size'], dropout_rate=mc['dropout_rate'])

        if self.cfg.get('device', {}).get('gpu_count', 1) > 1 and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.log(f'Params: {sum(p.numel() for p in self.model.parameters()):,}')

        pw = self._pos_weight()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.tc['lr'], weight_decay=self.tc['weight_decay'])
        self.scheduler = self._build_scheduler()

        self.best_metric, self.best_epoch, self.patience_ctr, self.best_state = -np.inf, 0, 0, None
        self.save_dir = os.path.join(os.getcwd(), 'experiments', self.cfg['experiment']['name'])
        self.roc_dir  = os.path.join(self.save_dir, 'ROC_curves')
        os.makedirs(self.roc_dir, exist_ok=True)

        from ruamel.yaml import YAML as Y
        y = Y()
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as f:
            y.dump({k: v for k, v in self.cfg.items() if k != 'logger'}, f)

    def _pos_weight(self):
        cw = self.tc.get('class_weights')
        if cw == 'auto':
            pos = self.train_ds.labels.sum(0).clip(min=1)
            w = (len(self.train_ds.labels) - pos) / pos
            return torch.tensor(w, dtype=torch.float32).to(self.device)
        if isinstance(cw, list):
            return torch.tensor(cw, dtype=torch.float32).to(self.device)
        return None

    def _build_scheduler(self):
        s, ep, wu, mlr = self.tc.get('scheduler','cosine'), self.tc['epochs'], self.tc.get('warmup_epochs',0), self.tc.get('min_lr',1e-6)
        if s == 'cosine':  return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, max(ep-wu,1), eta_min=mlr)
        if s == 'step':    return torch.optim.lr_scheduler.StepLR(self.optimizer, self.tc.get('step_size',10), self.tc.get('gamma',0.1))
        if s == 'plateau': return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5, min_lr=mlr)
        return None

    # ── EPOCH ────────────────────────────────────────────────
    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total, n, preds, tgts = 0., 0, [], []
        for ecg, vit, lbl in tqdm(loader, desc='Train' if train else '  Val', leave=False):
            ecg, vit, lbl = ecg.to(self.device), vit.to(self.device), lbl.to(self.device)
            if train: self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                logits = self.model(ecg, vit)
                loss = self.criterion(logits, lbl)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            total += loss.item(); n += 1
            with torch.no_grad(): preds.append(torch.sigmoid(logits).cpu().numpy())
            tgts.append(lbl.cpu().numpy())
        return total / max(n, 1), np.concatenate(preds), np.concatenate(tgts)

    # ── METRICS ──────────────────────────────────────────────
    def _metrics(self, preds, tgts):
        lc, thr = self.dc['label_columns'], self.ec.get('threshold', 0.5)
        m, aurocs, auprcs = {}, [], []
        for i, c in enumerate(lc):
            p, n = tgts[:, i].sum(), len(tgts) - tgts[:, i].sum()
            if p > 0 and n > 0:
                a = roc_auc_score(tgts[:, i], preds[:, i]); aurocs.append(a); m[f'auroc_{c}'] = a
            if p > 0:
                a = average_precision_score(tgts[:, i], preds[:, i]); auprcs.append(a); m[f'auprc_{c}'] = a
        m['auroc_macro'] = float(np.mean(aurocs)) if aurocs else 0.
        m['auprc_macro'] = float(np.mean(auprcs)) if auprcs else 0.
        m['f1_macro'] = float(f1_score(tgts, (preds >= thr).astype(int), average='macro', zero_division=0))
        return m

    # ── TRAIN LOOP ───────────────────────────────────────────
    def train(self):
        epochs, wu, base_lr = self.tc['epochs'], self.tc.get('warmup_epochs', 0), self.tc['lr']
        patience = self.tc.get('patience', 10)
        hist = {k: [] for k in ['train_loss','val_loss','train_auroc','val_auroc','val_auprc','val_f1','lr']}

        self.log(f'\n{"═"*74}\nTRAINING START\n{"═"*74}')
        for ep in range(1, epochs + 1):
            t0 = time.time()
            if ep <= wu:
                for pg in self.optimizer.param_groups: pg['lr'] = base_lr * ep / wu

            tl, tp, tt = self._run_epoch(self.train_loader, True)
            vl, vp, vt = self._run_epoch(self.val_loader, False)
            tm, vm = self._metrics(tp, tt), self._metrics(vp, vt)
            lr = self.optimizer.param_groups[0]['lr']

            if ep > wu and self.scheduler:
                self.scheduler.step(vm['auroc_macro']) if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else self.scheduler.step()

            for k, v in zip(hist, [tl, vl, tm['auroc_macro'], vm['auroc_macro'], vm['auprc_macro'], vm['f1_macro'], lr]):
                hist[k].append(v)

            self.log(f'Ep {ep:>3}/{epochs} │ Loss {tl:.4f}/{vl:.4f} │ '
                     f'AUROC {tm["auroc_macro"]:.4f}/{vm["auroc_macro"]:.4f} │ '
                     f'AUPRC {vm["auprc_macro"]:.4f} │ F1 {vm["f1_macro"]:.4f} │ '
                     f'LR {lr:.2e} │ {time.time()-t0:.0f}s')

            cur = vm['auroc_macro']
            if cur > self.best_metric:
                self.best_metric, self.best_epoch, self.patience_ctr = cur, ep, 0
                self.best_state = copy.deepcopy(self.model.state_dict())
                self._save_ckpt(ep, vm)
                self.log(f'  ✓ Best (ep {ep}, AUROC {cur:.4f})')
            else:
                self.patience_ctr += 1
                if self.tc.get('early_stopping') and self.patience_ctr >= patience:
                    self.log(f'  ✗ Early stop at ep {ep}'); break

        self.log(f'\n{"═"*74}\nDONE — Best ep {self.best_epoch} (AUROC {self.best_metric:.4f})\n{"═"*74}')
        if self.best_state: self.model.load_state_dict(self.best_state)
        self._roc_curves(self.val_loader, 'val')
        self._save_history(hist)
        self.train_ds.close(); self.val_ds.close()

    # ── IO ───────────────────────────────────────────────────
    def _save_ckpt(self, ep, metrics):
        state = (self.model.module if isinstance(self.model, nn.DataParallel) else self.model).state_dict()
        clean_cfg = json.loads(json.dumps({k: v for k, v in self.cfg.items() if k != 'logger'}, default=str))
        torch.save(dict(epoch=ep, model_state_dict=state, optimizer_state_dict=self.optimizer.state_dict(),
                        metrics=metrics, config=clean_cfg),
                os.path.join(self.save_dir, 'best_model.pth'))
        with open(os.path.join(self.save_dir, 'best_val_metrics.json'), 'w') as f:
            json.dump({k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
                       for k, v in metrics.items()}, f, indent=2)

    def _roc_curves(self, loader, name):
        self.model.eval(); _, preds, tgts = self._run_epoch(loader, False)
        lc, nc = self.dc['label_columns'], preds.shape[1]
        cols = 5; rows = (nc + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        for i, ax in enumerate(axes.flatten()):
            if i >= nc: ax.set_visible(False); continue
            p, n = tgts[:, i].sum(), len(tgts) - tgts[:, i].sum()
            if p > 0 and n > 0:
                fpr, tpr, _ = roc_curve(tgts[:, i], preds[:, i])
                ax.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_score(tgts[:, i], preds[:, i]):.3f}')
                ax.plot([0, 1], [0, 1], 'k--', alpha=.3); ax.legend(fontsize=8)
            else: ax.text(.5, .5, 'N/A', ha='center', va='center')
            ax.set_title(lc[i], fontsize=9)
        fig.suptitle(f'ROC — {name} (best ep {self.best_epoch})'); plt.tight_layout()
        plt.savefig(os.path.join(self.roc_dir, f'roc_{name}.png'), dpi=150, bbox_inches='tight'); plt.close()

    def _save_history(self, hist):
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f: json.dump(hist, f, indent=2)
        eps = range(1, len(hist['train_loss']) + 1)
        fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14, 10))
        for ax, ks, t in [(a1, ['train_loss','val_loss'], 'Loss'),
                          (a2, ['train_auroc','val_auroc'], 'AUROC'),
                          (a3, ['val_auprc','val_f1'], 'AUPRC & F1')]:
            for k in ks: ax.plot(eps, hist[k], label=k)
            ax.set_title(t); ax.legend(); ax.grid(True, alpha=.3)
        a4.plot(eps, hist['lr'], 'r'); a4.set_title('LR'); a4.set_yscale('log'); a4.grid(True, alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=150); plt.close()