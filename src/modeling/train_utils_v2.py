import os, time, json, copy, random
import numpy as np, torch, torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, roc_curve, precision_recall_curve)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataloader.ecg_dataset import ECGVitalsDataset
from src.modeling.models.mobilenetv2_vitals_11lead import MobileNetV2_1D_Vitals


# ── LOSS ─────────────────────────────────────────────────────
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = (1.0 - xs_pos + self.clip).clamp(max=1.0) if self.clip > 0 else 1.0 - xs_pos
        loss = targets * torch.log(xs_pos.clamp(min=self.eps)) + \
               (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1 - targets)
            loss *= torch.pow(1 - pt, self.gamma_pos * targets + self.gamma_neg * (1 - targets))
        return -loss.mean()


def _worker_init(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# ── SESSION BATCH SAMPLER ─────────────────────────────────────
class SessionBatchSampler:
    """
    Each yielded batch = indices of all (or up to max_segs) segments of one session.
    max_segs=None → use all segments (for validation).
    max_segs=N    → randomly sample N segments (for training, memory safety).
    """
    def __init__(self, session_ids, shuffle=True, max_segs=None, seed=42):
        groups = defaultdict(list)
        for i, s in enumerate(session_ids):
            groups[s].append(i)
        self.batches  = list(groups.values())
        self.shuffle  = shuffle
        self.max_segs = max_segs
        self.rng      = random.Random(seed)

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle: self.rng.shuffle(order)
        for i in order:
            b = self.batches[i]
            if self.max_segs and len(b) > self.max_segs:
                b = self.rng.sample(b, self.max_segs)
            yield b

    def __len__(self):
        return len(self.batches)


class Training:
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

    # ── SETUP ────────────────────────────────────────────────
    def setup(self, resume_path=None):
        self.log(f'Device: {self.device}')
        ds_kw = dict(data_dir=self.dc['data_dir'],
                     label_columns=self.dc['label_columns'],
                     vitals_columns=self.vc['columns'],
                     vitals_medians=self.vc['medians'])

        train_csv = os.path.join(self.dc['splits_dir'], self.dc['train_csv'])
        val_csv   = os.path.join(self.dc['splits_dir'], self.dc['val_csv'])

        self.train_ds = ECGVitalsDataset(train_csv, **ds_kw)
        self.val_ds   = ECGVitalsDataset(val_csv,   **ds_kw)
        self.log(f'Train: {len(self.train_ds):,}  Val: {len(self.val_ds):,} segments')

        nw       = self.tc['num_workers']
        max_segs = self.tc.get('max_segs_per_session', 32)
        seed     = self.cfg['experiment']['seed']

        def _make_loader(ds, csv_path, shuffle, cap):
            sids = pd.read_csv(csv_path, usecols=['SessionID'])['SessionID'].tolist()
            samp = SessionBatchSampler(sids, shuffle=shuffle, max_segs=cap, seed=seed)
            return DataLoader(ds, batch_sampler=samp, num_workers=nw,
                              pin_memory=True, worker_init_fn=_worker_init,
                              persistent_workers=False, prefetch_factor=1)

        self.train_loader = _make_loader(self.train_ds, train_csv, True,  max_segs)
        self.val_loader   = _make_loader(self.val_ds,   val_csv,   False, max_segs)  # all segs

        mc = self.mc
        self.model = MobileNetV2_1D_Vitals(
            input_channels=mc['input_channels'], alpha=mc['alpha'],
            num_classes=mc['num_classes'], vitals_dim=mc['vitals_dim'],
            vitals_hidden_dim=mc['vitals_hidden_dim'],
            stride_size=list(mc['stride_size']),
            kernel_size=mc['kernel_size'], dropout_rate=mc['dropout_rate'])

        if self.cfg.get('device', {}).get('gpu_count', 1) > 1 and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.log(f'Params: {sum(p.numel() for p in self.model.parameters()):,}')

        self.criterion = self._build_loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.tc['lr'], weight_decay=self.tc['weight_decay'])
        self.scheduler = self._build_scheduler()

        self.best_metric, self.best_epoch = -np.inf, 0
        self.patience_ctr, self.best_state = 0, None
        self.save_dir = os.path.join(os.getcwd(), 'experiments', self.cfg['experiment']['name'])
        self.start_epoch = 1
        if resume_path:
            self._resume(resume_path)
        self.roc_dir  = os.path.join(self.save_dir, 'ROC_curves')
        os.makedirs(self.roc_dir, exist_ok=True)

        from ruamel.yaml import YAML as Y
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as f:
            Y().dump({k: v for k, v in self.cfg.items() if k != 'logger'}, f)


    def _resume(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch   = ckpt['epoch'] + 1
        self.best_metric   = ckpt['metrics']['auroc_macro']
        self.best_epoch    = ckpt['epoch']
        self.log(f'Resumed from {path} (epoch {ckpt["epoch"]}, AUROC {self.best_metric:.4f})')
        
    # ── LOSS & SCHEDULER (identical to v1) ───────────────────
    def _build_loss(self):
        name = self.tc.get('loss', 'BCEWithLogitsLoss')
        if name == 'AsymmetricLoss':
            gn, gp, cl = self.tc.get('asl_gamma_neg',4), self.tc.get('asl_gamma_pos',1), self.tc.get('asl_clip',0.05)
            self.log(f'Loss: AsymmetricLoss (γ_neg={gn}, γ_pos={gp}, clip={cl})')
            return AsymmetricLoss(gamma_neg=gn, gamma_pos=gp, clip=cl)
        pw = self._pos_weight()
        self.log(f'Loss: BCEWithLogitsLoss (pos_weight={"auto" if pw is not None else "none"})')
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    def _pos_weight(self):
        cw = self.tc.get('class_weights')
        if cw == 'auto':
            pos = self.train_ds.labels.sum(0).clip(min=1)
            return torch.tensor(np.sqrt((len(self.train_ds.labels) - pos) / pos),
                                dtype=torch.float32).to(self.device)
        if isinstance(cw, list):
            return torch.tensor(cw, dtype=torch.float32).to(self.device)
        return None

    def _build_scheduler(self):
        s, ep, wu, ml = (self.tc.get('scheduler','cosine'), self.tc['epochs'],
                         self.tc.get('warmup_epochs',0), self.tc.get('min_lr',1e-6))
        if s == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, max(ep-wu,1), eta_min=ml)
        if s == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, self.tc.get('step_size',10), self.tc.get('gamma',0.1))
        if s == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5, min_lr=ml)
        return None

    # ── CORE: SESSION-LEVEL SOFT VOTING ──────────────────────
    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total, n, preds, tgts = 0., 0, [], []
        accum = self.tc.get('accum_sessions', 1)  # gradient accumulation across sessions

        if train: self.optimizer.zero_grad()

        for step, (ecg, vit, lbl) in enumerate(
                tqdm(loader, desc='Train' if train else '  Val', leave=False)):

            ecg = ecg.to(self.device)  # (n_segs, 11, 2500)
            vit = vit.to(self.device)  # (n_segs, 3)
            lbl = lbl.to(self.device)  # (n_segs, C) — identical rows per session

            with torch.set_grad_enabled(train):
                logits     = self.model(ecg, vit)          # (n_segs, C)
                avg_logits = logits.mean(0, keepdim=True)  # (1, C)  ← soft vote
                ses_lbl    = lbl[0:1]                      # (1, C)
                loss       = self.criterion(avg_logits, ses_lbl)
                if train: (loss / accum).backward()

            if train and ((step + 1) % accum == 0):
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            total += loss.item(); n += 1
            with torch.no_grad():
                preds.append(torch.sigmoid(avg_logits).cpu().numpy())
            tgts.append(ses_lbl.cpu().numpy())

        # flush remaining accumulated gradients
        if train and (n % accum != 0):
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step(); self.optimizer.zero_grad()

        return total / max(n, 1), np.concatenate(preds), np.concatenate(tgts)

    # ── THRESHOLD TUNING ─────────────────────────────────────
    def _find_best_thresholds(self, loader):
        self.model.eval()
        _, preds, tgts = self._run_epoch(loader, False)
        lc = self.dc['label_columns']
        best_thr = np.full(len(lc), 0.5)
        self.log('\n── Per-Class Threshold Tuning ──')
        for i, name in enumerate(lc):
            if tgts[:, i].sum() == 0: continue
            prec, rec, thrs = precision_recall_curve(tgts[:, i], preds[:, i])
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            best_thr[i] = thrs[np.argmax(f1)]
            self.log(f'  {name:20s}: thr={best_thr[i]:.3f}  F1={f1.max():.4f}')
        with open(os.path.join(self.save_dir, 'best_thresholds.json'), 'w') as f:
            json.dump({n: float(best_thr[i]) for i, n in enumerate(lc)}, f, indent=2)
        return best_thr

    # ── METRICS ──────────────────────────────────────────────
    def _metrics(self, preds, tgts, thresholds=None):
        lc = self.dc['label_columns']
        if thresholds is None:
            thresholds = np.full(len(lc), self.ec.get('threshold', 0.5))
        m, aurocs, auprcs = {}, [], []
        for i, c in enumerate(lc):
            p, ng = tgts[:, i].sum(), len(tgts) - tgts[:, i].sum()
            if p > 0 and ng > 0:
                a = roc_auc_score(tgts[:, i], preds[:, i]); aurocs.append(a); m[f'auroc_{c}'] = a
            if p > 0:
                a = average_precision_score(tgts[:, i], preds[:, i]); auprcs.append(a); m[f'auprc_{c}'] = a
        m['auroc_macro'] = float(np.mean(aurocs)) if aurocs else 0.
        m['auprc_macro'] = float(np.mean(auprcs)) if auprcs else 0.
        m['f1_macro']    = float(f1_score(tgts, (preds >= thresholds).astype(int),
                                           average='macro', zero_division=0))
        return m

    # ── TRAIN LOOP ───────────────────────────────────────────
    def train(self):
        epochs, wu, base_lr = self.tc['epochs'], self.tc.get('warmup_epochs', 0), self.tc['lr']
        patience = self.tc.get('patience', 10)
        hist = {k: [] for k in ['train_loss','val_loss','train_auroc','val_auroc',
                                  'val_auprc','val_f1','lr']}

        self.log(f'\n{"═"*74}\nTRAINING START — Session-Level Soft Voting\n{"═"*74}')
        for ep in range(self.start_epoch, epochs + 1):
            t0 = time.time()
            if ep <= wu:
                for pg in self.optimizer.param_groups: pg['lr'] = base_lr * ep / wu

            tl, tp, tt = self._run_epoch(self.train_loader, True)
            vl, vp, vt = self._run_epoch(self.val_loader, False)
            tm, vm     = self._metrics(tp, tt), self._metrics(vp, vt)
            lr         = self.optimizer.param_groups[0]['lr']

            if ep > wu and self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(vm['auroc_macro'])
                else:
                    self.scheduler.step()

            for k, v in zip(hist, [tl, vl, tm['auroc_macro'], vm['auroc_macro'],
                                     vm['auprc_macro'], vm['f1_macro'], lr]):
                hist[k].append(v)

            self.log(f'Ep {ep:>3}/{epochs} │ Loss {tl:.4f}/{vl:.4f} │ '
                     f'AUROC {tm["auroc_macro"]:.4f}/{vm["auroc_macro"]:.4f} │ '
                     f'AUPRC {vm["auprc_macro"]:.4f} │ F1 {vm["f1_macro"]:.4f} │ '
                     f'LR {lr:.2e} │ {time.time()-t0:.0f}s')

            if vm['auroc_macro'] > self.best_metric:
                self.best_metric, self.best_epoch = vm['auroc_macro'], ep
                self.patience_ctr = 0
                self.best_state   = copy.deepcopy(self.model.state_dict())
                self._save_ckpt(ep, vm)
                self.log(f'  ✓ Best (ep {ep}, AUROC {vm["auroc_macro"]:.4f})')
            else:
                self.patience_ctr += 1
                if self.tc.get('early_stopping') and self.patience_ctr >= patience:
                    self.log(f'  ✗ Early stop at ep {ep}'); break

        self.log(f'\n{"═"*74}\nDONE — Best ep {self.best_epoch} '
                 f'(AUROC {self.best_metric:.4f})\n{"═"*74}')
        if self.best_state: self.model.load_state_dict(self.best_state)
        self.best_thresholds = self._find_best_thresholds(self.val_loader)
        self._roc_curves(self.val_loader, 'val')
        self._save_history(hist)
        self.train_ds.close(); self.val_ds.close()

    # ── IO (identical to v1) ─────────────────────────────────
    def _save_ckpt(self, ep, metrics):
        state = (self.model.module if isinstance(self.model, nn.DataParallel)
                 else self.model).state_dict()
        clean_cfg = json.loads(json.dumps(
            {k: v for k, v in self.cfg.items() if k != 'logger'}, default=str))
        torch.save(dict(epoch=ep, model_state_dict=state,
                        optimizer_state_dict=self.optimizer.state_dict(),
                        metrics=metrics, config=clean_cfg),
                   os.path.join(self.save_dir, 'best_model.pth'))
        with open(os.path.join(self.save_dir, 'best_val_metrics.json'), 'w') as f:
            json.dump({k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
                       for k, v in metrics.items()}, f, indent=2)

    def _roc_curves(self, loader, name):
        self.model.eval()
        _, preds, tgts = self._run_epoch(loader, False)
        lc = self.dc['label_columns']
        nc, cols = preds.shape[1], 5
        rows = (nc + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        for i, ax in enumerate(axes.flatten()):
            if i >= nc: ax.set_visible(False); continue
            p, n = tgts[:, i].sum(), len(tgts) - tgts[:, i].sum()
            if p > 0 and n > 0:
                fpr, tpr, _ = roc_curve(tgts[:, i], preds[:, i])
                ax.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_score(tgts[:,i],preds[:,i]):.3f}')
                ax.plot([0,1],[0,1],'k--',alpha=.3); ax.legend(fontsize=8)
            else:
                ax.text(.5,.5,'N/A',ha='center',va='center')
            ax.set_title(lc[i], fontsize=9)
        fig.suptitle(f'ROC — {name} (best ep {self.best_epoch})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.roc_dir, f'roc_{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _save_history(self, hist):
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(hist, f, indent=2)
        eps = range(1, len(hist['train_loss']) + 1)
        fig, ((a1,a2),(a3,a4)) = plt.subplots(2, 2, figsize=(14, 10))
        for ax, ks, t in [(a1,['train_loss','val_loss'],'Loss'),
                           (a2,['train_auroc','val_auroc'],'AUROC'),
                           (a3,['val_auprc','val_f1'],'AUPRC & F1')]:
            for k in ks: ax.plot(eps, hist[k], label=k)
            ax.set_title(t); ax.legend(); ax.grid(True, alpha=.3)
        a4.plot(eps, hist['lr'], 'r'); a4.set_title('LR')
        a4.set_yscale('log'); a4.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=150)
        plt.close()