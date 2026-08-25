import os, time, json, copy, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, precision_recall_curve
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataloader.ecg_dataset import ECGVitalsDataset
from src.modeling.models.mobilenetv2_vitals_11lead import MobileNetV2_1D_Vitals


# ── LOSS FUNCTIONS ───────────────────────────────────────────
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    Reference: https://arxiv.org/abs/2009.14119

    Args:
        gamma_neg: focusing parameter for negative samples (higher = more suppression)
        gamma_pos: focusing parameter for positive samples
        clip:      probability margin to shift (clip) negative samples
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # Probabilities
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Asymmetric clipping — shifts negative distribution
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Binary cross-entropy components
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # Asymmetric focal modulation
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = (self.gamma_pos * targets
                               + self.gamma_neg * (1 - targets))
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


def _worker_init(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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
        self.train_ds = ECGVitalsDataset(
            os.path.join(self.dc['splits_dir'], self.dc['train_csv']), **ds_kw)
        self.val_ds = ECGVitalsDataset(
            os.path.join(self.dc['splits_dir'], self.dc['val_csv']), **ds_kw)
        self.log(f'Train: {len(self.train_ds):,}  Val: {len(self.val_ds):,} segments')

        nw, bs = self.tc['num_workers'], self.tc['batch_size']
        ldr_kw = dict(batch_size=bs, num_workers=nw, pin_memory=True,
                      persistent_workers=False, worker_init_fn=_worker_init, prefetch_factor=1)
        self.train_loader = DataLoader(
            self.train_ds, shuffle=True, drop_last=True, **ldr_kw)
        self.val_loader = DataLoader(
            self.val_ds, shuffle=False, drop_last=False, **ldr_kw)

        mc = self.mc
        self.model = MobileNetV2_1D_Vitals(
            input_channels=mc['input_channels'], alpha=mc['alpha'],
            num_classes=mc['num_classes'], vitals_dim=mc['vitals_dim'],
            vitals_hidden_dim=mc['vitals_hidden_dim'],
            stride_size=list(mc['stride_size']),
            kernel_size=mc['kernel_size'],
            dropout_rate=mc['dropout_rate'])

        if (self.cfg.get('device', {}).get('gpu_count', 1) > 1
                and torch.cuda.device_count() > 1):
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.log(f'Params: {sum(p.numel() for p in self.model.parameters()):,}')

        # ── Loss (now config-driven) ────────────────────────
        self.criterion = self._build_loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.tc['lr'],
            weight_decay=self.tc['weight_decay'])
        self.scheduler = self._build_scheduler()

        self.best_metric, self.best_epoch = -np.inf, 0
        self.patience_ctr, self.best_state = 0, None
        self.save_dir = os.path.join(
            os.getcwd(), 'experiments', self.cfg['experiment']['name'])
        self.start_epoch = 1
        if resume_path:
            self._resume(resume_path)
        self.roc_dir = os.path.join(self.save_dir, 'ROC_curves')
        os.makedirs(self.roc_dir, exist_ok=True)

        from ruamel.yaml import YAML as Y
        y = Y()
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as f:
            y.dump({k: v for k, v in self.cfg.items() if k != 'logger'}, f)


    def _resume(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_metric = ckpt['metrics']['auroc_macro']
        self.best_epoch  = ckpt['epoch']
        self.log(f'Resumed from {path} (epoch {ckpt["epoch"]}, AUROC {self.best_metric:.4f})')

    # ── LOSS BUILDER ─────────────────────────────────────────
    def _build_loss(self):
        loss_name = self.tc.get('loss', 'BCEWithLogitsLoss')

        if loss_name == 'AsymmetricLoss':
            gamma_neg = self.tc.get('asl_gamma_neg', 4)
            gamma_pos = self.tc.get('asl_gamma_pos', 1)
            clip = self.tc.get('asl_clip', 0.05)
            self.log(f'Loss: AsymmetricLoss '
                     f'(γ_neg={gamma_neg}, γ_pos={gamma_pos}, clip={clip})')
            return AsymmetricLoss(
                gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)

        # Default: BCEWithLogitsLoss with optional pos_weight
        pw = self._pos_weight()
        self.log(f'Loss: BCEWithLogitsLoss '
                 f'(pos_weight={"auto" if pw is not None else "none"})')
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    def _pos_weight(self):
        cw = self.tc.get('class_weights')
        if cw == 'auto':
            pos = self.train_ds.labels.sum(0).clip(min=1)
            w = (len(self.train_ds.labels) - pos) / pos
            w = np.sqrt(w)
            return torch.tensor(w, dtype=torch.float32).to(self.device)
        if isinstance(cw, list):
            return torch.tensor(cw, dtype=torch.float32).to(self.device)
        return None

    def _build_scheduler(self):
        s = self.tc.get('scheduler', 'cosine')
        ep = self.tc['epochs']
        wu = self.tc.get('warmup_epochs', 0)
        mlr = self.tc.get('min_lr', 1e-6)
        if s == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, max(ep - wu, 1), eta_min=mlr)
        if s == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, self.tc.get('step_size', 10),
                self.tc.get('gamma', 0.1))
        if s == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5,
                factor=0.5, min_lr=mlr)
        return None

    # ── EPOCH ────────────────────────────────────────────────
    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total, n, preds, tgts = 0., 0, [], []
        for ecg, vit, lbl in tqdm(loader,
                                  desc='Train' if train else '  Val',
                                  leave=False):
            ecg = ecg.to(self.device)
            vit = vit.to(self.device)
            lbl = lbl.to(self.device)
            if train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                logits = self.model(ecg, vit)
                loss = self.criterion(logits, lbl)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            total += loss.item(); n += 1
            with torch.no_grad():
                preds.append(torch.sigmoid(logits).cpu().numpy())
            tgts.append(lbl.cpu().numpy())
        return total / max(n, 1), np.concatenate(preds), np.concatenate(tgts)

    # ── THRESHOLD TUNING ─────────────────────────────────────
    def _find_best_thresholds(self, loader):
        self.model.eval()
        _, preds, tgts = self._run_epoch(loader, False)

        lc = self.dc['label_columns']
        best_thresholds = np.full(len(lc), 0.5)

        self.log('\n── Per-Class Threshold Tuning (on validation set) ──')
        for i, name in enumerate(lc):
            pos = tgts[:, i].sum()
            if pos == 0:
                continue
            precision, recall, thresholds = precision_recall_curve(
                tgts[:, i], preds[:, i])
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            best_idx = np.argmax(f1)
            best_thresholds[i] = thresholds[best_idx]
            self.log(f'  {name:20s}: threshold={thresholds[best_idx]:.3f}'
                     f'  F1={f1[best_idx]:.4f}')

        thr_dict = {name: float(best_thresholds[i])
                    for i, name in enumerate(lc)}
        with open(os.path.join(self.save_dir, 'best_thresholds.json'), 'w') as f:
            json.dump(thr_dict, f, indent=2)

        self.log(f'  Saved to best_thresholds.json')
        return best_thresholds

    # ── METRICS ──────────────────────────────────────────────
    def _metrics(self, preds, tgts, thresholds=None):
        lc = self.dc['label_columns']
        if thresholds is None:
            thr = self.ec.get('threshold', 0.5)
            thresholds = np.full(len(lc), thr)

        m, aurocs, auprcs = {}, [], []
        for i, c in enumerate(lc):
            p = tgts[:, i].sum()
            n = len(tgts) - p
            if p > 0 and n > 0:
                a = roc_auc_score(tgts[:, i], preds[:, i])
                aurocs.append(a); m[f'auroc_{c}'] = a
            if p > 0:
                a = average_precision_score(tgts[:, i], preds[:, i])
                auprcs.append(a); m[f'auprc_{c}'] = a
        m['auroc_macro'] = float(np.mean(aurocs)) if aurocs else 0.
        m['auprc_macro'] = float(np.mean(auprcs)) if auprcs else 0.

        pred_binary = (preds >= thresholds).astype(int)
        m['f1_macro'] = float(f1_score(
            tgts, pred_binary, average='macro', zero_division=0))
        return m

    # ── TRAIN LOOP ───────────────────────────────────────────
    def train(self):
        epochs = self.tc['epochs']
        wu = self.tc.get('warmup_epochs', 0)
        base_lr = self.tc['lr']
        patience = self.tc.get('patience', 10)
        hist = {k: [] for k in [
            'train_loss', 'val_loss', 'train_auroc', 'val_auroc',
            'val_auprc', 'val_f1', 'lr']}

        self.log(f'\n{"═" * 74}\nTRAINING START\n{"═" * 74}')
        for ep in range(self.start_epoch, epochs + 1):
            t0 = time.time()
            if ep <= wu:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = base_lr * ep / wu

            tl, tp, tt = self._run_epoch(self.train_loader, True)
            vl, vp, vt = self._run_epoch(self.val_loader, False)
            tm = self._metrics(tp, tt)
            vm = self._metrics(vp, vt)
            lr = self.optimizer.param_groups[0]['lr']

            if ep > wu and self.scheduler:
                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(vm['auroc_macro'])
                else:
                    self.scheduler.step()

            for k, v in zip(hist, [
                    tl, vl, tm['auroc_macro'], vm['auroc_macro'],
                    vm['auprc_macro'], vm['f1_macro'], lr]):
                hist[k].append(v)

            self.log(
                f'Ep {ep:>3}/{epochs} │ Loss {tl:.4f}/{vl:.4f} │ '
                f'AUROC {tm["auroc_macro"]:.4f}/{vm["auroc_macro"]:.4f} │ '
                f'AUPRC {vm["auprc_macro"]:.4f} │ '
                f'F1 {vm["f1_macro"]:.4f} │ '
                f'LR {lr:.2e} │ {time.time() - t0:.0f}s')

            cur = vm['auroc_macro']
            if cur > self.best_metric:
                self.best_metric = cur
                self.best_epoch = ep
                self.patience_ctr = 0
                self.best_state = copy.deepcopy(self.model.state_dict())
                self._save_ckpt(ep, vm)
                self.log(f'  ✓ Best (ep {ep}, AUROC {cur:.4f})')
            else:
                self.patience_ctr += 1
                if (self.tc.get('early_stopping')
                        and self.patience_ctr >= patience):
                    self.log(f'  ✗ Early stop at ep {ep}')
                    break

        self.log(f'\n{"═" * 74}\n'
                 f'DONE — Best ep {self.best_epoch} '
                 f'(AUROC {self.best_metric:.4f})\n{"═" * 74}')
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        self.best_thresholds = self._find_best_thresholds(self.val_loader)
        self.log('\nThresholds ready. '
                 'Use self.best_thresholds for test evaluation.')
        self._roc_curves(self.val_loader, 'val')
        self._save_history(hist)
        self.train_ds.close(); self.val_ds.close()

    # ── IO ───────────────────────────────────────────────────
    def _save_ckpt(self, ep, metrics):
        state = (self.model.module
                 if isinstance(self.model, nn.DataParallel)
                 else self.model).state_dict()
        clean_cfg = json.loads(json.dumps(
            {k: v for k, v in self.cfg.items() if k != 'logger'},
            default=str))
        torch.save(
            dict(epoch=ep, model_state_dict=state,
                 optimizer_state_dict=self.optimizer.state_dict(),
                 metrics=metrics, config=clean_cfg),
            os.path.join(self.save_dir, 'best_model.pth'))
        with open(os.path.join(self.save_dir,
                               'best_val_metrics.json'), 'w') as f:
            json.dump({k: round(float(v), 6)
                       if isinstance(v, (float, np.floating)) else v
                       for k, v in metrics.items()}, f, indent=2)

    def _roc_curves(self, loader, name):
        self.model.eval()
        _, preds, tgts = self._run_epoch(loader, False)
        lc = self.dc['label_columns']
        nc = preds.shape[1]
        cols = 5
        rows = (nc + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        for i, ax in enumerate(axes.flatten()):
            if i >= nc:
                ax.set_visible(False); continue
            p = tgts[:, i].sum()
            n = len(tgts) - p
            if p > 0 and n > 0:
                fpr, tpr, _ = roc_curve(tgts[:, i], preds[:, i])
                auc = roc_auc_score(tgts[:, i], preds[:, i])
                ax.plot(fpr, tpr, lw=2, label=f'AUC={auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', alpha=.3)
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, 'N/A', ha='center', va='center')
            ax.set_title(lc[i], fontsize=9)
        fig.suptitle(f'ROC — {name} (best ep {self.best_epoch})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.roc_dir, f'roc_{name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _save_history(self, hist):
        with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
            json.dump(hist, f, indent=2)
        eps = range(1, len(hist['train_loss']) + 1)
        fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14, 10))
        for ax, ks, t in [
                (a1, ['train_loss', 'val_loss'], 'Loss'),
                (a2, ['train_auroc', 'val_auroc'], 'AUROC'),
                (a3, ['val_auprc', 'val_f1'], 'AUPRC & F1')]:
            for k in ks:
                ax.plot(eps, hist[k], label=k)
            ax.set_title(t); ax.legend(); ax.grid(True, alpha=.3)
        a4.plot(eps, hist['lr'], 'r')
        a4.set_title('LR'); a4.set_yscale('log'); a4.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'),
                    dpi=150)
        plt.close()