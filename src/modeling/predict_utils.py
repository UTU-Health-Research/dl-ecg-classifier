# import numpy as np
# import os, sys
# import time
# import torch
# from torch import nn
# import pandas as pd
# from torch.utils.data import DataLoader
# from .models.seresnet18 import resnet18
# from ..dataloader.dataset import ECGDataset, get_transforms
# from .metrics import cal_multilabel_metrics, roc_curves
# import pickle

# class Predicting(object):
#     def __init__(self, args):
#         self.args = args
    
#     def setup(self):
#         ''' Initializing the device conditions and dataloader,
#         loading trained model
#         '''
#         # Consider the GPU or CPU condition
#         if torch.cuda.is_available():
#             self.device = torch.device("cuda")
#             self.device_count = self.args.device_count
#             self.args.logger.info('using {} gpu(s)'.format(self.device_count))
#         else:
#             self.device = torch.device("cpu")
#             self.device_count = 1
#             self.args.logger.info('using {} cpu'.format(self.device_count))
        
#         # Find test files based on the test csv (for naming saved predictions)
#         # The paths for these files are in the 'path' column
#         filenames = pd.read_csv(self.args.test_path, usecols=['path']).values.tolist()
#         self.filenames = [f for file in filenames for f in file]

#         # Load the test data
#         testing_set = ECGDataset(self.args.test_path, 
#                                  get_transforms(dataset_type='test'))
#         channels = testing_set.channels
#         self.test_dl = DataLoader(testing_set,
#                                   batch_size=1,
#                                   shuffle=False,
#                                   pin_memory=(True if self.device == 'cuda' else False),
#                                   drop_last=True)
        
#         # Load the trained model
#         self.model = resnet18(in_channel=channels,
#                          out_channel=len(self.args.labels))
#         self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))

#         self.sigmoid = nn.Sigmoid()
#         self.sigmoid.to(self.device)
#         self.model.to(self.device)
        
#     def predict(self):
#         ''' Make predictions
#         '''
#         self.args.logger.info('predict() called: model={}, device={}'.format(
#               type(self.model).__name__,
#               self.device))

#         # Saving the history
#         history = {}
#         history['test_micro_avg_prec'] = 0.0
#         history['test_micro_auroc'] = 0.0
#         history['test_macro_avg_prec'] = 0.0
#         history['test_macro_auroc'] = 0.0
        
#         history['labels'] = self.args.labels
#         history['test_csv'] = self.args.test_path
#         history['threshold'] = self.args.threshold
        
#         start_time_sec = time.time()
 
#         # --- EVALUATE ON TESTING SET ------------------------------------- 
#         self.model.eval()
#         labels_all = torch.tensor((), device=self.device)
#         logits_prob_all = torch.tensor((), device=self.device)  
        
#         for i, (ecgs, ag, labels) in enumerate(self.test_dl):
#             ecgs = ecgs.to(self.device) # ECGs
#             ag = ag.to(self.device) # age and gender
#             labels = labels.to(self.device) # diagnoses in SMONED CT codes 

#             with torch.set_grad_enabled(False):  
                
#                 logits = self.model(ecgs, ag)
#                 logits_prob = self.sigmoid(logits)
#                 labels_all = torch.cat((labels_all, labels), 0)
#                 logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

#             if i % 1000 == 0:
#                 self.args.logger.info('{:<4}/{:>4} predictions made'.format(i+1, len(self.test_dl)))

#         # Predicting metrics
#         test_macro_avg_prec, test_micro_avg_prec, test_macro_auroc, test_micro_auroc = cal_multilabel_metrics(labels_all, logits_prob_all, self.args.labels, self.args.threshold)
        
#         self.args.logger.info('macro avg prec: {:<6.2f} micro avg prec: {:<6.2f} macro auroc: {:<6.2f} micro auroc: {:<6.2f} '.format(
#             test_macro_avg_prec,
#             test_micro_avg_prec,
#             test_macro_auroc,
#             test_micro_auroc))
        
#         # Draw ROC curve for predictions
#         roc_curves(labels_all, logits_prob_all, self.args.labels, save_path = self.args.output_dir)
        
#         # Add information to testing history
#         history['test_micro_auroc'] = test_micro_auroc
#         history['test_micro_avg_prec'] = test_micro_avg_prec
#         history['test_macro_auroc'] = test_macro_auroc
#         history['test_macro_avg_prec'] = test_macro_avg_prec
     
#         # Save the history
#         history_savepath = os.path.join(self.args.output_dir,
#                                         self.args.yaml_file_name + '_test_history.pickle')
#         with open(history_savepath, mode='wb') as file:
#             pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)

#         # Store labels and logits
#         filenames = [os.path.basename(file) for file in self.filenames]
        
#         logits_csv_path = os.path.join(self.args.output_dir,
#                                         self.args.yaml_file_name + '_test_logits.csv') 
#         logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
#         logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=filenames)
#         logits_df.to_csv(logits_csv_path, sep=',')

#         labels_csv_path = os.path.join(self.args.output_dir,
#                                         self.args.yaml_file_name + '_test_labels.csv') 
#         labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
#         labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=filenames)
#         labels_df.to_csv(labels_csv_path, sep=',')
            
#         torch.cuda.empty_cache()
        
#         end_time_sec = time.time()
#         total_time_sec = end_time_sec - start_time_sec
#         self.args.logger.info('Time total:     %5.2f sec' % (total_time_sec))


import os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, roc_curve, classification_report)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataloader.ecg_dataset import ECGVitalsDataset
from src.modeling.models.mobilenetv2_vitals_11lead import MobileNetV2_1D_Vitals


class Predicting:
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

        self.test_ds = ECGVitalsDataset(
            csv_path=os.path.join(self.dc['splits_dir'], self.dc['test_csv']),
            data_dir=self.dc['data_dir'], label_columns=self.dc['label_columns'],
            vitals_columns=self.vc['columns'], vitals_medians=self.vc['medians'])
        self.log(f'Test segments: {len(self.test_ds):,}')

        nw = self.tc['num_workers']
        self.test_loader = DataLoader(
            self.test_ds, batch_size=self.tc['batch_size'], shuffle=False,
            num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
            worker_init_fn=_worker_init)

        mc = self.mc
        self.model = MobileNetV2_1D_Vitals(
            input_channels=mc['input_channels'], alpha=mc['alpha'], num_classes=mc['num_classes'],
            vitals_dim=mc['vitals_dim'], vitals_hidden_dim=mc['vitals_hidden_dim'],
            stride_size=list(mc['stride_size']), kernel_size=mc['kernel_size'], dropout_rate=mc['dropout_rate'])

        # Load checkpoint
        ckpt_path = self.cfg['checkpoint_path']
        self.log(f'Loading checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.log(f'Loaded from epoch {ckpt.get("epoch", "?")}')

        self.save_dir = os.path.join(os.getcwd(), 'experiments', self.cfg['experiment']['name'])
        self.roc_dir  = os.path.join(self.save_dir, 'ROC_curves')
        os.makedirs(self.roc_dir, exist_ok=True)

    # ── INFERENCE ────────────────────────────────────────────
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

    # ── METRICS ──────────────────────────────────────────────
    # ── METRICS ──────────────────────────────────────────────
    def _metrics(self, preds, tgts):
        lc, thr = self.dc['label_columns'], self.ec.get('threshold', 0.5)
        m, aurocs, auprcs, per_class = {}, [], [], []
        for i, c in enumerate(lc):
            p, n = tgts[:, i].sum(), len(tgts) - tgts[:, i].sum()
            row = {'label': c, 'n_pos': int(p), 'n_neg': int(n)}
            if p > 0 and n > 0:
                a = roc_auc_score(tgts[:, i], preds[:, i]); aurocs.append(a); m[f'auroc_{c}'] = a; row['auroc'] = a
            if p > 0:
                a = average_precision_score(tgts[:, i], preds[:, i]); auprcs.append(a); m[f'auprc_{c}'] = a; row['auprc'] = a
            bi = (preds[:, i] >= thr).astype(int)
            row['f1'] = float(f1_score(tgts[:, i], bi, zero_division=0))
            per_class.append(row)

        binary_preds = (preds >= thr).astype(int)

        # Macro
        m['auroc_macro'] = float(np.mean(aurocs)) if aurocs else 0.
        m['auprc_macro'] = float(np.mean(auprcs)) if auprcs else 0.
        m['f1_macro']    = float(f1_score(tgts, binary_preds, average='macro', zero_division=0))

        # Micro
        m['auroc_micro'] = float(roc_auc_score(tgts, preds, average='micro'))
        m['auprc_micro'] = float(average_precision_score(tgts, preds, average='micro'))
        m['f1_micro']    = float(f1_score(tgts, binary_preds, average='micro', zero_division=0))

        return m, per_class

    # ── PREDICT (main entry) ─────────────────────────────────
    def predict(self):
        self.log(f'\n{"═"*74}\nTEST EVALUATION\n{"═"*74}')
        loss, preds, tgts = self._run_inference()
        metrics, per_class = self._metrics(preds, tgts)

        self.log(f'\nTest Loss   : {loss:.4f}')
        self.log(f'\n{"Metric":<16} {"Macro":>10} {"Micro":>10}')
        self.log('─' * 38)
        self.log(f'{"AUROC":<16} {metrics["auroc_macro"]:>10.4f} {metrics["auroc_micro"]:>10.4f}')
        self.log(f'{"AUPRC":<16} {metrics["auprc_macro"]:>10.4f} {metrics["auprc_micro"]:>10.4f}')
        self.log(f'{"F1":<16} {metrics["f1_macro"]:>10.4f} {metrics["f1_micro"]:>10.4f}')

        self.log(f'\n{"Label":<20} {"AUROC":>8} {"AUPRC":>8} {"F1":>8} {"Pos":>8} {"Neg":>8}')
        self.log('─' * 60)
        for r in per_class:
            self.log(f'{r["label"]:<20} {r.get("auroc",0):>8.4f} {r.get("auprc",0):>8.4f} '
                    f'{r["f1"]:>8.4f} {r["n_pos"]:>8} {r["n_neg"]:>8}')

        # Save results
        results = {'test_loss': loss, **metrics, 'per_class': per_class}
        with open(os.path.join(self.save_dir, 'test_results.json'), 'w') as f:
            json.dump({k: round(float(v), 6) if isinstance(v, (float, np.floating)) else v
                       for k, v in results.items()}, f, indent=2)

        # Save raw predictions
        np.savez(os.path.join(self.save_dir, 'test_predictions.npz'), preds=preds, targets=tgts)

        # ROC curves
        self._roc_curves(preds, tgts)

        self.test_ds.close()
        self.log(f'\nResults saved to {self.save_dir}')

    # ── ROC CURVES ───────────────────────────────────────────
    def _roc_curves(self, preds, tgts):
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
        fig.suptitle(f'ROC — Test Set'); plt.tight_layout()
        plt.savefig(os.path.join(self.roc_dir, 'roc_test.png'), dpi=150, bbox_inches='tight'); plt.close()
        self.log(f'ROC curves saved to {self.roc_dir}')


def _worker_init(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)