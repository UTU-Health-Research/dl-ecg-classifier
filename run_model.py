'''
python run_model.py experiment_001.yaml                          # V1, default checkpoint
python run_model.py experiment_001.yaml --version 2              # V2 majority voting
python run_model.py experiment_001.yaml --checkpoint path/to.pth # custom checkpoint
'''

import os, sys, torch, random, logging, argparse
import numpy as np
from ruamel.yaml import YAML
import json


def main(config_path, checkpoint_override=None, version=1):
    cfg  = YAML().load(open(config_path))
    seed = cfg['experiment']['seed']
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True

    cfg['checkpoint_path'] = checkpoint_override or os.path.join(
        'experiments', cfg['experiment']['name'], 'best_model.pth')
    if not os.path.isfile(cfg['checkpoint_path']):
        raise FileNotFoundError(f'Checkpoint not found: {cfg["checkpoint_path"]}')

    thr_path = os.path.join('experiments', cfg['experiment']['name'], 'best_thresholds.json')
    if os.path.isfile(thr_path):
        with open(thr_path) as f: cfg['tuned_thresholds'] = json.load(f)
        print(f'Loaded per-class thresholds from {thr_path}')
    else:
        cfg['tuned_thresholds'] = None
        print('No thresholds file found, using default 0.5')

    save_dir = os.path.join(os.getcwd(), 'experiments', cfg['experiment']['name'])
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_dir, f'{cfg["experiment"]["name"]}_test.log'),
        format='%(asctime)s %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S')
    cfg['logger'] = logging.getLogger(cfg['experiment']['name'] + '_test')
    cfg['logger'].setLevel(logging.DEBUG)

    if version == 1:
        from src.modeling.predict_utils    import Predicting
    else:
        from src.modeling.predict_utils_v2 import Predicting

    pred = Predicting(cfg)
    pred.setup()
    pred.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--version', type=int, default=1, choices=[1, 2],
                        help='1=segment-level  2=session majority voting')
    args = parser.parse_args()
    path = next((p for p in [args.config, os.path.join('configs', args.config)]
                 if os.path.isfile(p)), None)
    if not path: raise FileNotFoundError(f'Config not found: {args.config}')
    main(path, args.checkpoint, args.version)
    print('Done.')