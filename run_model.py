'''
# Default — loads experiments/experiment_001/best_model.pth automatically
python run_model.py experiment_001.yaml

# Custom checkpoint
python run_model.py experiment_001.yaml --checkpoint experiments/experiment_001/best_model.pth
'''

import os, sys, torch, random, logging, argparse
import numpy as np
from ruamel.yaml import YAML
from src.modeling.predict_utils import Predicting


def main(config_path, checkpoint_override=None):
    cfg = YAML().load(open(config_path))
    seed = cfg['experiment']['seed']
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True

    # Resolve checkpoint path
    if checkpoint_override:
        cfg['checkpoint_path'] = checkpoint_override
    else:
        default = os.path.join('experiments', cfg['experiment']['name'], 'best_model.pth')
        cfg['checkpoint_path'] = default

    if not os.path.isfile(cfg['checkpoint_path']):
        raise FileNotFoundError(f'Checkpoint not found: {cfg["checkpoint_path"]}')

    # Logging
    save_dir = os.path.join(os.getcwd(), 'experiments', cfg['experiment']['name'])
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(save_dir, f'{cfg["experiment"]["name"]}_test.log'),
                        format='%(asctime)s %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S')
    cfg['logger'] = logging.getLogger(cfg['experiment']['name'] + '_test')
    cfg['logger'].setLevel(logging.DEBUG)

    pred = Predicting(cfg)
    pred.setup()
    pred.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test evaluation')
    parser.add_argument('config', help='Config YAML (path or name in configs/)')
    parser.add_argument('--checkpoint', default=None, help='Override checkpoint path')
    args = parser.parse_args()

    path = next((p for p in [args.config, os.path.join('configs', args.config)] if os.path.isfile(p)), None)
    if not path: raise FileNotFoundError(f'Config not found: {args.config}')
    main(path, args.checkpoint); print('Done.')
