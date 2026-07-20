import os, sys, torch, random, logging
import numpy as np
from ruamel.yaml import YAML
from src.modeling.train_utils import Training


def main(config_path):
    cfg = YAML().load(open(config_path))
    seed = cfg['experiment']['seed']
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True

    save_dir = os.path.join(os.getcwd(), 'experiments', cfg['experiment']['name'])
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(save_dir, f'{cfg["experiment"]["name"]}_train.log'),
                        format='%(asctime)s %(message)s', filemode='w', datefmt='%Y-%m-%d %H:%M:%S')
    cfg['logger'] = logging.getLogger(cfg['experiment']['name'])
    cfg['logger'].setLevel(logging.DEBUG)

    trainer = Training(cfg)
    trainer.setup()
    trainer.train()


if __name__ == '__main__':
    if len(sys.argv) < 2: print('Usage: python train_model.py <config.yaml>'); sys.exit(1)
    arg = sys.argv[1]
    path = next((p for p in [arg, os.path.join('configs', arg)] if os.path.isfile(p)), None)
    if not path: raise FileNotFoundError(f'Config not found: {arg}')
    main(path); print('Done.')