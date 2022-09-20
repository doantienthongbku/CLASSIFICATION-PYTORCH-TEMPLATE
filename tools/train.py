import os
import sys
import random
import munch
import yaml

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.load_config import parse_config, load_config, copy_config
from core.utils.utils import print_config, print_msg, set_random_seed
from core.models.generate_model import generate_model
from core.data.builder import generate_dataset
from core.metric.metric import Estimator

from core.engine.trainer import training
from core.engine.evaluator import evaluate


def main():
    args = parse_config()
    cfg = load_config(args.config)

    # create folder
    save_path = cfg.base.save_path
    log_path = cfg.base.log_path
    if os.path.exists(save_path):
        warning = 'Save path {} exists.\nDo you want to overwrite it? (y/n)\n'.format(save_path)
        if not (args.overwrite or input(warning) == 'y'):
            sys.exit(0)
    else:
        os.makedirs(save_path)

    logger = SummaryWriter(log_path)
    copy_config(args.config, save_path)

    if args.print_config:
        print_config({
            'BASE CONFIG': cfg.base,
            'DATA CONFIG': cfg.data,
            'TRAIN CONFIG': cfg.train
        })
    else:
        print_msg('LOADING CONFIG FILE: {}'.format(args.config))

    # train
    set_random_seed(cfg.base.random_seed)
    model = generate_model(cfg)
    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    estimator = Estimator(cfg.train.criterion, cfg.data.num_classes)
    
    training(cfg=cfg, model=model, train_dataset=train_dataset,
             val_dataset=val_dataset, estimator=estimator, logger=logger)

    # test
    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(save_path, 'best_validation_weights.pt')
    evaluate(cfg, model, checkpoint, test_dataset, estimator)
    print('This is the performance of the final model:')
    checkpoint = os.path.join(save_path, 'final_weights.pt')
    evaluate(cfg, model, checkpoint, test_dataset, estimator)


if __name__ == '__main__':
    main()