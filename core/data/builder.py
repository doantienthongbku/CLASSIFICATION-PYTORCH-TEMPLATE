import os
import sys
import pickle

from torchvision import datasets
from PIL import Image
from torch.utils.data import DataLoader

from utils import auto_statistics, generate_dataset_from_pickle, generate_dataset_from_folder, print_dataset_info
from transform import data_transforms_torchvision


def generate_dataset(cfg):
    if cfg.data.mean == 'auto' or cfg.data.std == 'auto':
        mean, std = auto_statistics(
            cfg.base.data_path,
            cfg.base.data_index,
            cfg.data.input_size,
            cfg.train.batch_size,
            cfg.train.num_workers
        )
        cfg.data.mean = mean
        cfg.data.std = std

    train_transform, test_transform = data_transforms_torchvision(cfg)
    if cfg.base.data_index:
        datasets = generate_dataset_from_pickle(
            cfg.base.data_index,
            train_transform,
            test_transform
        )
    else:
        datasets = generate_dataset_from_folder(
            cfg.base.data_path,
            train_transform,
            test_transform
        )

    print_dataset_info(datasets)
    return datasets


# define data loader
def initialize_dataloader(cfg, train_dataset, val_dataset, weighted_sampler):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(weighted_sampler is None),
        sampler=weighted_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


# define data loader
def initialize_dataloader(cfg, train_dataset, val_dataset, weighted_sampler):
    batch_size = cfg.train.batch_size
    num_workers = cfg.train.num_workers
    pin_memory = cfg.train.pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(weighted_sampler is None),
        sampler=weighted_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
