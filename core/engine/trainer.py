import os
import sys
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from core.solver.optimizer import initialize_optimizer
from core.solver.sampler import initialize_sampler
from core.solver.lr_scheduler import initialize_lr_scheduler
from core.metric.loss import initialize_loss
from core.data.builder import initialize_dataloader
from core.utils.utils import select_target_type, inverse_normalize, save_checkpoint, print_msg


def training(cfg, model, train_dataset, val_dataset, estimator, logger=None):
    device = cfg.base.device
    optimizer = initialize_optimizer(cfg, model)
    weighted_sampler = initialize_sampler(cfg, train_dataset)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(cfg, optimizer)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset, weighted_sampler)

    # start training
    model.train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(1, cfg.train.epochs + 1):
        # resampling weight update
        if weighted_sampler:
            weighted_sampler.step()

        # update loss weights
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X, y = train_data
            X, y = X.to(device), y.to(device)
            y = select_target_type(y, cfg.train.criterion)

            # forward
            y_pred = model(X)
            loss = loss_function(y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update(y_pred, y)
            avg_acc = estimator.get_accuracy(6)
            avg_kappa = estimator.get_kappa(6)

            # visualize samples
            if cfg.train.sample_view and step % cfg.train.sample_view_interval == 0:
                samples = torchvision.utils.make_grid(X)
                samples = inverse_normalize(samples, cfg.data.mean, cfg.data.std)
                logger.add_image('input samples', samples, 0, dataformats='CHW')

            progress.set_description(
                'epoch: [{} / {}], loss: {:.6f}, acc: {:.4f}, kappa: {:.4f}'
                .format(epoch, cfg.train.epochs, avg_loss, avg_acc, avg_kappa)
            )

        # validation performance
        if epoch % cfg.train.eval_interval == 0:
            eval(model, val_loader, cfg.train.criterion, estimator, device)
            acc = estimator.get_accuracy(6)
            kappa = estimator.get_kappa(6)
            print('validation accuracy: {}, kappa: {}'.format(acc, kappa))
            if logger:
                logger.add_scalar('validation accuracy', acc, epoch)
                logger.add_scalar('validation kappa', kappa, epoch)

            # save model
            indicator = kappa if cfg.train.kappa_prior else acc
            if indicator > max_indicator:
                save_checkpoint(model,
                                epoch=epoch,
                                network=cfg.train.network,
                                acc1 = acc,
                                optimizer=optimizer,
                                save_path=os.path.join(cfg.base.save_path, 'best_validation_weights.pt'))
                max_indicator = indicator
                print_msg('Best in validation set. Model save at {}'.format(cfg.base.save_path))

        if epoch % cfg.train.save_interval == 0:
            save_checkpoint(model,
                            epoch=epoch,
                            network=cfg.train.network,
                            acc1 = acc,
                            optimizer=optimizer,
                            save_path=os.path.join(cfg.base.save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        curr_lr = optimizer.param_groups[0]['lr']
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if cfg.solver.lr_scheduler == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)
            logger.add_scalar('training accuracy', avg_acc, epoch)
            logger.add_scalar('training kappa', avg_kappa, epoch)
            logger.add_scalar('learning rate', curr_lr, epoch)

    # save final model
    save_checkpoint(model,
                    epoch=epoch,
                    network=cfg.train.network,
                    acc1 = acc,
                    optimizer=optimizer,
                    save_path=os.path.join(cfg.base.save_path, 'final_weights.pt'))
    
    if logger:
        logger.close()
