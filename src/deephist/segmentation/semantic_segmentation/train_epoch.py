"""
Run supervised ML-experiment
"""
from typing import Dict

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from src.exp_management import tracking
from src.exp_management.data_provider import HoldoutSet
from src.exp_management.evaluation.dice import dice_coef, dice_denominator, dice_nominator
from src.exp_management.experiment.Experiment import Experiment
from src.pytorch_datasets.label_handler import LabelHandler



def train_epoch(exp: Experiment,
                holdout_set: HoldoutSet,
                model: nn.Module,
                criterion: _Loss,
                optimizer: torch.optim.Optimizer,
                label_handler: LabelHandler,
                epoch: int,
                args: Dict,
                writer: SummaryWriter) -> float:
    """Train the model on train-dataloader. Evaluate on val-dataloader

    Args:
        data_loaders (List[DataLoader]): [description]
        model (nn.Module): [description]
        criterion (_Loss): [description]
        optimizer (torch.optim.Optimizer): [description]
        label_handler (LabelHandler): [description]
        epoch (int): [description]
        args (Dict): [description]
        writer (writer): [description]

    Returns:
        float: Average validation loss after training step
    """

    for phase in ['train', 'vali']:
        
        if phase == 'train':
            data_loader = holdout_set.train_loader
        else:
            data_loader = holdout_set.vali_loader

        metric_logger = tracking.MetricLogger(delimiter="  ",
                                              tensorboard_writer=writer,
                                              args=args)
        metric_logger.add_meter(f'{phase}_loss',
                                tracking.SmoothedValue(window_size=1,
                                                       type='global_avg'))
        metric_logger.add_meter(f'{phase}_pixel_accuracy',
                                tracking.SmoothedValue(window_size=1,
                                                       type='global_avg'))

        metric_logger.add_meter(f'{phase}_dice_coef',
                                tracking.SmoothedValue(window_size=1,
                                                       type='global_avg'))
        
        metric_logger.add_meter(f'{phase}_step_dice',
                                tracking.SmoothedValue(window_size=1,
                                                       type='global_avg',
                                                       to_tensorboard=False))

        viz = tracking.Visualizer(writer=writer)

        header = f'{phase} GPU {args.gpu} Epoch: [{epoch}]'

        if phase == 'train':
            # switch to train mode
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        epoch_dice_nominator = 0
        epoch_dice_denominator = 0
        sample_images = None
        sample_labels = None
        sample_preds = None
        
        for images, labels in metric_logger.log_every(data_loader, args.print_freq, epoch, header, phase):

            if args.gpu is not None:
                images_gpu = images.cuda(args.gpu, non_blocking=True)
                labels_gpu = labels.cuda(args.gpu, non_blocking=True)
            # compute output and loss
            logits = model(images_gpu)
            loss = criterion(logits, labels_gpu)
            # compute gradiemt and do SGD step
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if args.log_details:
                # measure data loading time
                if sample_images is None:
                    sample_images = exp.unnormalize(images)
                    sample_labels = labels
                    sample_preds = logits.cpu().argmax(axis=1)
                    
                batch_accuracy = torch.sum(logits.cpu().argmax(axis=1) == labels)/(len(images)*256*256)
                
                step_dice_nominator = dice_nominator(y_true=labels_gpu,
                                                    y_pred=torch.argmax(logits, dim=1),
                                                    n_classes=args.number_of_classes)
                step_dice_denominator = dice_denominator(y_true=labels_gpu,
                                                    y_pred=torch.argmax(logits, dim=1),
                                                    n_classes=args.number_of_classes)
                
                # add up dice nom and denom over one epoch to get "epoch-dice-score" - different to WSI-dice score!
                epoch_dice_nominator += step_dice_nominator
                epoch_dice_denominator += step_dice_denominator
            
                step_dice, _ = dice_coef(dice_nominator=step_dice_nominator,
                                        dice_denominator=step_dice_denominator,
                                        n_classes=args.number_of_classes)
            else:    
                step_dice = 0
                batch_accuracy = 0
                
            if phase == 'train':
                metric_logger.update(train_pixel_accuracy=(batch_accuracy, len(images)),
                                     train_loss=(loss.item(), len(images)),
                                     train_step_dice=step_dice)
            else:
                metric_logger.update(vali_pixel_accuracy=(batch_accuracy, len(images)),
                                     vali_loss=(loss.item(), len(images)),
                                     vali_step_dice=step_dice)
        
        if args.log_details:
            epoch_dice, _ = dice_coef(dice_nominator=epoch_dice_nominator,
                                      dice_denominator=epoch_dice_denominator,
                                      n_classes=args.number_of_classes)
        else:
            epoch_dice = 0 
        
        if phase == 'train':
            metric_logger.update(train_dice_coef=epoch_dice)
        else:
            metric_logger.update(vali_dice_coef=epoch_dice)
        
        metric_logger.send_meters_to_tensorboard(step=epoch)
        if args.log_details:
            
            viz.plot_samples(tag=f'samples/{phase}_patch_samples',
                            images=sample_images,
                            col_size=8,
                            row_size=4,                       
                            epoch=epoch)
            
            viz.plot_masks(tag=f'samples/{phase}_mask_samples',
                        masks=sample_labels,
                        label_handler=label_handler,
                        col_size=8,
                        row_size=4,                       
                        epoch=epoch)
            
            viz.plot_masks(tag=f'samples/{phase}_pred_samples',
                        masks=sample_preds,
                        label_handler=label_handler,
                        col_size=8,
                        row_size=4,   
                        epoch=epoch)
            
        print(f"Averaged {phase} stats:", metric_logger.global_str())

    if args.performance_metric == 'dice':
        # performance set to (negative) Dice 
        performance_metric = -1 * metric_logger.vali_dice_coef.global_avg
    elif args.performance_metric == 'loss':
        performance_metric = metric_logger.vali_loss.global_avg
    return performance_metric
