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

from src.deephist.segmentation.attention_segmentation.logging import initialize_logging, log_epoch, log_step
from src.exp_management import tracking
from src.exp_management.data_provider import HoldoutSet
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
    """Given the holdout-set: Train the model on train-dataloader. Evaluate on val-dataloader

    Args:
        holdout_set (holdout_set]): [description]
        model (nn.Module): [description]
        criterion (_Loss): [description]
        optimizer (torch.optim.Optimizer): [description]
        label_handler (LabelHandler): [description]
        epoch (int): [description]
        args (Dict): [description]
        writer (writer): [description]

    Returns:
        float: Average validation performance (smaller is better) after training step
    """
    
    for phase in ['train', 'vali']:

        if phase == 'train':
            data_loader = holdout_set.train_loader
            # for fast embedding inference
            big_data_loader = holdout_set.big_train_loader
            wsi_dataset = holdout_set.train_wsi_dataset
        else:
            data_loader = holdout_set.vali_loader
            big_data_loader = data_loader
            wsi_dataset= holdout_set.vali_wsi_dataset

        
        metric_logger = tracking.MetricLogger(delimiter="  ",
                                              tensorboard_writer=writer,
                                              args=args)
        viz = tracking.Visualizer(writer=writer)
        
        initialize_logging(metric_logger=metric_logger,
                           phase=phase,
                           num_heads=args.num_attention_heads)

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
        
        # get memory of dataset
        memory = wsi_dataset.embedding_memory
        
        if args.memory_to_gpu is True:
            memory.to_gpu(args.gpu)
             
        # in first epoch, ignore embeddings memory
        # then, fill embedding memory
        if epoch > 0: 
            memory.fill_memory(data_loader=big_data_loader,
                               model=model,
                               gpu=args.gpu)
                    
            if args.log_details:
                # T-sne viz of embedding memory
                viz.plot_tsne(tag=f"{phase}_memory_tsne",
                              wsi_dataset=wsi_dataset,
                              memory=memory,
                              sample_size=1000,
                              label_handler=holdout_set.data_provider.label_handler,
                              epoch=epoch)
                 
        for images, labels, _, neighbours_idx in metric_logger.log_every(data_loader, args.print_freq, epoch, header, phase):
            
            if args.gpu is not None:
                images_gpu = images.cuda(args.gpu, non_blocking=True)
                labels_gpu = labels.cuda(args.gpu, non_blocking=True)
                neighbours_idx = neighbours_idx.cuda(args.gpu, non_blocking=True)
            
            if epoch > 0:
                k_neighbour_embedding, k_neighbour_mask = memory.get_k_neighbour_embeddings(neighbours_idx=neighbours_idx)
                
                if not k_neighbour_embedding.is_cuda:
                    k_neighbour_embedding = k_neighbour_embedding.cuda(args.gpu, non_blocking=True)
                    k_neighbour_mask = k_neighbour_mask.cuda(args.gpu, non_blocking=True)
            else:
                k_neighbour_embedding = k_neighbour_mask = None
          
            logits, attention = model(images=images_gpu, 
                                      neighbour_masks=k_neighbour_mask,
                                      neighbour_embeddings=k_neighbour_embedding,
                                      return_attention=True) 

            if args.combine_criterion_after_epoch is not None:
                loss = criterion(logits, labels_gpu, epoch)
            else:
                loss = criterion(logits, labels_gpu)
                
            # compute gradient and do SGD step
            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                                            
            step_dice_nominator, step_dice_denominator = log_step(phase=phase,   
                                                                metric_logger=metric_logger,
                                                                loss=loss,
                                                                logits_gpu=logits,
                                                                labels_gpu=labels_gpu,
                                                                images=images,
                                                                attention_gpu=attention, 
                                                                neighbour_masks=k_neighbour_mask,
                                                                args=args)
            
            # add up dice nom and denom over one epoch to get "epoch-dice-score" - different to WSI-dice score!
            epoch_dice_nominator += step_dice_nominator
            epoch_dice_denominator += step_dice_denominator
                
            if sample_images is None:
                sample_images = exp.unnormalize(images)
                sample_labels = labels
                sample_preds = logits.detach().argmax(axis=1).cpu()
                
        #after epoch is finished:        
        log_epoch(phase=phase,
                    metric_logger=metric_logger, 
                    viz=viz, 
                    epoch_dice_nominator=epoch_dice_nominator,
                    epoch_dice_denominator=epoch_dice_denominator,
                    model=model,
                    sample_images=sample_images,
                    sample_labels=sample_labels,
                    sample_preds=sample_preds,
                    label_handler=label_handler,
                    epoch=epoch,
                    args=args)
            
        print(f"Averaged {phase} stats:", metric_logger.global_str())

    if args.performance_metric == 'dice':
        # performance set to (negative) Dice 
        performance_metric = -1 * metric_logger.vali_dice_coef.global_avg
    elif args.performance_metric == 'loss':
        performance_metric = metric_logger.vali_loss.global_avg
    return performance_metric

