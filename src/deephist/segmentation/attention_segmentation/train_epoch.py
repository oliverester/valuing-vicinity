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
from deephist.segmentation.attention_segmentation.attention_inference import fill_memory

from src.exp_management.data_provider import HoldoutSet
from src.exp_management.evaluation.dice import dice_coef, dice_denominator, dice_nominator
from src.exp_management import tracking
from src.pytorch_datasets.label_handler import LabelHandler


def train_epoch(holdout_set: HoldoutSet,
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

    timer = tracking.Timer(verbose=False)

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
        
        # add attention logger per head
        for i in range(args.num_attention_heads):
            metric_logger.add_meter(f'{phase}_ex_con_central_attention/head_{i}',
                                tracking.SmoothedValue(window_size=1,
                                                       type='global_avg'))
            metric_logger.add_meter(f'{phase}_coeff_var_neighbour_attention/head_{i}',
                                tracking.SmoothedValue(window_size=1,
                                                       type='global_avg'))

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
        
        # get memory of dataset
        memory = wsi_dataset.embedding_memory
        k_neighbours = memory.k
        
        if args.memory_to_gpu is True:
            memory.to_gpu(args.gpu)
             
        # in first epoch, ignore embeddings memory
        # then, fill embedding memory
        if epoch > 0: 
            timer.start()

            memory = fill_memory(data_loader=big_data_loader,
                                 memory=memory,
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
                                            
            if args.log_details:
                
                logits_cpu = logits.cpu()
                labels_gpu.detach()
                logits.detach()
                
                if sample_images is None:
                    sample_images = images
                    sample_labels = labels
                    sample_preds = logits_cpu.argmax(axis=1)
                    
                # statistics over attention scores:
                if attention is not None:   
                    attention = attention.detach()             
                    # ensure central patch is not considered:
                    neighbour_masks[:, k_neighbours, k_neighbours] = 0
                    neighbour_masks = neighbour_masks.view(-1,1,1,(k_neighbours*2+1)*(k_neighbours*2+1)).cuda(args.gpu, non_blocking=True) # over all heads, neighbour to 1d
                    
                    number_of_attentions = torch.sum(neighbour_masks, dim=-1)
                    if args.use_self_attention:
                        # we have a centre patch which gets one attention score. 
                        # now, we want to determine if this attention score differs from the "expected" attention part (1/#attention_objects)
                        ratio_neighbour_patches = (number_of_attentions/(number_of_attentions+1))
                    else:
                        # we dont have a central patch attention score
                        ratio_neighbour_patches = 1
                    excess_contribution_central_attention = torch.mean(-((torch.sum(attention * neighbour_masks,-1)-(ratio_neighbour_patches)))/ratio_neighbour_patches,dim=0)
                    
                    mean_of_attentions_per_head = (torch.sum(attention * neighbour_masks,-1)/number_of_attentions)
                    att_deviation_from_mean = attention - mean_of_attentions_per_head.unsqueeze(3) 
                    var_attention_per_head_and_neighbourhood = torch.sum((att_deviation_from_mean * att_deviation_from_mean) * neighbour_masks, dim=-1) / \
                        torch.sum(neighbour_masks, dim=-1)
                    # coefficient of variance: sd/mean -> mean of attention differs per #neighbour patches 
                    coeff_var_attention = torch.mean(torch.sqrt(var_attention_per_head_and_neighbourhood) / mean_of_attentions_per_head, dim=0)
                    
                    excess_contribution_central_attention = excess_contribution_central_attention.cpu().numpy()
                    coeff_var_neighbour_attention = coeff_var_attention.cpu().numpy()
                else:
                    excess_contribution_central_attention = None
                    coeff_var_neighbour_attention = None
                    
                batch_accuracy = torch.sum(logits_cpu.argmax(axis=1) == labels)/(len(images)*256*256)

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
                excess_contribution_central_attention = None
                coeff_var_neighbour_attention = None
                    
            if phase == 'train':
                metric_logger.update(train_pixel_accuracy=(batch_accuracy, len(images)),
                                     train_loss=(loss.item(), len(images)),
                                     train_step_dice=step_dice)
                if excess_contribution_central_attention is not None:
                    # update attention logger per head
                    for i in range(args.num_attention_heads):
                        k = f'train_ex_con_central_attention_slash_head_{i}'
                        metric_logger.meters[k].update(excess_contribution_central_attention[i], len(images))
                        k = f'train_coeff_var_neighbour_attention_slash_head_{i}'
                        metric_logger.meters[k].update(coeff_var_neighbour_attention[i], len(images))
            else:
                metric_logger.update(vali_pixel_accuracy=(batch_accuracy, len(images)),
                                     vali_loss=(loss.item(), len(images)),
                                     vali_step_dice=step_dice)
                if excess_contribution_central_attention is not None:
                    # update attention logger per head
                    for i in range(args.num_attention_heads):
                        k = f'vali_ex_con_central_attention_slash_head_{i}'
                        metric_logger.meters[k].update(excess_contribution_central_attention[i], len(images))
                        k = f'vali_coeff_var_neighbour_attention_slash_head_{i}'
                        metric_logger.meters[k].update(coeff_var_neighbour_attention[i], len(images))
            
            timer.stop(key='logging')

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
            if phase == 'train':
                viz.plot_position_embeddings(tag=f'pos_embeddings',
                                             model=model,
                                             epoch=epoch)
             
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

