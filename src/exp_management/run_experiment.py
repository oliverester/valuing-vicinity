"""
Run supervised ML-experiment
"""

import math
import multiprocessing as mp
import os
from pathlib import Path
import random
from typing import List

from prettytable import PrettyTable
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from src.exp_management.experiment.Experiment import Experiment
from src.exp_management.data_provider import CvSet, DataProvider, HoldoutSet
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder


def run_experiment(exp: Experiment):
    """Run a supervised ML-experiment from config file

    Args:
        config_path (str): path to config file
    """
    args = exp.args
    
    writer = SummaryWriter(args.log_path)

    # prepare data
    data_provider = exp.data_provider

    args.number_of_classes = data_provider.number_classes
    exp.exp_log(number_of_classes=args.number_of_classes)
  
    if exp.args.reload_model_folder is None:
        model = train_model(data_provider=data_provider,
                            exp=exp,
                            writer=writer)
        reload_from = exp.args.log_path
    else:
        print(f"Skipping training step and restoring best model from {exp.args.reload_model_folder}")
        reload_from = Path(exp.args.logdir) / exp.args.reload_model_folder
        
    model = exp.model
    reload_epoch = reload_model(model=model,
                                model_path=reload_from,
                                gpu=exp.args.gpu)
    
    # evaluate best model on val set
    if exp.args.reload_model_folder is None or exp.args.include_val_inference is True:
        print("Evaluating final validation wsis")
        evaluate_model(exp=exp,
                       model=model,
                       wsis=data_provider.holdout_set.vali_wsi_dataset.wsis,
                       writer=writer,
                       epoch=reload_epoch,
                       save_to_folder=True,
                       log_metrics=True,
                       tag='vali_best')
    
    # test model on testset
    if exp.args.test_data is not None:
        ## evaluate WSI
        evaluate_model(exp=exp,
                       model=model,
                       wsis=data_provider.test_wsi_dataset.wsis,
                       writer=writer,
                       tag='test',
                       save_to_folder=True,
                       log_metrics=True)
        
    writer.close()

def train_model(data_provider: DataProvider,
                exp: Experiment,
                writer: SummaryWriter):

    if data_provider.holdout_set is not None:
        model = train_holdout_model(holdout_set=data_provider.holdout_set,
                                   exp=exp,
                                   writer=writer)
    else:
        model = train_kfold_model(cv_set=data_provider.cv_set,
                                  exp=exp)
    return model

def train_kfold_model(cv_set: CvSet,
                      exp: Experiment):
    m = mp.Manager()
    gpu_queue = m.Queue()
    # initialize queue
    for gpu_id in range(exp.args.ngpus):
        gpu_queue.put(gpu_id)
    
    def _exception_callback(e):
        pool.terminate()
        raise e
    
    # spawn processes per available gpus: use nestable Pool to avoid daemonic subprocesses
    pool = mp.get_context('spawn').Pool(processes=exp.args.ngpus) 
    for holdout_set in cv_set.holdout_sets:
        pool.apply_async(train_holdout_model_in_parallel,
                         (holdout_set,
                          gpu_queue,
                          exp),
                         error_callback=_exception_callback)
    pool.close()
    pool.join()
    

def train_holdout_model_in_parallel(holdout_set: HoldoutSet,
                                    gpu_queue: mp.Queue,
                                    exp: Experiment):
    
    # tread safe gpu id:
    exp.args.gpu = gpu_queue.get()
    
    # settings for folds
    exp.set_fold(fold=holdout_set.fold)
    
    writer = SummaryWriter(exp.args.log_path)
    
    exp.args.fold = holdout_set.fold
    print(f"Starting {holdout_set.fold}. fold run on gpu {exp.args.gpu}")
    train_holdout_model(holdout_set=holdout_set,
                        exp=exp,
                        writer=writer)
    print(f"Finished fold {holdout_set.fold}")
    # free gpu again
    gpu_queue.put(exp.args.gpu)

def train_holdout_model(holdout_set: HoldoutSet,
                        exp: Experiment,
                        writer: SummaryWriter):
    """
    Train a model on given data.

    Args:
        model (nn.Module): Pytorch module
        data_provider (DataProvider): DataProvider holding the train and val dataloader
        args (Dict): Train parameters
    """
    
    #torch.cuda.set_device(exp.args.gpu)
    model = exp.model
    model = model.cuda(exp.args.gpu)

    print(model) # print model after SyncBatchNorm  
    count_parameters(model)
    
    if exp.args.gpu is not None:
        print("Use GPU: {} for training".format(exp.args.gpu))

    # infer learning rate before changing batch size
    init_lr = exp.args.lr

    # define loss function (criterion) and optimizer
    criterion = exp.get_criterion()
    optimizer = exp.get_optimizer(model)
    
    if exp.args.adjust_lr is True:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp.args.lr_gamma, verbose=True)

    cudnn.benchmark = True

    val_performance =  float("inf")
    bad_epochs = 0 # counter for early stopping

    for epoch in range(exp.args.epochs):
        if exp.args.adjust_lr is True:
            scheduler.step()
            #adjust_learning_rate(optimizer, init_lr, epoch, exp.args)

        new_val_performance = exp.run_train_vali_epoch(holdout_set=holdout_set,
                                                       model=model,
                                                       criterion=criterion,
                                                       optimizer=optimizer,
                                                       label_handler=holdout_set.data_provider.label_handler,
                                                       epoch=epoch,
                                                       writer=writer,
                                                       args=exp.args)
                
        if (exp.args.warm_up_epochs is None \
                or epoch >= exp.args.warm_up_epochs):
            if new_val_performance < val_performance:
                
                bad_epochs = 0 # reset early stopping
                val_performance = new_val_performance
                exp.exp_log(best_epoch = epoch)
                exp.exp_log(best_val_performance = new_val_performance)
                torch.save(obj={'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'label_handler': holdout_set.data_provider.label_handler,
                                'args': exp.args
                                },
                        f=os.path.join(exp.args.checkpoint_path,
                                                'model_best.pth.tar'))
            else: 
                bad_epochs += 1
                if exp.args.early_stopping_epochs is not None and bad_epochs >= exp.args.early_stopping_epochs:
                    print(f"Early stopping at epoch {epoch}. {bad_epochs} times no val loss improvement.")
                    exp.exp_log(early_stopping_epoch=epoch)
                    break # stop epoch loop
            
        if exp.args.evaluate_every != 0 and (epoch+1) % exp.args.evaluate_every == 0:
            
            # sample wsis for evaluation once 
            if exp.args.n_eval_wsis is not None:
                
                n_train_wsis = len(holdout_set.train_wsi_dataset.wsis)
                train_wsi_idx = list(range(n_train_wsis))
                random.Random(exp.args.seed).shuffle(train_wsi_idx)
                train_wsis = [holdout_set.train_wsi_dataset.wsis[idx] for idx in 
                            train_wsi_idx[:min(exp.args.n_eval_wsis, n_train_wsis)]]
                
                n_val_wsis = len(holdout_set.vali_wsi_dataset.wsis)
                val_wsi_idx = list(range(n_val_wsis))
                random.Random(exp.args.seed).shuffle(val_wsi_idx)
                val_wsis = [holdout_set.vali_wsi_dataset.wsis[idx] for idx in 
                            val_wsi_idx[:min(exp.args.n_eval_wsis, n_val_wsis)]]
                
            else:
                train_wsis = holdout_set.train_wsi_dataset.wsis
                val_wsis = holdout_set.vali_wsi_dataset.wsis
                
            # evaluate validation set
            evaluate_model(exp=exp,
                           model=model,
                           wsis=train_wsis,
                           writer=writer,
                           epoch=epoch,
                           tag='train')

            evaluate_model(exp=exp,
                           model=model,
                           wsis=val_wsis,
                           writer=writer,
                           epoch=epoch,
                           tag='vali')

    return model 

def evaluate_model(exp: Experiment,
                   model: nn.modules,
                   wsis: List[WSIFromFolder],
                   writer: SummaryWriter,
                   tag: str,
                   epoch: int = None,
                   save_to_folder: bool = False,
                   log_metrics: bool = False):
    
    print(f"Evaluating {tag} WSIs.")
          
    # inference for all patches
    exp.wsi_inference(wsis=wsis,
                      model=model,
                      data_provider=exp.data_provider,
                      gpu=exp.args.gpu)
    
    # evaluate on WSI level
    evaluation = exp.evaluate_wsis(wsis=wsis,
                                   data_provider=exp.data_provider,
                                   log_path=Path(exp.args.log_path),
                                   tag=tag,
                                   writer=writer,
                                   save_to_folder=save_to_folder,
                                   epoch=epoch)  
    
    if log_metrics == True:
        writer.add_scalar(tag=f'evaluation/{tag}_wsi_dice', scalar_value=evaluation['wsi_mean_dice_scores'])
        writer.add_scalar(tag=f'evaluation/{tag}_wsi_precision', scalar_value=evaluation['wsi_mean_precision'])
        writer.add_scalar(tag=f'evaluation/{tag}_wsi_recall', scalar_value=evaluation['wsi_mean_recall'])
        writer.add_scalar(tag=f'evaluation/{tag}_wsi_jaccard', scalar_value=evaluation['wsi_mean_jaccard_scores'])
        
    exp.exp_log(key=f'evaluation_wsi_{tag}_set',
                value=evaluation)
    

def reload_model(model: nn.Module,
                 model_path: str,
                 gpu: int) -> int:
    """
    Reload the model in the best val loss state

    Args:
        model (nn.Module): module
        model_path (Path): path to model log folder (must contain 'checkpoints/model_best.pth.tar')

    Raises:
        Exception: [description]
        
    Returns: epoch of reloaded model
    """
    # try to detect checkpoint files in model_path
    checkpoint_path = Path(model_path) / 'checkpoints' / 'model_best.pth.tar'
    if not checkpoint_path.exists():
        raise Exception(f"Cannot find best checkpoint file {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    return checkpoint['epoch']


def adjust_learning_rate(optimizer,
                         init_lr,
                         epoch,
                         args):
    """
    Decay the learning rate based on schedule
    """
    #cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    cur_lr = init_lr * math.pow((1 - epoch/args.epochs),0.9) 
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params