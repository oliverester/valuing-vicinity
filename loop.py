from datetime import datetime
import logging
import logging.config
import logging.handlers
import threading
import time
#import multiprocessing as mp
import torch.multiprocessing as mp
import yaml
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

from pathlib import Path
import traceback
from typing import List

import torch

from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment


def run_job_queue(config_folder: str,
                  gpu_file: str,
                  kwargs):
    q = mp.Queue()
    
    d = {
        'version': 1,
        'formatters': {
            'detailed': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s'
            },
             'printer': {
                'class': 'logging.Formatter',
                'format': '%(processName)-10s %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'printer',
            },
            'loop': {
                'class': 'logging.FileHandler',
                'filename': 'loop.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
            'success': {
                'class': 'logging.FileHandler',
                'filename': 'success.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
            'error': {
                'class': 'logging.FileHandler',
                'filename': 'error.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
        },
        'loggers': {
            'success_logger': {
                'handlers': ['success', 'console']
            },
             'error_logger': {
                'handlers': ['error', 'console']
            },
             'loop_logger': {
                'handlers': ['loop', 'console']
            }
        }
    }
    # overwrite log files
    Path("logs").mkdir(exist_ok=True)
    n = datetime.now().strftime("%Y-%m-%d_%H-%H-%S")
    d['handlers']['success']['filename'] = f'logs/log_{n}_success.log'
    d['handlers']['error']['filename'] = f'logs/log_{n}_error.log'
    d['handlers']['loop']['filename'] = f'logs/log_{n}_loop.log'
    
    logging.config.dictConfig(d)
    logger = logging.getLogger('loop_logger')
    logger.setLevel(logging.DEBUG)

    config_queue = mp.Manager().Queue()
    gpu_queue  = mp.Manager().Queue()
    used_gpu_queue  = mp.Manager().Queue()

    gt = threading.Thread(target=gpu_resource_thread, args=(gpu_queue, used_gpu_queue, gpu_file, logger))
    gt.start()
    ct = threading.Thread(target=config_sync_thread, args=(config_queue, config_folder, logger))
    ct.start()
    lp = threading.Thread(target=logger_thread, args=(q,))
    lp.start()
    
    torch.multiprocessing.spawn(run_job,
                                args=(config_queue, gpu_queue, used_gpu_queue, q, kwargs), 
                                nprocs=6, # max parallel jobs
                                join=True, 
                                daemon=False,
                                start_method='spawn')
    print("Done")
    
    # finish log thread
    q.put(None)
    used_gpu_queue.put(None)
    lp.join()
    gt.join()
    ct.join()
    

def run_job(proc_idx: int,
            config_queue,
            gpu_queue,
            used_gpu_queue,
            logger_config,
            kwargs):
        
    qh = logging.handlers.QueueHandler(logger_config)
 
    success_logger = logging.getLogger('success_logger')
    success_logger.addHandler(qh)
    success_logger.setLevel(logging.DEBUG)
    error_logger = logging.getLogger('error_logger')
    error_logger.addHandler(qh)
    error_logger.setLevel(logging.DEBUG)
    loop_logger = logging.getLogger('loop_logger')
    loop_logger.addHandler(qh)
    loop_logger.setLevel(logging.DEBUG)

    # run until all configs are processed
    while config_queue.qsize() > 0:
        
        config_file = config_queue.get()
        # get gpu resource from queue
        gpu = gpu_queue.get()
        
        loop_logger.info(f"Running {config_file}")
        loop_logger.info(f"{config_queue.qsize()} tasks left")
        
        try:
            run_experiment(exp=SegmentationExperiment(config_path=config_file,
                                                      gpu=gpu,
                                                      **kwargs
                                                      )
                           )
            success_logger.info(f"Successful: {config_file}")
            
        except Exception as e:
            error_logger.error(traceback.format_exc())
            error_logger.error(e)
            error_logger.error(f"Error: {config_file}")
            
        # free gpu resource again    
        used_gpu_queue.put(gpu)

        
def logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)
        
def config_sync_thread(config_queue,
                       config_folder,
                       logger):
    # check for new configs and put into config_queue
    
    # initial load
    base = Path(config_folder)
    base_configs_files = [str(p) for p in list(base.rglob("*")) if p.is_file()]
    
    logger.info(f"Starting job queue for configs in folder {config_folder}")
    logger.info(f"Task list: {base_configs_files}")
      
    # initialize config queue
    for config in base_configs_files:
        config_queue.put(config)
        
    # run until all configs are done
    while config_queue.qsize() > 0:
        time.sleep(5)
        
        # check for new configs
        base = Path(config_folder)
        curr_configs_files = [str(p) for p in list(base.rglob("*")) if p.is_file()]
        new_config_files = [file for file in curr_configs_files if file not in base_configs_files]
        if len(new_config_files) > 0:
            logger.info(f"Detecting {len(new_config_files)} new configs")
            logger.info(f"New: {new_config_files}")
            
            for config in base_configs_files:
                config_queue.put(config)
                
            # update base_config_files
            base_configs_files = curr_configs_files
    
def gpu_resource_thread(gpu_queue,
                        used_gpu_queue, 
                        gpu_file,
                        logger):
    # manage a queue that holds usable gpus:
    base_gpus = get_gpus_from_file(gpu_file, logger)
    logger.info(f"Using gpus {base_gpus} for job queue.")
    for gpu in base_gpus:
        gpu_queue.put(gpu)
        
    while True:
        time.sleep(5)
        curr_gpus = get_gpus_from_file(gpu_file, logger)
        
        # check for new gpus
        add_gpus = [gpu for gpu in curr_gpus if gpu not in base_gpus]
        if len(add_gpus) > 0:
            for add_gpu in add_gpus:
                gpu_queue.put(add_gpu)
                logger.info(f"Providing new gpu {add_gpu} for job queue.")
     
        while used_gpu_queue.qsize() > 0:
            used_gpu = used_gpu_queue.get()
            # finish thread
            if used_gpu is None: 
                break
            if used_gpu in curr_gpus:
                gpu_queue.put(used_gpu)
            else:
                logger.info(f"Removing gpu {used_gpu} after job was finished.")
            
        base_gpus = curr_gpus

def get_gpus_from_file(path, logger, initial=False):
    with open(path, "r") as stream:
        try:
            gpus = (yaml.safe_load(stream))['gpus']
            return gpus
        except yaml.YAMLError as exc:
            # info to
            if initial:
                raise exc
            else:
                logger.error(exc)
                logger.error("Cannot load gpus from file")
                
if __name__ == '__main__':
    run_job_queue(gpu_file="gpus.yml",
                  config_folder="configs_paper/configs_rcc/semantic",
                #   kwargs=dict(
                #     sample_size= 5,
                #     epochs=1,
                #     warm_up_epochs=0,
                #     nfold=5,
                #     folds=[0],
                #     logdir="logdir_paper/test_runs")
                 )