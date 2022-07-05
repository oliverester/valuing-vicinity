from datetime import datetime
import logging
import logging.config
import logging.handlers
import threading
#import multiprocessing as mp
import torch.multiprocessing as mp
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
                  gpus: List[int],
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
    n = datetime.now().strftime("%Y-%m-%d_%H-%H-%S")
    d['handlers']['success']['filename'] = f'log_{n}_success.log'
    d['handlers']['error']['filename'] = f'log_{n}_error.log'
    d['handlers']['loop']['filename'] = f'log_{n}_loop.log'
    
    base = Path(config_folder)
    configs_files = [str(p) for p in list(base.rglob("*")) if p.is_file()]
    
    config_queue = mp.Manager().Queue()
    # initialize config queue
    for config in configs_files:
        config_queue.put(config)
      
    logging.config.dictConfig(d)
    lp = threading.Thread(target=logger_thread, args=(q,))
    lp.start()
    
    logger = logging.getLogger('loop_logger')
    logging.getLogger('exp').info(f"Task list: {configs_files}")
      
    torch.multiprocessing.spawn(run_job,
                                args=(config_queue, gpus, q, kwargs), 
                                nprocs=len(gpus),
                                join=True, 
                                daemon=False,
                                start_method='spawn')
    
    print("Done")
    
    # finish log thread
    q.put(None)
    lp.join()

def run_job(proc_idx: int,
            config_queue,
            gpus,
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

    # get subprocess gpu
    gpu = gpus[proc_idx]
    
    # run until all configs are processed
    while config_queue.qsize() > 0:
        config_file = config_queue.get()
        
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
        
def logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)
        
if __name__ == '__main__':
    # needed because only works in spawn mode (fork is default)
    #torch.multiprocessing.set_start_method('spawn', force=True)
    run_job_queue(gpus=[2,4],
                  config_folder="configs_paper",
                  kwargs=dict(
                    sample_size= 5,
                    epochs=2,
                    warm_up_epochs=0,
                    nfold=5,
                    folds=[0,1],
                    logdir="logdir_paper/test_runs")
                 )