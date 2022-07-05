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
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'mplog.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
            'success': {
                'class': 'logging.FileHandler',
                'filename': 'mplog-foo.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
            'error': {
                'class': 'logging.FileHandler',
                'filename': 'mplog-errors.log',
                'mode': 'w',
                'formatter': 'detailed',
            },
        },
        'loggers': {
            'success_logger': {
                'handlers': ['success']
            },
             'error_logger': {
                'handlers': ['error']
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file']
        },
    }
    # overwrite log files
    #n = datetime.now().strftime("%Y-%m-%d_%H-%H-%S")

    
    logging.config.dictConfig(d)
    lp = threading.Thread(target=logger_thread, args=(q,))
    lp.start()
    
    base = Path(config_folder)
    configs_files = [str(p) for p in list(base.glob("*")) if p.is_file()]
    
    config_queue = mp.Manager().Queue()
    # initialize config queue
    for config in configs_files:
        config_queue.put(config)
      
    logging.config.dictConfig(d)
    lp = threading.Thread(target=logger_thread, args=(q,))
    lp.start()
      
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
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(qh)
    
    success_logger = logging.getLogger('success_logger')
    error_logger = logging.getLogger('error_logger')

    # get subprocess gpu
    gpu = gpus[proc_idx]
    
    # run until all configs are processed
    while config_queue.qsize() > 0:
        config_file = config_queue.get()
        
        success_logger.info(f"Running {config_file}")
        try:
            run_experiment(exp=SegmentationExperiment(config_path=config_file,
                                                      gpu=gpu,
                                                      **kwargs
                                                      ))
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
    run_job_queue(gpus=[4,5],
                 config_folder="configs_paper_test/configs_rcc/semantic/deeplab_resnet50",
                 kwargs=dict(
                    sample_size= 5,
                    epochs=2,
                    warm_up_epochs=0,
                    nfold=2,
                    logdir="logdir_paper/test_runs")
                 )