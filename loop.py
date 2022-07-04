from datetime import datetime
import logging
import multiprocessing as mp
from pathlib import Path
import traceback
from typing import List

from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment
from src.lib.NestablePool.nestable_pool import NestablePool


def run_job_queue(config_folder: str,
                  gpus: List[int],
                  kwargs):
    
    base = Path(config_folder)
    
    n = datetime.now().strftime("%Y-%m-%d_%H-%H-%S")
    
    success_logger = logging.getLogger('success_loop')
    fh = logging.FileHandler(f'loop_{n}_success.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to logger
    success_logger.addHandler(fh)
    
    error_logger = logging.getLogger('error_loop')
    fh = logging.FileHandler(f'loop_{n}_error.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to logger
    error_logger.addHandler(fh)

    configs_files = [str(p) for p in list(base.glob("*")) if p.is_file()]
    
    gpu_queue = mp.Manager().Queue()
    # initialize gpu queue
    for gpu_id in gpus:
        gpu_queue.put(gpu_id)
    
    # needed because this pool creates nondaemonic processes (needed for pytorch)
    pool = NestablePool(processes=len(gpus))

    for config_file in configs_files:
        pool.apply_async(run_job, (config_file, gpu_queue, success_logger, error_logger, kwargs))
   
    pool.close()
    pool.join()
    

def run_job(config_file, gpu_queue, success_logger, error_logger, kwargs):
    
    gpu = gpu_queue.get()
    success_logger.info(f"Running {config_file}")
    
    try:
        run_experiment(exp=SegmentationExperiment(config_path=config_file,
                                                  **kwargs
                                                  ))
        success_logger.info(f"Successful: {config_file}")
        
    except Exception as e:
        error_logger.error(traceback.format_exc())
        error_logger.error(e)
        error_logger.error(f"Error: {config_file}")
    
    # done: free gpu again
    gpu_queue.put(gpu)

if __name__ == '__main__':
    # needed because only works in spawn mode (fork is default)
    mp.set_start_method('spawn', force=True)
    run_job_queue(gpus=[4,5],
                 config_folder="configs_paper_test/configs_rcc/semantic/deeplab_resnet50",
                 kwargs=dict(
                    sample_size= 5,
                    epochs=2,
                    warm_up_epochs=0,
                    nfold=2,
                    logdir="logdir_paper/test_runs")
                 )