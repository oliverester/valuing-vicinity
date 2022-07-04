import logging
import os
from pathlib import Path
from time import sleep
import traceback

from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment

if __name__ == '__main__':
   
    base = Path("configs_loop")
    (base / 'error').mkdir(exist_ok=True)
    (base / 'success').mkdir(exist_ok=True)
    
    logger = logging.getLogger('loop')
    fh = logging.FileHandler('loop.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    
    while True:
        # get configs
        configs_files = [str(p) for p in list(base.glob("*")) if p.is_file()]
        if len(configs_files) > 0:
            config_file = configs_files[0]
            logger.info(f"Running {config_file}")
            try:
                run_experiment(exp=SegmentationExperiment(config_path=config_file))
                # move to success
                os.rename(config_file, (base / 'success' / Path(config_file).name))
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(e)
                logger.error(f"Error: {config_file}")
                # move to error
                os.rename(config_file, (base / 'error' / Path(config_file).name))
                
            logger.info(f"Successful: {config_file}")
        else:
            sleep(2)
            logger.info("waiting for configs")
   
