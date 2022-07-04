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
    
    while True:
        # get configs
        configs_files = [str(p) for p in list(base.glob("*")) if p.is_file()]
        if len(configs_files) > 0:
            config_file = configs_files[0]
            logging.info(f"Running {config_file}")
            try:
                run_experiment(exp=SegmentationExperiment(config_path=config_file))
                # move to success
                os.rename(config_file, (base / 'success' / Path(config_file).name))
            except Exception as e:
                logging.info(traceback.format_exc())
                logging.info(e)
                logging.info(f"error at {config_file}")
                # move to error
                os.rename(config_file, (base / 'error' / Path(config_file).name))
        else:
            sleep(2)
            logging.info("waiting for configs")
   
