from pathlib import Path
import traceback

from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment

if __name__ == '__main__':
   
    base = Path("configs/loop")
    configs_files = [str(p) for p in list(base.glob("*"))]
    for config_file in configs_files:
        print(f"Starting with {config_file}")
        try:
            run_experiment(exp=SegmentationExperiment(config_path=config_file))
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print(f"error at {config_file}")
