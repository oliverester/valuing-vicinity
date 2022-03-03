from src.deephist.attention_segmentation.AttentionSegmentationExperiment import AttentionSegmentationExperiment
from src.deephist.run_experiment import run_experiment
from pathlib import Path
import traceback

if __name__ == '__main__':
   
    base = Path("configs/loop")
    configs_files = [str(p) for p in list(base.glob("*"))]
    for config_file in configs_files:
        print(f"Starting with {config_file}")
        try:
            run_experiment(exp=AttentionSegmentationExperiment(config_path=config_file))
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print(f"error at {config_file}")
