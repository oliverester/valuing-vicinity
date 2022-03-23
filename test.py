from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment

if __name__ == '__main__':
    run_experiment(exp=SegmentationExperiment(config_path='configs_rcc/attention_segmentation_config_test.yml'))