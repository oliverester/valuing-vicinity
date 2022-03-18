from src.exp_management.experiment.SegmentationExperiment import AttentionSegmentationExperiment
from src.exp_management.run_experiment import run_experiment

if __name__ == '__main__':
    run_experiment(exp=AttentionSegmentationExperiment(config_path='configs_cy16/config.yml'))
 