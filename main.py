from src.deephist.attention_segmentation.AttentionSegmentationExperiment import AttentionSegmentationExperiment
from src.deephist.run_experiment import run_experiment

if __name__ == '__main__':
  
    run_experiment(exp=AttentionSegmentationExperiment(config_path='configs_cy16/baseline_segmentation_config.yml'))
 