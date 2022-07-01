from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment

if __name__ == '__main__':
    run_experiment(exp=SegmentationExperiment(config_path='configs_paper/configs_rcc/attention/mha/deeplab_res50/d16_k8_deeplab_config.yml'))
 