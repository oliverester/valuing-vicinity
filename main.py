from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment

if __name__ == '__main__':
    run_experiment(exp=SegmentationExperiment(config_path='configs_paper/configs_rcc/semantic/deeplab_resnet50/d16_deeplab_config.yml',
                                              folds=[0],
                                              gpu=2),
                  )                           
 