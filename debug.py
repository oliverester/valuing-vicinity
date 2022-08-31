from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment

if __name__ == '__main__':
    run_experiment(exp=SegmentationExperiment(config_path='configs_test/d16_c64_deeplab_config.yml',
                                              gpu=3,
                                              sample_size=5,
                                              warm_up_epochs=0,
                                              nfold=2)
                   )