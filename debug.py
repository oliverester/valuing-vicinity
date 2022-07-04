from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import run_experiment

if __name__ == '__main__':
    run_experiment(exp=SegmentationExperiment(config_path='configs_rcc/cv/deeplab_maf_config_test.yml',
                                              sample_size=5,
                                              epochs=2,
                                              warm_up_epochs=0,
                                              nfold=2))