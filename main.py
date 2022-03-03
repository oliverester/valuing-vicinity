import torch
from src.deephist.attention_segmentation.AttentionSegmentationExperiment import AttentionSegmentationExperiment
from src.deephist.semantic_segmentation.SemanticSegmentationExperiment import SemanticSegmentationExperiment
from src.deephist.segmentation.SegmentationExperiment import SegmentationExperiment
from src.deephist.evaluate.tsne.TsneExperiment import TsneExperiment
from src.exp_management.Experiment import Experiment
from src.deephist.evaluate.tsne.run_tsne import run_tsne
from src.deephist.embedding.nic.run_nic import run_nic
from src.deephist.embedding.create_embedding import create_embedding

from src.deephist.run_experiment import run_experiment
from src.deephist.embedding.clam.ClamExperiment import ClamExperiment
from src.deephist.supervised.SupervisedExperiment import SupervisedExperiment

if __name__ == '__main__':
    #run_tsne(config_path='configs/tsne_config.yml')

    #create_embedding(config_path='configs/embedding_config.yml')

    #run_experiment(exp=ClamExperiment(config_path='configs/clam_config.yml'))
    
    #run_experiment(exp=SupervisedExperiment(config_path='configs/supervised_config_patch_test.yml'))

    #run_experiment(exp=SegmentationExperiment(config_path='configs/segmentation_config.yml'))
    
    #run_experiment(exp=SemanticSegmentationExperiment(config_path='configs/semantic_segmentation_config.yml'))
    
    run_experiment(exp=AttentionSegmentationExperiment(config_path='configs/loop/attention_segmentation_config_k4.yml'))
    
    #run_tsne(exp=TsneExperiment(config_path='configs/tsne_config.yml'))

    #run_nic(config_path='configs/nic_config.yml') 