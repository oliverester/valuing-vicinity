import logging
import numpy as np
from pathlib import Path
import torch

from src.exp_management.run_experiment import reload_model
from src.exp_management.experiment.Experiment import Experiment
from src.deephist.segmentation.attention_segmentation.attention_inference import memory_inference
from src.exp_management.tracking import Visualizer

def run_attention_analysis(exp: Experiment):
    """Anaylse attention 

    Args:
        exp (Experiment): _description_
    """

    model = exp.get_model()
    if exp.args.nfold is not None:
        for fold in range(exp.args.nfold):
            logging.getLogger('exp').info(f"Inference for fold {fold}")
            
            reload_from = Path(exp.args.reload_model_folder) / f"fold_{fold}"
            exp.log_path = reload_from #str(Path(exp.args.logdir) / f"fold_{fold}")
            
            reload_model(model=model,
                         model_path=reload_from,
                         gpu=exp.args.gpu)
            
            attention_analysis(exp=exp, 
                               model=model, 
                               wsis=exp.data_provider.cv_set.holdout_sets[fold].test_wsi_dataset.wsis)
            
def attention_analysis(exp,
                       model,
                       wsis):
          
    k = exp.args.k_neighbours  
    l = k*2+1

    # get original attention matrices
    
    # softmax?
    
    # distribute over distances
    dist_matrix = get_dist_matrix(k=exp.args.k_neighbours, dist='manhattan')
    
    # sum up distances

    viz = Visualizer(save_to_folder=True)
    
    att_distances = None
    
    model.eval()
    for wsi in wsis:
        with wsi.inference_mode(): # initializes memory
            logging.getLogger('exp').info("Building memory")
            wsi_loader = exp.data_provider.get_wsi_loader(wsi=wsi)
            
            # fill memory
            model.initialize_memory(**wsi.meta_data['memory'], gpu=exp.args.gpu)
            model.fill_memory(data_loader=wsi_loader, gpu=exp.args.gpu)
            
            outputs, labels, attentions, n_masks = memory_inference(data_loader=wsi_loader,
                                                                    model=model,
                                                                    gpu=exp.args.gpu)  
            # merge "batches"
            outputs, labels, attentions, n_masks = torch.cat(outputs), torch.cat(labels), torch.cat(attentions), torch.cat(n_masks)
                
            # attention dim: patches, heads, 1, token
            n_patches, heads, _, _ = attentions.shape
            
            attentions = torch.mean(attentions.view((n_patches, heads, (k*2+1),(k*2+1))), dim=1)
            
            # dot-product to apply distance masks
            dist_cube =  dist_matrix.unsqueeze(0).expand((n_patches,l,l,k)) 
            attention_cube = attentions.unsqueeze(-1).expand((n_patches,l,l,k))
            a = attention_cube * dist_cube
            # all weights must still sum up to 1
            assert torch.round(torch.sum(a)) == 1 * n_patches, 'all attention weigths of one WSI must still sum up to 1'
            # a shape: n_patches, l, l, k 
            # -> now weighted sum over l x l (only attention score for one distance value)
            # attention ratio per patch with distance
            
            new_att_distances = torch.sum(a,(1,2)) / torch.sum(dist_cube,(1,2))
            
            if att_distances is None:
                att_distances = new_att_distances
            else:
                att_distances = torch.cat((new_att_distances, att_distances))
                
    att_distance_mean = torch.mean(att_distances, dim=0)
    #att_distance_mean / torch.sum(att_distance_mean)
    att_distance_std = torch.std(att_distances, dim=0)
    
    print(f"Patch Attention per distance of {str(len(wsis))} WSIs:")
    print(att_distance_mean)
            
            
            
    
def get_dist_matrix(k=8, dist='manhattan'):
    assert dist in ['manhattan'], 'dist must be one of [manhattan]'
    
    if dist == 'manhattan':
        # create distance mask depeneding on neighbourhood size
        l = k*2+1
        dist_mask = torch.zeros(size=(l, l, k))
        # fill distance mask: for each k, value is 1 when x or/and y coord equals k
        for dist in range(1, k+1):
            # e.g. fill like this for dist=1 and k=2
            # 0 0 0 0 0
            # 0 1 1 1 0
            # 0 1 0 1 0
            # 0 1 1 1 0
            # 0 0 0 0 0
            for x in range(k-dist,l-(k-dist)):
                y1 = k-dist
                y2 = (k+dist)

                dist_mask[x,y1,dist-1] = 1
                dist_mask[x,y2,dist-1] = 1

            for y in range(k-dist,l-(k-dist)):
                x1 = (k-dist)
                x2 = (k+dist)
                
                dist_mask[x1,y,dist-1] = 1
                dist_mask[x2,y,dist-1] = 1
    
    return dist_mask