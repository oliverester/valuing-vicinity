from collections import defaultdict
import logging
import numpy as np
from pathlib import Path
import torch

from src.exp_management.run_experiment import reload_model
from src.exp_management.experiment.Experiment import Experiment
from src.exp_management.tracking import Visualizer
from src.deephist.segmentation.attention_segmentation.attention_inference import memory_inference
from src.deephist.segmentation.attention_segmentation.AttentionPatchesDistAnalysisDataset import AttentionPatchesDistAnalysisDataset

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
                               wsis=exp.data_provider.test_wsi_dataset.wsis)
            
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
    
    att_sum_per_dist = defaultdict(int)
    neighbour_count_per_dist = defaultdict(int)
    central_patch_counter = defaultdict(int)
    
    model.eval()
    for wsi in wsis:
        with wsi.inference_mode(): # initializes memory
            logging.getLogger('exp').info("Building memory")
            # loader wsi with special dataset including neighbourhood patch distribution 
            wsi_loader = exp.data_provider.get_wsi_loader(wsi=wsi, 
                                                          dataset_type=AttentionPatchesDistAnalysisDataset)
            
            # fill memory
            model.initialize_memory(**wsi.meta_data['memory'], gpu=exp.args.gpu)
            model.fill_memory(data_loader=wsi_loader, gpu=exp.args.gpu)
            
            # TODO: get neighbourhood class distribution / soft- one-hot?
            outputs, labels, attentions, n_masks, n_dists = memory_inference(data_loader=wsi_loader,
                                                                             model=model,
                                                                             gpu=exp.args.gpu,
                                                                             return_cls_dist=True)
    
            # merge "batches"
            outputs, labels, attention_per_head, n_masks, n_dists = \
                torch.cat(outputs), torch.cat(labels), torch.cat(attentions), torch.cat(n_masks), torch.cat(n_dists)
            # select center patch class distributions 
            center_dists = n_dists[:,k,k,:]
               
             # attention dim: patches, heads, 1, token
            n_patches, n_heads, _, _ = attention_per_head.shape
            
            # mean over heads per patch
            attentions = torch.mean(attention_per_head.view((n_patches, n_heads, (k*2+1),(k*2+1))), dim=1)
                
            # determine attention per center patch cls
            for cls in range(n_dists.shape[-1]):
               
                # % of cls in center patch 
                center_patch_selector = center_dists[:,cls].unsqueeze(-1).unsqueeze(-1).expand(n_patches,l,l)
                
                # weight / select attentions by center patch cls
                weighted_attentions = attentions * center_patch_selector
                # weight / select neighbourmasks for center patch cls
                weighted_masks = n_masks * center_patch_selector
                
                # dot-product to apply distance masks
                dist_cube = dist_matrix.unsqueeze(0).expand((n_patches,l,l,k)) 
                attention_cube = weighted_attentions.unsqueeze(-1).expand((n_patches,l,l,k))
                neighbour_masks_cube = weighted_masks.unsqueeze(-1).expand((n_patches,l,l,k))
                
                attention_per_dist = attention_cube * dist_cube # select attention per distance dimension
                # all weights must still sum up to #center patches
                assert round(torch.sum(attention_per_dist).item()) == round(torch.sum(center_dists[:,cls]).item()), \
                    'all attention weigths of one WSI must still sum up to 1'
                
                # count neighbour patches per distance / filter the distance cube by existing neighbours
                neighbour_per_dist =  dist_cube * neighbour_masks_cube
                
                # sum over all patches for each dist
                count_neighbours_per_dist = torch.sum(neighbour_per_dist, dim=(0,1,2))
                sum_att_per_dist = torch.sum(attention_per_dist, dim=(0,1,2))
                
                # cumulate for all WSIs
                att_sum_per_dist[cls] += sum_att_per_dist
                neighbour_count_per_dist[cls] += count_neighbours_per_dist
                central_patch_counter[cls] += torch.sum(center_dists[:,cls]) 
       
    for cls in att_sum_per_dist.keys():
        # normalized with mean of patches per dist
        att_distance_mean =  att_sum_per_dist[cls] / neighbour_count_per_dist[cls] * (neighbour_count_per_dist[cls] / central_patch_counter[cls])

        print(f"{exp.data_provider.label_handler.decode(cls)} distance attention of {str(len(wsis))} WSIs:")
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