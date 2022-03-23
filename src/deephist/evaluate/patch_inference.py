 
 
from pathlib import Path
from typing import Dict

import src.deephist.evaluate.evaluate_attention as att
import deephist.evaluate.evaluate_multiscale as ms
import src.deephist.evaluate.evaluate_baseline as bs
from src.exp_management.experiment.Experiment import Experiment
from src.exp_management.run_experiment import reload_model

def run_patch_inference(exp: Experiment,
                        patch_coordinates: Dict,
                        k: int):
    """Render patch inference for given coordinates + neighbourhood size k

    Args:
        exp (Experiment): _description_
        patch_coordinates (Dict): dictionary of WSI-name as key and list of patch-coord-tuple as value
        k (int): Neighbourhood size
    """

    model = exp.model
    reload_from = Path(exp.args.logdir) / exp.args.reload_model_folder
    
    reload_model(model=model,
                 model_path=reload_from,
                 gpu=exp.args.gpu)
    
    if exp.args.attention_on:
        eval = att.evaluate_details
    elif exp.args.multiscale_on:
        eval = ms.evaluate_details
    else:
        eval = bs.evaluate_details
        
    eval(patch_coordinates=patch_coordinates,
         include_k = k,
         exp=exp, 
         model=model, 
         wsis=exp.data_provider.test_wsi_dataset.wsis)