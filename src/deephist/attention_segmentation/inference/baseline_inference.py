
from pathlib import Path

import torch
from tqdm import tqdm

from src.deephist.attention_segmentation.AttentionSegmentationExperiment import AttentionSegmentationExperiment
from src.deephist.attention_segmentation.models.attention_segmentation_model import get_k_neighbour_embeddings, update_embeddings
from src.deephist.run_experiment import reload_model
from src.exp_management.tracking import Visualizer

def evaluate_details(patch_coordinates,
                     include_k,
                     exp,
                     model,
                     wsis):
    
        viz = Visualizer(save_to_folder=True)

        for wsi_name, patch_coordinates in patch_coordinates.items():
            # select WSI from wsis
            try:
                selected_wsi = [wsi for wsi in wsis if wsi.name == wsi_name][0]
            except Exception as e:
                print(f"Warning: Cannot find WSI {wsi_name}. Contueing")
                continue
            # build memory on that WSI
            with selected_wsi.inference_mode(): # sets wsi to idx 0 for memory
                for x, y in patch_coordinates:
                    try:
                        patch = selected_wsi.get_patch_from_position(x,y)
                        context_patches, _  = patch.get_neighbours(k=include_k)
                        
                        patches = [p for p in list(context_patches.flatten()) if p is not None]
                        
                        patches_loader = exp.data_provider.get_wsi_loader(patches=patches)

                        outputs, labels = baseline_inference(data_loader=patches_loader,
                                                             model=model,
                                                             gpu=exp.args.gpu)  
                        # append results to patch object
                        for i, patch in enumerate(patches):
                            patch.prediction = exp.mask_to_img(mask=outputs[i],
                                                               label_handler=exp.data_provider.label_handler,
                                                               org_size=True) 
                            patch.mask = exp.mask_to_img(mask=labels[i],
                                                         label_handler=exp.data_provider.label_handler,
                                                         org_size=True) 
                            
                        viz.plot_wsi_section(section=context_patches,
                                        mode='org',
                                        log_path=exp.args.logdir)
                        # gt
                        viz.plot_wsi_section(section=context_patches,
                                            mode='gt',
                                            log_path=exp.args.logdir)
                        # pred
                        viz.plot_wsi_section(section=context_patches,
                                            mode='pred',
                                            log_path=exp.args.logdir)
                        
                    except Exception as e:
                        print(f"Could not visualize patch {x}, {y} of WSI {wsi_name}")
                        raise e
   
def baseline_inference(data_loader: torch.utils.data.DataLoader,
                       model: torch.nn.Module,
                       gpu: int = None,
                       args = None):
    """Apply model to data to receive model output

    Args:
        data_loader (torch.utils.data.DataLoader): A pytorch DataLoader
            that holds the inference data
        model (torch.nn.Module): A pytorch model
        args (Dict): args

    Returns:
        [type]: [description]
    """

    outputs = []
    labels = []
    
    with torch.no_grad():
        # switch to evaluate mode
        model.eval()
        m = torch.nn.Softmax(dim=1).cuda(gpu)

        # second loop: attend freezed neighbourhood memory     
        for images, targets in data_loader:
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            
            logits = model(images)  
            probs = m(logits)
    
            outputs.extend(torch.argmax(probs,dim=1).cpu())
            labels.extend(targets.cpu())

    return outputs, labels
    
if __name__ == "__main__":

    # patch_coordinates = {'RCC-TA-033.001~C': [(14,19), (15,20), (20,20)],
    #                      'RCC-TA-011.001~J': [(20, 15), (20, 17)],
    #                      'RCC-TA-004.001~C': [(21, 35)]
    #                      }
    
    patch_coordinates = {'RCC-TA-163.001~B': [(7,12), (8,12), (8,11), (7,11)],
                         }
    exp_baseline=AttentionSegmentationExperiment(config_path='/homes/oester/repositories/prae/src/deephist/attention_segmentation/analysis/baseline_segmentation_config_inference.yml')

    model_baseline = exp_baseline.model
    reload_baseline_from = Path(exp_baseline.args.logdir) / exp_baseline.args.reload_model_folder
    reload_model(model=model_baseline,
                 model_path=reload_baseline_from,
                 gpu=exp_baseline.args.gpu)
    
    evaluate_details(patch_coordinates=patch_coordinates,
                     include_k = 8,
                     exp=exp_baseline, 
                     model=model_baseline, 
                     wsis=exp_baseline.data_provider.test_wsi_dataset.wsis)

