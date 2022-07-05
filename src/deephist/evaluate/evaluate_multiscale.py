
import logging
from pathlib import Path

from src.deephist.segmentation.multiscale_segmentation.multiscale_inference import do_inference
from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import reload_model
from src.exp_management.tracking import Visualizer

logger = logging.getLogger('exp')

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
                logger.error(f"Warning: Cannot find WSI {wsi_name}. Contueing")
                continue
            # build memory on that WSI
            with selected_wsi.inference_mode(): # sets wsi to idx 0 for memory
                for x, y in patch_coordinates:
                    try:
                        patch = selected_wsi.get_patch_from_position(x,y)
                        context_patches, _  = patch.get_neighbours(k=include_k)
                        
                        patches = [p for p in list(context_patches.flatten()) if p is not None]
                        
                        patches_loader = exp.data_provider.get_wsi_loader(patches=patches)

                        outputs, labels = do_inference(data_loader=patches_loader,
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
                        logger.error(f"Could not visualize patch {x}, {y} of WSI {wsi_name}")
                        raise e

    
if __name__ == "__main__":

    # patch_coordinates = {'RCC-TA-033.001~C': [(14,19), (15,20), (20,20)],
    #                      'RCC-TA-011.001~J': [(20, 15), (20, 17)],
    #                      'RCC-TA-004.001~C': [(21, 35)]
    #                      }
    
    #patch_coordinates = {'RCC-TA-163.001~B': [(7,12), (8,12), (8,11), (7,11)],
    #                     }
    patch_coordinates = {'RCC-TA-163.001~B': [(14,24), (16,24), (16,22), (14,22)],
                         }
    exp_multiscale = SegmentationExperiment(config_path='/src/deephist/evaluate/configs/multiscale_segmentation_config_inference.yml')

    model_multiscale = exp_multiscale.model
    reload_multiscale_from = Path(exp_multiscale.args.logdir) / exp_multiscale.args.reload_model_folder
    reload_model(model=model_multiscale,
                 model_path=reload_multiscale_from,
                 gpu=exp_multiscale.args.gpu)
    
    evaluate_details(patch_coordinates=patch_coordinates,
                     include_k = 16,
                     exp=exp_multiscale, 
                     model=model_multiscale, 
                     wsis=exp_multiscale.data_provider.test_wsi_dataset.wsis)

