
from pathlib import Path

from src.deephist.segmentation.attention_segmentation.attention_inference import fill_memory, memory_inference
from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment
from src.exp_management.run_experiment import reload_model
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
            with selected_wsi.inference_mode(): # initializes memory
                print("Building memory")
                wsi_loader = exp.data_provider.get_wsi_loader(wsi=selected_wsi)
                embedding_memory = fill_memory(data_loader=wsi_loader,
                                               memory=selected_wsi.memory,
                                               model=model,
                                               gpu=exp.args.gpu)
                # select patches 
                for x, y in patch_coordinates:
                    patch = selected_wsi.get_patch_from_position(x,y)
                    
                    context_patches, _  = patch.get_neighbours(k=include_k)
                    context_patches=[p for p in list(context_patches.flatten()) if p is not None]
                    
                    #patches_loader = exp.data_provider.get_wsi_loader(patches=[p for p in list(context_patches.flatten()) if p is not None])
                    with selected_wsi.restrict_patches(context_patches):
                        outputs, labels, attentions, n_masks = memory_inference(data_loader=wsi_loader,
                                                                                memory=embedding_memory,
                                                                                model=model,
                                                                                gpu=exp.args.gpu)  
                    # append results to patch object
                    for i, patch in enumerate(context_patches):
                        patch.prediction = exp.mask_to_img(mask=outputs[i],
                                                        label_handler=exp.data_provider.label_handler,
                                                        org_size=True) 
                        patch.mask = exp.mask_to_img(mask=labels[i],
                                                    label_handler=exp.data_provider.label_handler,
                                                    org_size=True)
                        patch.attention = attentions[i]
                        patch.neighbour_mask = n_masks[i]
                
                    # gt
                    viz.plot_wsi_section(section=context_patches,
                                        mode='gt',
                                        log_path=exp.args.logdir)
                    # pred
                    viz.plot_wsi_section(section=context_patches,
                                        mode='pred',
                                        log_path=exp.args.logdir)
                    # att + gt
                    viz.plot_wsi_section(section=context_patches,
                                        mode='gt',
                                        attention=True,
                                        log_path=exp.args.logdir)

  
if __name__ == "__main__":

    # patch_coordinates = {'RCC-TA-033.001~C': [(14,19), (15,20), (20,20)],
    #                      'RCC-TA-011.001~J': [(20, 15), (20, 17)],
    #                      'RCC-TA-004.001~C': [(21, 35)]
    #                      }
    
    patch_coordinates = {'RCC-TA-163.001~B': [(7,12), (8,12), (8,11), (7,11)],
                         }
    exp = SegmentationExperiment(config_path='/src/deephist/evaluate/configs/attention_segmentation_config_inference.yml')

    model = exp.model
    reload_from = Path(exp.args.logdir) / exp.args.reload_model_folder
    reload_model(model=model,
                 model_path=reload_from,
                 gpu=exp.args.gpu)
    
    evaluate_details(patch_coordinates=patch_coordinates,
                     include_k = 8,
                     exp=exp, 
                     model=model, 
                     wsis=exp.data_provider.test_wsi_dataset.wsis)