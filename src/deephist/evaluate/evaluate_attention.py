

import torch
from src.deephist.segmentation.attention_segmentation.attention_inference import memory_inference
from src.exp_management.tracking import Visualizer

def evaluate_details(patch_coordinates,
                     include_k,
                     exp,
                     model,
                     wsis):
    
        viz = Visualizer(save_to_folder=True)

        model.eval()
        for wsi_name, patch_coordinates in patch_coordinates.items():
            # select WSI from wsis
            try:
                selected_wsi = [wsi for wsi in wsis if wsi.name == wsi_name][0]
            except Exception as e:
                print(f"Warning: Cannot find WSI {wsi_name}. Continuing")
                continue
            # build memory on that WSI
            with selected_wsi.inference_mode(): # initializes memory
                print("Building memory")
                wsi_loader = exp.data_provider.get_wsi_loader(wsi=selected_wsi)
                
                # fill memory
                model.initialize_memory(**selected_wsi.meta_data['memory'], gpu=exp.args.gpu)
                model.fill_memory(data_loader=wsi_loader, gpu=exp.args.gpu)
                    
                # select patches 
                for x, y in patch_coordinates:
                    patch = selected_wsi.get_patch_from_position(x,y)
                    if patch is None:
                        print(f"Patch {x}, {y} does not exist.")
                        continue
                    
                    context_patches, _  = patch.get_neighbours(k=include_k)
                    context_patches_list = [p for p in list(context_patches.flatten()) if p is not None]
                    
                    #patches_loader = exp.data_provider.get_wsi_loader(patches=[p for p in list(context_patches.flatten()) if p is not None])
                    with selected_wsi.restrict_patches(context_patches_list):
                        outputs, labels, attentions, n_masks = memory_inference(data_loader=wsi_loader,
                                                                                model=model,
                                                                                gpu=exp.args.gpu)  
                    # merge "batches"
                    outputs, labels, attentions, n_masks = torch.cat(outputs), torch.cat(labels), torch.cat(attentions), torch.cat(n_masks)
                    
                    # append results to patch object
                    for i, patch in enumerate(context_patches_list):
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
                                        log_path=exp.args.log_path)
                    # pred
                    viz.plot_wsi_section(section=context_patches,
                                        mode='pred',
                                        log_path=exp.args.log_path)
                    # att + gt
                    viz.plot_wsi_section(section=context_patches,
                                        mode='gt',
                                        attention=True,
                                        log_path=exp.args.log_path)
