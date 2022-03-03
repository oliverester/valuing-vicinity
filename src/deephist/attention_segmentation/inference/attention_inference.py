
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
                print("Building memory")
                wsi_loader = exp.data_provider.get_wsi_loader(wsi=selected_wsi)
                embedding_memory = build_memory(wsi_loader, model, exp.args)
                assert len(wsi_loader.dataset.wsi_dataset.get_patches()) == torch.sum(torch.max(embedding_memory, dim=-1)[0] != 0).item(), \
                    'memory is not completely build-up.'
                # select patches 
                for x, y in patch_coordinates:
                    patch = selected_wsi.get_patch_from_position(x,y)
                    
                    context_patches, _  = patch.get_neighbours(k=include_k)
                
                    patches_loader = exp.data_provider.get_wsi_loader(patches=[p for p in list(context_patches.flatten()) if p is not None])

                    patches, outputs, labels, attentions, n_masks = attention_inference(data_loader=patches_loader,
                                                                                        model=model,
                                                                                        k=wsi_loader.dataset.wsi_dataset.k_neighbours,
                                                                                        embedding_memory=embedding_memory,
                                                                                        gpu=exp.args.gpu)  
                    # append results to patch object
                    for i, patch in enumerate(patches):
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
            
def build_memory(data_loader, model, args):
    
    # switch to evaluate mode
    model.eval()
    # compute output
    embedding_memory = data_loader.dataset.wsi_dataset.embedding_memory
        
    if args.memory_to_gpu is True:
        embedding_memory = embedding_memory.cuda(args.gpu, non_blocking=True)
        
    # first loop: create neighbourhood embedding memory
    with torch.no_grad():
        for patches, images, _, _, _ in tqdm(data_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
    
            embeddings = model(images,
                               return_embeddings=True)
            update_embeddings(embedding_memory=embedding_memory,
                              patches=patches, 
                              embeddings=embeddings)
    return embedding_memory

   
def attention_inference(data_loader: torch.utils.data.DataLoader,
                        model: torch.nn.Module,
                        k: int,
                        embedding_memory,
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
    attentions = []
    patches = []
    n_masks = []
    
    with torch.no_grad():
        # switch to evaluate mode
        model.eval()
        m = torch.nn.Softmax(dim=1).cuda(gpu)

        # second loop: attend freezed neighbourhood memory     
        for batch_patches, images, targets, neighbours, neighbour_masks in data_loader:
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
                neighbour_masks = neighbour_masks.cuda(gpu, non_blocking=True)
            
            k_neighbour_embedding = get_k_neighbour_embeddings(k=k,
                                                               embedding_memory=embedding_memory,
                                                               neighbour_patches_list=neighbours)
            
            if not k_neighbour_embedding.is_cuda:
                k_neighbour_embedding = k_neighbour_embedding.cuda(gpu, non_blocking=True)
            
            logits, attention = model(images, 
                                      neighbour_masks,
                                      neighbour_embeddings=k_neighbour_embedding,
                                      return_attention=True)  
            
            probs = m(logits)
    
            outputs.extend(torch.argmax(probs,dim=1).cpu())
            labels.extend(targets.cpu())
            attentions.extend(attention.cpu())
            patches.extend(batch_patches)
            n_masks.extend(neighbour_masks.cpu())

    return patches, outputs, labels, attentions, n_masks
    
if __name__ == "__main__":

    # patch_coordinates = {'RCC-TA-033.001~C': [(14,19), (15,20), (20,20)],
    #                      'RCC-TA-011.001~J': [(20, 15), (20, 17)],
    #                      'RCC-TA-004.001~C': [(21, 35)]
    #                      }
    
    patch_coordinates = {'RCC-TA-163.001~B': [(7,12), (8,12), (8,11), (7,11)],
                         }
    exp=AttentionSegmentationExperiment(config_path='/homes/oester/repositories/prae/src/deephist/attention_segmentation/analysis/attention_segmentation_config_inference.yml')

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