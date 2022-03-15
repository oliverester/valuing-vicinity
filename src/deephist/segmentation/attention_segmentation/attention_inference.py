import torch
from torch import nn


def do_inference(data_loader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 gpu: int = None,
                 return_attention: bool = False,
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

    memory = data_loader.dataset.wsi_dataset.embedding_memory
     
    if args.memory_to_gpu is True:
        memory.to_gpu(gpu)
        
    # first loop: create neighbourhood embedding memory
    with torch.no_grad():
        memory = fill_memory(data_loader=data_loader,
                             memory=memory,
                             model=model,
                             gpu=gpu)
        outputs, labels, attentions, n_masks = memory_inference(data_loader=data_loader,
                                                                memory=memory,
                                                                model=model,
                                                                gpu=gpu)
    if return_attention:
        return outputs, labels, attentions, n_masks
    else:
        return outputs, labels
  
  
def memory_inference(data_loader,
                     memory,
                     model,
                     gpu):
    outputs = []
    labels = []
    attentions = []
    neighbour_masks = []
    
    m = nn.Softmax(dim=1).cuda(gpu)
    # second loop: attend freezed neighbourhood memory   
    with torch.no_grad():
        model.eval()  
        for  images, targets, _, neighbours_idx in data_loader:
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            
            k_neighbour_embedding, k_neighbour_mask = memory.get_k_neighbour_embeddings(neighbours_idx=neighbours_idx)
            
            if not k_neighbour_embedding.is_cuda:
                k_neighbour_embedding = k_neighbour_embedding.cuda(gpu, non_blocking=True)
                k_neighbour_mask = k_neighbour_mask.cuda(gpu, non_blocking=True)

            logits, attention = model(images, 
                                      neighbour_masks=k_neighbour_mask,
                                      neighbour_embeddings=k_neighbour_embedding,
                                      return_attention=True)  
            probs = m(logits)

            outputs.append(torch.argmax(probs,dim=1).cpu())
            labels.append(targets.cpu())
            attentions.append(attention.cpu())
            neighbour_masks.append(k_neighbour_mask.cpu())
            
    return outputs, labels, attentions, neighbour_masks
  
def fill_memory(data_loader, memory, model, gpu): 
    model.eval()
    # no matter what, enforce all patch mode to create complete memory
    with data_loader.dataset.wsi_dataset.all_patch_mode():
        with torch.no_grad():
            for images, _, patches_idx, _ in data_loader:
                if gpu is not None:
                    images = images.cuda(gpu, non_blocking=True)
                    
                embeddings = model(images,
                                   return_embeddings=True)
                memory.update_embeddings(patches_idx=patches_idx,
                                         embeddings=embeddings)
     
    assert len(data_loader.dataset.wsi_dataset.get_patches()) == torch.sum(torch.max(memory, dim=-1)[0] != 0).item(), \
        'memory is not completely build-up.'
    model.train()
    return memory
    
    