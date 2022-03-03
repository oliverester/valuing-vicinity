
import torch 

def do_inference(data_loader: torch.utils.data.DataLoader,
                 model: torch.nn.Module,
                 gpu: int = None,
                 out: str = 'list',
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
    # switch to evaluate mode
    model.eval()
    m = torch.nn.Softmax(dim=1).cuda(gpu)

    with torch.no_grad():
        for images, context_images, targets in data_loader:
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
                context_images = context_images.cuda(gpu, non_blocking=True)
                
            # compute output
            logits, _ = model(images, context_images)
            probs = m(logits)
            if out == 'list':
                outputs.extend(probs.cpu().numpy())
                labels.extend(targets.cpu().numpy())
            elif out == 'torch':
                #targets = targets.cuda(gpu, non_blocking=True)
                outputs.append(probs.cpu())
                labels.append(targets.cpu())

    return outputs, labels
