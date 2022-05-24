from re import I
from segmentation_models_pytorch import create_model
from segmentation_models_pytorch.base.modules import Conv2dReLU
import torch
from torch import nn
from tqdm import tqdm

from src.deephist.segmentation.attention_segmentation.models.Memory import Memory
from src.deephist.segmentation.attention_segmentation.models.multihead_attention_model import \
    MultiheadAttention
from src.deephist.segmentation.attention_segmentation.models.transformer_model import ViT


class AttentionSegmentationModel(torch.nn.Module):
    
    def __init__(self, 
                 arch: str,
                 encoder_name: str,
                 encoder_weights: str,
                 number_of_classes: int,
                 attention_input_dim: int,
                 k: int,
                 attention_hidden_dim: int = 1024,
                 mlp_hidden_dim: int = 2048,
                 num_attention_heads: int = 8,
                 transformer_depth: int = 8,
                 emb_dropout: float = 0.,
                 dropout: float = 0.,
                 use_ln: bool = False,
                 use_pos_encoding: bool = False,
                 use_central_attention: bool = False,
                 learn_pos_encoding: bool = False,
                 attention_on: bool = True,
                 use_transformer: bool = False,
                 online: bool = False
                 ) -> None:
        super().__init__()
        
        self.online = online
        self.attention_on = attention_on
        self.use_central_attention = use_central_attention
        self.attention_input_dim = attention_input_dim
        self.use_transformer = use_transformer
        
        if self.attention_on:
            self.block_memory = False # option to skip memory attention in forward-pass
            
            self.kernel_size = k*2+1
            
            if not self.use_central_attention and not self.use_transformer:
                # mask central patch
                mask_central = torch.full((self.kernel_size,self.kernel_size), fill_value=1)
                mask_central[k,k] = 0
                self.register_buffer('mask_central', mask_central, persistent=False)
                    
        # Pytorch Segmentation Models: Baseline
        self.base_model = create_model(arch=arch,
                                       encoder_name=encoder_name,
                                       encoder_weights=encoder_weights,
                                       classes=number_of_classes)
        
        if self.attention_on:
            # f_emb
            self.pooling = nn.AdaptiveAvgPool2d(1)
            # number of feature maps of encoder output: e.g. 2048 for U-net 5 layers 
            self.lin_proj = nn.Linear(self.base_model.encoder._out_channels[-1], attention_input_dim)
            
            # f_fuse
            self.conv1x1 =  Conv2dReLU(self.base_model.encoder._out_channels[-1]+attention_input_dim, 
                                       self.base_model.encoder._out_channels[-1], (1,1))
            if self.use_transformer:
                self.transformer = ViT(kernel_size=self.kernel_size,
                                       dim=attention_input_dim,
                                       depth=transformer_depth,
                                       heads=num_attention_heads,
                                       mlp_dim=mlp_hidden_dim,
                                       hidde_dim=attention_hidden_dim,
                                       emb_dropout=emb_dropout,
                                       dropout=dropout,
                                       use_pos_encoding=use_pos_encoding,
                                       )
            else: # use MHA
                self.msa = MultiheadAttention(input_dim=attention_input_dim, 
                                              hidden_dim=attention_hidden_dim,
                                              num_heads=num_attention_heads,
                                              kernel_size= self.kernel_size,
                                              use_ln=use_ln,
                                              use_pos_encoding=use_pos_encoding,
                                              learn_pos_encoding=learn_pos_encoding)
                
    def initialize_memory(self,
                          gpu: int,
                          reset=True,
                          **memory_params):
        """Initializes (train/eval) Memory - use model.eval() to initialize validation/evaluation Memory.

        Args:
            is_eval (bool, optional): If true, creates eval Memory - else
            train Memory. Later on, controlled by model.eval(). Defaults to False.
        """
        if self.training:
            if not hasattr(self, 'train_memory') or reset:
                print("Initializing train memory")
                train_memory = Memory(**memory_params, is_eval=False, gpu=gpu)
                super(AttentionSegmentationModel, self).add_module('train_memory', train_memory)
        else:
            if not hasattr(self, 'val_memory') or reset:
                print("Initializing eval memory")
                val_memory = Memory(**memory_params, is_eval=True, gpu=gpu)
                super(AttentionSegmentationModel, self).add_module('val_memory', val_memory)

    def fill_memory(self, 
                    data_loader: torch.utils.data.dataloader.DataLoader,
                    gpu: int):
        """Fill the memory by providing a dataloader that iterates the patches. 
        Iterate must provide all n_p patches.
        
        Args:
            data_loader (torch.utils.data.dataloader.DataLoader): DataLoader
        """
        if self.block_memory:
            raise Exception("Memory is blocked. If you really want to fill memory, set 'block_memory' to False")
        
        #reset memory first to ensure consistency
        self.memory._reset()
        
        print("Filling memory..")
        
        # no matter what, enforce all patch mode to create complete memory
        with data_loader.dataset.all_patch_mode():
            with torch.no_grad():
                for images, _, patches_idx, _ in tqdm(data_loader):
                    images = images.cuda(gpu, non_blocking=True)
                        
                    embeddings = self(images,
                                      return_embeddings=True)
                    self.memory.update_embeddings(patches_idx=patches_idx,
                                                  embeddings=embeddings)
            # flag memory ready to use
            self.memory.set_ready(n_patches=data_loader.dataset.__len__())
        
    @property
    def memory(self):
        if self.training:
            if not hasattr(self, 'train_memory'):
                raise Exception("""Train Memory is not initialized yet. Please use the initialize_memory 
                function of the AttentionSegmentationModel and specify the required dimensions""")
            return self.train_memory
        else:
            if not hasattr(self, 'val_memory'):
                raise Exception("""Train Memory is not initialized yet. Please use the initialize_memory 
                function of the AttentionSegmentationModel and specify the required dimensions""")
            return self.val_memory
        
    def forward(self, 
                images: torch.Tensor, 
                neighbours_idx: torch.Tensor = None,
                neighbour_imgs: torch.Tensor = None,
                return_embeddings: bool = False):
        """ 
        Attention segmentation model:
        If return_embeddings is True, only images must be provided and the model returns the compressed
        patch representation after the encoder + pooling + lineanr projection.
        
        If return_emebeddings is False, neighbour_idx must be provided to point to the coordinates in the memory.
        
        If model is set to online, instead of neighbour_idx you have to provide neighbour_imgs, as the neighbourhood
        patch embeddings will be derived, simultaneously.
   
        Args:
            images (torch.Tensor, optional): B x C x h x w normalized image tensor.
            neighbours_idx (torch.Tensor, optional): Indixes of neihgbourhood memory. Must be provided as long as return_embeddings is False. Defaults to None.
            neighbour_imgs (torch.Tensor, optional): Must be provided if model is set to online. Defaults to None.
            return_embeddings (bool, optional): Set to True to receive the patch embeddings, only. Defaults to False.

        Returns:
            [torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: 
            Either embeddings tensor - or tuple of prediction masks, attention maps and binary neighbour masks tensors
        """
        # set default values
        attention = neighbour_masks = None
        
        # segmentation encoder
        features = self.base_model.encoder(images)
        
        if self.attention_on and not self.block_memory:
            # add context to deepest feature maps feat_l by attending neighbourhood embeddings
            encoder_map = features[-1]
            
            # f_emb:
            # pool feature maps + linear proj to patch emb
            pooled = self.pooling(encoder_map)
            embeddings = self.lin_proj(pooled.flatten(1))
            
            # sanity check
            assert not torch.any(torch.max(embeddings, dim=1)[0] == 0), "Embeddings must have values."
                                
            if return_embeddings:
                return embeddings
            else:
                tmp_batch_size, ch, x, y = images.shape
                # online: get neighbour embeddings (with grads) concurrently
                if self.online:
                    with torch.no_grad():
                        neighbour_features = self.base_model.encoder(neighbour_imgs.view(-1,ch,x,y))
                    encoder_map = neighbour_features[-1]
                    # f_emb:
                    # pool feature maps + linear proj to patch emb
                    pooled = self.pooling(encoder_map)
                    neighbour_embeddings = self.lin_proj(pooled.flatten(1))
                    neighbour_embeddings = neighbour_embeddings.view(tmp_batch_size,self.kernel_size*self.kernel_size,-1)
                else: 
                    # query memory for context information
                    neighbour_embeddings, neighbour_masks = self.memory.get_k_neighbour_embeddings(neighbours_idx=neighbours_idx)
                
                embeddings = torch.unsqueeze(embeddings, 1)

                # sanity check: all embeddings should have values  
                if not self.online:                  
                    assert torch.sum(neighbour_masks).item() == torch.sum(torch.max(neighbour_embeddings, dim=-1)[0] != 0).item(), \
                        'all embeddings should have values'
                        
                if not self.use_central_attention and not self.use_transformer:  
                    # add empty central patches - happens when self-attention is turned off
                    neighbour_masks = neighbour_masks * self.mask_central

                # from "2d" to "1d"
                k_neighbour_masks = neighbour_masks.view(tmp_batch_size, 1, 1, -1)
                neighbour_embeddings = neighbour_embeddings.view(tmp_batch_size, -1, self.attention_input_dim)
            
                if self.use_transformer:
                    # replace central patch embedding with current embedding
                    c_pos = (self.kernel_size*self.kernel_size-1)//2
                    neighbour_embeddings[:,c_pos:(c_pos+1),:] = embeddings
                    attended_embeddings, attention = self.transformer(x=neighbour_embeddings,
                                                                        mask=k_neighbour_masks,
                                                                        return_attention=True) 
                else: #MHA
                    attended_embeddings, attention =  self.msa(q=embeddings,
                                                                kv=neighbour_embeddings,
                                                                mask=k_neighbour_masks,
                                                                return_attention=True)
                # f_fuse:
                # concatinate attended embeddings to encoded features      
                attended_embeddings = torch.squeeze(attended_embeddings, 1)
                # expand over e.g 8x8-convoluted feature map for Unet - or 32x32 for deeplabv3
                attended_embeddings = attended_embeddings[:,:,None, None].expand(-1, -1, encoder_map.shape[-2], encoder_map.shape[-1])
                features_with_neighbour_context = torch.cat((features[-1], attended_embeddings),1)
                # 1x1 conv to merge features to 2048 again
                features_with_neighbour_context = self.conv1x1(features_with_neighbour_context)
                
                # exchange feat_l with feat_l'
                features[-1] = features_with_neighbour_context
        
        # segmentation decoder    
        decoder_output = self.base_model.decoder(*features)
        # segmentation head
        masks = self.base_model.segmentation_head(decoder_output)
        
        if self.attention_on:
            return masks, attention, neighbour_masks
        else:
            return masks

           
            