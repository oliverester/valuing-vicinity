from re import M
import torch
from segmentation_models_pytorch import create_model
from segmentation_models_pytorch.base.modules import Conv2dReLU
from torch import nn

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
                 use_ln: bool = False,
                 use_pos_encoding: bool = False,
                 use_central_attention: bool = False,
                 learn_pos_encoding: bool = False,
                 attention_on: bool = True,
                 use_transformer: bool = False) -> None:
        super().__init__()
        
        self.attention_on = attention_on
        self.use_central_attention = use_central_attention
        self.attention_input_dim = attention_input_dim
        self.use_transformer = use_transformer
        self.kernel_size = k*2+1
        
        if not self.use_central_attention and not use_transformer:
            # mask central patch
            mask_central = torch.full((self.kernel_size,self.kernel_size), fill_value=1)
            mask_central[k,k] = 0
            self.register_buffer('mask_central', mask_central, persistent=False)
                
        # Pytorch Segmentation Models: Baseline
        self.model = create_model(arch=arch,
                                  encoder_name=encoder_name,
                                  encoder_weights=encoder_weights,
                                  classes=number_of_classes)
        
        if self.attention_on:
            # f_emb
            self.pooling = nn.AdaptiveAvgPool2d(1)
            # number of feature maps of encoder output: e.g. 2048 for U-net 5 layers 
            self.lin_proj = nn.Linear(self.model.encoder._out_channels[-1], attention_input_dim)
            
            # f_fuse
            self.conv1x1 =  Conv2dReLU(self.model.encoder._out_channels[-1]+attention_input_dim, 
                                       self.model.encoder._out_channels[-1], (1,1))
            if self.use_transformer:
                self.transformer = ViT(kernel_size=self.kernel_size,
                                       dim=attention_input_dim,
                                       depth=transformer_depth,
                                       heads=num_attention_heads,
                                       mlp_dim=mlp_hidden_dim,
                                       hidde_dim=attention_hidden_dim
                                       )
            else: # use MHA
                self.msa = MultiheadAttention(input_dim=attention_input_dim, 
                                              hidden_dim=attention_hidden_dim,
                                              num_heads=num_attention_heads,
                                              kernel_size= self.kernel_size,
                                              use_ln=use_ln,
                                              use_pos_encoding=use_pos_encoding,
                                              learn_pos_encoding=learn_pos_encoding)
            
    def forward(self, 
                images, 
                neighbour_masks=None,
                neighbour_embeddings=None,
                return_attention=False,
                return_embeddings=False):
        """Sequentially pass `x` through model`s encoder, decoder and heads"""
        # segmentation encoder
        features = self.model.encoder(images)
        
        if self.attention_on:
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
                tmp_batch_size = embeddings.shape[0]

                embeddings = torch.unsqueeze(embeddings, 1)
                
                # attend neighbour embeddings if exist:
                if neighbour_embeddings is not None:
                    # sanity check: all embeddings should have values                    
                    assert torch.sum(neighbour_masks).item() == torch.sum(torch.max(neighbour_embeddings, dim=-1)[0] != 0).item(), \
                        'all embeddings should have values'
                        
                    if not self.use_central_attention:  
                        # add empty central patches - happens when self-attention is turned off
                        neighbour_masks = neighbour_masks * self.mask_central

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
                    attended_embeddings = attended_embeddings[:,:,None,None].expand(-1,-1,encoder_map.shape[-2],encoder_map.shape[-1])
                    features_with_neighbour_context = torch.cat((features[-1], attended_embeddings),1)
                    # 1x1 conv to merge features to 2048 again
                    features_with_neighbour_context = self.conv1x1(features_with_neighbour_context)
                    
                    # exchange feat_l with feat_l'
                    features[-1] = features_with_neighbour_context
                else:
                    attention = None
        # segmentation decoder    
        decoder_output = self.model.decoder(*features)
        # segmentation head
        masks = self.model.segmentation_head(decoder_output)

        if return_attention:
            return masks, attention
        else:
            return masks

           
            