import itertools
from typing import List

import torch

from src.pytorch_datasets.patch.patch_from_file import PatchFromFile


class Memory(torch.nn.Module):
    
    def __init__(self,
                 n_x: int,
                 n_y: int,
                 n_w: int,
                 D: int,
                 k: int,
                 n_p: int = None,
                 ) -> None:
        """Memory to store compressed patch information and access patch neighbourhood.

        Args:
            n_x (int): Number of spatial patch-dimension in x
            n_y (int): Number of spatial patch-dimension in y
            n_w (int): Number of WSIs to fit into the memory
            D (int): Dimension of embedding
            k (int): Neighbourhood size
            n_p (int): Number of total patches that will be added to the memory. Only needed for sanity check. Optional.
        """
        super().__init__()

        self.metadata = dict()
        
        self.n_x = n_x
        self.n_y = n_y
        self.n_w = n_w
        self.n_p = n_p # only for sanity check to ensure every patch is in memory
        self.D = D
        self.k = k
        
        self._initialize_emb_memory()
    
    def _initialize_emb_memory(self):
        # plus k-boarder 
        memory = torch.full(size=(self.n_w, self.n_x+2*self.k, self.n_y+2*self.k, self.D), 
                                fill_value=0,
                                dtype=torch.float32, pin_memory=True)
        mask = torch.full(size=(self.n_w, self.n_x+2*self.k, self.n_y+2*self.k), 
                                fill_value=0,
                                dtype=torch.float32, pin_memory=True)
    
        print(f"Creating embedding memory with dim: {memory.shape}")
        size_in_gb = (memory.element_size() * memory.nelement()) / 1024 / 1024 / 1024
        print(f"Embedding memory size: {str(round(size_in_gb, 2))} GB")
        
        self.metadata['memory_size_in_gb'] = size_in_gb
        self.metadata['n_x'] = self.n_x
        self.metadata['n_y'] = self.n_y
        self.metadata['n_w'] = self.n_w
        self.metadata['D'] = self.D
        self.metadata['k'] = self.k

        self.register_buffer('_memory', memory, persistent=False)
        self.register_buffer('_mask', mask, persistent=False)
        
    def _reset(self):
        self._memory[...] = 0
        self._mask[...] = 0
            
    def to_gpu(self,
               gpu):
        self._memory = self._memory.cuda(gpu, non_blocking=False)
        self._mask = self._mask.cuda(gpu, non_blocking=False)

    def update_embeddings(self,
                          embeddings: torch.Tensor,
                          patches_idx = None):
        
        if not self._memory.is_cuda:
            embeddings = embeddings.detach().cpu()
        
        # batch update of embeddings     
        self._memory[patches_idx[0], # wsi idx
                     patches_idx[1], # x idx
                     patches_idx[2], # y idx
                     :] = embeddings
        self._mask[patches_idx[0], # wsi idx
                   patches_idx[1], # x idx
                   patches_idx[2], # y idx
                  ] = 1
            
    def get_k_neighbour_embeddings(self,
                                   neighbours_idx):
        batch_size = neighbours_idx.shape[-2]
       
            # select corresponding embeddings across batch and create a view on the memory with: batch, kernel size (k*2+1)^2, D       
        neighbour_embeddings = self._memory[neighbours_idx[0], # wsi idx
                                            neighbours_idx[1], # x idx
                                            neighbours_idx[2], # y idx
                                            :].view(batch_size,
                                                self.k*2+1,
                                                self.k*2+1,
                                                self.D)
                                                
        # after emb insert, there must be values != 0 for emb
        neighbour_mask = self._mask[neighbours_idx[0],
                                    neighbours_idx[1],
                                    neighbours_idx[2]].view(batch_size,
                                                            self.k*2+1,
                                                            self.k*2+1)
        return neighbour_embeddings, neighbour_mask
                
    def get_embeddings(self, patches: List[PatchFromFile]):
        
        wsi_idxs = []
        x_idxs = []
        y_idxs = []
        
        for patch in patches:
            wsi_idx, x, y = self.get_memory_idx(patch=patch, 
                                                k=self.k)
            wsi_idxs.append(wsi_idx)
            x_idxs.append(x)
            y_idxs.append(y)

        embeddings = self._memory[wsi_idxs, x_idxs, y_idxs, :]
        return embeddings
         
    @staticmethod
    def get_neighbour_memory_idxs(k: int,
                                  patch: PatchFromFile):
        wsi_idxs = []
        x_idxs = []
        y_idxs = []
    
        x, y = patch.get_coordinates()
        # adjust for memory boarders
        x, y = x + k, y + k 
         
        for coord in itertools.product(range(x-k,x+k+1), range(y-k,y+k+1)): 
            wsi_idxs.append(patch.wsi.idx)
            x_idxs.append(coord[0])
            y_idxs.append(coord[1])
                
        return wsi_idxs, x_idxs, y_idxs
    
    @staticmethod
    def get_memory_idx(k: int,
                       patch: PatchFromFile):
        x, y = patch.get_coordinates()
        # adjust for memory boarders
        x, y = x + k, y + k 
        wsi_idx = patch.wsi.idx
        return wsi_idx, x, y
