"""
Provide a CustomPatchesDataset to work with WSI and Patches objects.

"""

from contextlib import contextmanager
from typing import List, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.deephist.segmentation.attention_segmentation.models.Memory import Memory
from src.pytorch_datasets.patch.patch_from_file import PatchFromFile
from src.pytorch_datasets.label_handler import LabelHandler
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder
from src.pytorch_datasets.wsi_dataset.wsi_dataset_from_folder import WSIDatasetFolder


class AttentionPatchesDataset(Dataset):
    """
    CustomPatchesDataset is a pytorch dataset to work with WSI patches
    provides by eihter WSI objects or WSIDatset objects.
    """

    def __init__(self,
                 wsi_dataset: Union[WSIDatasetFolder, WSIFromFolder] = None,
                 patches: List[PatchFromFile] = None,
                 transform: transforms.Compose = None):
        """
        Create a AttentionPatchesDataset from a WSI or WSI dataset.

        Args:
            wsi_dataset Union[AbstractWSIDataset, AbstractWSI]: Either a WSI or a WSIDataset
                to get patches from.
            transform (transforms.Compose, optional): Augmentation pipeline. Defaults to None.
        """
        if patches is not None and wsi_dataset is not None:
            raise Exception("Either provide patches or wsis.")
        elif patches is None and wsi_dataset is not None:
            self.use_patches = False
        else:
            self.use_patches = True
            
        self.wsi_dataset = wsi_dataset
        self.patches = patches
        self.transform = transform
                
    def get_label_handler(self) -> LabelHandler:
        """Get the label handler of the WSIs to access map to original labels.

        Returns:
            LabelHandler: LabelHandler that was created during the WSI datset building.
        """
        return self.wsi_dataset.label_handler

    def __len__(self):
        
        if self.use_patches:
            n_patches = len(self.patches)
        else:
            n_patches = len(self.wsi_dataset.get_patches())
        return n_patches
    
    def __getitem__(self, idx):
                
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # __call__ of patch provides image and label
        if self.use_patches:
            patch = self.patches[idx]
        else:
            if idx >= len(self.wsi_dataset.get_patches()):
                print("stop") 
            patch = self.wsi_dataset.get_patches()[idx]
        
        # get patch idx in memory
        patch_idx = Memory.get_memory_idx(patch=patch,
                                          k=self.wsi_dataset.k_neighbours)
        # get k-neighbourhood patch idxs in memory
        patch_neighbour_idxs = Memory.get_neighbour_memory_idxs(k=self.wsi_dataset.k_neighbours,
                                                                patch=patch)
        
        patch_img, label = patch()
      
        if self.transform is not None:
            patch_img = self.transform(patch_img)

        return patch_img, label, patch_idx, patch_neighbour_idxs
    
    @contextmanager
    def all_patch_mode(self):
        """Set context to receive all patches from wsi (instead of sampled ones).
        """
        with self.wsi_dataset.all_patch_mode():
            yield(self)
