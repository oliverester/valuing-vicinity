import json
import random
from pathlib import Path
from typing import List, Tuple, Union

import torch
import yaml

from src.exp_management.Experiment import Experiment
from src.deephist.CustomPatchesDataset import CustomPatchesDataset
from src.pytorch_datasets.label_handler import LabelHandler
from src.pytorch_datasets.patch.patch_from_file import PatchFromFile
from src.pytorch_datasets.wsi_dataset.wsi_dataset_from_folder import WSIDatasetFolder
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder

class DataProvider():
    """
    Create Training and Validation DataLoader given the configs. Handles WSI splits.
    """

    def __init__(self,
                 exp: Experiment = None,
                 train_data: str = None,
                 test_data: str = None,
                 embeddings_root: str = None,
                 overlay_polygons: bool = False,
                 image_label_in_path: bool = False,
                 patch_sampling: str = None,
                 patch_label_type: str = 'patch',
                 vali_split: float = None,
                 exclude_classes: List[int] = None,
                 include_classes: List[int] = None,
                 merge_classes: List[List[int]] = None,
                 draw_patches_per_class: int = None,
                 draw_patches_per_wsi: int = None,
                 label_map_file: str = None,
                 hue_aug_ratio: float = None,
                 normalize: bool = False,
                 batch_size: int = None,
                 val_batch_size: int = None,
                 test_batch_size: int = None,
                 nfold: int = None,
                 mc_runs: int = None,
                 workers: int = 16,
                 distributed: bool = False,
                 gpu: int = None,
                 dataset_type = CustomPatchesDataset,
                 collate_fn = None,
                 attention_on: bool = False,
                 embedding_dim: int = None,
                 k_neighbours: int = None,
                 multiscale_on: bool = False,
                 ):

        # avoid holdout and cv at the same time
        assert(not (nfold is not None and (vali_split is not None and mc_runs is None)))
        assert(not (nfold is not None and mc_runs is not None))
        assert(patch_label_type in ['patch', 'image', 'mask', 'distribution'])
        if exclude_classes is not None:
            assert(all([isinstance(ex, str) for ex in exclude_classes]))
        if include_classes is not None:
            assert(all([isinstance(inc, str) for inc in include_classes]))
        
        # detect label mapping
        if label_map_file is not None:
            with Path(label_map_file).open("r") as jfile:
                self.label_map = json.load(jfile)
        else:
            self.label_map = None

        self.exp = exp
        
        self.patch_label_handler = LabelHandler(prehist_label_map = self.label_map,
                                                merge_classes=merge_classes,
                                                include_classes=include_classes)
        self.image_label_handler = LabelHandler(include_classes=[0])
        self.mask_label_handler = LabelHandler(prehist_label_map = self.label_map, 
                                               merge_classes=merge_classes,
                                               include_classes=include_classes)
        
        self.patch_label_handler.lock() # to ensure no new labels are added
        self.image_label_handler.lock() # to ensure no new labels are added
        self.mask_label_handler.lock() # to ensure no new labels are added


        self.image_label_in_path = image_label_in_path
        self.patch_sampling = patch_sampling
        self.patch_label_type = patch_label_type
        self.embeddings_root = embeddings_root
        
        # check for pre-histo config
        try:
            prehisto_config_path = Path(train_data) / "prehisto_config.yml"
            with open(prehisto_config_path) as f:
                # use safe_load instead load
                self.prehisto_config = yaml.safe_load(f)
            if overlay_polygons is True:   
                self.polygons_root = self.prehisto_config['annotation_paths']
            else:
                self.polygons_root = None

            # determine thumbnail adjustment ratio to align patch plotting
            # (rescaling patches with fixed thumbnail size can result in rounding errors)
            expected_patch_img_size = (self.prehisto_config['patch_size'] * self.prehisto_config['downsample']) / 100
            self.exp.args.thumbnail_correction_ratio =  1 / (expected_patch_img_size / round(expected_patch_img_size))
        except Exception as e:
            if overlay_polygons is True:
                raise Exception("Cannot find prehist_config.yml in data root. \
                                Please provide to allow overlay_polygon-option.")
            else:
                self.polygons_root = None
                self.prehisto_config = None
                self.exp.args.thumbnail_correction_ratio = 1

        self.train_data = train_data
        self.test_data = test_data
        self.val_ratio = vali_split
        self.exclude_patch_class = exclude_classes
        self.include_patch_class = include_classes
        self.merge_classes = merge_classes
        self.draw_patches_per_class = draw_patches_per_class
        self.draw_patches_per_wsi = draw_patches_per_wsi
        self.batch_size = batch_size
        
        if val_batch_size is None:
            self.val_batch_size = batch_size
        else:
            self.val_batch_size = val_batch_size
        if test_batch_size is None:
            self.test_batch_size = self.val_batch_size
        else:
            self.test_batch_size = test_batch_size
        self.nfold = nfold
        self.mc_runs = mc_runs

        self.workers = workers
        self.distributed = distributed
        self.gpu = gpu

        self.collate_fn = collate_fn
        self.dataset_type = dataset_type
        
        #attention
        self.attention_on = attention_on
        self.embedding_dim = embedding_dim
        self.k_neighbours = k_neighbours

        #multiscale
        self.multiscale_on = multiscale_on
        
        #augmentation
        self.hue_aug_ratio = hue_aug_ratio
        self.normalize = normalize
        self._set_augmentation()

        # on reload with provided test data, do not prepare train data
        if self.exp.args.reload_model_folder is None or self.exp.args.include_val_inference is True:
            self._setup_data()
            self.holdout_set = self._set_holdout_set()
            self.train_set = self._set_train_set()
            self.cv_set = self._set_cv_set()
        self.test_wsi_dataset = self._set_test_wsis()

        
        if patch_label_type in ['patch', 'distribution']:
            self.label_handler = self.patch_label_handler
            self.number_classes = len(self.label_handler.classes)
        elif patch_label_type == 'image':
            self.label_handler = self.image_label_handler
            self.number_classes = len(self.label_handler.classes)
        elif patch_label_type == 'mask':
            self.label_handler = self.mask_label_handler
            self.number_classes = len(self.mask_label_handler.classes)

    def _set_augmentation(self):
        if self.train_data is not None:
            self.train_aug_transform, self.vali_aug_transform = self.exp.get_augmention()

    def _setup_data(self):
        if self.train_data is not None:

            self.wsi_dataset = WSIDatasetFolder(dataset_root=self.train_data,
                                                embeddings_root=self.embeddings_root,
                                                polygons_root=self.polygons_root,
                                                root_contains_wsi_label=self.image_label_in_path,
                                                patch_sampling=self.patch_sampling,
                                                exclude_patch_class=self.exclude_patch_class,
                                                include_patch_class=self.include_patch_class,
                                                merge_classes = self.merge_classes,
                                                patch_label_handler=self.patch_label_handler,
                                                image_label_handler=self.image_label_handler,
                                                mask_label_handler=self.mask_label_handler,
                                                patch_label_type=self.patch_label_type,
                                                draw_patches_per_class=self.draw_patches_per_class,
                                                draw_patches_per_wsi=self.draw_patches_per_wsi,
                                                prehisto_config=self.prehisto_config,
                                                attention_on=self.attention_on,
                                                embedding_dim=self.embedding_dim,
                                                k_neighbours=self.k_neighbours,
                                                multiscale_on=self.multiscale_on,
                                                exp=self.exp)

    def _set_cv_set(self):
        if self.nfold is not None:
            return CvSet(wsi_dataset=self.wsi_dataset,
                         data_provider=self,
                         nfold=self.nfold)
        elif self.mc_runs is not None and self.val_ratio is not None:
            return McSet(wsi_dataset=self.wsi_dataset,
                         data_provider=self,
                         runs=self.mc_runs,
                         val_ratio=self.val_ratio)
        else:
            return None

    def _set_holdout_set(self):
        if self.train_data is not None and \
           self.val_ratio is not None and \
           self.nfold is None and \
           self.mc_runs is None:
               
            train_wsi_dataset, vali_wsi_dataset = self.wsi_dataset.split_wsi_dataset_by_ratios(split_ratios= [1-self.val_ratio,
                                                                                                              self.val_ratio])

            holdout_set =  HoldoutSet(train_wsi_dataset=train_wsi_dataset,
                                      vali_wsi_dataset=vali_wsi_dataset,
                                      data_provider=self)

            return holdout_set
        else:
            return None

    def _set_train_set(self):
        if self.train_data is not None and \
           self.val_ratio is  None and \
           self.nfold is None:
               return self.wsi_dataset
        else:
            return None

    def _set_test_wsis(self):
        if self.test_data is not None:
            test_dataset = WSIDatasetFolder(dataset_root=self.test_data,
                                            embeddings_root=self.embeddings_root,
                                            polygons_root=self.polygons_root,
                                            root_contains_wsi_label=self.image_label_in_path,
                                            exclude_patch_class=self.exclude_patch_class,
                                            include_patch_class=self.include_patch_class,
                                            merge_classes = self.merge_classes,
                                            patch_label_handler=self.patch_label_handler,
                                            image_label_handler=self.image_label_handler,
                                            mask_label_handler=self.mask_label_handler,
                                            patch_label_type=self.patch_label_type,
                                            prehisto_config=self.prehisto_config,
                                            attention_on=self.attention_on,
                                            embedding_dim=self.embedding_dim,
                                            k_neighbours=self.k_neighbours,
                                            multiscale_on=self.multiscale_on,
                                            exp=self.exp)
            
            return test_dataset
        else:
            return None

    def get_train_loader(self):
        # Data loading code
        if self.train_set is not None:

            train_dataset = self.dataset_type(dataset_root=self.train_data,
                                              embeddings_root=self.embeddings_root,
                                              root_contains_wsi_label=self.image_label_in_path,
                                              patch_sampling=self.patch_sampling,
                                              exclude_patch_class=self.exclude_patch_class,
                                              include_patch_class=self.include_patch_class,
                                              patch_label_handler=self.patch_label_handler,
                                              image_label_handler=self.image_label_handler,
                                              patch_label_type=self.patch_label_type,
                                              draw_patches_per_class=self.draw_patches_per_class,
                                              draw_patches_per_wsi=self.draw_patches_per_wsi)

            if self.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                print(f"GPU {self.gpu}")
            else:
                train_sampler = None

            print(f"Train Data set length {len(train_dataset)}")
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=int(self.batch_size),
                                                       shuffle=(train_sampler is None),
                                                       num_workers=self.workers,
                                                       pin_memory=True,
                                                       sampler=train_sampler,
                                                       drop_last=True,
                                                       collate_fn=self.collate_fn)

            return train_loader, train_sampler, train_dataset.get_label_handler()

    def get_test_loader(self):

        if self.test_set is not None:

            test_dataset = self.dataset_type(wsi_dataset=self.test_set,
                                             transform=self.vali_aug_transform,
                                             patch_label_type=self.patch_label_type
                                             )
            # validation loader does not have distributed loader. So, each GPU runs a full validation run. However, only rank 0 prints
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=self.test_batch_size,
                                                      shuffle=False,
                                                      num_workers=self.workers,
                                                      pin_memory=True,
                                                      collate_fn=self.collate_fn)

            return test_loader, test_dataset.get_label_handler()


    def get_wsi_loader(self, 
                       wsi: Union[WSIFromFolder, WSIDatasetFolder] = None,
                       patches: List[PatchFromFile] = None) -> torch.utils.data.DataLoader:
        """Generates a DataLoader from a WSI object (generated by pre-histo) or a WSI dataset object

        Returns:
            torch.utils.data.DataLoader: DataLoader
        """
        wsi_pytorch_dataset = self.dataset_type(wsi_dataset=wsi,
                                                patches=patches,
                                                transform=self.vali_aug_transform)
        # validation loader does not have distributed loader. So, each GPU runs a full validation run. However, only rank 0 prints
        wsi_loader = torch.utils.data.DataLoader(wsi_pytorch_dataset,
                                                 batch_size=self.val_batch_size,
                                                 shuffle=False, # IMPORTANT, do not change for CLAM
                                                 num_workers=self.workers,
                                                 pin_memory=True,
                                                 collate_fn=self.collate_fn)

        return wsi_loader


class HoldoutSet():

    def __init__(self,
                 train_wsi_dataset: WSIDatasetFolder,
                 vali_wsi_dataset: WSIDatasetFolder,
                 data_provider: DataProvider,
                 fold: int = None
                ):

        self.train_wsi_dataset = train_wsi_dataset
        self.vali_wsi_dataset = vali_wsi_dataset
        self.data_provider = data_provider
        self.fold = fold
        
        if self.data_provider.embedding_dim is not None:
            self.train_wsi_dataset.initialize_memory()
            self.vali_wsi_dataset.initialize_memory()

        self._create_loader()


    def _create_loader(self):
        # log all metadata of wsi dataset
        self.data_provider.exp.exp_log(train_wsi_dataset = self.train_wsi_dataset.metadata)

        self.train_torch_dataset = self.data_provider.dataset_type(
            wsi_dataset=self.train_wsi_dataset,
            transform=self.data_provider.train_aug_transform)

        self.data_provider.exp.exp_log(vali_wsi_dataset = self.vali_wsi_dataset.metadata)
        self.vali_torch_dataset = self.data_provider.dataset_type(
            wsi_dataset=self.vali_wsi_dataset,
            transform=self.data_provider.vali_aug_transform)

        print(f"Train Data set length {self.train_wsi_dataset.metadata['n_drawn_patches']}"
            f" patches from {self.train_wsi_dataset.metadata['n_wsis']} wsis")
        print(f"Vali Data set length {self.vali_wsi_dataset.metadata['n_drawn_patches']}"
            f" patches from {self.vali_wsi_dataset.metadata['n_wsis']} wsis")

        self.train_loader = torch.utils.data.DataLoader(
            self.train_torch_dataset,
            batch_size=int(self.data_provider.batch_size),
            shuffle=True,
            num_workers=self.data_provider.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.data_provider.collate_fn,
            persistent_workers=True if self.data_provider.workers > 0 else False)
        
        self.big_train_loader = torch.utils.data.DataLoader(
            self.train_torch_dataset,
            batch_size=int(self.data_provider.val_batch_size),
            shuffle=True,
            num_workers=self.data_provider.workers,
            pin_memory=True,
            collate_fn=self.data_provider.collate_fn,
            persistent_workers=True if self.data_provider.workers > 0 else False)

        self.vali_loader = torch.utils.data.DataLoader(
            self.vali_torch_dataset,
            batch_size=int(self.data_provider.val_batch_size),
            shuffle=True,
            num_workers=self.data_provider.workers,
            pin_memory=True,
            collate_fn=self.data_provider.collate_fn)

class McSet():
    """
    Create a Monte-Carlo data set
    """
    def __init__(self,
                 data_provider: DataProvider,
                 wsi_dataset: WSIDatasetFolder,
                 runs: int = 3,
                 val_ratio: float = 0.3):

        self.runs = runs
        self.val_ratio = val_ratio
        self.wsi_dataset= wsi_dataset
        self.data_provider = data_provider
        self.holdout_sets = self._create_mc_set()

    def _create_mc_set(self):
        print("Creating Monte-Carlo-Sets")
        runs = []
        wsi_labels = self.wsi_dataset.get_wsi_labels()
        wsis = self.wsi_dataset.get_wsis()

        train_wsis = [ [] for _ in range(self.runs) ] 
        val_wsis = [ [] for _ in range(self.runs) ] 
        
        for lbl in set(wsi_labels):
            lbl_wsis = [wsi for wsi in wsis if wsi.get_label() == lbl]
            # shuffle list
      
            for run in range(self.runs):
                random.Random(5).shuffle(lbl_wsis)

                cutoff = int(round(self.val_ratio*len(lbl_wsis)))
                val_wsis[run].extend(lbl_wsis[:cutoff])
                train_wsis[run].extend(lbl_wsis[cutoff:])
                
        for run in range(self.runs):
            print(f"Monte-Carlo-Set {run}")
            runs.append(HoldoutSet(
                train_wsi_dataset=self.wsi_dataset.get_wsi_dataset_subset(train_wsis[run]),
                vali_wsi_dataset=self.wsi_dataset.get_wsi_dataset_subset(val_wsis[run]),
                data_provider= self.data_provider,
                fold = run))

        #self.data_provider.exp.exp_log(splitting=fold_splits)
        return runs


class CvSet():
    """
    Create a K-Fold-Cross-Validation data set
    """

    def __init__(self,
                 data_provider: DataProvider,
                 wsi_dataset: WSIDatasetFolder,
                 nfold: int = 3):
        self.nfold = nfold
        self.wsi_dataset= wsi_dataset
        self.data_provider = data_provider
        self.holdout_sets = self._create_kfold_set()

    def _create_kfold_set(self):
        """
        Create stratified cross validation datasets.
        """
        print("Creating Cross-Validation-Sets")
        folds = []
        wsi_labels = self.wsi_dataset.get_wsi_labels()
        wsis = self.wsi_dataset.get_wsis()

        label_splits = dict()
        label_wsis = dict()
        for lbl in set(wsi_labels):
            lbl_wsis = [wsi for wsi in wsis if wsi.get_label() == lbl]
            # shuffle list
            random.Random(5).shuffle(lbl_wsis)
            # save the shuffled list of wsis
            label_wsis[lbl] = lbl_wsis

            # draw cutoffs for folds
            number_wsis = len(lbl_wsis)
            # find split points
            split_points = [int(round(fold / self.nfold * number_wsis)) for fold in list(range(1,self.nfold+1))]
            split_points = zip(([0]+split_points[:-1]), split_points) # zip with starting index
            label_splits[lbl] = list(split_points)

        fold_splits = dict()
        all_val_wsis = []
        for fold in range(self.nfold):
            print(f"CV-Fold {fold}")
            
            train_wsis = []
            val_wsis = []
            fold_splits[fold] = dict()
            fold_splits[fold]['train'] = dict()
            fold_splits[fold]['vali'] = dict()

            for lbl, lbl_split in label_splits.items():
                # add val indices per lbl
                lbl_val_wsis = label_wsis[lbl][lbl_split[fold][0]:lbl_split[fold][1]]
                val_wsis.extend(lbl_val_wsis)
                # all other incides per lbl go into the train set
                lbl_train_wsis = [lbl_idx for idx, lbl_idx in enumerate(label_wsis[lbl])
                                   if idx not in range(lbl_split[fold][0],lbl_split[fold][1])]
                train_wsis.extend(lbl_train_wsis)

                fold_splits[fold]['train'][lbl] = [wsi.name for wsi in lbl_train_wsis]
                fold_splits[fold]['vali'][lbl] = [wsi.name for wsi in lbl_val_wsis]
            
            folds.append(HoldoutSet(
                train_wsi_dataset=self.wsi_dataset.get_wsi_dataset_subset(train_wsis),
                vali_wsi_dataset=self.wsi_dataset.get_wsi_dataset_subset(val_wsis),
                data_provider= self.data_provider,
                fold = fold))

            # collect val incides to assert uniqueness later
            all_val_wsis.extend(val_wsis)
            # no val data in test data
            assert(all(val_wsi not in train_wsis for val_wsi in val_wsis))

        # every wsi only once in val set
        assert(len(all_val_wsis) == len(set(all_val_wsis)))
        assert(len(all_val_wsis) == len(wsi_labels))

        self.data_provider.exp.exp_log(splitting=fold_splits)
        return folds


def split_wsi_dataset_root(dataset_root: str,
                           val_ratio: List[int],
                           image_label_in_path: bool
                           ) -> Tuple[List[str]]:
    """
    Providing a WSI dataset root, the WSI roots are split into N lists of WSI roots.

    Args:
        dataset_root (str): root of prehist-outcome WSI dataset
        val_ratio (float): validation ratio
        image_label_in_path (bool): Weither or not the image label is included in the WSI path

    Returns:
        Tuple[List[str]]: Returns a tuple of lists of WSI roots.
    """

    dataset_path = Path(dataset_root)

    if image_label_in_path:
        wsi_root_paths = [d for d in dataset_path.glob('*/*') if d.is_dir()]
    else:
        wsi_root_paths = [d for d in dataset_path.iterdir() if d.is_dir()]

    n_wsis = len(wsi_root_paths)
    random.Random(10).shuffle(wsi_root_paths)
    train_vali_split_index = round(n_wsis * (1-val_ratio))
    train_roots = wsi_root_paths[:train_vali_split_index]
    val_roots = wsi_root_paths[train_vali_split_index:]

    assert(len(train_roots) > 0 and len(val_roots) > 0)

    return train_roots, val_roots