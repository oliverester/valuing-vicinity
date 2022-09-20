"""
Loads a WSI from a folder - generated by the prehisto repository.

"""
import collections
from contextlib import contextmanager
import logging
import random
from typing import Dict, List, Tuple, Union

import geojson
import numpy as np
from pathlib import Path
from PIL import Image
from shapely.affinity import scale
from shapely.geometry import Polygon, shape, GeometryCollection
import yaml

from src.pytorch_datasets.patch.patch_from_file import PatchFromFile


class WSIFromFolder():
    """
    WSIFromFolder provides functionality to load and access
    a wsi which was preprocessed via the prehist-repo.

    """

    def __init__(self,
                 wsi_dataset,
                 root: str,
                 idx: int,
                 wsi_label: str = 'unknown',
                 embedding_path: str = None,
                 annotation_path: Path = None):
        """[summary]

        Args:
            wsi_dataset (WSIDatsetFolder): The WSI dataset the WSI belongs to
            root (str): Provide the root of the (prehist) WSI folder.
            idx (int): Unique index of WSI in WSIDataset.
            wsi_label (str, optional): Provide the WSI label. Defaults to None.
            embedding_file (str): Provide an embedding npy-file holding the WSI patch embeddings
        """
        
        self.idx = idx

        self.wsi_dataset = wsi_dataset
        self.wsi_label = wsi_label
        
        #self.wsi_dataset.image_label_handler.add_label(self.wsi_label) #register label in image label handler

        self.root_path = Path(root)
        self.name = Path(root).name
        # now load embeddings if available and add to corresponding patch
        self._embeddings = self._load_embedding(embedding_path)
        self.annotations = self._load_annoations(annotation_path)
        
        self._patches = self._create_patches()
        self._drawn_patches =  self._draw_patches()
        self.meta_data = self._load_metadata()

        self.n_patches_col = self.get_metadata('org_n_tiles_col')
        self.n_patches_row = self.get_metadata('org_n_tiles_row')
        
        # needed, when one WSI is in DataLoader
        self.embedding_dim = self.wsi_dataset.embedding_dim
        self.k_neighbours = self.wsi_dataset.k_neighbours
        
        self._patch_map, self._pad_size = self._create_patch_map(pad_size=self.k_neighbours)
        
        self.thumbnail = self._load_thumbnail()
        self._prediction = None
    
        self._all_patch_mode = False
        
        self._restricted_patches = None
        
        self.label_handler = self.wsi_dataset.label_handler
        
        self._initialize_memory()

        
    def _initialize_memory(self):
        # inference embedding memory: one wsi
        n_wsis = 1
        n_x = self.n_patches_col 
        n_y = self.n_patches_row
        n_patches = len(self._patches)
    
        self.memory_params = dict(n_x=n_x,
                                    n_y=n_y,
                                    n_w=n_wsis,
                                    n_p=n_patches,
                                    D=self.embedding_dim,
                                    k=self.k_neighbours)
        
        self.meta_data['memory'] = self.memory_params

    def _load_thumbnail(self):
        thumbnail_path = self.root_path / (self.name + "_thumbnail.png")
        if thumbnail_path.exists():
            thumbnail_img = Image.open(thumbnail_path)
            # adjust for patch ploting
            f = self.wsi_dataset.exp.args.thumbnail_correction_ratio
            tmp_size = thumbnail_img.size
            thumbnail_img = thumbnail_img.resize((round(tmp_size[0]*f), round(tmp_size[1]*f)))
        else:
            thumbnail_img = None
        return thumbnail_img
    
    def _load_annoations(self,
                         annotation_path: Path):
        
        if annotation_path is not None:
            polygons, labels = get_regions_json(path=annotation_path)
            # scale to fit wsi thumbnail (default: downscale 100x)
            scaled_polygons = [scale(
                                    poly,
                                    xfact=1 / 100 * self.wsi_dataset.exp.args.thumbnail_correction_ratio,
                                    yfact=1 / 100 * self.wsi_dataset.exp.args.thumbnail_correction_ratio,
                                    origin=(0, 0),
                                ) for poly in polygons
            ]
            # remove labels which were excluded in preprocessing:
            prehist_exclude = self.wsi_dataset.prehisto_config['exclude_classes']
            if prehist_exclude is None:
                prehist_exclude = []
            active_polygons = [(p, sp, l) for p, sp, l in zip(polygons, scaled_polygons, labels) 
                            if l not in prehist_exclude and
                            l != 'tissue']
        else:
            active_polygons = None
        return active_polygons
        
    def _load_embedding(self,
                        embedding_path: str):
        
        if embedding_path is not None:
            # load embeddings for this wsi
            embeddings, wsi_label = np.load(embedding_path, allow_pickle=True)
            # sanity check
            assert(self.wsi_label == wsi_label)
        else:
            embeddings = None
            
        return embeddings
        
    def _get_embedding(self,
                       x_coord: int,
                       y_coord: int) -> np.ndarray:
        """
        Return embeddings for given patch coordinates. Returns None if coordinates do not exit.

        Args:
            x_coord (int): x-coordinate of patch embedding
            y_coord (int): y-coordinate of patch embedding

        Returns:
            np.ndarray: Patch embedding as a numpy array
        """
        
        if self._embeddings is not None:
            if (x_coord, y_coord) in self._embeddings.keys():
                return self._embeddings[(x_coord, y_coord)]
            else:
                return None

    def get_patch_distribution(self, relative=False) -> Dict[str, int]:
        """
        Get patch label distribution as a dictionary with label as key and number of patches as value.
        Always contains 'n_unique_patches' value to count the total number of distinct patch locations.

        Returns:
            Dict[str: int]: Patch count per label
        """
        n_unique_patch_coordinates = len(set([patch.get_coordinates() for patch in self.get_patches()]))
        patch_dist = dict(collections.Counter(self.get_patch_labels(org=True)))
        n_patches = sum(patch_dist.values())
        
        if not relative:
            patch_dist = {lbl: int(count) for lbl, count in patch_dist.items()}
            patch_dist['n_patches'] = n_patches
            patch_dist['n_unique_patches'] = n_unique_patch_coordinates
        else:
            patch_dist = {lbl: round(count/n_unique_patch_coordinates,4) for lbl, count in patch_dist.items()}
            patch_dist['n_patches'] = round(n_patches/n_unique_patch_coordinates,4)
            patch_dist['n_unique_patches'] = 1
            
        return patch_dist
        
        
    def _load_metadata(self) -> Dict[str, Union[str, int]]:
        """Loads metadata from yaml file in WSI folder - if exists

        Returns:
            Dict[str, Union[str, int]]: dictionary containing all metadata from yaml file.
        """

        config_path = self.root_path / (self.name + "_metadata.yaml")
        if config_path.exists():
            with config_path.open() as stream:
                meta_data = yaml.safe_load(stream)
        else:
            meta_data = dict()
        return meta_data

    def get_metadata(self, key: str):
        """Returns value of key's WSI metadata. If key does not exists, returns None
        """

        if key in self.meta_data.keys():
            return self.meta_data[key]
        else:
            return None
        
    def _create_patch_map(self, 
                          pad_size: int) -> np.array:
        """
        Creates a 2d numpy array of size of WSI patches with the corresponding patch
        object.

        Args:
            pad_size (int): Add empty padding -  be used for k-neighbour-patch-selection

        Returns:
            np.array: 2d numpy array of WSI patches
        """
        patch_map = np.empty((self.n_patches_col, self.n_patches_row), dtype=object)
        # loop over patches and put into map by their coordinates:
        for patch in self._patches:
            x_coord, y_coord =  patch.get_coordinates()
            patch_map[x_coord, y_coord] = patch

        if pad_size is not None:
            patch_map = np.pad(patch_map, pad_size, mode='empty')
        
        return patch_map, pad_size
    
    def _create_coordinate_map(self, 
                               pad_size: int) -> np.array:
        """
        Creates a 2d numpy array of size of WSI patches with the corresponding patch
        object.

        Args:
            pad_size (int): Add empty padding -  be used for k-neighbour-patch-selection

        Returns:
            np.array: 2d numpy array of WSI patches
        """
        patch_map = np.empty((self.n_patches_col, self.n_patches_row), dtype=object)
        # loop over patches and put into map by their coordinates:
        for patch in self._patches:
            x_coord, y_coord =  patch.get_coordinates()
            patch_map[x_coord, y_coord] = patch

        patch_map = np.pad(patch_map, pad_size, mode='empty')
        
        return patch_map, pad_size
    
    def _get_k_neighbours(self,
                          x_coord, 
                          y_coord):
        # assigned b reference!
        k_neighbours, k_neighbours_mask = self._get_neighbours(x_coord, 
                                                               y_coord,
                                                               k=self._pad_size)     
        # remove self.patch
        if not self.wsi_dataset.exp.args.use_self_attention:
            k_neighbours_mask[self._pad_size, self._pad_size] = 0
                
        return k_neighbours, k_neighbours_mask
    
    def _get_neighbours(self,
                        x_coord, 
                        y_coord,
                        k):
        # watch out! map has a 0-padding of border size self._pad_size to avoid out of array selection
        # so we first shift coords by pad_size, then we adjust for k.
        k_neighbours = self._patch_map[max((x_coord+self._pad_size-k),0):min((x_coord+self._pad_size+k+1),self._patch_map.shape[0]),
                                       max((y_coord+self._pad_size-k),0):min((y_coord+self._pad_size+k+1),self._patch_map.shape[1])
                                       ]
        # pad in case of a large "k"
        pad_left = max(-(x_coord+self._pad_size-k),0)
        pad_right = max((x_coord+self._pad_size+k+1)-self._patch_map.shape[0],0) 
        pad_top = max(-(y_coord+self._pad_size-k),0)
        pad_bottom = max((y_coord+self._pad_size+k+1)-self._patch_map.shape[1],0)
        
        if (pad_left + pad_right + pad_top + pad_bottom) > 0:
            k_neighbours = np.pad(k_neighbours,((pad_left, pad_right),(pad_top, pad_bottom)),mode='empty')
        
        k_neighbours_mask = (k_neighbours != None).astype(int)          
        return k_neighbours, k_neighbours_mask
        

    def _create_patches(self) -> List[PatchFromFile]:
        """
        Creates the patches objects for the wsi.

        Returns:
            List[PatchFromFile]: [description]
        """
        patches = list()
        # select patch files
        patches_files = [f for f in self.root_path.glob('*/*')
                         if f.is_file() and 
                         f.suffix in ['.png', '.jpg'] and
                         'context' not in str(f.name)]
        
        for patch_file in patches_files:
            
            x_coord, y_coord, patch_label, corresponding_wsi_name = PatchFromFile.parse_file(file_path=patch_file)
            if self.wsi_dataset.patch_label_type != 'mask':
                if self.wsi_dataset.exclude_patch_class:
                    if patch_label in self.wsi_dataset.exclude_patch_class:
                        continue
                if self.wsi_dataset.include_patch_class:
                    if patch_label not in self.wsi_dataset.include_patch_class:
                        continue

            # sanity check: patch file must start with WSI name
            assert(corresponding_wsi_name == self.name)
            
            if self._embeddings is not None:
                embedding = self._get_embedding(x_coord=x_coord,
                                                y_coord=y_coord)
                if embedding is None:
                    raise Exception("Cannot find embedding for patch "
                                    f" with x {x_coord}, y {y_coord}, label {patch_label}")
            else:
                embedding = None
                
            if self.wsi_dataset.patch_label_type == 'mask':
                mask_file = self.root_path / 'masks' / f"{corresponding_wsi_name}_{y_coord}_{x_coord}_mask.npy"
                if not mask_file.exists():
                    raise Exception("Cannot find masks for patch "
                                    f" with x {x_coord}, y {y_coord}")
                ##mask = np.load(mask_file)
            else:
                #mask = None
                mask_file = None
                
            # get metadata file for patch    
            metadata_file = self.root_path / 'metadata' / f"{corresponding_wsi_name}_{y_coord}_{x_coord}_metadata.yml"
            if not metadata_file.exists():
                metadata_file = None

            patches.append(PatchFromFile(file_path=patch_file,
                                         x_coord=x_coord,
                                         y_coord=y_coord,
                                         org_label=patch_label,
                                         embedding=embedding,
                                         mask_path=mask_file,
                                         metadata_file=metadata_file,
                                         #mask=mask,
                                         wsi=self
                                         )
                           )
        return patches
    
    
    def _draw_patches(self) -> List[PatchFromFile]:
        if self.wsi_dataset.draw_patches_per_class is not None:
            draw_idxs = []
            logging.getLogger('exp').info(f"Drawing {self.wsi_dataset.draw_patches_per_class} random patches per patch "
                  "class without repetition.")
            patch_labels = [ptc.org_label for ptc in self._patches]
            for tmp_lbl in set(patch_labels):
                tmp_lbl_idxs = [idx for idx, lbl in enumerate(patch_labels) if tmp_lbl == lbl]
                # in case the draw size is equal or above the label occurence, select all
                if self.wsi_dataset.draw_patches_per_class >= len(tmp_lbl_idxs):
                    logging.getLogger('exp').info(f"Warning: draw_patches_per_class ({self.wsi_dataset.draw_patches_per_class})"
                          f"exceeds patches with class {tmp_lbl} for wsi {self.name} ({len(tmp_lbl_idxs)})")
                    draw_idxs.extend(tmp_lbl_idxs)
                else:
                    tmp_draw = random.sample(tmp_lbl_idxs, self.wsi_dataset.draw_patches_per_class)
                    draw_idxs.extend(tmp_draw)
            drawn_patches = [self._patches[idx] for idx in draw_idxs]
            
        elif self.wsi_dataset.draw_patches_per_wsi is not None:
            draw_idxs = random.sample(range(len(self._patches)), self.wsi_dataset.draw_patches_per_wsi)
            drawn_patches = [self._patches[idx] for idx in draw_idxs]
        else:
            drawn_patches = self._patches
        return drawn_patches
    
    def get_patches(self) -> List[PatchFromFile]:
        if self._restricted_patches is not None:
            return self._restricted_patches 
        else:
            if self._all_patch_mode:
                return self._patches
            else:
                return self._drawn_patches
        
    def get_patch_from_position(self, x, y) -> PatchFromFile:
        # shifted because of border 
        return self._patch_map[x+self._pad_size,y+self._pad_size]
    
    def get_patch_predictions(self) -> List[Tuple[float]]:
        return [ptc.get_prediction() for ptc in self.get_patches()]   
        
    def get_embeddings(self) -> List[np.ndarray]:
        return [ptc.embedding for ptc in self.get_patches()]
        
    def get_patch_labels(self, org=False) -> List[int]:
        return [ptc.get_label(org=org) for ptc in self.get_patches()]   
          
    def get_label(self, org=True) -> str:
        """Returns original (medical) image label

        Returns:
            str: Medical original image label
        """
        if org: 
            return self.wsi_label
        else:
            return self.wsi_dataset.image_label_handler.encode(self.wsi_label)

    def set_prediction(self, prediction):
        self._prediction = prediction
        
    def get_prediction(self):
        """
        Pool patch predictions to one wsi prediction. Consider all_patch_mode context manager to include all or drawn patches
        """
        if self._prediction is None:
            n_classes = len(self.label_handler.classes)
            n_patches = len(self.get_patches())
            predictions = np.zeros(shape=(n_patches, n_classes))

            for i, patch in enumerate(self.get_patches()):
                predictions[i,:] = patch.get_prediction()

            prediction = np.average(predictions, axis=0)
        else:
            prediction = self._prediction
        return prediction
        
    def get_pred_dict(self):
        return {self.label_handler.decode(pytorch_class): round(pred.item(),4)
                for pytorch_class, pred in enumerate(self.get_prediction())}
        
    @contextmanager
    def all_patch_mode(self):
        tmp_patch_mode = self._all_patch_mode
        self._all_patch_mode = True
        yield(self)
        self._all_patch_mode = tmp_patch_mode
        
    @contextmanager
    def inference_mode(self):
        tmp_idx = self.idx
        self.idx = 0 # needed for memory wsi idx
        tmp_patch_mode = self._all_patch_mode
        self._all_patch_mode = True
        #import: set each wsi index to 0 - dataloader holds one WSI only in each interation
        if self.wsi_dataset.attention_on:
            if self.k_neighbours is not None:
                self._patch_map, self._pad_size = self._create_patch_map(pad_size=self.k_neighbours)
        yield(self)
        self.idx = tmp_idx
        self._all_patch_mode = tmp_patch_mode
        
    @contextmanager
    def restrict_patches(self, patches):
        self._restricted_patches = patches
        yield(self)
        self._restricted_patches = None
        
    @contextmanager
    def patch_label_mode(self, label_type):
        """Set context to receive all patches from wsi (instead of sampled ones).
        """
        assert(label_type in ['patch', 'image', 'mask', 'distribution'])

        default_type = self.wsi_dataset.patch_label_type
        self.wsi_dataset.patch_label_type = label_type
        yield(self)
        self.wsi_dataset.patch_label_type = default_type

    
def get_regions_json(path: Union[str, Path],
                     exclude_classes: List[str] = None) -> Tuple[List[Polygon], List[str]]:

    """
    Parses a JSON file (as obtained by QuPath) and returns the stored polygons and their labels.

    :param path: The path where the XML file is located
    :type path: Union[str, Path]
    :param exclude_classes: A list of annotation classes to be excluded, default to None.
    :type exclude_classes: str, optional
    :return: A list of polygons (each described as an ndarray) and a list of their corresponding
    labels
    :rtype: Tuple[List[Polygon], List[str]]
    """
    with open(path) as f:
        gj = geojson.load(f)
    region_labels = []
    all_geometries = list(GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in gj]))
    
    geometries = []
    # build region labels from geometries: 
    for feature, geom in zip(gj, all_geometries):
        if 'classification' in feature.properties:
            label = feature.properties['classification']['name'].lower()
            if exclude_classes is not None and label in exclude_classes:
                continue
            region_labels.append(label)
        else:
            region_labels.append("tumor")  
        geometries.append(geom)
    
    return geometries, region_labels
