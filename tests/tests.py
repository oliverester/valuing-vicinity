import unittest

from numpy import False_

from src.deephist.evaluate.tsne.TsneExperiment import TsneExperiment
from src.deephist.evaluate.tsne.run_tsne import run_tsne
from src.deephist.data_provider import DataProvider, split_wsi_dataset_root
from src.deephist.supervised.SupervisedExperiment import SupervisedExperiment
from src.deephist.run_experiment import run_experiment

from src.pytorch_datasets.wsi_dataset.wsi_dataset_from_folder import WSIDatasetFolder

WSI_DATASET_PATH_W_IMAGELABEL = 'tests/testdata/PDL1'
WSI_DATASET_PATH_WO_IMAGELABEL = 'tests/testdata/PDL1/bad'

class GeneralTestCase(unittest.TestCase):

    def test_wsi_loaders(self):
        wsi_dataset = WSIDatasetFolder(dataset_root=WSI_DATASET_PATH_W_IMAGELABEL,
                                       patch_label_type='patch')
        wsi = wsi_dataset.get_wsis()[0]
        print(wsi.name)
        #wsi.label_handler.lock()
        patches = wsi.get_patches()
        assert(patches[0].get_label() == 5)
        assert(patches[0].get_label(org=True) == '14')

        wsi_dataset = WSIDatasetFolder(dataset_root=WSI_DATASET_PATH_W_IMAGELABEL,
                                       patch_label_type='image')
        wsi = wsi_dataset.get_wsis()[0]
        patches = wsi.get_patches()
        assert(patches[0].get_label() == 1)
        assert(patches[0].get_label(org=True) == 'good')

        wsi_dataset = WSIDatasetFolder(dataset_root=WSI_DATASET_PATH_W_IMAGELABEL,
                                       patch_label_type='patch',
                                       include_patch_class=['9'])
        wsi = wsi_dataset.get_wsis()[0]
        patches = wsi.get_patches()

        assert(patches[0].get_label() == 0)
        assert(patches[0].get_label(org=True) == '9')
        assert(wsi.get_label() == 'good')

        wsi_dataset = WSIDatasetFolder(dataset_root=WSI_DATASET_PATH_W_IMAGELABEL,
                                       patch_label_type='patch',
                                       include_patch_class=['9'],
                                       draw_patches_per_class=10,)
        wsi = wsi_dataset.get_wsis()[0]
        patches = wsi.get_patches()
        org_labels = wsi.get_patch_labels(org=True)
        labels = wsi.get_patch_labels(org=False)

        assert(len(labels) == len(org_labels) == len(patches) == 10)
        assert(labels[0] == 0)
        assert(org_labels[0] == '9')
        
        #switch label to image-patch-label:
        wsi_dataset.set_patch_label_type("img")
        org_labels = wsi.get_patch_labels(org=True)
        labels = wsi.get_patch_labels(org=False)
        assert(labels[0] == 0)
        assert(org_labels[0] == '9')

    def test_wsi_dataset_loader(self):
        wsi_dataset_i_lbl = WSIDatasetFolder(dataset_root=WSI_DATASET_PATH_W_IMAGELABEL,
                                             patch_label_type='image')
        patch_labels = wsi_dataset_i_lbl.wsis[0].get_patch_labels()
        assert(isinstance(patch_labels[0], int))
        assert(patch_labels.count(patch_labels[0]) == len(patch_labels)) ## all labels must be equal (image label)


        wsi_dataset_p_lbl = WSIDatasetFolder(dataset_root=WSI_DATASET_PATH_WO_IMAGELABEL,
                                             patch_label_type='patch',
                                             root_contains_wsi_label=False)
        patch_labels = wsi_dataset_p_lbl.wsis[0].get_patch_labels()
        assert(isinstance(patch_labels[0], int))
        assert(patch_labels.count(patch_labels[0]) != len(patch_labels)) ## different labels (patch label)

        wsi_dataset_pi_lbl = WSIDatasetFolder(dataset_root=WSI_DATASET_PATH_W_IMAGELABEL,
                                              patch_label_type='patch&image')
        patch_labels = wsi_dataset_pi_lbl.wsis[0].get_patch_labels()
        assert(isinstance(patch_labels[0], tuple))

    def test_wsi_dataset_splitting(self):

        wsi_dataset_i_lbl = WSIDatasetFolder(dataset_root=WSI_DATASET_PATH_W_IMAGELABEL,
                                             patch_label_type='image')
        assert(len(wsi_dataset_i_lbl.wsis) == 4)

        # split dataset
        wsi_datasets = wsi_dataset_i_lbl.split_wsi_dataset_by_ratios(split_ratios=[0.5,0.5])
        splitted_patches = []
        for wsi_dataset in wsi_datasets:
            assert(len(wsi_dataset.wsis) == 2)
            splitted_patches.extend(wsi_dataset.get_patches())

        assert(len(splitted_patches) == len(wsi_dataset_i_lbl.get_patches()))

        # split files, then create WSIDatasets
        splitted_patches = []
        splitted_roots = split_wsi_dataset_root(dataset_root=WSI_DATASET_PATH_W_IMAGELABEL,
                                                val_ratio=0.5,
                                                image_label_in_path=True)
        for split_roots in splitted_roots:
            wsi_dataset = WSIDatasetFolder(wsi_roots=split_roots)
            assert(len(wsi_dataset.wsis) == 2)
            splitted_patches.extend(wsi_dataset.get_patches())

        assert(len(splitted_patches) == len(wsi_dataset_i_lbl.get_patches()))


    def test_data_provider(self):
        data_provider = DataProvider(train_data = WSI_DATASET_PATH_W_IMAGELABEL,
                                     image_label_in_path=True,
                                     batch_size=128)
        data_provider = DataProvider(train_data = WSI_DATASET_PATH_W_IMAGELABEL,
                                     image_label_in_path=True,
                                     vali_split=0.5,
                                     batch_size=128)
        
        assert(len(data_provider.holdout_set.train_wsi_dataset.get_wsis()) ==
               len(data_provider.holdout_set.vali_wsi_dataset.get_wsis()))
        assert(data_provider.holdout_set.train_wsi_dataset.get_wsis()[0].name != 
               data_provider.holdout_set.vali_wsi_dataset.get_wsis()[0].name)
        
    def test_data_provider_drawn(self):
        data_provider = DataProvider(train_data = WSI_DATASET_PATH_W_IMAGELABEL,
                                     image_label_in_path=True,
                                     vali_split=0.5,
                                     batch_size=128,
                                     include_classes=['9'],
                                     draw_patches_per_class=10)
        
        assert(len(data_provider.holdout_set.train_wsi_dataset.get_wsis()[0].get_patches()) == 10)
                

    def test_image_supervised_training(self):

        run_experiment(exp=SupervisedExperiment(config_path='tests/testconfigs/supervised_config_images_classification.yml',
                       testmode=True))
    
    def test_patch_supervised_training(self):

        run_experiment(exp=SupervisedExperiment(config_path='tests/testconfigs/supervised_config_patches_classification.yml',
                       testmode=True))

    def test_draw_patch_supervised_training(self):

        run_experiment(exp=SupervisedExperiment(config_path='tests/testconfigs/supervised_config_images_classification_draw.yml',
                       testmode=True))
 
    def test_image_cv__supervised_training(self):

        run_experiment(exp=SupervisedExperiment(config_path='tests/testconfigs/supervised_config_images_classification_cv_parallel.yml',
                       testmode=True))
    
    def test_clam_training(self):
        from src.deephist.embedding.clam.ClamExperiment import ClamExperiment
        
        run_experiment(exp=ClamExperiment(config_path='tests/testconfigs/clam_config_test.yml', testmode=True))
        
    def test_tsne(self):
        
        run_tsne(exp=TsneExperiment(config_path='tests/testconfigs/tsne_config_patchclass.yml', testmode=True))
        run_tsne(exp=TsneExperiment(config_path='tests/testconfigs/tsne_config_wsi.yml', testmode=True))
        run_tsne(exp=TsneExperiment(config_path='tests/testconfigs/tsne_config_wsidataset.yml', testmode=True))

if __name__ == "__main__":
    unittest.main()