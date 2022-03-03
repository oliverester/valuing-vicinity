
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

from src.deephist.data_provider import DataProvider

data_provider = DataProvider(test_data='/homes/oester/repositories/prae/data/segmentation/preprocessed/HEEL_seg',
                             label_map_file='/homes/oester/repositories/prae/data/segmentation/label_map.json',
                             exclude_classes=['12'])

patch_dist_df = data_provider.test_wsi_dataset.get_patch_dist(relative=True)
fig, ax = plt.subplots(figsize=(13,len(patch_dist_df))) 
ax = sns.heatmap(patch_dist_df, annot=True, fmt = '.3f', square=1, cmap='viridis', vmax=1, linewidths=.5, cbar=False)
ax.figure.savefig("scripts/exploration/patch_distribution_rel_wo_tumor.png")
ax.figure.clf()

patch_dist_df = data_provider.test_wsi_dataset.get_patch_dist(relative=False)
fig, ax = plt.subplots(figsize=(13,len(patch_dist_df))) 
ax2 = sns.heatmap(patch_dist_df, annot=True, square=True, fmt = '.0f', cmap='viridis', 
                  vmax=patch_dist_df.loc[patch_dist_df.index != 'count_patches'][patch_dist_df.columns.difference(['n_patches', 'n_unique_patches'])].max().max(),
                  linewidths=.5, 
                  cbar=False)
ax2.figure.savefig("scripts/exploration/patch_distribution_abs_wo_tumor.png")
ax.figure.clf()
