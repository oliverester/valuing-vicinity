
print_labels = {
    "cyst": "Cyst",
    "tumor_bleeding": "Tumour Bleeding", 
    "tumor_necrosis": "Tumour Necrosis", 
    "tumor_vital": "Tumour Vital", 
    "extrarenal": "Extrarenal", 
    "cortex": "Cortex", 
    "angioinvasion": "Angioinvasion", 
    "mark": "Mark", 
    "tumor_regression": "Tumour Regression", 
    "capsule": "Capsule"}

from collections import defaultdict
import numpy as np
import pandas as pd
from plotnine import ggplot, scales, geom_col, aes, geom_errorbar, theme, element_text, xlab, ylab, theme_matplotlib, geom_text

from src.deephist.attention_segmentation.AttentionSegmentationExperiment import AttentionSegmentationExperiment


exp=AttentionSegmentationExperiment(testmode=True,
                                    config_path='scripts/exploration/attention_segmentation_config.yml')

wsis = exp.data_provider.wsi_dataset.wsis

area_dict = defaultdict(list)
mean_area_dict = defaultdict(int)
std_area_dict = defaultdict(int)
class_wsi_count_dict = defaultdict(int)
for wsi in wsis:
    annotations = wsi.annotations
    class_exists_in_wsi = list()
    for polygon, _, medical_label in annotations:
        # handles merged classes
        torch_label = wsi.wsi_dataset.label_handler.encode(medical_label, medical=True)
        label = wsi.wsi_dataset.label_handler.decode(torch_label)
        # 0.1729 micrometer / pixel
        size = polygon.area * (0.1729* 0.1729) / (1000*1000) # micrometer in mm
        area_dict[label].append(size)
        if label not in class_exists_in_wsi:
            class_exists_in_wsi.append(label)
    for label in class_exists_in_wsi:
        class_wsi_count_dict[label] += 1
    
for label, size_list in area_dict.items():
    mean_area_dict[label] = np.mean(size_list)
    std_area_dict[label] = np.std(size_list)
    
print(area_dict)
print(class_wsi_count_dict)

# create pandas dataframe
classes = mean_area_dict.keys()
means = [mean_area_dict[cls] for cls in classes]
stds = [std_area_dict[cls] for cls in classes]
counts = [class_wsi_count_dict[cls] for cls in classes]

print_cls = [print_labels[cls] for cls in classes]

class_dist_tbl = pd.DataFrame(list(zip(print_cls, classes, means, stds, counts)), columns=['print_cls', 'cls', 'mean', 'std', 'count'])
class_dist_tbl['cls'] = pd.Categorical(class_dist_tbl['cls'], categories=[cls for _, cls in sorted(zip(means, classes))])
class_dist_tbl['print_cls'] = pd.Categorical(class_dist_tbl['print_cls'], categories=[cls for _, cls in sorted(zip(means, print_cls))])

class_dist_tbl['error_up'] = class_dist_tbl['mean'] + class_dist_tbl['std'] 
class_dist_tbl['error_down'] = class_dist_tbl['mean'] - class_dist_tbl['std'] 

col_array = wsi.wsi_dataset.label_handler.get_color_array(type='hex')
col_dict = {wsi.wsi_dataset.label_handler.decode(idx): col for idx, col in enumerate(list(col_array))}
plot = ggplot(class_dist_tbl, aes(x='print_cls', y='mean', fill='cls')) + geom_col() + scales.scale_fill_manual(values=col_dict, type="qual", name="Tissue class", guide=False) + \
    geom_errorbar(aes(ymax='error_up', ymin='error_down')) + theme_matplotlib() + theme(axis_text_x=element_text(angle=45, hjust=1), text=element_text(family="STIXGeneral")) + \
    xlab("") + ylab("Tissue area in $\mathrm{mm}^2$") + \
    geom_text(aes(x='print_cls', y='error_up', label='count'), size=10, nudge_y=5, format_string="N: {:}") 
    
plot.save("scripts/exploration/class_distribution.png", dpi=400, width=5, height=3)


