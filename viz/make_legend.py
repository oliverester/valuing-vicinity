import sys
sys.path.append("/homes/oester/repositories/vv/")

from src.settings import COLOR_PALETTE
from src.exp_management.evaluation.plot_wsi import create_legend_img

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

legend = {-1: "none",
          0: "Background",
          1: "Cyst",
          2: "Tumour bleeding", 
          3: "Tumour necrosis", 
          4: "Papillom", 
          5: "Lymph node", 
          6: "Diffuse tumour growth in soft tissue", 
          7: "Cortex atrophy", 
          8: "Tumour vital", 
          9: "Extrarenal", 
          10: "Cortex", 
          11: "Tissue", 
          12: "Tumour", 
          13: "Angioinvasion", 
          14: "medullary spindel cell nodule", 
          15: "Mark                      ", 
          16: "Tumour regression", 
          17: "Capsule"}

# legend = {-1: "none",
#           2: "Healthy",
#           1: "Tumour"}


include_classes = [1, 2, 3, 8, 9, 10, 13, 15, 16, 17] 
# tumour
#include_classes = [13, 17, 1, 2, 3, 16, 8] 
# no tumour
include_classes = [10, 9, 15] 

#include_classes = [0, 3, 8, 9, 16, 17] 
#include_classes = [2, 1] 

colors = [COLOR_PALETTE[cls] for cls in include_classes]
labels = [legend[cls] for cls in include_classes]
    
legend = create_legend_img(colors=colors,
                           labels=labels,
                           loc="lower left",
                           dpi=400,
                           ncol=1,
                           title='Non-Tumour'
                           )

legend.save("legend_non_tumour.png", )