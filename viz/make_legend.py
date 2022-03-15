from src.settings import COLOR_PALETTE
from src.exp_management.evaluation.plot_wsi import create_legend_img

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

legend = {-1: "none",
          0: "background",
          1: "cyst",
          2: "tumour bleeding", 
          3: "tumour necrosis", 
          4: "papillom", 
          5: "lymph node", 
          6: "diffuse tumour growth in soft tissue", 
          7: "cortex atrophy", 
          8: "tumour vital", 
          9: "extrarenal", 
          10: "cortex", 
          11: "tissue", 
          12: "tumour", 
          13: "angioinvasion", 
          14: "medullary spindel cell nodule", 
          15: "mark", 
          16: "tumour regression", 
          17: "capsule"}

#include_classes = [0, 1, 2, 3, 8, 9, 10, 13, 15, 16, 17] 
include_classes = [0, 3, 8, 9, 16, 17] 

colors = [COLOR_PALETTE[cls] for cls in include_classes]
labels = [legend[cls] for cls in include_classes]
    
legend = create_legend_img(colors=colors,
                           labels=labels,
                           loc=1,
                           dpi=400)

legend.save("legend_163.png", )