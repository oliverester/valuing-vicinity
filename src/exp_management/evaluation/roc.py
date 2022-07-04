
import logging
from pathlib import Path

import numpy as np
import PIL
from plotnine import *
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd


def plot_roc_curve(y_true,
                   y_probas,
                   label_handler,
                   log_path: str,
                   true_class=1):
    """[summary]

    Args:
        y_true ([type]): ground truth labels
        y_probas ([type]): predicted probabilities generated by sklearn classifier
    """
    log_path = Path(log_path)
    y_probas = np.array(y_probas)
    y_true = np.array(y_true)

    auc = roc_auc_score(y_true=y_true, y_score=y_probas[:,true_class])
    logging.info(f'AUC: {auc}')   
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_probas[:,true_class])
    
    df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    p = ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed') 
    p = p + ggtitle(f'AUC for class {label_handler.decode(true_class)}: {round(auc,4)}')
    ggsave(plot = p, filename = log_path / 'tmp_auc.jpg', path = ".", verbose = False)
    image = PIL.Image.open(log_path / 'tmp_auc.jpg')
    ar = np.asarray(image)
    #conv_matrix_array = conv_matrix_array.reshape(conv_matrix_fig.canvas.get_width_height()[::-1] + (3,)) 
    ar = np.transpose(ar, (2, 0, 1)) # 3 x H x W
    return ar