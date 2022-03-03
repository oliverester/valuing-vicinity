import unittest

import numpy as np
import torch

from src.exp_management.evaluation.dice import dice_coef, dice_denominator, dice_nominator

class GeneralTestCase(unittest.TestCase):

    def test_dice_score(self):
        
        # true images
        trues = torch.Tensor(np.array(
            [
            [[0,0,0],[0,0,0],[0,0,0]],
            [[0,0,0],[0,0,0],[0,0,0]]
            ]
                                      )
                             ).type(dtype=torch.int64)
        denom = dice_denominator(y_true=trues,y_pred=trues,n_classes=2)
        nom = dice_nominator(y_true=trues,y_pred=trues,n_classes=2)
        dice = dice_coef(dice_denominator=denom, dice_nominator=nom, n_classes=2)
        # perfect prediction -> trues == preds
        assert(dice[0] == 1)
        # dice score for class 1 is nan 
        assert(np.isnan(dice[1][1]))
        
        preds = torch.Tensor(np.array(
            [
            [[1,1,1],[1,1,1],[1,1,1]],
            [[1,1,1],[1,1,1],[1,1,1]]
            ]
                                      )
                             ).type(dtype=torch.int64)
        
        denom = dice_denominator(y_true=trues,y_pred=preds,n_classes=2)
        nom = dice_nominator(y_true=trues,y_pred=preds,n_classes=2)
        dice = dice_coef(dice_denominator=denom, dice_nominator=nom, n_classes=2)
        # perfect prediction -> trues == preds
        assert(dice[0] == 0)
        assert(dice[1][0] == 0)
        assert(dice[1][1] == 0)
        
        
        trues = torch.Tensor(np.array(
            [
            [[0,0],[0,0]],
            [[0,0],[0,0]]
            ]
                                      )
                             ).type(dtype=torch.int64)
        preds = torch.Tensor(np.array(
            [
            [[0,0],[0,0]],
            [[1,1],[1,1]]
            ]
                                      )
                             ).type(dtype=torch.int64)
         
        denom = dice_denominator(y_true=trues,y_pred=preds,n_classes=2)
        nom = dice_nominator(y_true=trues,y_pred=preds,n_classes=2)
        dice = dice_coef(dice_denominator=denom, dice_nominator=nom, n_classes=2)
        # perfect prediction -> trues == preds
        assert(dice[0] == 1/3)


if __name__ == "__main__":
    unittest.main()