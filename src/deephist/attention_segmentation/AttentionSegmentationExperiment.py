
from collections import defaultdict
from copy import copy
from pathlib import Path

from fastai.losses import DiceLoss, FocalLoss
from PIL import Image
import numpy as np
from torch import nn
from typing import Dict, List, Tuple
import torch
from torchvision import transforms

from src.deephist.attention_segmentation.AttentionPatchesDataset import AttentionPatchesDataset
from src.deephist.attention_segmentation.models.attention_segmentation_model import AttentionSegmentationModel
from src.deephist.attention_segmentation.train_epoch import train_epoch
from src.deephist.attention_segmentation.attention_segmentation_parser import AttentionSegmentationConfig
import src.deephist.semantic_segmentation.train_epoch as train_semantic
import src.deephist.multiscale_segmentation.train_epoch as train_multiscale
from src.deephist.multiscale_segmentation.MultiscalePatchesDataset import MultiscalePatchesDataset
from src.deephist.multiscale_segmentation.multiscale.models.YclassRes18Net import YclassRes18Net
from src.deephist.multiscale_segmentation.multiscale.models.Ymm2classRes18Net import Ymm2classRes18Net
from src.deephist.data_provider import DataProvider
from src.deephist.CustomPatchesDataset import CustomPatchesDataset
from src.exp_management import tracking
from src.exp_management.MLExperiment import MLExperiment
from src.exp_management.evaluation.confusion_matrix import torch_conf_matrix
from src.exp_management.evaluation.dice import dice_coef, dice_denominator, dice_nominator
from src.exp_management.evaluation.jaccard import jaccard, jaccard_denominator, jaccard_nominator
from src.exp_management.evaluation.precision_recall import positives, precision, pred_positives, recall, true_positives
from src.exp_management.losses import CombinedLoss, FocalTverskyLoss
from src.pytorch_datasets.label_handler import LabelHandler
from src.pytorch_datasets.wsi.wsi_from_folder import WSIFromFolder
from src.settings import get_class_weights



class AttentionSegmentationExperiment(MLExperiment):
    

    def __init__(self, config_path, testmode=False):
        super().__init__(config_path=config_path,
                         config_parser=AttentionSegmentationConfig,
                         prefix='attention_segmentation',
                         testmode=testmode)

        self.data_provider = self.get_data_provider()
        self.set_model()
    
    def get_data_provider(self):
        args = self.args
        
        assert(not(args.attention_on and args.multiscale_on)) , 'Either attention mode or multiscale mode can be activated.'
        
        if args.attention_on:
            dataset_type = AttentionPatchesDataset
            collate_fn = self.collate_neighbour_patches
        elif args.multiscale_on:
            dataset_type = MultiscalePatchesDataset
            collate_fn = self.collate_multiscale_patches            
        else:
            dataset_type = CustomPatchesDataset
            collate_fn = None # torch default collate
            
        data_provider = DataProvider(train_data=args.train_data,
                                     test_data=args.test_data,
                                     image_label_in_path=args.image_label_in_path,
                                     patch_sampling=args.patch_sampling,
                                     patch_label_type=args.patch_label_type,
                                     vali_split=args.vali_split,
                                     nfold=args.nfold,
                                     exclude_classes=args.exclude_classes,
                                     include_classes=args.include_classes,
                                     merge_classes=args.merge_classes,
                                     draw_patches_per_class=args.draw_patches_per_class,
                                     draw_patches_per_wsi=args.draw_patches_per_wsi,
                                     normalize=args.normalize,
                                     hue_aug_ratio=args.hue_aug_ratio,
                                     label_map_file=args.label_map_file,
                                     batch_size=args.batch_size,
                                     val_batch_size=args.val_batch_size,
                                     test_batch_size=args.test_batch_size,
                                     overlay_polygons=args.overlay_polygons,
                                     workers=args.workers,
                                     gpu=args.gpu,
                                     dataset_type=dataset_type,
                                     collate_fn=collate_fn,
                                     attention_on=args.attention_on,
                                     embedding_dim=args.embedding_dim,
                                     k_neighbours=args.k_neighbours,
                                     multiscale_on=args.multiscale_on,
                                     exp=self)

        self.args.number_of_classes = data_provider.number_classes
        return data_provider
    
    def get_criterion(self):
        """Basic criterion

        Returns:
            [type]: [description]
        """
        if self.args.criterion is None or self.args.criterion == 'cross_entropy':
            # basic cross entropy
            if self.args.use_ce_weights:
                weights =  get_class_weights(label_handler=self.data_provider.label_handler)
                criterion = super().get_criterion(weight=weights)
            else:
                criterion = super().get_criterion()
            return criterion
        elif self.args.criterion == 'dice':
            return DiceLoss(reduction='mean')
        elif self.args.criterion == 'focal':
            return FocalLoss(gamma=self.args.gamma) 
        elif self.args.criterion == 'focal_tversky':
            return FocalTverskyLoss(alpha=self.args.alpha, 
                                    beta=self.args.beta, 
                                    gamma=self.args.gamma)
        elif self.args.criterion == 'ce+dice':
            return CombinedLoss(l1=super().get_criterion(),
                                l2=DiceLoss(reduction='mean'),
                                weight2=self.args.combine_weight,
                                add_l2_at_epoch=self.args.combine_criterion_after_epoch) 
        elif self.args.criterion == 'focal+dice':
            return CombinedLoss(l1=DiceLoss(reduction='mean'),
                                l2= FocalLoss(gamma=self.args.gamma),
                                weight2=self.args.combine_weight,
                                add_l2_at_epoch=self.args.combine_criterion_after_epoch) 
        elif self.args.criterion == 'focal+focal_tversky':
            return CombinedLoss(l1=FocalLoss(gamma=self.args.gamma),
                                l2= FocalTverskyLoss(gamma=self.args.gamma,
                                                        alpha=self.args.alpha, 
                                                        beta=self.args.beta),
                                weight2=self.args.combine_weight,
                                add_l2_at_epoch=self.args.combine_criterion_after_epoch)     
        else:
            raise Exception("Criterion is invalid")
            

    
    def set_model(self):
        if not self.args.use_self_attention and self.args.k_neighbours == 0:
            raise Exception("If no self-attention than 0 neighbours is forbidden.")

        if self.args.multiscale_on:
            cfg = {
                    'num_classes': self.args.number_of_classes,
                    'num_channels': 3,
                    'activation_function': 'ReLU',
                    'num_base_featuremaps': 64,
                    'encoder_featuremap_delay': 2,
                    'decoder_featuremaps_out': [512, 256, 256, 128, -1],
                    'conv_norm_type': 'None',
                    'depth_levels_down_main': [[2, 3], [0, 3], [0, 1], [0, 1], [0, 1]],  # [Convs,Res] each
                    'depth_levels_down_side': [[2, 3], [0, 3], [0, 1], [0, 1], [0, 1]],  # [Convs,Res] each
                    'depth_levels_down_tail': [[2, 3], [0, 3], [0, 1], [0, 1], [0, 1]],  # [Convs,Res] each
                    'depth_levels_up': [1, 1, 1, 1, 1],  # Convs
                    'depth_bottleneck': [0, 0, 0],  # [Conv,Res,Conv]
                    'internal_prediction_activation': 'None',  # Softmax, Sigmoid or None. None for use with BCEWithLogitsLoss etc.
                    'gpu': self.args.gpu,
                    'scale_list': [4],
                    'patch_size': 256
                }
            self.model = YclassRes18Net(cfg=cfg)
        else:
            self.model = AttentionSegmentationModel(arch=self.args.arch,
                                                    encoder_name=self.args.encoder,
                                                    encoder_weights='imagenet',
                                                    number_of_classes=self.args.number_of_classes,
                                                    attention_input_dim=self.args.embedding_dim,
                                                    attention_hidden_dim=self.args.attention_hidden_dim,
                                                    num_attention_heads=self.args.num_attention_heads,
                                                    k=self.args.k_neighbours,
                                                    use_ln=self.args.use_ln,
                                                    use_pos_encoding=self.args.use_pos_encoding,
                                                    use_central_attention=self.args.use_self_attention,
                                                    learn_pos_encoding=self.args.learn_pos_encoding,
                                                    attention_on=self.args.attention_on
                                                    )
    
    def run_train_vali_epoch(self,
                             holdout_set,
                             model,
                             criterion,
                             optimizer,
                             label_handler,
                             epoch,
                             writer,
                             args,
                             **kwargs):
        
        if self.args.attention_on:
            new_performance = train_epoch(holdout_set=holdout_set,
                                          model=model,
                                          criterion=criterion,
                                          optimizer=optimizer,
                                          label_handler=label_handler,
                                          epoch=epoch,
                                          args=args,
                                          writer=writer)
        elif self.args.multiscale_on:
            new_performance = train_multiscale.train_epoch(holdout_set=holdout_set,
                                                           model=model,
                                                           criterion=criterion,
                                                           optimizer=optimizer,
                                                           label_handler=label_handler,
                                                           epoch=epoch,
                                                           args=args,
                                                           writer=writer)
        else:
            new_performance = train_semantic.train_epoch(holdout_set=holdout_set,
                                                         model=model,
                                                         criterion=criterion,
                                                         optimizer=optimizer,
                                                         label_handler=label_handler,
                                                         epoch=epoch,
                                                         args=args,
                                                         writer=writer)
            
        return new_performance
    
    def set_predictions(self,
                        wsis: List[WSIFromFolder],
                        model: nn.Module,
                        data_provider: DataProvider,
                        gpu: int):
        
        if self.args.attention_on: 
            inference_fun = attention_inference
        elif self.args.multiscale_on:
            from src.deephist.multiscale_segmentation.multiscale_inference import do_inference
            inference_fun = do_inference
        else:
            # basic inference
            from src.deephist.run_experiment import do_inference
            inference_fun = do_inference
            
        for wsi in wsis:
            
            # free space on gpu
            torch.cuda.empty_cache()
            wsi_dice_nominator = 0
            wsi_dice_denominator = 0
            
            wsi_jaccard_nominator = 0
            wsi_jaccard_denominator = 0
            
            wsi_true_positives = 0
            wsi_positives = 0
            wsi_pred_positives = 0
            
            conf_matrix = 0
            
            patch_counter = 0
            
            with wsi.inference_mode():
                patches = wsi.get_patches()
                print(f"Inference for WSI {wsi.name}")       
                wsi_loader = data_provider.get_wsi_loader(wsi=wsi)

                mask_predictions, masks = inference_fun(wsi_loader,
                                                        model,
                                                        gpu,
                                                        out='torch',
                                                        args=self.args)
                    
                assert(len([mask for mask_batch in mask_predictions for mask in mask_batch]) == len(wsi.get_patches()))

                # determine Dice nominator and denominator per batch. Later, calc Dice score
                for mask_batch, mask_pred_batch in zip(masks, mask_predictions):
                    # move to cuda 
                    mask_batch = mask_batch.cuda(gpu, non_blocking=True)
                    mask_pred_batch = mask_pred_batch.cuda(gpu, non_blocking=True)
                    
                    mask_argmax_batch = torch.argmax(mask_pred_batch, dim=1)
                    # dice score
                    wsi_dice_nominator += dice_nominator(y_true=mask_batch,
                                                        y_pred=mask_argmax_batch,
                                                        n_classes=data_provider.number_classes)
                    
                    wsi_dice_denominator += dice_denominator(y_true=mask_batch,
                                                            y_pred=mask_argmax_batch,
                                                            n_classes=data_provider.number_classes)
                    
                    # jaccard index
                    wsi_jaccard_nominator += jaccard_nominator(y_true=mask_batch,
                                                            y_pred=mask_argmax_batch,
                                                            n_classes=data_provider.number_classes)
                    
                    wsi_jaccard_denominator += jaccard_denominator(y_true=mask_batch,
                                                                y_pred=mask_argmax_batch,
                                                                n_classes=data_provider.number_classes)
                    
                    # confusion matrix:
                    conf_matrix += torch_conf_matrix(y_true=mask_batch,
                                                    y_pred=mask_argmax_batch,
                                                    n_classes=data_provider.number_classes)
                    # precision / recall
                    wsi_positives += positives(y_true=mask_batch,
                                            n_classes=data_provider.number_classes)
                    wsi_pred_positives += pred_positives(y_pred=mask_argmax_batch,
                                                        n_classes=data_provider.number_classes)
                    wsi_true_positives += true_positives(y_true=mask_batch,
                                                        y_pred=mask_argmax_batch,
                                                        n_classes=data_provider.number_classes)
                    
                    true_positive_mask_batch = mask_batch == mask_argmax_batch
                    
                    for mask_batch, mask_output, true_positive_mask in zip(mask_batch, mask_argmax_batch, true_positive_mask_batch):
                        # seg mask pred
                        img = self.mask_to_img(mask=mask_output.cpu(),
                                            label_handler=data_provider.label_handler)
                        patches[patch_counter].set_prediction(img)
                        
                        # seg true mask
                        true_img = self.mask_to_img(mask=mask_batch.cpu(),
                                                    label_handler=data_provider.label_handler)
                        patches[patch_counter].true_mask = true_img
                        
                        # heatmap
                        heatmap = self.mask_to_img(mask=true_positive_mask.cpu(),
                                                binary=True)
                        patches[patch_counter].heatmap = heatmap
                        
                        patch_counter += 1
                        
                assert(patch_counter == len(wsi.get_patches()))
                        
                wsi.conf_matrix = conf_matrix
                                
                # determine WSI-Dice score
                wsi.dice_score, wsi.dice_score_per_class = dice_coef(dice_nominator=wsi_dice_nominator,
                                                                     dice_denominator=wsi_dice_denominator,
                                                                     n_classes=data_provider.number_classes)
                
                wsi.jaccard_score, wsi.jaccard_score_per_class = jaccard(jaccard_nominator=wsi_jaccard_nominator,
                                                                         jaccard_denominator=wsi_jaccard_denominator,
                                                                         n_classes=data_provider.number_classes)
                        
                # determine WSI-precision score
                wsi.precision_score, wsi.precision_score_per_class = precision(true_positives=wsi_true_positives,
                                                                               pred_positives=wsi_pred_positives,
                                                                               n_classes=data_provider.number_classes)
                
                # determine WSI-recall score
                wsi.recall_score, wsi.recall_score_per_class = recall(true_positives=wsi_true_positives,
                                                                      positives=wsi_positives,
                                                                      n_classes=data_provider.number_classes)
                                
     
    def mask_to_img(self,
                    mask: np.ndarray,
                    label_handler: LabelHandler = None,
                    org_size: bool = False,
                    binary: bool = False):
        
        if not binary:
            color_array = label_handler.get_color_array()
        else:
            #binary:
            color_array = np.array([(0.705, 0.015, 0.149),(0.349, 0.466, 0.890)])
            
        mask_img = Image.fromarray((color_array[mask.int()]*255).astype(np.uint8))
        # scale to patch size which fits the thumbnail of wsi (1/100 of org size)
        # therefore, first scale up to org patch size, then downsample by 100
        if not org_size:
            prehisto_downsample = self.data_provider.prehisto_config['downsample']
            target_size= round(mask.shape[0] * prehisto_downsample / 100), \
                        round(mask.shape[1] * prehisto_downsample / 100)
               
            mask_img = mask_img.resize(target_size, Image.ANTIALIAS)
        return mask_img


    def evaluate_wsis(
        self,
        wsis: List[WSIFromFolder],
        data_provider: DataProvider,
        log_path: Path,
        epoch: int = None,
        writer = None,
        save_to_folder = False,
        tag: str = ""
        ) -> List[Dict]:
        """
        Run patch inference and returns WSI predictions

        Args:
            config_path (str): Path to yaml-config in PatchInferenceConfig style
        """

        viz = tracking.Visualizer(writer=writer,
                                  save_to_folder=save_to_folder)

        # Dice
        wsi_dice_scores = list()
        wsi_dice_scores_per_class = list()
        per_class_dice = list()
        
        wsi_dice_scores_dict = dict()
        wsi_dice_scores_per_class_dict = dict()

        #Jaccard
        wsi_jaccard_scores = list()
        wsi_jaccard_scores_per_class = list()
        per_class_jaccard = list()
        
        wsi_jaccard_scores_dict = dict()
        wsi_jaccard_scores_per_class_dict = dict()

        # Precision
        wsi_precision_scores = list()
        wsi_precision_scores_per_class = list()
        per_class_precision = list()
        
        wsi_precision_scores_dict = dict()
        wsi_precision_scores_per_class_dict = dict()

        # Recall
        wsi_recall_scores = list()
        wsi_recall_scores_per_class = list()
        per_class_recall = list()
        
        wsi_recall_scores_dict = dict()
        wsi_recall_scores_per_class_dict = dict()
        
        global_conf_matrix = 0
        for wsi in wsis:
            print(f"Evaluating WSI {wsi.name}")

            viz.wsi_plot(tag=tag + "_wsi",
                         mode='truewsi+wsi+heatmap+thumbnail',
                         wsi=wsi,
                         log_path=log_path,
                         epoch=epoch)
            
            viz.confusion_matrix_img(tag=tag + f"_conf_matrix/conf_matrix_{wsi.name}",
                                     confm=wsi.conf_matrix,
                                     label_handler=data_provider.label_handler,
                                     epoch=epoch,
                                     fz=8)
            
            global_conf_matrix += wsi.conf_matrix
            
            # collect scores over all WSIs
            wsi_dice_scores.append(wsi.dice_score)
            wsi_dice_scores_per_class.extend(wsi.dice_score_per_class.values())
            per_class_dice.extend(wsi.dice_score_per_class.keys())
           
            wsi_jaccard_scores.append(wsi.jaccard_score)
            wsi_jaccard_scores_per_class.extend(wsi.jaccard_score_per_class.values())
            per_class_jaccard.extend(wsi.jaccard_score_per_class.keys())
             
            wsi_precision_scores.append(wsi.precision_score)
            wsi_precision_scores_per_class.extend(wsi.precision_score_per_class.values())
            per_class_precision.extend(wsi.precision_score_per_class.keys())
            
            wsi_recall_scores.append(wsi.recall_score)
            wsi_recall_scores_per_class.extend(wsi.recall_score_per_class.values())
            per_class_recall.extend(wsi.recall_score_per_class.keys())
            
            # Dice 
            wsi_dice_scores_per_class_dict[wsi.name] = {data_provider.label_handler.decode(cls): round(score,4) for
                                                        cls, score in wsi.dice_score_per_class.items()}
            wsi_dice_scores_dict[wsi.name] = round(wsi.dice_score, 4)
            
            # Jaccard 
            wsi_jaccard_scores_per_class_dict[wsi.name] = {data_provider.label_handler.decode(cls): round(score,4) for
                                                        cls, score in wsi.jaccard_score_per_class.items()}
            wsi_jaccard_scores_dict[wsi.name] = round(wsi.jaccard_score, 4)
            
            # Precision
            wsi_precision_scores_per_class_dict[wsi.name] = {data_provider.label_handler.decode(cls): round(score,4) for
                                                        cls, score in wsi.precision_score_per_class.items()}
            wsi_precision_scores_dict[wsi.name] = round(wsi.precision_score, 4)
            
            # Recall 
            wsi_recall_scores_per_class_dict[wsi.name] = {data_provider.label_handler.decode(cls): round(score,4) for
                                                        cls, score in wsi.recall_score_per_class.items()}
            wsi_recall_scores_dict[wsi.name] = round(wsi.recall_score, 4)
            
            
            print(f"{wsi.name} dice scores: {wsi.dice_score_per_class} ({wsi.dice_score})")
        
        #tbd: normalize to 100 % actual per class
        viz.confusion_matrix_img(tag=tag + f"_conf_matrix/conf_matrix_all",
                                 confm=global_conf_matrix,
                                 label_handler=data_provider.label_handler,
                                 epoch=epoch,
                                 fz=8)
        
        # plot wsi dice score distribution and per class
        viz.score_hist(tag=tag + "_wsi_dice_per_class",
                       score_value=wsi_dice_scores_per_class,
                       score_name='dice_score',
                       label_handler=data_provider.label_handler,
                       log_path=log_path,
                       score_classes=per_class_dice,
                       epoch=epoch)
        
        viz.score_hist(tag=tag + "_wsi_dice",
                       score_value=wsi_dice_scores,
                       score_name='dice_score',
                       label_handler=data_provider.label_handler,
                       log_path=log_path,
                       epoch=epoch)
        
        assert(per_class_precision == per_class_recall) # ensure scores are ordered in same class sequence!
        viz.score_2dhist(tag=tag + "_wsi_precision_recall_per_class",
                         score_value1=wsi_precision_scores_per_class,
                         score_name1='precision',
                         score_value2=wsi_recall_scores_per_class,
                         score_name2='recall',
                         label_handler=data_provider.label_handler,
                         log_path=log_path,
                         score_classes=per_class_precision,
                         epoch=epoch)
        
        viz.score_2dhist(tag=tag + "_wsi_precision_recall",
                         score_value1=wsi_precision_scores,
                         score_name1='precision',
                         score_value2=wsi_recall_scores,
                         score_name2='recall',
                         label_handler=data_provider.label_handler,
                         log_path=log_path,
                         epoch=epoch)
        
        # Dice 
        
        # logging of mean dice score per class        
        dice_grouped = defaultdict(list)
        for k, v in zip(per_class_dice, wsi_dice_scores_per_class):
            dice_grouped[k].append(v)
            
        mean_dice_scores_per_class = {data_provider.label_handler.decode(a): round(np.nanmean(b).item(),4) for a, b in dice_grouped.items()}
        std_dice_scores_per_class = {data_provider.label_handler.decode(a): round(np.nanstd(b).item(),4) for a, b in dice_grouped.items()}
        
        wsi_mean_dice_score = round(np.nanmean(list(wsi_dice_scores_dict.values())).item(),4)
        wsi_std_dice_score = round(np.nanstd(list(wsi_dice_scores_dict.values())).item(),4)
        class_mean_dice_score = round(np.nanmean(list(mean_dice_scores_per_class.values())).item(), 4)
        class_std_dice_score = round(np.nanstd(list(mean_dice_scores_per_class.values())).item(), 4)
        
        log_wsi_preds = dict()
        log_wsi_preds['epoch'] = epoch
        # per Wsi: (class-wise)mean Dice Score
        log_wsi_preds['wsi_dice_score'] = wsi_dice_scores_dict
        # per Wsi: class Dice Score
        log_wsi_preds['wsi_dice_score_per_class'] = wsi_dice_scores_per_class_dict

        # all Wsis: (wsi-wise-)mean class Dice Score
        log_wsi_preds['mean_dice_scores_per_class'] = mean_dice_scores_per_class
        # all Wsis: (wsi-wise-)std class Dice Score
        log_wsi_preds['std_dice_scores_per_class'] = std_dice_scores_per_class
        
        # wsi-mean of class-wise-mean Dice Score
        log_wsi_preds['wsi_mean_dice_scores'] = wsi_mean_dice_score
        log_wsi_preds['wsi_std_dice_scores'] = wsi_std_dice_score
        # class-mean of wsi-wise-mean Dice Score
        log_wsi_preds['class_mean_dice_scores'] = class_mean_dice_score
        log_wsi_preds['class_std_dice_scores'] = class_mean_dice_score
        
        # Jaccard 
        
        # logging of mean dice score per class        
        jaccard_grouped = defaultdict(list)
        for k, v in zip(per_class_jaccard, wsi_jaccard_scores_per_class):
            jaccard_grouped[k].append(v)
            
        mean_jaccard_scores_per_class = {data_provider.label_handler.decode(a): round(np.nanmean(b).item(),4) for a, b in jaccard_grouped.items()}
        std_jaccard_scores_per_class = {data_provider.label_handler.decode(a): round(np.nanstd(b).item(),4) for a, b in jaccard_grouped.items()}
        
        wsi_mean_jaccard_score = round(np.nanmean(list(wsi_jaccard_scores_dict.values())).item(),4)
        wsi_std_jaccard_score = round(np.nanstd(list(wsi_jaccard_scores_dict.values())).item(),4)
        class_mean_jaccard_score = round(np.nanmean(list(mean_jaccard_scores_per_class.values())).item(),4)
        class_std_jaccard_score = round(np.nanstd(list(mean_jaccard_scores_per_class.values())).item(),4)
        
        # per Wsi: (class-wise)mean jaccard Score
        log_wsi_preds['wsi_jaccard_score'] = wsi_jaccard_scores_dict
        # per Wsi: class jaccard Score
        log_wsi_preds['wsi_jaccard_score_per_class'] = wsi_jaccard_scores_per_class_dict

        # all Wsis: (wsi-wise-)mean class jaccard Score
        log_wsi_preds['mean_jaccard_scores_per_class'] = mean_jaccard_scores_per_class
        # all Wsis: (wsi-wise-)std class jaccard Score
        log_wsi_preds['std_jaccard_scores_per_class'] = std_jaccard_scores_per_class
        
        # wsi-mean of class-wise-mean jaccard Score
        log_wsi_preds['wsi_mean_jaccard_scores'] = wsi_mean_jaccard_score
        log_wsi_preds['wsi_std_jaccard_scores'] = wsi_std_jaccard_score
        # class-mean of wsi-wise-mean jaccard Score
        log_wsi_preds['class_mean_jaccard_scores'] = class_mean_jaccard_score
        log_wsi_preds['class_std_jaccard_scores'] = class_std_jaccard_score
        
        # Precision
        
        # logging of mean precision score per class        
        precision_grouped = defaultdict(list)
        for k, v in zip(per_class_precision, wsi_precision_scores_per_class):
            precision_grouped[k].append(v)
            
        mean_precision_scores_per_class = {data_provider.label_handler.decode(a): round(np.nanmean(b).item(),4) for a, b in precision_grouped.items()}
        std_precision_scores_per_class = {data_provider.label_handler.decode(a): round(np.nanstd(b).item(),4) for a, b in precision_grouped.items()}
        
        wsi_mean_precision_score = round(np.nanmean(list(wsi_precision_scores_dict.values())).item(),4)
        wsi_std_precision_score = round(np.nanstd(list(wsi_precision_scores_dict.values())).item(),4)
        class_mean_precision_score = round(np.nanmean(list(mean_precision_scores_per_class.values())).item(),4)
        class_std_precision_score = round(np.nanstd(list(mean_precision_scores_per_class.values())).item(),4)
        
        # per Wsi: (class-wise)mean precision
        log_wsi_preds['wsi_precision'] = wsi_precision_scores_dict
        # per Wsi: class Dice Score
        log_wsi_preds['wsi_precision_per_class'] = wsi_precision_scores_per_class_dict

        # all Wsis: (wsi-wise-)mean class precision
        log_wsi_preds['mean_precision_per_class'] = mean_precision_scores_per_class
        # all Wsis: (wsi-wise-)std class precision
        log_wsi_preds['std_precision_per_class'] = std_precision_scores_per_class
        
        # wsi-mean of class-wise-mean precision
        log_wsi_preds['wsi_mean_precision'] = wsi_mean_precision_score
        log_wsi_preds['wsi_std_precision'] = wsi_std_precision_score
        # class-mean of wsi-wise-mean Dice Score
        log_wsi_preds['class_mean_precision'] = class_mean_precision_score
        log_wsi_preds['class_std_precision'] = class_std_precision_score
        
        # Recall
        
        # logging of mean precision score per class        
        recall_grouped = defaultdict(list)
        for k, v in zip(per_class_recall, wsi_recall_scores_per_class):
            recall_grouped[k].append(v)
            
        mean_recall_scores_per_class = {data_provider.label_handler.decode(a): round(np.nanmean(b).item(),4) for a, b in recall_grouped.items()}
        std_recall_scores_per_class = {data_provider.label_handler.decode(a): round(np.nanstd(b).item(),4) for a, b in recall_grouped.items()}
        
        wsi_mean_recall_score = round(np.nanmean(list(wsi_recall_scores_dict.values())).item(),4)
        wsi_std_recall_score = round(np.nanstd(list(wsi_recall_scores_dict.values())).item(),4)
        class_mean_recall_score = round(np.nanmean(list(mean_recall_scores_per_class.values())).item(),4)
        class_std_recall_score = round(np.nanstd(list(mean_recall_scores_per_class.values())).item(),4)
        
        # per Wsi: (class-wise)mean recall
        log_wsi_preds['wsi_recall'] = wsi_recall_scores_dict
        # per Wsi: class Dice Score
        log_wsi_preds['wsi_recall_per_class'] = wsi_recall_scores_per_class_dict

        # all Wsis: (wsi-wise-)mean class recall
        log_wsi_preds['mean_recall_per_class'] = mean_recall_scores_per_class
        # all Wsis: (wsi-wise-)std class recall
        log_wsi_preds['std_recall_per_class'] = std_recall_scores_per_class
        
        # wsi-mean of class-wise-mean recall
        log_wsi_preds['wsi_mean_recall'] = wsi_mean_recall_score
        log_wsi_preds['wsi_std_recall'] = wsi_std_recall_score
        # class-mean of wsi-wise-mean recall
        log_wsi_preds['class_mean_recall'] = class_mean_recall_score
        log_wsi_preds['class_std_recall'] = class_std_recall_score
        
        print(f"Performance: {log_wsi_preds['wsi_mean_dice_scores']} mean-Dice (WSI-wise)")
        
        return log_wsi_preds



    def get_augmention(self) -> Tuple[transforms.Compose, transforms.Compose]:
        
        # pytorch resnet normalization values https://pytorch.org/hub/pytorch_vision_resnet/
        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        ## validation
        val_transforms = [
            transforms.Resize(256),
            transforms.ToTensor(),
            ]

        if self.args.normalize is True:
            val_transforms.append(self.normalization)

        val_aug_transform = transforms.Compose(val_transforms)

        ##train
        train_transforms = []
        if self.args.hue_aug_ratio is not None:
            train_transforms.append(transforms.ColorJitter(hue=self.args.hue_aug_ratio))

        train_transforms.extend([
            transforms.Resize(256),
            transforms.ToTensor(),
        ])

        if self.args.normalize is True:
            train_transforms.append(self.normalization)

        train_aug_transform = transforms.Compose(train_transforms)
        
        return train_aug_transform, val_aug_transform
    
    def collate_neighbour_patches(self, batch):
        return NeighbourBatch(batch)
    
    def collate_multiscale_patches(self, batch):
        return MultiscaleBatch(batch)
        
     
def attention_inference(data_loader: torch.utils.data.DataLoader,
                        model: torch.nn.Module,
                        gpu: int = None,
                        out: str = 'list',
                        return_attention: bool = False,
                        args = None):
    """Apply model to data to receive model output

    Args:
        data_loader (torch.utils.data.DataLoader): A pytorch DataLoader
            that holds the inference data
        model (torch.nn.Module): A pytorch model
        args (Dict): args

    Returns:
        [type]: [description]
    """

    outputs = []
    labels = []
    attentions = []
    
    # switch to evaluate mode
    model.eval()
    m = nn.Softmax(dim=1).cuda(gpu)

        # compute output
    memory = data_loader.dataset.wsi_dataset.embedding_memory
    k_neighbours = data_loader.dataset.wsi_dataset.k_neighbours
        
    if args.memory_to_gpu is True:
        memory.to_gpu(gpu)
        
    # first loop: create neighbourhood embedding memory
    with torch.no_grad():
        for patches, images, _, _, _ in data_loader:
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
    
            embeddings = model(images,
                               return_embeddings=True)
            memory.update_embeddings(patches=patches, 
                                     embeddings=embeddings)
            
        # second loop: attend freezed neighbourhood memory     
        for patches, images, targets in data_loader:
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            
            k_neighbour_embedding, k_neighbour_mask = memory.get_k_neighbour_embeddings(k=k_neighbours,
                                                                      patches=patches)
            
            if not k_neighbour_embedding.is_cuda:
                k_neighbour_embedding = k_neighbour_embedding.cuda(gpu, non_blocking=True)
            
            logits, attention = model(images, 
                                      neighbour_masks=k_neighbour_mask,
                                      neighbour_embeddings=k_neighbour_embedding,
                                      return_attention=True)  
            
            probs = m(logits)
            if out == 'list':
                outputs.extend(probs.cpu().numpy())
                labels.extend(targets.cpu().numpy())
                attentions.extend(attention.cpu().numpy())
            elif out == 'torch':
                #targets = targets.cuda(gpu, non_blocking=True)
                outputs.append(probs.cpu())
                labels.append(targets.cpu())
                attentions.append(attention.cpu())
                
    if return_attention:
        return outputs, labels, attention
    else:
        return outputs, labels
    
class NeighbourBatch:
    def __init__(self, batch) -> None:    
        self.img = torch.stack([item[0] for item in batch])
        self.mask = torch.stack([torch.LongTensor(item[1]) for item in batch])
        self.patch_idx = torch.stack([torch.LongTensor(l) for l in list(zip(*[(item[2]) for item in batch]))])
        self.patch_neighbour_idxs = torch.stack([torch.LongTensor(l) for l in list(zip(*[(item[3]) for item in batch]))]) 
        
    # custom memory pinning method on custom type
    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.mask = self.mask.pin_memory()
        self.patch_idx = self.patch_idx.pin_memory()
        self.patch_neighbour_idxs = self.patch_neighbour_idxs.pin_memory()
        return self.img, self.mask, self.patch_idx, self.patch_neighbour_idxs
    
class MultiscaleBatch:
    def __init__(self, batch) -> None:    
        self.img = torch.stack([item[0] for item in batch])
        self.context_img = torch.stack([item[1] for item in batch])
        self.mask = torch.stack([torch.LongTensor(item[2]) for item in batch])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.context_img = self.context_img.pin_memory()
        self.mask = self.mask.pin_memory()
        return self.img, self.context_img, self.mask