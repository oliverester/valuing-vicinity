from pathlib import Path
from statistics import mean, stdev
from torch.utils.tensorboard import SummaryWriter
import yaml

def fix_logs(exp_path: str):
    
    
    test_tumor_dices = []
    val_tumor_dices = []
    for f in range(5):
        with open(Path(exp_path) / f'fold_{f}' / 'attention_segmentation_log.yml', "r") as stream:
            try:
                fold_logs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
            test_tumor_dices.append(fold_logs['evaluation_wsi_test_set']['mean_dice_scores_per_class']['tumor'])
            val_tumor_dices.append(fold_logs['evaluation_wsi_vali_best_set']['mean_dice_scores_per_class']['tumor'])
            
    test_mean_tumor_dice = round(mean(test_tumor_dices),4)
    test_std_tumor_dice = round(stdev(test_tumor_dices),4)
    val_mean_tumor_dice = round(mean(val_tumor_dices),4)
    val_std_tumor_dice = round(stdev(val_tumor_dices),4)
        
    # add to aggregate log
    with open(Path(exp_path) / 'attention_segmentation_log.yml', "r") as stream:
        try:
            logs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    logs['fold_aggregates']['vali_best']['wsi_tumor_dice_scores'] = {}
    logs['fold_aggregates']['vali_best']['wsi_tumor_dice_scores']['mean'] = val_mean_tumor_dice
    logs['fold_aggregates']['vali_best']['wsi_tumor_dice_scores']['std'] = val_std_tumor_dice
    
    logs['fold_aggregates']['test']['wsi_tumor_dice_scores'] =  {}
    logs['fold_aggregates']['test']['wsi_tumor_dice_scores']['mean'] = test_mean_tumor_dice
    logs['fold_aggregates']['test']['wsi_tumor_dice_scores']['std'] = test_std_tumor_dice
   
    with open(Path(exp_path) / 'attention_segmentation_log.yml', "w") as stream:
        yaml.dump(logs, stream, default_flow_style=False, sort_keys=False)
    
    # log to tensorboard
    writer = SummaryWriter(exp_path)
    writer.add_scalar(tag=f'fold_aggregation/vali_best_fold_mean_wsi_tumor_dice',
                              scalar_value=val_mean_tumor_dice)
    writer.add_scalar(tag=f'fold_aggregation/test_fold_mean_wsi_tumor_dice',
                              scalar_value=test_mean_tumor_dice)
    writer.close()
        

if __name__ == "__main__":
    fix_logs(exp_path="logdir_paper/configs_paper/configs_cy16/attention/deeplab_resnet18/a_d4_k8_deeplab_config-2022-09-19_11_22_34_3")
    