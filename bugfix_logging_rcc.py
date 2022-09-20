from pathlib import Path
from statistics import mean, stdev
from torch.utils.tensorboard import SummaryWriter
import yaml

def fix_logs(exp_path: str):
    
    classes = ['background',
    'cyst',
    'tumor_bleeding',
    'tumor_necrosis',
    'tumor_vital',
    'extrarenal',
    'cortex',
    'angioinvasion',
    'mark',
    'tumor_regression',
    'capsule']
    
    test_dices = {'background': [],
                'cyst': [],
                'tumor_bleeding': [],
                'tumor_necrosis': [],
                'tumor_vital': [],
                'extrarenal': [],
                'cortex': [],
                'angioinvasion': [],
                'mark': [],
                'tumor_regression': [],
                'capsule': []}
    val_dices = {'background': [],
                 'cyst': [],
                 'tumor_bleeding': [],
                 'tumor_necrosis': [],
                 'tumor_vital': [],
                 'extrarenal': [],
                 'cortex': [],
                 'angioinvasion': [],
                 'mark': [],
                 'tumor_regression': [],
                 'capsule': []}
    
    # each fold
    for f in range(5):
        with open(Path(exp_path) / f'fold_{f}' / 'attention_segmentation_log.yml', "r") as stream:
            try:
                fold_logs = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
            for cls in classes:
                test_dices[cls].append(fold_logs['evaluation_wsi_test_set']['mean_dice_scores_per_class'][cls])
                val_dices[cls].append(fold_logs['evaluation_wsi_vali_best_set']['mean_dice_scores_per_class'][cls])
    
    test_mean_dice = {}
    test_std_dice = {}
    val_mean_dice = {}
    val_std_dice = {}
    
    for cls in classes:
        test_mean_dice[cls] = round(mean(test_dices[cls]),4)
        test_std_dice[cls] = round(stdev(test_dices[cls]),4)
        val_mean_dice[cls] = round(mean(val_dices[cls]),4)
        val_std_dice[cls] = round(stdev(val_dices[cls]),4)
    
        
    # add to aggregate log
    with open(Path(exp_path) / 'attention_segmentation_log.yml', "r") as stream:
        try:
            logs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    
    for cls in classes:
        logs['fold_aggregates']['vali_best'][f'wsi_{cls}_dice_scores'] = {}
        logs['fold_aggregates']['vali_best'][f'wsi_{cls}_dice_scores']['mean'] = val_mean_dice[cls]
        logs['fold_aggregates']['vali_best'][f'wsi_{cls}_dice_scores']['std'] = val_std_dice[cls]
        
        logs['fold_aggregates']['test'][f'wsi_{cls}_dice_scores'] =  {}
        logs['fold_aggregates']['test'][f'wsi_{cls}_dice_scores']['mean'] = test_mean_dice[cls]
        logs['fold_aggregates']['test'][f'wsi_{cls}_dice_scores']['std'] = test_std_dice[cls]
        
   
    with open(Path(exp_path) / 'attention_segmentation_log.yml', "w") as stream:
        yaml.dump(logs, stream, default_flow_style=False, sort_keys=False)
    
    # log to tensorboard
    writer = SummaryWriter(exp_path)
    for cls in classes:
        writer.add_scalar(tag=f'fold_aggregation_class/vali_best_fold_mean_wsi_{cls}_dice',
                                scalar_value=val_mean_dice[cls])
        writer.add_scalar(tag=f'fold_aggregation_class/test_fold_mean_wsi_{cls}_dice',
                                scalar_value=test_mean_dice[cls])
    writer.close()
        

if __name__ == "__main__":
    fix_logs(exp_path="logdir_paper/configs_paper/configs_rcc/attention/unet_res50/a_d16_k8_unet_config-2022-09-13_07_03_59_1")
    