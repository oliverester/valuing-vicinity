from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import yaml

def fix_logs(exp_path: str):
    
    with open(Path(exp_path) / 'attention_segmentation_log.yml', "r") as stream:
        try:
            logs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    scores = logs['fold_aggregates']
   
    # log to tensorboard
    writer = SummaryWriter(exp_path)
    for phase in ['vali_best', 'test']:
        for score in scores[phase].keys():
            writer.add_scalar(tag=f'fold_aggregation/{phase}_fold_mean_{score}',
                                scalar_value=scores[phase][score]['mean'])
    
    writer.close()
        

if __name__ == "__main__":
    fix_logs(exp_path="logdir_paper/configs_paper/configs_rcc/attention/deeplab_res50/variants/a_d16_k8_deeplab_transformer_config-2022-08-17_14_25_11_1")
    