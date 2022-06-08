 
from src.deephist.evaluate.patch_inference import run_patch_inference
from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment

if __name__ == "__main__":

    # patch_coordinates = {'RCC-TA-033.001~C': [(14,19), (15,20), (20,20)],
    #                      'RCC-TA-011.001~J': [(20, 15), (20, 17)],
    #                      'RCC-TA-004.001~C': [(21, 35)]
    #                      }
    
    patch_coordinates = {'RCC-TA-163.001~B': [(7,12), (8,12), (8,11), (7,11)],
                         }
    # patch_coordinates = {'tumor026': [(8,34)],
    #                     }
    exp = SegmentationExperiment(config_path='configs_rcc/attention_segmentation_config_inference.yml')
    
    run_patch_inference(exp, patch_coordinates, k=8)