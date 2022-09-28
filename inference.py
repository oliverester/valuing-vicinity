 
from src.deephist.evaluate.attention_analysis import run_attention_analysis
from src.deephist.evaluate.patch_inference import run_patch_inference
from src.exp_management.experiment.SegmentationExperiment import SegmentationExperiment

if __name__ == "__main__":

    # patch_coordinates = {'RCC-TA-033.001~C': [(13,24),(13,24),(11,24),(14,19), (15,20), (20,20)],
    #                      'RCC-TA-011.001~J': [(20, 15), (20, 17), (16,16),(21,13)],
    #                      'RCC-TA-004.001~C': [(21, 35), (18,36), (18,37)],
    #                      'RCC-TA-163.001~B': [(8,11),(16,13),(18,3),(13,10),(10,14),(13,14),(16,10), (8,10), (10,11), (14,4)],
    #                      'RCC-TA-045.001~C': [(20,50), (19,42), (26,42), (24,57),(5,41)],
    #                      'RCC-TA-158.001~B': [(13,27)],
    #                      'RCC-TA-022.001~F': [(16,18),(5,19),(9,13),(13,15),(16,22),(26,26),(13,32)],
    #                      'RCC-TA-066.001~F': [(20,18)],
    #                      'RCC-TA-115.001~C': [(12,24)]
    #                      }
    
    # # patch_coordinates = {'RCC-TA-022.001~F': [(16,18),(5,19),(9,13),(13,15),(16,22),(26,26),(13,32)]
    # #                     }
    
    # # patch_coordinates = {
    # #                      'RCC-TA-045.001~C': [(20,50), (19,42), (26,42), (24,57),(5,41)],
    # #                      }
    # patch_coordinates = {'RCC-TA-163.001~B': [(8,11),(18,3),(13,10),(13,14),(16,10), (8,10), (10,11), (14,4)],
    #                      }
    # # patch_coordinates = {'tumor026': [(8,34)],
    # #                     }
    
    # exp = SegmentationExperiment(config_path='configs_inference/attention_segmentation_config_inference.yml')
    
    # #run_attention_analysis(exp)
    # run_patch_inference(exp, patch_coordinates, radius=25)
    
    
    patch_coordinates = {'tumor088': [(80,110),(180,30),(130,100),(130,140),(160,100), (80,100), (100,110), (140,40)],
                         }
    
    exp = SegmentationExperiment(config_path='configs_inference/cy16_attention_segmentation_config_inference.yml')
    
    #run_attention_analysis(exp)
    run_patch_inference(exp, patch_coordinates, radius=25)