
from pathlib import Path
import json
from typing import List

IGNORE_LIST = []



def validate_annotations(annotation_folder: str,
                         annotation_output_folder: str = None,
                         wsi_folder: str = None):
    
    if wsi_folder is not None:
        annotation_exists(wsi_folder=wsi_folder,
                          annotation_folder=annotation_folder)
        
    tissue_in_annotations(annotation_folder=annotation_folder,
                          annotation_output_folder=annotation_output_folder)
    

def annotation_exists(wsi_folder: str, 
                      annotation_folder: str,
                      wsi_type: List[str] = ['svs']):
    
    annotation_path = Path(annotation_folder)
    annotation_stems = [el.stem for el in annotation_path.glob('**/*') if el.is_file() and el.suffix == '.json']
    wsi_path = Path(wsi_folder)
    wsi_files = [el for el in wsi_path.glob('**/*') if el.is_file() and el.suffix in ["." + _type for _type in wsi_type]]
    print(f"Found {len(wsi_files)} wsi images")
    
    annotation_not_exists = [wsi_file for wsi_file in wsi_files if wsi_file.stem not in annotation_stems]
    
    print_res = '\n'.join([str(file) for file in annotation_not_exists])
    if len(annotation_not_exists) > 0:
        print("Cannot find annotation file for \n" + print_res)
    else:
        print("Annotation for each wsi exists")
    

def tissue_in_annotations(annotation_folder: str, annotation_output_folder: str = None):
    
    annotation_path = Path(annotation_folder)
    annotation_files = [el for el in annotation_path.glob('**/*') if el.is_file() and el.suffix == '.json']
    
    if annotation_output_folder is not None:
        annotation_output_path = Path(*(annotation_path.parts[:-1] + (annotation_output_folder,)))
        annotation_output_path.mkdir(parents=True, exist_ok=True)
        
    file_with_other = [] 
    file_with_missing_classfication = [] 
    file_with_no_tissue = [] 
    file_with_multi_tissue = []

    for annotation_file in annotation_files:
        if annotation_file.name in IGNORE_LIST:
            continue
        print(f'Loading {annotation_file}')
        with open(annotation_file) as f:
            has_tissue = []          
            annotation = json.load(f)
            for idx, _ in enumerate(annotation):
                if 'classification' in annotation[idx]['properties']:
                    clas = annotation[idx]['properties']['classification']['name']
                else:
                    print('Found annotation without classification. Renaming in tissue')
                    annotation[idx]['properties']['classification'] = {}
                    annotation[idx]['properties']['classification']['name'] = 'tissue'
                    has_tissue.append(True)
                    file_with_missing_classfication.append(annotation_file.name)
                if clas.lower() == 'tissue':
                    print('Found tissue annotation')
                    has_tissue.append(True)
                if clas.lower() == 'other':
                    print("Found 'other' - converting to tissue")
                    annotation[idx]['properties']['classification']['name'] = 'tissue'
                    has_tissue.append(True)
                    file_with_other.append(annotation_file.name)
            if len(has_tissue) == 0:
                file_with_no_tissue.append(annotation_file.name)
                print(f"Cannot find tissue for {annotation_file}")    
            if len(has_tissue) > 1:
                file_with_multi_tissue.append(annotation_file.name)
            
        # save annotation file:
        if annotation_output_folder is not None:
            with open(annotation_output_path / annotation_file.name , 'w') as f:
                json.dump(annotation, f)
    
            
    print(f"Files with 'other'-Annotation: {file_with_other}")          
    print(f"Files with missing tissue classifcation: {file_with_missing_classfication}")
    print(f"Files with no tissue annotation: {file_with_no_tissue}")
    print(f"Files with multiple tissue annotation: {file_with_multi_tissue}")  

    
def create_tissue_map(annotation_folder: str):
    annotation_path = Path(annotation_folder)
    annotation_files = [el for el in annotation_path.glob('**/*') if el.is_file() and el.suffix == '.json']
    
    tissue_type_set = set()
    for annotation_file in annotation_files:
        if annotation_file.name in IGNORE_LIST:
            continue
        print(f'Loading {annotation_file}')
        with open(annotation_file) as f:
            annotation = json.load(f)
            for idx, _ in enumerate(annotation):
                if 'classification' in annotation[idx]['properties']:
                    tmp_class = annotation[idx]['properties']['classification']['name'].lower()
                    tissue_type_set.add(tmp_class)
    # convert to dict for class numbers:
    class_dict = dict()
    for idx, cls in enumerate(tissue_type_set):
        class_dict[cls] = idx
    
    print(json.dumps(class_dict))   
            
if __name__ == '__main__':
    create_tissue_map(annotation_folder='data/segmentation/annotations_heel_seg')
    #validate_annotations(annotation_folder='data/annotations_heel_seg',
                         #annotation_output_folder="annotations_corrected",
     #                    wsi_folder='/projects/praediknika/data/from_mhh/RCCs_HE-El_svs')