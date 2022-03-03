
import glob
import PIL
from openslide import OpenSlide
from pathlib import Path

def create_wsi_thumbnails(source_folder: str, target_folder: str, wsi_type='svs', img_type='png'):
    """
    Searches through source folder and converts each wsi into a thumbnail.
    The thumbnail is stored with the same relative path to the target folder.
    """
    
    # root_dir needs a trailing slash (i.e. /root/dir/)
    for filename in glob.iglob(source_folder + f'/**/*.{wsi_type}', recursive=True):
        print(filename)
        file_pth = Path(filename)
        src_pth = Path(source_folder)
        trg_pth_end = file_pth.parts[-(len(file_pth.parts)-len(src_pth.parts)):-1] + (file_pth.stem + f".{img_type}",)
        thumbnail = get_wsi_thumbnail(filename)
        save_wsi_thumbnail(thumbnail, dst_path=Path(target_folder, *trg_pth_end), img_type=img_type)

def get_wsi_thumbnail(image_filepath) -> PIL.Image:
    '''
    Generates a thumbnail from the specified image.
    '''
    # open image with OpenSlide library
    image_file = OpenSlide(image_filepath)
    # extract image dimensions
    image_dims = image_file.dimensions
    # make thumbnail 100 times smaller
    thumb_dims = tuple( (x/100 for x in image_dims) )
    # create thumbnail
    thumb_file = image_file.get_thumbnail(thumb_dims)
    # cleanup
    image_file.close()
    
    return thumb_file

def save_wsi_thumbnail(thumbnail: PIL.Image, dst_path: Path, img_type='png'):

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)

    # save file with desired path, format
    thumbnail.save(dst_path, img_type)
    
if __name__ == "__main__":
    create_wsi_thumbnails(source_folder='/projects/praediknika/data/wsis/HEEL',
    target_folder='/homes/oester/datasets/praediknika/thumbnails_new')