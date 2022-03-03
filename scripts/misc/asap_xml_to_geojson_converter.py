"""
Convert ASAP xml annotations to geojson
"""

import os

from pathlib import Path
from typing import Dict
from xml.dom import minidom

from geojson import Polygon
from geojson.codec import dump
from geojson.feature import Feature

def convert_asap_xmls(source_folder: str,
                      target_folder: str,
                      target_format: str = 'json'):
    """Converts all asap-xml-annotation files in source-Folder into geojson for QuPath

    Args:
        source_folder ([type]): Path to source folder of xml-files
        target_folder ([type]): Path to target folder for converted geojson-files
        target_format (str, optional): One of [json]. Defaults to 'json'.
            More annotation file types could be implemented in futere
    """

    # loop all files
    if target_format not in ['json']:
        raise Exception("Target foramt must be one of [json].")

    # create target folder if not yet exists
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith(".xml"):
            geo_json = convert_asap_xml_to_geo_json(os.path.join(source_folder, filename))
            serialize_geo_json(geo_json, target_folder, filename[0:-3]+'json')
            print(f"Successfully converted {filename} into geojson")

def serialize_geo_json(geo_json: Dict,
                       target_folder: str,
                       target_file_name: str):
    """Serializes and stores a geo_json

    Args:
        geo_json (Dict): dictionary in geojson format
        target_folder (str): path to target folder
        target_file_name (str): file name
    """

    with open(os.path.join(target_folder, target_file_name), 'w') as f:
        dump(geo_json, f,)


def convert_asap_xml_to_geo_json(path: Path) -> Dict:
    """Convert asap xml annotation file to geojson annotation file.

    Args:
        path (Path): Path object with path to XML-file

    Returns:
        [Dict]: annotations in dictionary in geojosn format
    """

    try:
        xml = minidom.parse(str(path))

        # The first region marked is always the tumour delineation
        polygons_ = xml.getElementsByTagName("Annotation")
        feature_list = []
        for polygon in polygons_:
            coordinate_list = []
            coordinates_ = polygon.getElementsByTagName('Coordinate')
            for coord in coordinates_:
                x = float(coord.getAttribute("X"))
                y = float(coord.getAttribute("Y"))
                coordinate_list.append((x,y))
            # check if coordinates form a closed ring: last coord = first colormode
            # which seems to be a requirement at least for QuPath
            if coordinate_list[0] != coordinate_list[-1]:
                coordinate_list.append(coordinate_list[0])

            # in list: qupath requirement
            feature = Feature(geometry=Polygon(coordinates=[coordinate_list]), 
                             id='PathAnnotationObject',
                             properties=dict(isLocked=False,
                                             measurements=[]))
            feature_list.append(feature)
        #feature_collection = FeatureCollection(feature_list)

        return feature_list
    except Exception as error:
        print(f"Error converting {path} into geojosn")
        raise error

if __name__ == '__main__':

    source = '/homes/oester/datasets/camelyon16/testing/lesion_annotations'
    target = '/homes/oester/datasets/camelyon16/testing/geojson_lesion_annotations'

    convert_asap_xmls(source_folder=source,
                      target_folder=target,
                      target_format='json')