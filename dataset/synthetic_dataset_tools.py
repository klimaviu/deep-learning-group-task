import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import random
import PIL
from io import BytesIO
import base64
import cv2
#from . import mask as maskUtils
import os
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


# Load the dataset json
class SyntheticDataset():
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.colors = ['blue', 'purple', 'red', 'green', 'orange', 'salmon', 'pink', 'gold',
                                'orchid', 'slateblue', 'limegreen', 'seagreen', 'darkgreen', 'olive',
                               'teal', 'aquamarine', 'steelblue', 'powderblue', 'dodgerblue', 'navy',
                               'magenta', 'sienna', 'maroon', 'yellow', 'brown', 'gray', 'amaranth',
                               'amber', 'bluetiful', 'blush', 'buff', 'burgandy']
        
        json_file = open(self.annotation_path)
        self.anno = json.load(json_file)
        json_file.close()
        
        self.process_info()
        self.process_licenses()
        self.process_categories()
        self.process_images()
        self.process_segmentations()

    def getImgInfo(self, scene):
        """ Given a list of scene names, return dictionaries corresponding to image info
        """
        for d in self.anno["images"]:
            if d["scene"] == scene:
                return d

    def getAnnoInfo(self, imgIds):
        """ Given a list of image ids, return list of corresponding annotation dicts
        """
        annos = []
        for ann in self.anno["annotations"]:
            if ann["image_id"] in imgIds:
                annos.append(ann)
        return annos

    def getImgIds(self, scenes):
        """
        Get image ids that correspond to scenes
        @param scenes (list): List of scene name strings, <store>_<id>_<cam-num>
        """
        image_ids = []
        for scene in scenes:
            for d in self.anno["images"]:
                if d["scene"] == scene:
                    image_ids.append(d["id"])
        return image_ids

    def getAnnIds(self, imgIds):
        """ Given a list of image ids get list of corresponding annotations ids
        @param imgIds (list): List of ints corresponding to image ids
        """
        annIds = []
        for ann in self.anno["annotations"]:
            if ann["image_id"] in imgIds:
                annIds.append(ann["id"])
        return annIds

    def mapScenesToInfo(self, scenes):
        """ Generate dictionary that maps each scene to the following:
            "img1": name of before image
            "img2": name of after image
            "label": list of Polygons
            "actions": list of action values/integers,
            "bbox",
            "depth1"
            "depth2"
        """
        sceneToInfo = dict()
        for scene in scenes:
            sceneToInfo[scene] = dict()
            # Get dict corresponding to image id
            img_dict = self.getImgInfo(scene)
            sceneToInfo[scene]["img1"] = img_dict["image1"]
            sceneToInfo[scene]["img2"] = img_dict["image2"]
            sceneToInfo[scene]["label"] = []
            sceneToInfo[scene]["actions"] = []
            sceneToInfo[scene]["bbox"] = []
            anno_dicts = self.getAnnoInfo([img_dict["id"]])
            for anno in anno_dicts:
                sceneToInfo[scene]["label"].append(anno["segmentation"])
                sceneToInfo[scene]["actions"].append(anno["action"])
                sceneToInfo[scene]["bbox"].append(anno["bbox"])
            sceneToInfo[scene]["depth1"] = img_dict["depth1"]
            sceneToInfo[scene]["depth2"] = img_dict["depth2"]

        return sceneToInfo

    def display_info(self):
        print('Dataset Info:')
        print('=============')
        for key, item in self.info.items():
            print('  {}: {}'.format(key, item))
        
        requirements = [['description', str],
                        ['url', str],
                        ['version', str],
                        ['year', int],
                        ['contributor', str],
                        ['date_created', str]]
        for req, req_type in requirements:
            if req not in self.info:
                print('ERROR: {} is missing'.format(req))
            elif type(self.info[req]) != req_type:
                print('ERROR: {} should be type {}'.format(req, str(req_type)))
        print('')

        
    def display_licenses(self):
        print('Licenses:')
        print('=========')
        
        requirements = [['id', int],
                        ['url', str],
                        ['name', str]]
        for license in self.licenses:
            for key, item in license.items():
                print('  {}: {}'.format(key, item))
            for req, req_type in requirements:
                if req not in license:
                    print('ERROR: {} is missing'.format(req))
                elif type(license[req]) != req_type:
                    print('ERROR: {} should be type {}'.format(req, str(req_type)))
            print('')
        print('')
        
    def display_categories(self):
        print('Categories:')
        print('=========')
        for sc_key, sc_val in self.super_categories.items():
            print('  super_category: {}'.format(sc_key))
            for cat_id in sc_val:
                print('    id {}: {}'.format(cat_id, self.categories[cat_id]['name']))
            print('')
    
    def display_image(self, image_id, show_polys=True, show_bbox=True, actions=['take','put','shift']):
        print('Image:')
        print('======')
        if image_id == 'random':
            image_id = random.choice(list(self.images.keys()))
        
        # Print the image info
        image = self.images[image_id]
        for key, val in image.items():
            print('  {}: {}'.format(key, val))
            
        # Open the image
        image_path = os.path.join(self.image_dir, image['image2'])
        image_pil = PIL.Image.open(image_path)

        buffer = BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)

        data_uri = base64.b64encode(buffer.read()).decode('ascii')
        image_path = "data:image/png;base64,{0}".format(data_uri)
            
        # Calculate the size and adjusted display size
        max_width = 1280
        image_width, image_height = image_pil.size
        adjusted_width = min(image_width, max_width)
        adjusted_ratio = adjusted_width / image_width
        adjusted_height = adjusted_ratio * image_height
        
        # Create list of polygons to be drawn
        polygons = {}
        bbox_polygons = {}
        rle_regions = {}
        poly_colors = {}
        all_formatted_points = []
        print('  segmentations ({}):'.format(len(self.segmentations[image_id])))
        image = cv2.imread(os.path.join(self.image_dir, image['image2']))
        overlay = image.copy()
        for i, segm in enumerate(self.segmentations[image_id]):
            if segm['action'] in actions:
                polygons_list = []
                # Add the polygon segmentation
                for segmentation_points in segm['segmentation']:
                    segmentation_points = np.multiply(segmentation_points, adjusted_ratio).astype(int)
                    # Switching x,y coords in segmentation points
                    formatted_points = []
                    for i in range(0, len(segmentation_points), 2):
                        formatted_points.append((segmentation_points[i+1], segmentation_points[i]))
                polygons[segm['id']] = polygons_list
                if i < len(self.colors):
                    poly_colors[segm['id']] = self.colors[i]
                else:
                    poly_colors[segm['id']] = 'white'

                if show_polys:
                    polygon = Polygon(formatted_points)
                    int_coords = lambda x: np.array(x).round().astype(int)
                    exterior = [int_coords(polygon.exterior.coords)]
                    overlay = cv2.fillPoly(overlay, exterior, color=(5*i, 5*i, 0))#color=(255, 255, 0))
                    alpha=0.5
                    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                bbox = segm['bbox']
                bbox_points = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1],
                               bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0], bbox[1] + bbox[3],
                               bbox[0], bbox[1]]
                bbox_points = np.multiply(bbox_points, adjusted_ratio).astype(int)
                bbox_polygons[segm['id']] = str(bbox_points).lstrip('[').rstrip(']')

                if show_bbox:
                    cv2.rectangle(overlay, (int(bbox[1]), int(bbox[0])), (int(bbox[1]+bbox[3]), int(bbox[0]+bbox[2])), (255, 0, 0), 2)

                # Print details
                print('    {}:{}'.format(segm['id'], self.categories[segm['category_id']]))
        
        fig = plt.figure(figsize=(8,6), dpi=200, facecolor='w', edgecolor='k')
        plt.imshow(image, )
        plt.axis('off')
        
        return None
        
    def process_info(self):
        self.info = self.anno['info']
    
    def process_licenses(self):
        self.licenses = self.anno['licenses']
    
    def process_categories(self):
        self.categories = {}
        self.super_categories = {}
        for category in self.anno['categories']:
            cat_id = category['id']
            
            # Add category to the categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print("ERROR: Skipping duplicate category id: {}".format(category))
                
    def process_images(self):
        self.images = {}
        for image in self.anno['images']:
            image_id = image['id']
            if image_id in self.images:
                print("ERROR: Skipping duplicate image id: {}".format(image))
            else:
                self.images[image_id] = image
                
    def process_segmentations(self):
        self.segmentations = {}
        for segmentation in self.anno['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)