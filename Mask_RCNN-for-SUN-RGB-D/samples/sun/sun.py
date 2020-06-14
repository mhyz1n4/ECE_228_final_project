"""
Mask R-CNN
Train on the SUN dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Mask RCNN is Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 sun.py train --dataset=/Users/ekaterina/Desktop/diploma/mask_rcnn/datasets/SUNRGBD/train --weights=coco


    # Resume training a model that you had trained earlier
    python3 sun.py train --dataset=/path/to/sun/dataset/train --weights=last

    # Train a new model starting from ImageNet weights
    python3 sun.py train --dataset=/path/to/sun/dataset/train --weights=imagenet

    # Apply color splash to an image
    python3 sun.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 sun.py splash --weights=last --video=<URL or path to file>ß
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
from mrcnn import model as modellib, utils
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "hmeng/mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


#  Configurations

class SunConfig(Config):
    # Give the configuration a recognizable name
    NAME = "sun"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7
    
    #LEARNING_RATE = 0.001
    LEARNING_RATE = 0.0005


#  Dataset

class SunDataset(utils.Dataset):
    
#     def load_sun(self, dataset_dir, subset):
#         self.add_class("sun", 1, "bed")
#         self.add_class("sun", 2, "books")
#         self.add_class("sun", 3, "ceiling")
#         self.add_class("sun", 4, "chair")
#         self.add_class("sun", 5, "floor")
#         self.add_class("sun", 6, "furniture")
#         self.add_class("sun", 7, "objects")
#         self.add_class("sun", 8, "picture")
#         self.add_class("sun", 9, "sofa")
#         self.add_class("sun", 10, "table")
#         self.add_class("sun", 11, "tv")
#         self.add_class("sun", 12, "wall")
#         self.add_class("sun", 13, "window")
        
#         img_dir = os.path.join(dataset_dir, subset)
#         if subset == "train":
#             annot_dir = os.path.join(dataset_dir, "train_13")
#         else:
#             annot_dir = os.path.join(dataset_dir, "val_13")
        
#         for img in os.listdir(annot_dir):
#             #class_ids = [int(n['class']) for n in objects]
#             mask = skimage.io.imread(os.path.join(annot_dir, img))
#             classes = np.unique(mask)
#             img_id = img[12:-4]  #img13labels-004151.png
#             img_name = "img-" + img_id + ".jpg"
#             img_path = os.path.join(img_dir, img_name)
#             #train_image = skimage.io.imread(image_path)
#             height, width = mask.shape[0], mask.shape[1],
#             self.add_image(
#                 "sun",
#                 image_id=img_id,  # use file name as a unique image id
#                 path=img_path,
#                 width=width, height=height,
#                 mask_path=os.path.join(annot_dir, img),
#                 class_ids=classes)
                
#     def load_mask(self, image_id):
#         info = self.image_info[image_id]
#         if info["source"] != "sun":
#             return super(self.__class__, self).load_mask(image_id)
#         class_ids = info['class_ids']
        
#         mask_path = info["mask_path"]
        
#         img_mask = skimage.io.imread(mask_path)
#         h,w = info["height"], info["width"]
        
#         all_mask = np.zeros((h, w, len(class_ids)), dtype=bool)
#         for i in range(len(class_ids)):
#             #print(class_ids[i])
#             mask_img = (img_mask==class_ids[i])
#             #print(mask_img)
#             all_mask[:, :, i] = mask_img
#         #all_mask.astype(np.bool)
#         #exit()
#         class_ids = np.array(class_ids, dtype=np.int32)
        
#         return all_mask, class_ids
        
    def load_sun(self, dataset_dir, subset):
        self.add_class("sun", 1, "bed")
        self.add_class("sun", 2, "books")
        self.add_class("sun", 3, "ceiling")
        self.add_class("sun", 4, "chair")
        self.add_class("sun", 5, "floor")
        self.add_class("sun", 6, "furniture")
        self.add_class("sun", 7, "objects")
        self.add_class("sun", 8, "picture")
        self.add_class("sun", 9, "sofa")
        self.add_class("sun", 10, "table")
        self.add_class("sun", 11, "tv")
        self.add_class("sun", 12, "wall")
        self.add_class("sun", 13, "window")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values()) 
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]
            class_ids = [int(n['class']) for n in objects]
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            #image = skimage.transform.resize(image, (image.shape[0]+1, image.shape[1]+1))
            height, width = image.shape[:2]

            self.add_image(
                "sun",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        if info["source"] != "sun":
            return super(self.__class__, self).load_mask(image_id)
        class_ids = info['class_ids']
        
        # Convert polygons to a bitmap mask of shape
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            tmp_mask = np.zeros((info["height"], info["width"]),dtype=np.uint8)
            x = p['all_points_x']
            y = p['all_points_y']
            pts2 = np.array([x,y], np.int32)
            pts2 = np.transpose(pts2)
            color = [1, 0, 0]
            #rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            cv2.fillPoly(tmp_mask, [pts2], color)
            tmp_mask[tmp_mask > 0] = 1
            mask[:, :, i] = tmp_mask

        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sun":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, flag):
    """Train the model."""
    # Training dataset.
    
    if flag == "old":
        dataset_train = SunDataset()
        dataset_train.load_sun_old(args.dataset, "train")
        dataset_train.prepare()

        dataset_val = SunDataset()
        dataset_val.load_sun_old(args.dataset, "val")
        dataset_val.prepare()
    else:
        dataset_train = SunDataset()
        dataset_train.load_sun(args.dataset, "train")
        dataset_train.prepare()

        dataset_val = SunDataset()
        dataset_val.load_sun(args.dataset, "val")
        dataset_val.prepare()
        
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')

def detect_and_color_splash(model, image_path=None, video_path=None, image_name=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print("r shape")
        print(r['class_ids'])
        print(r['masks'].shape)
        with open("./detect/mask_" + image_name + ".txt", "w") as p_file:
            for i in range(len(r['class_ids'])):
                data_slice = r['masks'][:, :, i]
                #print(np.sum(data_slice))
                np.savetxt(p_file, data_slice.astype(float))
    else:
        print("empty image_path")
        exit(1)

        
#  Training

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SunConfig()
    else:
        class InferenceConfig(SunConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, "new")
    elif args.command == "test":
        image_name = "".join([i for i in args.image if i.isdigit()])
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video, image_name=image_name)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))
