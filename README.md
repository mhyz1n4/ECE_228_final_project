# ECE_228_final_project
For this project, we use the SUNRGBD dataset, which can be downloaded using:

```
wget http://rgbd.cs.princeton.edu/data/SUNRGBD.zip
unzip SUNRGBD.zip
rm SUNRGBD.zip
```

## Mask-RCNN
In the `Mask_RCNN-for-SUN-RGB-D` folder, you will find the training and testing code for Mask-RCNN. For package requirements, please refer to [link](https://github.com/matterport/Mask_RCNN) (requirements including installing cocoapi). To start training, please refer to shell scripts `train_new.sh` and `train.sh`. To start testing and generating predictions, please refer to `test_new.sh` and `test.sh`. Training the model would require COCO pretrain weights form this [link](https://github.com/matterport/Mask_RCNN/releases).

For generate train/val/test dataset for Mask-RCNN, run `Mask_RCNN-for-SUN-RGB-D/sample/sun/Preprocess_dataset.ipynb` first. To exam the predicted mask, please refer to `Mask_RCNN-for-SUN-RGB-D/exam results.ipynb`.


## Coordinate Recovery
In `Coordinate Recovery` folder, you will find the code and test images. To run the test, please see `CoordinateRecovery.ipynb`. For complete code, see `CoordinateRecovery.py`. RGB image, depth image, mask, intrinsic matrix and extrinsic matrix are needed.

## DeepLabV3
In `DeepLabV3` folder, `aspp.py,deeplabv3.py,resnet.py` contains the code for building the DeepLabV3 model. To initialize the model, you need to import all classes in aspp,resnet, and deeplabv3 py files. the file - `train.py` contains the code for training the model. The visulization and learning curves are done by the code in `viusalization_curve.py`

