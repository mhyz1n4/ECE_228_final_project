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
