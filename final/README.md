# DeepLiDAR (Python3, Pytorch 1.4.0)
This repository is the implementation for [DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qiu_DeepLiDAR_Deep_Surface_Normal_Guided_Depth_Prediction_for_Outdoor_Scene_CVPR_2019_paper.pdf). There are some difference between author's repo and mine.

(1) Rewrite the code referenced from [author's repo](https://github.com/JiaxiongQ/DeepLiDAR) with python3.6 and newest version of pytorch. 

(2) Clarify the structure of KITTI depth completion data

(3) Make it easier to reproduce my result (**not author's result**, because original model is too large to put onto single GPU and use lots of time to train. So, I **reduced the parameters** of model and **used less data** to train. Also, there might be **slight difference** between author's implementation and mine)

(4) I add comments on the code and make it more flexible and readable. 

(5) I add tensorboard visualization for every epoch



## Requirements
* Python 3.6.8
```
pip3 install -r requirements.txt
```

## Data Preparation
- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset from their website.
- Use the following scripts to extract corresponding RGB images from the raw dataset. 
```bash
./download/rgb_train_downloader.sh
./download/rgb_val_downloader.sh
```
* The overall code, data, and results directory is structured as follows
  * data_depth_annotated: ground truth data (dense depth)
  * data_depth_velodyne: sparse data (LiDAR)
  * data_rgb: RGB image
  * data_depth_annotated: Used to train surface normals, and generated from **data_depth_annotated** by using **generate_normals.py**

```
.
├── KITTI_data
|   ├── data_depth_annotated
|   |   ├── train
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |       ├── proj_depth
|   |   |           ├── groundtruch
|   |   |               ├── image02
|   |   |               ├── image03
|   |   ├── val
|   |   |   ├── (the same as train)
|   ├── data_depth_velodyne
|   |   ├── train
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |       ├── proj_depth
|   |   |           ├── velodyne_raw
|   |   |               ├── image_02
|   |   |               ├── image_03
|   |   ├── val
|   |   |   ├── (the same as train)
|   ├── data_rgb
|   |   ├── train
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |   |   ├── image_02
|   |   |   |   ├── image_03
|   |   ├── val
|   |   |   ├── (the same as train)
|   ├── data_depth_normals
|   |   ├── (the same as data_depth_annotated)
|   └── depth_selection
|   |   ├── test_depth_completion_anonymous
|   |   ├── test_depth_prediction_anonymous
|   |   ├── val_selection_cropped

```
### To generate **data_depth_normals**
First, enter **surface-normal/** to build and install library. 

Second, set path in **generate_normals.py** and run the following script to generate **data_depth_normals**
```
python3 generate_normals.py
```


## Usage


### Train and validation
```
python3 main.py
```
There are three different stages.
1. (N) Train surface normal
2. (D) Train depth of color pathway and normal pathway
3. (A) Train whole network

The result provided below is train N stage for 15 epoch, then train D stage for 15 epoch, and then train A stage for 15 epoch

If you don't want to use surface normal, or you don't have surface normal data, you can directly train A stage

### Test
```
python3 test.py
```
RMSE: 1191.6127 on **depth_selection/val_selection_cropped** data

## Tensorboard Visualization
```
tensorboard --logdir runs/
```

## Experiment
Input data (rgb image, lidar image)
![image](https://github.com/ChingYenShih/EECS-545-Final/blob/master/final/figure/input.png)

Groundtruth (surface normal, dense depth)
![image](https://github.com/ChingYenShih/EECS-545-Final/blob/master/final/figure/gt.png)

Predicted result (surface normal, dense depth)
![image](https://github.com/ChingYenShih/EECS-545-Final/blob/master/final/figure/pred.png)
