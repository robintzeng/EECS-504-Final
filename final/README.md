# Mask-Assisted Depth Completion with Multi-Resolution Predictions Based onAttention Mechanism from Color Image and Sparse LiDAR

## Introduction
**Abstract Depth  completion**  is  essential  for  autonomous  driving applications.  In  this  paper,   we  propose  the  end-to-end learning architecture to effectively complete depth from acolor image and sparse LiDAR data.   We develop a local pathway  and  a  global  pathway  to  extract  high-resolution features and low-resolution features respectively.   The local pathway is FuseNet without 3D representation ,and the global pathway is made up of our proposed stacked U-Block. Our architecture combines predictions from thesepathways  based  on  the  attention  mechanism. With  the learned confidence map, our model can put attention on local or global pathway depending on their confidence, and experiment result shows that local pathway has higher confidence on the edge,  and global pathway has higher confidence inside the object.  Furthermore, we apply a binary mask to help our model know positions of valid values in sparse LiDAR data, and it can boost the performance of local pathway.   We evaluate our model on the KITTI depth completion  dataset.   To  make  comparison,  we  implement two models ranking 8th and 11th on the KITTI depth completion benchmark. Also, we conduct an ablation study and qualitative analysis to demonstrate the effectiveness of proposed U-Block and our methods

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
|   |   |       ├── image_02
|   |   |       ├── image_03
|   |   ├── val
|   |   |   ├── (the same as train)
|   ├── data_depth_normals
|   |   ├── (the same as data_depth_annotated)
|   └── depth_selection
|   |   ├── test_depth_completion_anonymous
|   |   ├── test_depth_prediction_anonymous
|   |   ├── val_selection_cropped

```

### Path setting
Please set the path in the **env.py** first

```
SAVED_MODEL_PATH = './saved_model' # save model in this directory
KITTI_DATASET_PATH = /PATH/TO/KITTI_data/ # path to KITTI_data as structured in the above
PREDICTED_RESULT_DIR = './predicted_dense' # path to save predicted figures (used in test.py)
```

## Pretrained Model Download
See the next section

## Usage

### Train and validation
```
python3 main.py -b <BATCH_SIZE> -e <EPOCH> -m <SAVED_MODEL_NAME> -l <MODEL_PATH> -n <NUM_DATA> -cpu
    -b <BATCH_SIZE>
        batch size used for training and validation
    -e <EPOCH>
        the number of epoch for training and validation
    -m <SAVED_MODEL_NAME>
        the model name (be saved in SAVED_MODEL_PATH)
    -l <MODEL_PATH>
        specified the model path if you want to load previous model
    -n <NUM_DATA>
        the number of data used for training. (set -1 if you want to use all the training data (85898))
    -cpu
        if you want to use CPU to train
```
### Test
Test on **depth_selection/val_selection_cropped** data
```
python3 test.py -m <MODEL_PATH> -n <NUM_DATA> -cpu
    -n <NUM_DATA>
        the number of data used for testing. (set -1 if you want to use all the testing data (1000))
    -m <MODEL_PATH>
        the path of loaded model
    -cpu
        if you want to use CPU to test
    -s
        if you want to save predicted figure in PREDICTED_RESULT_DIR
```
### Test a pair of inputs
Run a pair of rgb and lidar image as input, and then save the predicted dense depth
```
python3 test_a_pair.py --model_path </PATH/TO/PRETRAIN_MODEL> --rgb <PATH/TO/RGB_IMAGE> --lidar <PATH/TO/LIDAR_IMAGE>
                       --saved_path </SAVED_FIGURE/PATH>
    --model_path <MODEL_PATH>
        the path of pretrained model  
    --rgb <PATH>
        the path of rgb image
    --lidar <PATH>
        the path of lidar image
    --saved_path <PATH>
        the path of saved image
```
## Tensorboard Visualization
```
tensorboard --logdir runs/
```

