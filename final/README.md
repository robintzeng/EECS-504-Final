# DeepLiDAR

## Usage
### Train and validation
```
python3 main.py
```
### Test
```
python3 test.py
```

## Data
- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset from their website. Use the following scripts to extract corresponding RGB images from the raw dataset. 
```bash
./download/rgb_train_downloader.sh
./download/rgb_val_downloader.sh
```
The downloaded rgb files will be stored in the `../data/data_rgb` folder. The overall code, data, and results directory is structured as follows (updated on Oct 1, 2019)

data_depth_annotated: ground truth data (dense depth)

data_depth_velodyne: sparse data (LiDAR)

data_rgb: RGB image

```
.
├── CV_data
|   ├── data_depth_annotated
|   |   ├── train
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |   |   ├── proj_depth
|   |   |   |   |   ├── groundtruch
|   |   |   |   |   |   ├── image02
|   |   |   |   |   |   ├── image03
|   |   ├── val
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |   |   ├── proj_depth
|   |   |   |   |   ├── groundtruch
|   |   |   |   |   |   ├── image02
|   |   |   |   |   |   ├── image03
|   ├── data_depth_velodyne
|   |   ├── train
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |   |   ├── proj_depth
|   |   |   |   |   ├── velodyne_raw
|   |   |   |   |   |   ├── image_02
|   |   |   |   |   |   ├── image_03
|   |   ├── val
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |   |   ├── proj_depth
|   |   |   |   |   ├── velodyne_raw
|   |   |   |   |   |   ├── image_02
|   |   |   |   |   |   ├── image_03
|   ├── depth_selection
|   |   ├── test_depth_completion_anonymous
|   |   ├── test_depth_prediction_anonymous
|   |   ├── val_selection_cropped
|   └── data_rgb
|   |   ├── train
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |   |   ├── image_02
|   |   |   |   ├── image_03
|   |   ├── val
|   |   |   ├── 2011_09_26_drive_0001_sync
|   |   |   |   ├── image_02
|   |   |   |   ├── image_03
```