from tqdm import tqdm
import argparse
import os
import torch
import torch.nn as nn
from dataloader.image_reader import *
from model.DeepLidar import deepLidar
import torch.nn.functional as F
from PIL import Image
from training.utils import *
from env import PREDICTED_RESULT_DIR, KITTI_DATASET_PATH
from skimage import color
from surface_normal import normals_from_depth

parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-m', '--model_path', help='loaded model path')
parser.add_argument('-n', '--num_testing_image', type=int, default=10, 
                    help='The number of testing image to be runned')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-s', '--save_fig', action='store_true', help='save predicted result or not')

args = parser.parse_args()



DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'



def rmse(pred, gt):
    dif = gt[np.where(gt>0)] - pred[np.where(gt>0)]
    error = np.sqrt(np.mean(dif**2))
    return error   

def test(model, rgb, lidar, mask, lab):
    model.eval()

    model = model.to(DEVICE)
    rgb = rgb.to(DEVICE)
    lab = lab.to(DEVICE)
    lidar = lidar.to(DEVICE)
    mask = mask.to(DEVICE)

    with torch.no_grad():
        color_path_dense, lab_path_dense, color_attn, lab_attn = model(rgb, lidar, mask, lab)

        predicted_dense, pred_color_path_dense, pred_normal_path_dense = get_predicted_depth(color_path_dense, lab_path_dense, color_attn, lab_attn)

        
        return torch.squeeze(predicted_dense).cpu()


INTRINSICS = {
    "2011_09_26": (721.5377, 609.5593, 172.8540),
    "2011_09_28": (707.0493, 604.0814, 180.5066),
    "2011_09_29": (718.3351, 600.3891, 181.5122),
    "2011_09_30": (707.0912, 601.8873, 183.1104),
    "2011_10_03": (718.8560, 607.1928, 185.2157),
}

def get_testing_img_paths():
    gt_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'groundtruth_depth')
    rgb_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'image')
    lidar_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'velodyne_raw')

    gt_normal_folder = os.path.join(KITTI_DATASET_PATH, 'depth_selection', 'val_selection_cropped', 'gt_normal')
    if not os.path.exists(gt_normal_folder):
        os.makedirs(gt_normal_folder)
    for fn in os.listdir(gt_folder):
        if not os.path.exists(os.path.join(gt_normal_folder, fn)):
            intrinsic = INTRINSICS[fn[:10]]
            normals_from_depth(os.path.join(gt_folder, fn), os.path.join(gt_normal_folder, fn),
                               intrinsics=intrinsic,
                               window_size=15,
                               max_rel_depth_diff=0.1
                            )

    gt_filenames = sorted([img for img in os.listdir(gt_folder)])
    rgb_filenames = sorted([img for img in os.listdir(rgb_folder)])
    lidar_filenames = sorted([img for img in os.listdir(lidar_folder)])

    gt_paths = [os.path.join(gt_folder, fn) for fn in gt_filenames]
    rgb_paths = [os.path.join(rgb_folder, fn) for fn in rgb_filenames]
    lidar_paths = [os.path.join(lidar_folder, fn) for fn in lidar_filenames]
    gt_normal_paths = [os.path.join(gt_normal_folder, fn) for fn in gt_filenames]
    return rgb_paths, lidar_paths, gt_paths, gt_normal_paths

def main():
    # get image paths
    rgb_paths, lidar_paths, gt_paths, gt_normal_paths = get_testing_img_paths()

    # set the number of testing images
    num_testing_image = len(rgb_paths) if args.num_testing_image == -1 else args.num_testing_image

    # load model
    model = deepLidar()
    dic = torch.load(args.model_path, map_location=DEVICE)
    state_dict = dic["state_dict"]
    model.load_state_dict(state_dict)
    print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))
    print('The number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    transformer = image_transforms()
    pbar = tqdm(range(num_testing_image))
    running_error = 0

    for idx in pbar:
        # read image
        rgb = read_rgb(rgb_paths[idx]) # h x w x 3
        lidar, mask = read_lidar(lidar_paths[idx]) # h x w x 1
        gt = read_gt(gt_paths[idx]) # h x w x 1
        gt_normal, _ = read_normal(gt_normal_paths[idx]) # h x w x 1


        # transform numpy to tensor and add batch dimension
        rgb, gt_normal = transformer(rgb).unsqueeze(0), transformer(gt_normal).unsqueeze(0)
        lidar, mask = transformer(lidar).unsqueeze(0), transformer(mask).unsqueeze(0)
        
        # saved file path
        fn = os.path.basename(rgb_paths[idx])
        saved_path = os.path.join(PREDICTED_RESULT_DIR, fn)

        # run model
        pred = test(model, rgb, lidar, mask, gt_normal).numpy()
        pred = np.where(pred <= 0.0, 0.9, pred)

        gt = gt.reshape(gt.shape[0], gt.shape[1])
        rmse_loss = rmse(pred, gt)*1000

        running_error += rmse_loss
        mean_error = running_error / (idx + 1)
        pbar.set_description('Mean error: {:.4f}'.format(mean_error))

        if args.save_fig:
            # save image
            pred_show = pred * 256.0
            pred_show = pred_show.astype('uint16')
            res_buffer = pred_show.tobytes()
            img = Image.new("I", pred_show.T.shape)
            img.frombytes(res_buffer, 'raw', "I;16")
            img.save(saved_path)


if __name__ == '__main__':
    main()