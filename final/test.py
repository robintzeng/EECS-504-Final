from tqdm import tqdm
import argparse
import os
import torch
import torch.nn as nn
from dataloader.image_reader import *
from model.DeepLidar import deepLidar
import torch.nn.functional as F
from PIL import Image

parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('--model_path', default='/home/tmt/CV_final/cys/EECS-545-Final/final/saved_model/model.tar',
                    help='load model')
parser.add_argument('-n', '--num_testing_image', type=int, default=10, 
                    help='The number of testing image to be runned')
args = parser.parse_args()



SAVED_DIR = 'predicted_dense'
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'


def rmse(pred, gt):
    dif = gt[np.where(gt>0)] - pred[np.where(gt>0)]
    error = np.sqrt(np.mean(dif**2))
    return error   

def test(model, rgb, lidar, mask):
    model.eval()

    # to gpu
    model = model.to(DEVICE)
    rgb = rgb.to(DEVICE)
    lidar = lidar.to(DEVICE)
    mask = mask.to(DEVICE)

    criterion = nn.MSELoss()
    with torch.no_grad():
        color_path_dense, normal_path_dense, color_attn, normal_attn, surface_normal = model(rgb, lidar, mask)
        
        # get predicted dense depth from 2 pathways
        pred_color_path_dense = color_path_dense[:, 0, :, :] # b x 128 x 256
        pred_normal_path_dense = normal_path_dense[:, 0, :, :]

        # get attention map of 2 pathways
        color_attn = torch.squeeze(color_attn) # b x 128 x 256
        normal_attn = torch.squeeze(normal_attn) # b x 128 x 256

        # softmax 2 attention map
        pred_attn = torch.zeros_like(color_path_dense) # b x 2 x 128 x 256
        pred_attn[:, 0, :, :] = color_attn
        pred_attn[:, 1, :, :] = normal_attn
        pred_attn = F.softmax(pred_attn, dim=1) # b x 2 x 128 x 256

        color_attn, normal_attn = pred_attn[:, 0, :, :], pred_attn[:, 1, :, :]

        # get predicted dense from weighted sum of 2 path way
        predicted_dense = pred_color_path_dense * color_attn + pred_normal_path_dense * normal_attn # b x 128 x 256


        #loss = torch.sqrt(criterion(predicted_dense, gt.squeeze(1)))*1000

        return torch.squeeze(predicted_dense).cpu().numpy()

def get_testing_img_paths():
    gt_folder = os.path.join('/home', 'tmt', 'CV_data', 'selection', 'depth_selection', 'val_selection_cropped', 'groundtruth_depth')
    rgb_folder = os.path.join('/home', 'tmt', 'CV_data', 'selection', 'depth_selection', 'val_selection_cropped', 'image')
    lidar_folder = os.path.join('/home', 'tmt', 'CV_data', 'selection', 'depth_selection', 'val_selection_cropped', 'velodyne_raw')

    gt_filenames = sorted([img for img in os.listdir(gt_folder)])
    rgb_filenames = sorted([img for img in os.listdir(rgb_folder)])
    lidar_filenames = sorted([img for img in os.listdir(lidar_folder)])

    gt_paths = [os.path.join(gt_folder, fn) for fn in gt_filenames]
    rgb_paths = [os.path.join(rgb_folder, fn) for fn in rgb_filenames]
    lidar_paths = [os.path.join(lidar_folder, fn) for fn in lidar_filenames]

    return rgb_paths, lidar_paths, gt_paths

def main():
    # get image paths
    rgb_paths, lidar_paths, gt_paths = get_testing_img_paths()
    num_testing_image = len(rgb_paths) if args.num_testing_image == -1 else args.num_testing_image

    # load model
    model = deepLidar()
    state_dict = torch.load(args.model_path, map_location=DEVICE)["state_dict"]
    model.load_state_dict(state_dict)

    transformer = image_transforms()
    pbar = tqdm(range(num_testing_image))
    running_error = 0
    for idx in pbar:
        # read image
        rgb = read_rgb(rgb_paths[idx]) # h x w x 3
        lidar, mask = read_lidar(lidar_paths[idx]) # h x w x 1
        gt = read_gt(gt_paths[idx]) # h x w x 1

        # transform numpy to tensor and add batch dimension
        rgb = transformer(rgb).unsqueeze(0)
        lidar, mask = transformer(lidar).unsqueeze(0), transformer(mask).unsqueeze(0)
        
        # saved file path
        fn = os.path.basename(rgb_paths[idx])
        saved_path = os.path.join(SAVED_DIR, fn)

        # run model
        pred = test(model, rgb, lidar, mask)
        pred = np.where(pred <= 0.0, 0.9, pred)

        gt = gt.reshape(gt.shape[0], gt.shape[1])
        rmse_loss = rmse(pred, gt)*1000

        running_error += rmse_loss
        mean_error = running_error / (idx + 1)
        pbar.set_description('Mean error: {:.4f}'.format(mean_error))

        # save image
        pred_show = pred * 256.0
        pred_show = pred_show.astype('uint16')
        res_buffer = pred_show.tobytes()
        img = Image.new("I", pred_show.T.shape)
        img.frombytes(res_buffer, 'raw', "I;16")
        img.save(saved_path)
if __name__ == '__main__':
    main()