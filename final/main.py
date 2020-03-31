import torch
from dataloader.dataloader import get_loader
import argparse
from tqdm import tqdm
from training.train import train_val, EarlyStop
from model.DeepLidar import deepLidar
import torch.optim as optim
import os
from tensorboardX import SummaryWriter
from training.utils import *


parser = argparse.ArgumentParser(description='Depth Completion')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('-m', '--saved_model_name', type=str, default='model', help='saved_model_name')
parser.add_argument('-cpu', '--using_cpu', action='store_true', help='use cpu')
parser.add_argument('-s', '--stage', type=str, default='A', help='decide stage to train network')
parser.add_argument('-l', '--load_model', help='load model')
parser.add_argument('-n', '--num_data', type=int, default=20000, help='the number of data used to train')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.using_cpu else 'cpu'
STAGE = args.stage.upper()
SAVED_MODEL_PATH = 'saved_model'



def get_testing_img(writer):
    loader = get_loader('val', shuffle=False, num_data=1, crop=False)
    for rgb, lidar, mask, gt_depth, params, gt_surface_normal, gt_normal_mask in loader:
        writer.add_image('RGB input', rgb[0] / 255.0, 1)
        writer.add_image('lidar input', lidar[0], 1)
        writer.add_image('GroundTruth depth', normal_to_0_1(gt_depth[0]), 1)
        writer.add_image('GroundTruth surface normal', normal_to_0_1(gt_surface_normal[0]), 1)
        return rgb.to(DEVICE), lidar.to(DEVICE), mask.to(DEVICE)


def tensorboard_write(writer, epoch, train_losses, val_losses, predicted_dense, pred_surface_normal):
    loss_type = ['loss', 'loss_d', 'loss_c', 'loss_n', 'loss_normal']

    for i, t in enumerate(loss_type):
        writer.add_scalar('train_{}'.format(t), train_losses[i], epoch)
        writer.add_scalar('val_{}'.format(t), val_losses[i], epoch)
    writer.add_image('predicted_dense', normal_to_0_1(predicted_dense[0]), epoch)
    writer.add_image('pred_surface_normal', normal_to_0_1(pred_surface_normal[0]), epoch)



def main_train(model, stage):
    writer = SummaryWriter('runs/{}_{}'.format(stage, args.saved_model_name))

    loader = {'train': get_loader('train', num_data=args.num_data), \
              'val': get_loader('val', shuffle=False, num_data=1000)}
    
    testing_rgb, testing_lidar, testing_mask = get_testing_img(writer)


    early_stop = EarlyStop(patience=10, mode='min')
    for epoch in range(args.epoch):
        saved_model_path = os.path.join(SAVED_MODEL_PATH, "{}_{}_e{}".format(args.saved_model_name, stage, epoch+1))
        train_losses, val_losses = train_val(model, loader, epoch, DEVICE, stage)

        predicted_dense, pred_surface_normal = get_depth_and_normal(model, testing_rgb, testing_lidar, testing_mask)
        tensorboard_write(writer, epoch, train_losses, val_losses, predicted_dense, pred_surface_normal)

        if early_stop.stop(val_losses[0], model, epoch+1, saved_model_path):
            break

    writer.close()




def main():
    model = deepLidar().to(DEVICE)
    if args.load_model:
        dic = torch.load(args.load_model)
        state_dict = dic["state_dict"]
        model.load_state_dict(state_dict)
        print('Loss of loaded model: {:.4f}'.format(dic['val_loss']))

    main_train(model, 'N')
    main_train(model, 'D')
    main_train(model, 'A')


if __name__ == '__main__':
    main()