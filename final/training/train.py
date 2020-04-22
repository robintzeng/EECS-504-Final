from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from training.utils import get_loss, get_predicted_depth
import torch.nn.functional as F






def train_val(model, loader, epoch, device):
    """Train and validate the model

    Returns: training and validation loss
    """

    #model, optimizer, loss_weights = get_optimizer(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    loss_weights = [0.3, 0.3, 0.5]

    train_loss, val_loss = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

    for phase in ['train', 'val']:
        total_loss, total_loss_g, total_loss_l, total_loss_w, total_loss_normal = 0, 0, 0, 0, 0
        total_pic = 0 # used to calculate average loss
        data_loader = loader[phase]
        pbar = tqdm(iter(data_loader))

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for num_batch, (rgb, lidar, mask, gt_depth, gt_normal) in enumerate(pbar):
            """
            rgb: b x 3 x 128 x 256
            lidar: b x 1 x 128 x 256
            mask: b x 1 x 128 x 256
            gt: b x 1 x 128 x 256
            params: b x 128 x 256 x 3
            """
            rgb, lidar = rgb.to(device), lidar.to(device)
            gt_depth = gt_depth.to(device)

            if phase == 'train':
                x_global, x_local, global_attn, local_attn = model(rgb, lidar)
            else:
                with torch.no_grad():
                    x_global, x_local, global_attn, local_attn = model(rgb, lidar)
            # color_path_dense: b x 2 x 128 x 256
            # normal_path_dense: b x 2 x 128 x 256
            # color_mask: b x 1 x 128 x 256
            # normal_mask: b x 1 x 128 x 256
            # surface_normal: b x 3 x 128 x 256
            predicted_dense = get_predicted_depth(x_global, x_local, global_attn, local_attn)

            w_loss = get_loss(predicted_dense, gt_depth)
            local_loss = get_loss(x_local, gt_depth)
            global_loss = get_loss(x_global, gt_depth)


            loss = 0.25*local_loss+0.25*global_loss+0.5*w_loss

            total_loss += loss.item()
            total_loss_w += w_loss.item()
            total_loss_g += global_loss.item()
            total_loss_l += local_loss.item()
            total_pic += rgb.size(0)

            if phase == 'train':
                train_loss[0] = total_loss/total_pic
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            else:
                val_loss[0] = total_loss/total_pic


            pbar.set_description('[{}] Epoch: {}; loss: {:.4f}; loss_w: {:.4f}, loss_g: {:.4f}, loss_l: {:.4f}'.\
                format(phase.upper(), epoch + 1, total_loss/total_pic, total_loss_w/total_pic, total_loss_g/total_pic, total_loss_l/total_pic))

    return train_loss, val_loss

class EarlyStop():
    """Early stop training if validation loss didn't improve for a long time"""
    def __init__(self, patience, mode = 'min'):
        self.patience = patience
        self.mode = mode

        self.best = float('inf') if mode == 'min' else 0
        self.cur_patience = 0

    def stop(self, loss, model, epoch, saved_model_path):
        update_best = loss < self.best if self.mode == 'min' else loss > self.best

        if update_best:
            self.best = loss
            self.cur_patience = 0

            torch.save({'val_loss': loss, \
                        'state_dict': model.state_dict(), \
                        'epoch': epoch}, saved_model_path+'.tar')
            print('SAVE MODEL to {}'.format(saved_model_path))
        else:
            self.cur_patience += 1
            if self.patience == self.cur_patience:
                return True
        
        return False