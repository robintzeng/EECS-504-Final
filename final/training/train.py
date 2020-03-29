from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def cal_loss(dense, c_dense, n_dense, gt, params, normals):
    """
    dense: b x 1 x 128 x 256
    c_dense: b x 1 x 128 x 256
    n_dense: b x 1 x 128 x 256
    gt: b x 1 x 128 x 256
    params: b x 3 x 128 x 256
    normals: b x 128 x 256 x 3
    """
    valid_mask = (gt > 0.0).detach() # b x 1 x 128 x 256

    gt = gt[valid_mask]
    dense, c_dense, n_dense = dense[valid_mask], c_dense[valid_mask], n_dense[valid_mask]

    criterion = nn.MSELoss()
    loss_d = torch.sqrt(criterion(dense, gt))
    loss_c = torch.sqrt(criterion(c_dense, gt))
    loss_n = torch.sqrt(criterion(n_dense, gt))
    
    return loss_d, loss_c, loss_n



def train(model, optimizer, loader, epoch, device):
    model.train()
    pbar = tqdm(iter(loader))
    total_loss, total_loss_d, total_loss_c, total_loss_n = 0, 0, 0, 0
    total_pic = 0
    for num_batch, (rgb, lidar, mask, gt, params) in enumerate(pbar):
        """
        rgb: b x 3 x 128 x 256
        lidar: b x 1 x 128 x 256
        mask: b x 1 x 128 x 256
        gt: b x 1 x 128 x 256
        params: b x 128 x 256 x 3
        """
        rgb, lidar, mask = rgb.to(device), lidar.to(device), mask.to(device)
        gt, params = gt.to(device), params.to(device)

        color_path_dense, normal_path_dense, color_attn, normal_attn, surface_normal = model(rgb, lidar, mask)
        # color_path_dense: b x 2 x 128 x 256
        # normal_path_dense: b x 2 x 128 x 256
        # color_mask: b x 1 x 128 x 256
        # normal_mask: b x 1 x 128 x 256
        # surface_normal: b x 3 x 128 x 256

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

        predicted_dense = predicted_dense.unsqueeze(1)
        pred_color_path_dense = pred_color_path_dense.unsqueeze(1) 
        pred_normal_path_dense = pred_normal_path_dense.unsqueeze(1)

        # normalize surface normal
        b, c, h, w = surface_normal.size()
        surface_normal = surface_normal.permute(0, 2, 3, 1).contiguous().view(-1, c)
        surface_normal = F.normalize(surface_normal, p=2, dim=1) # perform Lp normalization over specific dimension
        surface_normal = surface_normal.view(b, h, w, c)

        # TODO
        output_normal = torch.zeros_like(surface_normal)
        output_normal[:, :, :, 0] = -surface_normal[:, :, :, 0]
        output_normal[:, :, :, 1] = -surface_normal[:, :, :, 2]
        output_normal[:, :, :, 2] = -surface_normal[:, :, :, 1]

        loss_d, loss_c, loss_n = cal_loss(predicted_dense, pred_color_path_dense, pred_normal_path_dense, gt, params, output_normal)
        loss = 0.5 * loss_d + 0.25 * loss_c + 0.25 * loss_n

        total_loss += loss.item()
        total_loss_d += loss_d.item()
        total_loss_c += loss_c.item()
        total_loss_n += loss_n.item()

        total_pic += b

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        pbar.set_description('[TRAIN] Epoch: {}; Avg loss: {:.4f}; loss_d: {:.2f}, loss_c: {:.2f}, loss_n: {:.2f}'.\
            format(epoch + 1, 1e3*total_loss/total_pic , 1e3*total_loss_d/total_pic, \
            1e3*total_loss_c/total_pic, 1e3*total_loss_n/total_pic))



def val(model, optimizer, loader, epoch, device):
    pbar = tqdm(iter(loader))
    model.eval()
    total_loss, total_loss_d, total_loss_c, total_loss_n = 0, 0, 0, 0
    total_pic = 0
    for num_batch, (rgb, lidar, mask, gt, params) in enumerate(pbar):
        """
        rgb: b x 3 x 128 x 256
        lidar: b x 1 x 128 x 256
        mask: b x 1 x 128 x 256
        gt: b x 1 x 128 x 256
        params: b x 128 x 256 x 3
        """
        with torch.no_grad():
            rgb, lidar, mask = rgb.to(device), lidar.to(device), mask.to(device)
            gt, params = gt.to(device), params.to(device)

            color_path_dense, normal_path_dense, color_attn, normal_attn, surface_normal = model(rgb, lidar, mask)
            # color_path_dense: b x 2 x 128 x 256
            # normal_path_dense: b x 2 x 128 x 256
            # color_mask: b x 1 x 128 x 256
            # normal_mask: b x 1 x 128 x 256
            # surface_normal: b x 3 x 128 x 256

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

            predicted_dense = predicted_dense.unsqueeze(1)
            pred_color_path_dense = pred_color_path_dense.unsqueeze(1) 
            pred_normal_path_dense = pred_normal_path_dense.unsqueeze(1)

            # normalize surface normal
            b, c, h, w = surface_normal.size()
            surface_normal = surface_normal.permute(0, 2, 3, 1).contiguous().view(-1, c)
            surface_normal = F.normalize(surface_normal, p=2, dim=1) # perform Lp normalization over specific dimension
            surface_normal = surface_normal.view(b, h, w, c)

            # TODO
            output_normal = torch.zeros_like(surface_normal)
            output_normal[:, :, :, 0] = -surface_normal[:, :, :, 0]
            output_normal[:, :, :, 1] = -surface_normal[:, :, :, 2]
            output_normal[:, :, :, 2] = -surface_normal[:, :, :, 1]

            loss_d, loss_c, loss_n = cal_loss(predicted_dense, pred_color_path_dense, pred_normal_path_dense, gt, params, output_normal)
            loss = 0.5 * loss_d + 0.25 * loss_c + 0.25 * loss_n

            total_loss += loss.item()
            total_loss_d += loss_d.item()
            total_loss_c += loss_c.item()
            total_loss_n += loss_n.item()

            total_pic += b

            pbar.set_description('[VAL] Epoch: {}; Avg loss: {:.4f}; loss_d: {:.2f}, loss_c: {:.2f}, loss_n: {:.2f}'.\
                format(epoch + 1, 1e3*total_loss/total_pic , 1e3*total_loss_d/total_pic, \
                1e3*total_loss_c/total_pic, 1e3*total_loss_n/total_pic))
    return total_loss /  total_pic


class EarlyStop():
    def __init__(self, saved_model_path, patience, mode = 'min'):
        self.saved_model_path = saved_model_path
        self.patience = patience
        self.mode = mode

        self.best = float('inf') if mode == 'min' else 0
        self.cur_patience = 0

    def stop(self, loss, model, epoch):
        update_best = loss < self.best if self.mode == 'min' else loss > self.best

        if update_best:
            self.best = loss
            self.cur_patience = 0

            torch.save({'val_loss': loss, 'state_dict': model.state_dict(), 'epoch': epoch}, self.saved_model_path)
            print('SAVE MODEL to {}'.format(self.saved_model_path))
        else:
            self.cur_patience += 1
            if self.patience == self.cur_patience:
                return True
        
        return False