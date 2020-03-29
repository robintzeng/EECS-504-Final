from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


MatI=np.zeros((8,128,256), dtype=np.float32)
for i in range(MatI.shape[1]):
    MatI[:,i,:]= i
MatJ = np.zeros((8,128,256), dtype=np.float32)
for j in range(MatJ.shape[2]):
    MatJ[:,:,j] = j

MatI = np.reshape(MatI, [8,128,256, 1]).astype(np.float32)
MatJ = np.reshape(MatJ, [8,128,256, 1]).astype(np.float32)
MatI = torch.FloatTensor(MatI).cuda()
MatJ = torch.FloatTensor(MatJ).cuda()
MatI = torch.squeeze(MatI)
MatJ = torch.squeeze(MatJ)


k = np.array([[0,1,0],[1,-4,1],[0,1,0]])
k1 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
k2 = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])

def nomal_loss(pred, targetN,params,depthI,depthJ):
    depthI = depthI.permute(0, 2, 3, 1)
    depthJ = depthJ.permute(0, 2, 3, 1)

    predN_1 = torch.zeros_like(targetN)
    predN_2 = torch.zeros_like(targetN)

    f = params[:, :, :, 0]
    cx = params[:, :, :, 1]
    cy = params[:, :, :, 2]

    z1 = depthJ - pred
    z1 = torch.squeeze(z1)
    depthJ = torch.squeeze(depthJ)
    predN_1[:, :, :, 0] = ((MatJ - cx) * z1 + depthJ) * 1.0 / f
    predN_1[:, :, :, 1] = (MatI - cy) * z1 * 1.0 / f
    predN_1[:, :, :, 2] = z1

    z2 = depthI - pred
    z2 = torch.squeeze(z2)
    depthI = torch.squeeze(depthI)
    predN_2[:, :, :, 0] = (MatJ - cx) * z2  * 1.0 / f
    predN_2[:, :, :, 1] = ((MatI - cy) * z2 + depthI) * 1.0 / f
    predN_2[:, :, :, 2] = z2

    predN = torch.cross(predN_1, predN_2)
    pred_n = F.normalize(predN)
    pred_n = pred_n.contiguous().view(-1, 3)
    target_n = targetN.contiguous().view(-1, 3)

    loss_function = nn.CosineEmbeddingLoss()
    loss = loss_function(pred_n, target_n, (torch.Tensor(pred_n.size(0)).cuda().fill_(1.0)))
    return loss


def cal_loss(pred,predC,predN,target,params,normal):
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    w = torch.from_numpy(k1).float().unsqueeze(0).unsqueeze(0).cuda()
    conv1.weight = nn.Parameter(w)
    depthJ1 = conv1(pred)
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    w2 = torch.from_numpy(k2).float().unsqueeze(0).unsqueeze(0).cuda()
    conv2.weight = nn.Parameter(w2)
    depthI1 = conv2(pred)

    valid_mask = (target > 0.0).detach()
    pred = pred.permute(0, 2, 3, 1)
    predN = predN.permute(0, 2, 3, 1)
    predC = predC.permute(0, 2, 3, 1)

    loss4 = nomal_loss(pred, normal, params, depthI1, depthJ1)

    pred_n = pred[valid_mask]
    predN_n = predN[valid_mask]
    predC_n = predC[valid_mask]
    target_n = target[valid_mask]

    loss2 = mse_loss(predC_n, target_n)
    loss3 = mse_loss(predN_n, target_n)
    loss1_function = nn.MSELoss(size_average=True)
    loss1 =  loss1_function(pred_n, target_n)

    loss = 0.5 * loss1 + 0.3 * loss2 + 0.3 * loss3 + 0.1 * loss4

    return loss,loss1,loss2,loss3,loss4


def train(model, optimizer, loader, epoch, device):
    pbar = tqdm(iter(loader))
    model.train()
    
    for rgb, lidar, mask, gt, params in pbar:
        rgb, lidar, mask = rgb.to(device), lidar.to(device), mask.to(device)
        gt, params = gt.to(device), params.to(device) 

        color_path_dense, normal_path_dense, color_mask, normal_mask, surface_normal = model(rgb, lidar, mask)
        # color_path_dense: 3 x 128 x 256
        # normal_path_dense: 2 x 128 x 256
        # color_mask: 1 x 128 x 256
        # normal_mask: 1 x 128 x 256
        # surface_normal: 3 x 128 x 256

        pred_color_path_dense = color_path_dense[:, 0, :, :] 
        pred_normal_path_dense = normal_path_dense[:, 0, :, :]

        color_mask = torch.squeeze(color_mask) # 128 x 256
        normal_mask = torch.squeeze(normal_mask) # 128 x 256

        temp_mask = torch.zeros_like(color_path_dense)
        temp_mask[:, 0, :, :] = color_mask
        temp_mask[:, 1, :, :] = normal_mask
        pred_mask = F.softmax(temp_mask)
        pred_mask_c = pred_mask[:, 0, :, :]
        pred_mask_n = pred_mask[:, 1, :, :]
        pred = pred_color_path_dense * pred_mask_c + pred_normal_path_dense * pred_mask_n

        pred = torch.unsqueeze(pred, 1)
        pred_color_path_dense = torch.unsqueeze(pred_color_path_dense, 1)
        pred_normal_path_dense = torch.unsqueeze(pred_normal_path_dense, 1)

        batch_size, c, h, w = surface_normal.size()
        surface_normal = surface_normal.permute(0, 2, 3, 1).contiguous().view(-1, c)
        surface_normal = F.normalize(surface_normal)
        surface_normal = surface_normal.view(batch_size, h, w, c)

        output_normal = torch.zeros_like(surface_normal)
        output_normal[:, :, :, 0] = -surface_normal[:, :, :, 0]
        output_normal[:, :, :, 1] = -surface_normal[:, :, :, 2]
        output_normal[:, :, :, 2] = -surface_normal[:, :, :, 1]

        #loss = cal_loss(pred, pred_color_path_dense, pred_normal_path_dense, gt, params, output_normal)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()