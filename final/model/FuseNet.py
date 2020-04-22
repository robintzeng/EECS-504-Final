import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import GlobalBlock, LocalBlock, maskBlock
class FuseNet(nn.Module):
    def __init__(self, N):
        # N: the number of fuseblock
        super(FuseNet, self).__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3 = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5 = nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv7 = nn.Conv2d(48, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.mask_block1 = maskBlock()
        self.mask_block2 = maskBlock()

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)
        self.bn6 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)

        global_layer = []
        for i in range(N):
            global_layer += [GlobalBlock(in_channel=48, C=64)]
        self.global_layer = nn.Sequential(*global_layer)


        local_layer = []
        for i in range(N):
            local_layer += [LocalBlock(in_channel=48, C=64)]
        self.local_layer = nn.Sequential(*local_layer)

    def forward(self, rgb, lidar, mask):
        b, c, w, h = rgb.size()

        lidar = torch.cat((lidar, mask), dim=1)


        l1 = self.relu(self.bn1(self.conv1(lidar)))
        l2 = self.relu(self.bn2(self.conv2(l1)))

        rl = torch.cat((rgb, lidar), dim=1)
        rl1 = self.relu(self.bn3(self.conv3(rl)))
        rl2 = self.relu(self.bn4(self.conv4(rl1)))

        x = torch.cat((rl2, l2), dim=1) # b x 48 x h/2 x w/2
        x_global = self.global_layer(x)
        x_global = F.interpolate(x_global, (w, h), mode='bilinear', align_corners=True)

        global_attn = self.mask_block1(x_global)
        x_global = self.relu(self.bn5(self.conv5(x_global))) # for dense
        x_global = self.conv6(x_global) # for dense


        x_local = self.local_layer(x)
        x_local = F.interpolate(x_local, (w, h), mode='bilinear', align_corners=True)

        local_attn = self.mask_block2(x_local)
        x_local = self.relu(self.bn6(self.conv7(x_local))) # for dense
        x_local = self.conv8(x_local) # for dense



        return x_global, x_local, global_attn, local_attn
if __name__ == "__main__":
    from dataloader.dataloader import get_loader
    loader = get_loader('val', shuffle=False, num_data=1, crop=False)
    net = FuseNet(12)
    for rgb, lidar, mask, gt_depth, gt_surface_normal in loader:
        print(rgb.size(), lidar.size())
        net(rgb, lidar)