import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module import FuseBlock
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

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(inplace=True)

        block_layer = []
        for i in range(N):
            block_layer += [FuseBlock(in_channel=48, C=64)]
        self.block_layer = nn.Sequential(*block_layer)


    def forward(self, rgb, lidar, mask):
        b, c, w, h = rgb.size()

        lidar = torch.cat((lidar, mask), dim=1)

        l1 = self.relu(self.bn1(self.conv1(lidar)))
        l2 = self.relu(self.bn2(self.conv2(l1)))

        rl = torch.cat((rgb, lidar), dim=1)
        rl1 = self.relu(self.bn3(self.conv3(rl)))
        rl2 = self.relu(self.bn4(self.conv4(rl1)))

        x = torch.cat((rl2, l2), dim=1) # b x 48 x h/2 x w/2
        x = self.block_layer(x)
        x = F.interpolate(x, (w, h), mode='bilinear', align_corners=True)

        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return x
if __name__ == "__main__":
    from dataloader.dataloader import get_loader
    loader = get_loader('val', shuffle=False, num_data=1, crop=False)
    net = FuseNet(12)
    for rgb, lidar, mask, gt_depth, gt_surface_normal in loader:
        print(rgb.size(), lidar.size())
        net(rgb, lidar)