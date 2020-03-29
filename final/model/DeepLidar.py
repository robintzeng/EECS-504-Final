import torch
import torch.nn as nn
from model.module import *


class deepCompletionUnit(nn.Module):
    def __init__(self):
        super(deepCompletionUnit, self).__init__()

        self.conv_sparse1 = ResBlock(channels_in=2, num_filters=32, stride=1)
        self.conv_sparse2 = ResBlock(channels_in=32, num_filters=99, stride=2)
        self.conv_sparse3 = ResBlock(channels_in=99, num_filters=195, stride=2)
        self.conv_sparse4 = ResBlock(channels_in=195, num_filters=387, stride=2)
        self.conv_sparse5 = ResBlock(channels_in=387, num_filters=515, stride=2)
        self.conv_sparse6 = ResBlock(channels_in=515, num_filters=512, stride=2)

        self.conv_rgb1 = ResBlock(channels_in=3, num_filters=32, stride=1)
        self.conv_rgb2 = ResBlock(channels_in=32, num_filters=64, stride=2)
        self.conv_rgb3 = ResBlock(channels_in=64, num_filters=128, stride=2)
        self.conv_rgb3_1 = ResBlock(channels_in=128, num_filters=128, stride=1)
        self.conv_rgb4 = ResBlock(channels_in=128, num_filters=256, stride=2)
        self.conv_rgb4_1 = ResBlock(channels_in=256, num_filters=256, stride=1)
        self.conv_rgb5 = ResBlock(channels_in=256, num_filters=256, stride=2)
        self.conv_rgb5_1 = ResBlock(channels_in=256, num_filters=256, stride=1)
        self.conv_rgb6 = ResBlock(channels_in=256, num_filters=512, stride=2)
        self.conv_rgb6_1 = ResBlock(channels_in=512, num_filters=512, stride=1)

        self.dense_conv5 = nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.dense_conv4 = nn.Conv2d(515, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.dense_conv3 = nn.Conv2d(387, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.dense_conv2 = nn.Conv2d(195, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.dense_conv1 = nn.Conv2d(99, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.upsample4 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsample3 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsample2 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsample1 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)
        self.upsample0 = nn.ConvTranspose2d(3, 3, 4, 2, 1, bias=False)

    def forward(self, rgb, lidar, mask):
        sparse_input = torch.cat((lidar, mask), 1)
        s = self.conv_sparse1(sparse_input)
        s = self.conv_sparse2(s)
        s = self.conv_sparse3(s)
        s = self.conv_sparse4(s)
        s = self.conv_sparse5(s)
        s = self.conv_sparse6(s)

        rgb_out = self.conv_rgb1(rgb)
        rgb_out1 = self.conv_rgb2(rgb_out)
        rgb_out2 = self.conv_rgb3_1(self.conv_rgb3(rgb_out1))
        rgb_out3 = self.conv_rgb4_1(self.conv_rgb4(rgb_out2))
        rgb_out4 = self.conv_rgb5_1(self.conv_rgb5(rgb_out3))
        rgb_out5 = self.conv_rgb6_1(self.conv_rgb6(rgb_out4)) + s

        dense5 = self.dense_conv5(rgb_out5)
        dense6_up = self.upsample4(dense5)
        dense7_up = self.upsample3(dense6_up)
        dense8_up = self.upsample2(dense7_up)
        dense = self.upsample1(dense8_up)
        dense = self.upsample0(dense)
        print(dense.size())
class deepLidar(nn.Module):
    def __init__(self):
        super(deepLidar, self).__init__()
        self.normal = deepCompletionUnit()
        self.color_path = deepCompletionUnit()
        self.normal_path = deepCompletionUnit()

    def forward(self, rgb, lidar, mask):
        self.normal(rgb, lidar, mask)



class test_model(nn.Module):
    def __init__(self):
        super(deepLidar, self).__init__()
        

    def forward(self, rgb, lidar, mask):
        sparse = torch.cat((lidar, mask), 1)