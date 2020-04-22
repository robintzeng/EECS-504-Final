import torch
import torch.nn as nn
import torch.nn.functional as F



class maskBlock(nn.Module):
    def __init__(self):
        super(maskBlock, self).__init__()
        self.mask_block = self.make_layers()

    def make_layers(self):
        in_channels = 48
        cfg = [48, 48]

        out_channels = 1
        layers = []

        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=True)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v

        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.mask_block(x)



class GlobalBlock(nn.Module):

    def __init__(self, in_channel, C):
        super(GlobalBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, C, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(C)

        self.conv2 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(C)

        self.conv3 = nn.Conv2d(in_channel, C, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(C)

        self.conv4 = nn.Conv2d(C, in_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(in_channel)

        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        b, c, w, h = x.size()

        # first branch
        x_1 = self.relu(self.bn3(self.conv3(x)))

        # second branch
        x_2 = self.relu(self.bn1(self.conv1(x)))
        x_2 = self.relu(self.bn2(self.conv2(x_2)))
        x_2 = F.interpolate(x_2, (w, h), mode='bilinear', align_corners=True)

        x_3 = x_1 + x_2
        x_4 = self.relu(self.bn4(self.conv4(x_3))) + x
        return x_4


class LocalBlock(nn.Module):

    def __init__(self, in_channel, C):
        super(LocalBlock, self).__init__()

        self.conv_down1 = nn.Conv2d(in_channel, C, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn_down1 = nn.BatchNorm2d(C)
        self.conv_down2 = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn_down2 = nn.BatchNorm2d(C)
        self.conv_down3 = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn_down3 = nn.BatchNorm2d(C)

        self.conv1 = nn.Conv2d(in_channel, C, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(C)
        self.conv3 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(C)
        self.conv4 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(C)
        self.conv5 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(C)
        self.conv6 = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(C)
        self.conv7 = nn.Conv2d(C, in_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7 = nn.BatchNorm2d(in_channel)


        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        b, c, w, h = x.size()

        x0_conv = self.relu(self.bn1(self.conv1(x)))

        x1 = self.relu(self.bn_down1(self.conv_down1(x)))
        x1_conv = self.relu(self.bn2(self.conv2(x1)))

        x2 = self.relu(self.bn_down2(self.conv_down2(x1)))
        x2_conv = self.relu(self.bn3(self.conv3(x2)))     

        x3 = self.relu(self.bn_down3(self.conv_down3(x2)))
        x3_conv = self.relu(self.bn4(self.conv4(x3)))

        b, c, w2, h2 = x2_conv.size()
        x5 = self.relu(self.bn5(self.conv5(x2_conv + F.interpolate(x3_conv, (w2, h2), mode='bilinear', align_corners=True))))

        b, c, w1, h1 = x1_conv.size()
        x6 = self.relu(self.bn6(self.conv6(x1_conv + F.interpolate(x5, (w1, h1), mode='bilinear', align_corners=True))))
        x7 = self.relu(self.bn7(self.conv7(x0_conv + F.interpolate(x6, (w, h), mode='bilinear', align_corners=True))))
        return x + x7
        """# first branch
        x_1 = self.relu(self.bn3(self.conv3(x)))

        # second branch
        x_2 = self.relu(self.bn1(self.conv1(x)))
        x_2 = self.relu(self.bn2(self.conv2(x_2)))
        x_2 = F.interpolate(x_2, (w, h), mode='bilinear', align_corners=True)

        x_3 = x_1 + x_2
        x_4 = self.relu(self.bn4(self.conv4(x_3))) + x"""
        return x_4

class ResBlock(nn.Module):
    
    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A', use_dropout=False):
        super(ResBlock, self).__init__()
        
        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        #else:
        #    if res_option == 'A':
        #        self.projection = IdentityPadding(num_filters, channels_in, stride)
        #    elif res_option == 'B':
        #        self.projection = ConvProjection(num_filters, channels_in, stride)
        #    elif res_option == 'C':
        #        self.projection = AvgPoolPadding(num_filters, channels_in, stride)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        residual = self.bn3(self.conv3(residual))

        if self.use_dropout:
            out = self.dropout(out)

        out += residual
        out = self.relu2(out)
        return out



class UpProject(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpProject, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, kernel_size=2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        batch_size = x.size(0)
        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))

        out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github

        out1_3 = self.conv1_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github

        out1_4 = self.conv1_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))

        out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github

        out2_3 = self.conv2_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github

        out2_4 = self.conv2_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out


def adaptive_cat(out_conv, out_deconv, out_depth_up):
    out_deconv = out_deconv[:, :, :out_conv.size(2), :out_conv.size(3)]
    out_depth_up = out_depth_up[:, :, :out_conv.size(2), :out_conv.size(3)]
    return torch.cat((out_conv, out_deconv, out_depth_up), 1)