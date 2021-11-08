import torch
import torch.nn as nn
from torch.nn import functional as F

from .do_conv_pytorch import DOConv2d


class BasicBlockdo(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlockdo, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = DOConv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = DOConv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = DOConv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                DOConv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                DOConv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)

def make_layersdo(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlockdo(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlockdo(c_out, c_out), ]
    return nn.Sequential(*blocks)

class Net128do(nn.Module):
    def __init__(self, num_classes=10, reid=False):
        super(Net128do, self).__init__()
        # 3 128 128
        self.conv = nn.Sequential(
            # nn.Conv2d(in,out,kernel);
            DOConv2d(3, 32, 3, stride=1, padding=1),  # conv 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DOConv2d(32, 32, 3, stride=1, padding=1),  # conv 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel,stried),
            nn.MaxPool2d(3, 2, padding=1),  # max pool 3
        )  # 32 64 64
        # make_layers(c_in,c_out,repeat_times,is_downsample)
        self.layer4 = make_layersdo(32, 32, 2, False)
        # 32 64 64
        self.layer5 = make_layersdo(32, 32, 2, False)
        # 32 64 64
        self.layer6 = make_layersdo(32, 64, 2, True)
        # 64 32 64
        self.layer7 = make_layersdo(64, 64, 2, False)
        # 64 32 64
        self.layer8 = make_layersdo(64, 128, 2, True)
        # 128 16 16
        self.layer9 = make_layersdo(128, 128, 2, False)
        # 128 16 16
        self.reid = reid
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128*16*16, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.feature = nn.Sequential(
            nn.Linear(128, 2),
            # nn.BatchNorm1d(8),
            nn.ReLU(inplace=True)
        )
        # 128
        # self.classifier = nn.Linear(128, 6)
        self.classifier = nn.Linear(128, num_classes)
        # nn.Sequential
        # nn.Linear(128*16*8, 128),
        # nn.BatchNorm1d(128),
        # nn.ReLU(inplace=True),
        # nn.Dropout(),
        # 10

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.dense(x)
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        y = self.classifier(x)
        x = self.feature(x)
        # f = x
        return x, y


class Net64do(nn.Module):
    def __init__(self, num_classes=10, reid=False):
        super(Net64do, self).__init__()
        # 3 128 128
        self.conv = nn.Sequential(
            # nn.Conv2d(in,out,kernel);
            DOConv2d(3, 32, 3, stride=1, padding=1),  # conv 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DOConv2d(32, 32, 3, stride=1, padding=1),  # conv 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel,stried),
            nn.MaxPool2d(3, 2, padding=1),  # max pool 3
        )  # 32 64 64
        # make_layers(c_in,c_out,repeat_times,is_downsample)
        self.layer4 = make_layersdo(32, 32, 2, False)
        # 32 64 64
        self.layer5 = make_layersdo(32, 32, 2, False)
        # 32 64 64
        self.layer6 = make_layersdo(32, 64, 2, True)
        # 64 32 64
        self.layer7 = make_layersdo(64, 64, 2, False)
        # 64 32 64
        self.layer8 = make_layersdo(64, 128, 2, True)
        # 128 16 16
        self.layer9 = make_layersdo(128, 128, 2, False)
        # 128 16 16
        self.reid = reid
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128*16*8, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.feature = nn.Sequential(
            nn.Linear(128, 2),
            # nn.BatchNorm1d(2),
            nn.ReLU(inplace=True)
        )
        # 128
        # self.classifier = nn.Linear(128, 6)
        self.classifier = nn.Linear(128, num_classes)
        # nn.Sequential
        # nn.Linear(128*16*8, 128),
        # nn.BatchNorm1d(128),
        # nn.ReLU(inplace=True),
        # nn.Dropout(),
        # 10

    def forward(self, x):
        x = self.conv(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        # classifier
        y = self.classifier(x)
        x = self.feature(x)
        # f = x
        return x, y

