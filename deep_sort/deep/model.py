import torch
import torch.nn as nn
import torch.nn.functional as F

from .center_loss import CenterLoss


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
                c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
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


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Net128(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        super(Net128, self).__init__()
        # 3 128 128
        self.conv = nn.Sequential(
            # nn.Conv2d(in,out,kernel);
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # 64 64 64
        self.layer1 = make_layers(64, 64, 2, False)
        # 64 64 64
        self.layer2 = make_layers(64, 128, 2, True)
        # 128 32 32
        self.layer3 = make_layers(128, 256, 2, True)
        # 256 16 16
        self.layer4 = make_layers(256, 512, 2, True)
        # 512 8 8
        self.avgpool = nn.AvgPool2d((8, 8), 1)
        # 512 1 1
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            # print(x.size())
            return x
        # classifier
        x = self.classifier(x)
        return x

    # def forward2(self, x):
        print("1:", x.shape)
        x = self.conv(x)
        print("2:", x.shape)
        x = self.layer1(x)
        print("3:", x.shape)
        x = self.layer2(x)
        print("4:", x.shape)
        x = self.layer3(x)
        print("5:", x.shape)
        x = self.layer4(x)
        print("6:", x.shape)
        x = self.avgpool(x)
        print("7:", x.shape)
        x = x.view(x.size(0), -1)
        # B x 128
        print("8:", x.shape)
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            # print(x.size())
            return x
        # classifier
        print("9:", x.div(x.norm(p=2, dim=1, keepdim=True)).shape)
        x = self.classifier(x)
        print("10:", x.shape)
        return x


class Net64(nn.Module):
    def __init__(self, num_classes=10, reid=False):
        super(Net64, self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            # nn.Conv2d(in,out,kernel);
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # conv 1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # conv 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel,stried),
            nn.MaxPool2d(3, 2, padding=1),  # max pool 3
        )  # 32 64 32
        # make_layers(c_in,c_out,repeat_times,is_downsample)
        self.layer4 = make_layers(32, 32, 2, False)
        # 32 64 32
        self.layer5 = make_layers(32, 32, 2, False)
        # 32 64 32
        self.layer6 = make_layers(32, 64, 2, True)
        # 64 32 16
        self.layer7 = make_layers(64, 64, 2, False)
        # 64 32 16
        self.layer8 = make_layers(64, 128, 2, True)
        # 128 16 8
        self.layer9 = make_layers(128, 128, 2, False)
        # 128 16 8
        self.reid = reid
        self.dense = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(128*16*8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.feature = nn.Sequential(
            nn.Linear(128, 2),
            nn.ReLU(inplace=True)
        )
        # 128
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

    def forward2(self, x):
        print("1:", x.shape)
        x = self.conv(x)
        print("2:", x.shape)
        x = self.layer4(x)
        print("3:", x.shape)
        x = self.layer5(x)
        print("4:", x.shape)
        x = self.layer6(x)
        print("5:", x.shape)
        x = self.layer7(x)
        print("6:", x.shape)
        x = self.layer8(x)
        print("7:", x.shape)
        x = self.layer9(x)
        print("8:", x.shape)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        print("9:", x.shape)
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            print("12:", x.size())
            return x
        # classifier
        print("10:", x.div(x.norm(p=2, dim=1, keepdim=True)).shape)
        y = self.classifier(x)
        print("11:", x.shape)
        f = self.feature(x)
        return f, y

    # def forward2(self, x):
        # print("1:",x.shape)
        x = self.conv(x)
        # print("2:",x.shape)
        x = self.layer1(x)
        # print("3:",x.shape)
        x = self.layer2(x)
        # print("4:",x.shape)
        x = self.layer3(x)
        # print("5:",x.shape)
        x = self.layer4(x)
        # print("6:",x.shape)
        x = self.avgpool(x)
        # print("7:",x.shape)
        x = x.view(x.size(0), -1)
        # B x 128
        # print("8:",x.shape)
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            # print(x.size())
            return x
        # classifier
        # print("9:",x.div(x.norm(p=2,dim=1,keepdim=True)).shape)
        x = self.classifier(x)
        # print("10:",x.shape)
        return x

class Net(nn.Module):
    def __init__(self, num_classes=751 ,reid=False):
        super(Net,self).__init__()
        # 3 128 64
        self.conv = nn.Sequential(
            nn.Conv2d(3,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,padding=1),
        )
        # 32 64 32
        self.layer1 = make_layers(64,64,2,False)
        # 32 64 32
        self.layer2 = make_layers(64,128,2,True)
        # 64 32 16
        self.layer3 = make_layers(128,256,2,True)
        # 128 16 8
        self.layer4 = make_layers(256,512,2,True)
        # 256 8 4
        self.avgpool = nn.AvgPool2d((8,4),1)
        # 256 1 1 
        self.reid = reid
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        # B x 128
        if self.reid:
            x = x.div(x.norm(p=2,dim=1,keepdim=True))
            return x
        # classifier
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = Net()
    x = torch.randn(4,3,128,64)
    y = net(x)

# if __name__ == '__main__':
#     # criterion = nn.CrossEntropyLoss()
#     # center_loss = CenterLoss(num_classes=10, feat_dim=2, use_gpu=False)

#     criterion_xent = nn.CrossEntropyLoss()
#     criterion_cent = CenterLoss(num_classes=10, feat_dim=128, use_gpu=False)
#     net = Net64()
#     optimizer_model = torch.optim.SGD(
#         net.parameters(), lr=0.001, weight_decay=1e-2, momentum=0.9)
#     optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=0.5)

#     # params = list(net.parameters()) + list(center_loss.parameters())
#     # optimizer = torch.optim.SGD(params, lr=0.1) # here lr is the overall learning rate

#     # optimizer = torch.optim.SGD(
#     #     params, lr=0.1, momentum=0.9, weight_decay=1e-2)

#     x = torch.randn(4, 3, 128, 64)
#     labels = torch.randint(0, 10, (4, 1)).squeeze(1)
#     features, outputs = net(x)

#     # print(f.shape)
#     # print(y.shape)

#     alpha = 1
#     # print("!!!!!!!! a = ", center_loss(f, y_gt))
#     # print("!!!!!!!! b = ", criterion(y, y_gt))
#     # criterion()
#     # loss = center_loss(f, y_gt) * alpha + criterion(y, y_gt)
#     # optimizer.zero_grad()

#     loss_xent = criterion_xent(outputs, labels.long())
#     loss_cent = criterion_cent(features, labels.long())
#     loss_cent *= 1
#     loss = loss_xent + loss_cent

#     optimizer_model.zero_grad()
#     optimizer_centloss.zero_grad()

#     loss.backward()

#     optimizer_model.step()
#     # by doing so, weight_cent would not impact on the learning of centers
#     for param in criterion_cent.parameters():
#         param.grad.data *= (1. / 1)
#     optimizer_centloss.step()

#     xent_losses = AverageMeter()
#     cent_losses = AverageMeter()
#     losses = AverageMeter()

#     losses.update(loss.item(), labels.size(0))
#     xent_losses.update(loss_xent.item(), labels.size(0))
#     cent_losses.update(loss_cent.item(), labels.size(0))

#     print(loss)

#     # l2_penalty = l2_weight * sum([(p**2).sum() for p in net.hidden.parameters()])
#     # import ipdb; ipdb.set_trace()
#     # print("===================")
