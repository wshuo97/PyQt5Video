import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision

from center_loss import CenterLoss
from utils import AverageMeter, Logger

from model import Net64

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='data', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true')
args = parser.parse_args()

# device
device = "cuda:{}".format(
    args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    # torchvision.transforms.RandomCrop((128, 64), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=64, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=64, shuffle=True
)
num_classes = max(len(trainloader.dataset.classes),
                  len(testloader.dataset.classes))

# net definition
start_epoch = 0
net = Net64(num_classes=num_classes)
# if args.resume:
# assert os.path.isfile(
#     "./checkpoint/ckpt.vehicle.t7"), "Error: no checkpoint file found!"
# print('Loading from checkpoint/ckpt.t7')
# checkpoint = torch.load("./checkpoint/ckpt.vehicle.t7")
# # import ipdb; ipdb.set_trace()
# net_dict = checkpoint['net_dict']
# net.load_state_dict(net_dict)
# best_acc = checkpoint['acc']
# start_epoch = checkpoint['epoch']
net = net.to(device)

# loss and optimizer

criterion_xent = nn.CrossEntropyLoss()
criterion_cent = CenterLoss(num_classes=num_classes, feat_dim=2, use_gpu=True)
# net = Net64()
optimizer_model = torch.optim.SGD(
    net.parameters(), lr=args.lr, weight_decay=1e-2, momentum=0.9)
optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=0.5)
alpha = 1
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(
#     net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.

# train function for each epoch


def train(epoch):
    print("\nEpoch : %d" % (epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        features, outputs = net(inputs)
        # features, outputs = features.to(device), outputs.to(device)

        # print(inputs.device)
        # print(labels.device)
        # print(features.device)
        # print(outputs.device)
        # print(labels.long().device)
        # print(next(net.parameters()).device)

        loss_xent = criterion_xent(outputs, labels.long())
        loss_cent = criterion_cent(features, labels.long())
        loss_cent *= 1
        loss = loss_xent + loss_cent

        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()

        # loss = criterion(outputs, labels)

        # backward
        # optimizer.zero_grad()
        loss.backward()
        # optimizer.step()

        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / 1)
        optimizer_centloss.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

        # print
        if (idx+1) % interval == 0:
            end = time.time()
            # print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            #     100.*(idx+1)/len(trainloader), end-start, training_loss /
            #     interval, correct, total, 100.*correct/total
            # ))
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})"
                  .format(idx+1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, cent_losses.val, cent_losses.avg))
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss /
                interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()

    return train_loss/len(trainloader), 1. - correct/total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            features, outputs = net(inputs)
            features, outputs = features.to(device), outputs.to(device)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            loss_xent = criterion_xent(outputs, labels.long())
            loss_cent = criterion_cent(features, labels.long())
            loss_cent *= 1
            loss = loss_xent + loss_cent

            # loss = criterion(outputs, labels)
            optimizer_model.step()
            # not impact on the learning of centers
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / 1)
            optimizer_centloss.step()

            test_loss += loss.item()
            # correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        # print("Testing ...")
        # end = time.time()
        # print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
        #     100.*(idx+1)/len(testloader), end-start, test_loss /
        #     len(testloader), correct, total, 100.*correct/total
        # ))

    acc = correct*100./total
    err = 100.-acc

    # saving checkpoint
    # acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.person.net64.t7')

    # return test_loss/len(testloader), 1. - correct/total
    return test_loss/len(testloader), acc, err


# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")

# lr decay


def lr_decay():
    global optimizer
    for params in optimizer_model.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    for epoch in range(start_epoch, start_epoch+1000):
        train_loss, train_err = train(epoch)
        test_loss, test_acc, test_err = test(epoch)
        print("Accuracy (%): {}\t Error rate (%): {}".format(test_acc, test_err))
        # draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch+1) % 100 == 0:
            lr_decay()


if __name__ == '__main__':
    main()
