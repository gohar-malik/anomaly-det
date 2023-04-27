import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class FDResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,in_C=12,wave='sym17',mode='append'):
        super(FDResNet, self).__init__()

        self.wave = wave
        self.DWT = DWTForward(J=1, wave = self.wave, mode='symmetric',Requirs_Grad=True).cuda()
        self.FDmode = mode

        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_C, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.wavelets(x, self.FDmode)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def wavelets(self, x, FDmode):
        # 'haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus',\n
        # 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor'\n\n
        x = x.cuda().reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        Yl, Yh = self.DWT(x)
        output = self.plugdata(x, Yl, Yh, FDmode)
        return output
    
    def plugdata(self, x, Yl, Yh, mode):
        if mode == 'append':
            output = torch.zeros(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
            output = output.cuda()
            output[:, 0:3, :] = Yl[:, :, :]
            output[:, 3:6, :] = Yh[0][:, 0, :, :]
            output[:, 6:9, :] = Yh[0][:, 1, :, :]
            output[:, 9:12, :] = Yh[0][:, 2, :, :]
            output = output.reshape(x.shape[0], x.shape[1] * 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        elif mode == 'avg':
            output = torch.zeros(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
            output = output.cuda()
            output[:, 0, :] = torch.mean(Yl[:, :, :], axis=1)
            output[:, 1, :] = torch.mean(Yh[0][:, 0, :, :], axis=1)
            output[:, 2, :] = torch.mean(Yh[0][:, 1, :, :], axis=1)
            output[:, 3, :] = torch.mean(Yh[0][:, 2, :, :], axis=1)
            output = output.reshape(x.shape[0], 4, Yl[:, :, :].shape[2], Yl[:, :, :].shape[3])
        return output

def resnet18(num_classes, use_wavelet_transform=False, **kwargs):
    if use_wavelet_transform:
        return FDResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_C=12)
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)



if __name__ == "__main__":
    net = resnet18(10).cuda()
    net_dwt = resnet18(10,True).cuda()
    y = net(torch.randn(1, 3, 32, 32).cuda())
    y_dwt = net_dwt(torch.randn(1, 3, 32, 32).cuda())
    print(y)
    print(y_dwt)
