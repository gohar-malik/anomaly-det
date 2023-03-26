import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from resnet import ResNet18
from utils import init_weights, split_weights
from utils import WarmUpLR, AMP, LARS

def train(epoch):

    start = time.time()
    net.train()
    total_loss = 0.0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        labels = labels.to(device)
        images = images.to(device)
        
        # def closure():
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
            # return outputs, loss
        
        # outputs, loss = optimizer.step(closure)
        optimizer.step()
        total_loss += loss.item()
        
        if epoch <= warm:
            warmup_scheduler.step()

    finish = time.time()
    print(f'Epoch: {epoch}')
    print(f'Training:\tLoss: {total_loss/len(cifar100_training_loader):0.4f}\tLR: {optimizer.param_groups[0]["lr"]:0.6f}\tTime: {finish-start:.2f}s')

@torch.no_grad()
def eval_training(epoch=0):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        labels = labels.to(device)
        images = images.to(device)

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    print(f'Testing:\tLoss: {test_loss / len(cifar100_test_loader.dataset):.4f}\tAcc: {correct.float() / len(cifar100_test_loader.dataset):.4f}\tTime:{ finish - start:.2f}s')

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=0, help='gpu id to use')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-optim', default='sgd',help='optimizer to use: [sgd, amp or lars]')
    parser.add_argument('-xavier', action='store_true', default=False, help='whether to use xavier init')
    parser.add_argument('-nbd', action='store_true', default=False, help='whether to use no bias decay')
    parser.add_argument('-warm', type=int, default=0, help='number of warmup steps to use for LR')
    parser.add_argument('-trainsched', default='multistep', help='LR scheduler to use: [multistep or onecycle]')
    parser.add_argument('-ckpt', default='./checkpoints/model_ResNet18_cifar100',help='directory of model for saving checkpoint')
    parser.add_argument('-ckptepoch', type=int, default=25 ,help='directory of model for saving checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    print(f"Configuration: {args}\n")

    
    ### device config
    use_cuda = (args.gpu is not None) and (torch.cuda.is_available())
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    print(f"Using Device: {device}")

    
    ### network initialize
    net = ResNet18(num_classes=100).to(device)
    if args.xavier:
        net = init_weights(net)
    
    
    #### data loaders
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(cifar100_training, shuffle=True, num_workers=4, batch_size=args.b)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(cifar100_test, shuffle=True, num_workers=4, batch_size=args.b)

    
    ### training config
    loss_function = nn.CrossEntropyLoss()

    params = net.parameters()
    if args.nbd:
        params = split_weights(net)
    optimizer = {"sgd": optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4),
                 "amp": AMP(params, lr=args.lr, epsilon=0.5, momentum=0.9, weight_decay=5e-4),
                 "lars": LARS(params, lr=0.04, weight_decay=5e-4)}[args.optim]

    warm = args.warm
    warmup_ratio = warm * 1.0 / args.epochs
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    milestones = [150,225] #[100,150,180,210,240,270] #[60, 120, 160]

    train_scheduler = {"multistep": optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, 
                                                                   gamma=0.1), #0.2,
                       "onecycle": optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                        steps_per_epoch=iter_per_epoch, epochs=int(args.epochs*1.0), anneal_strategy='linear', 
                        pct_start=warmup_ratio, div_factor=25, final_div_factor=10, cycle_momentum=False,
                        base_momentum=0.9, max_momentum=0.9)}[args.trainsched]
    
    
    ### create checkpoint folder to save model
    checkpoint_path = args.ckpt
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{epoch}_{type}.pth')


    ### training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        
        train(epoch)
        acc = eval_training(epoch)
        if epoch > warm:
            train_scheduler.step(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > milestones[0] and best_acc < acc:
            weights_path = checkpoint_path.format(epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % args.ckptepoch:
            weights_path = checkpoint_path.format(epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)