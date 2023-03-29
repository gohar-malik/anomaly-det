import os
import argparse
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

import sys
sys.path.insert(0,'..')

from resnet import ResNet18

from autoattack import AutoAttack

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=0, help='gpu id to use')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--norm', type=str, default='Linf')
parser.add_argument('--epsilon', type=float, default=8./255.)
parser.add_argument('--model', type=str, default='./model_test.pth')
parser.add_argument('--n_ex', type=int, default=1000)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--log_path', type=str, default='./log_file.txt')
parser.add_argument('--version', type=str, default='standard')
parser.add_argument('--state-path', type=Path, default=None)

args = parser.parse_args()

#edit below lines to change norm, model and dataset
args.norm = 'Linf' #choose either 'Linf' or 'L2'
args.model = './cifar10.pth'
dataset = 'cifar10' #choose either 'cifar10' or 'cifar100'

### device config
use_cuda = (args.gpu is not None) and (torch.cuda.is_available())
device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
print(f"Using Device: {device}")

if dataset == 'cifar10':
    classes = 10
    mean = (0.49139968, 0.48215827 ,0.44653124)
    std = (0.24703233,0.24348505,0.26158768)
elif dataset == 'cifar100':
    classes = 100
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#load .pt model
model = ResNet18(num_classes=classes)
ckpt = torch.load(args.model)
model.load_state_dict(ckpt)
model.cuda()
model.eval()

transform_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ]
transform_chain = transforms.Compose(transform_list)

if dataset == 'cifar10':
    item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
elif dataset == 'cifar100':
    item = datasets.CIFAR100(root=args.data_dir, train=False, transform=transform_chain, download=True)

test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path, version=args.version)

l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)

with torch.no_grad():
    adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],bs=args.batch_size, state_path=args.state_path)

    torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}_{}_{}.pth'.format(args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon, args.norm, dataset))
