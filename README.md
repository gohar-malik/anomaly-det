# Adversarial Sample Detection Common Project

# Training Resnet18 on Cifar 10

# Training Resnet18 on Cifar 100

# Running Autoattack

1.Move the trained Resnet18 .pth checkpoint files into the directory folder AArate
Note there are already two Pre-Trained Resnet18 Models in the AArate one for cifar10 and cifar100 named cifar10.pth and cifar100.pth respectively

2.Run the following for sets of codes for the desired norm and dataset

For Cifar10 and norm of Linf
```
python AArate.py --dataset cifar10 --norm Linf\
    --model <name_of_.pth_file_to_be_used>
```

For Cifar10 and norm of L2
```
python AArate.py --dataset cifar10 --norm L2\
    --model <name_of_.pth_file_to_be_used>
```

For Cifar100 and norm of Linf
```
python AArate.py --dataset cifar100 --norm Linf\
    --model <name_of_.pth_file_to_be_used>
```

For Cifar100 and norm of L2
```
python AArate.py --dataset cifar100 --norm L2\
    --model <name_of_.pth_file_to_be_used>
```

# Partial and Asymmetric Contrastive Learning for Out-Of-Distribution Detection in Long-Tailed Recognition

# Detecting Adversarial Examples from Inconsistency of Spatial-Transform Domain


