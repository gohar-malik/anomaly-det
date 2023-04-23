# Adversarial Sample Detection Common Project

## Training Resnet18 on Cifar 10

## Training Resnet18 on Cifar 100

## Running Autoattack and generating adversarial samples

1.Move the trained Resnet18 .pth checkpoint files into the directory folder AArate
Note there are already two Pre-Trained Resnet18 Models in the AArate one for cifar10 and cifar100 named cifar10.pth and cifar100.pth respectively

2.cd into the AArate directory

3.Run the following sets of commands for the desired norm and dataset

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

Running the autoattack code will generate a .pth file in the results folder of adversarial samples that has the dataset and norm used contained in the name
for example for cifar10 dataset and linf norm the file would be called "aa_standard_1_1000_eps_0.03137_Linf_cifar10.pth"

## Partial and Asymmetric Contrastive Learning for Out-Of-Distribution Detection in Long-Tailed Recognition

1.Move the .pth files generated from autoattack into the OOD_detect_ICML folder, 
Important: make sure not to change the name of the .pth files from the autoattack adversarial sample generation otherwise the code will not work
Note there are already premade adversarial sample .pth files in the folder

2.cd into the OOD_detect_ICML folder

3.Run the following commands based on which dataset and norm is in use

For Cifar10 and norm Linf
```
python test.py --gpu 0  --dout cifar --ckpt_path ./results/ --drp ./datasets/\
    --ds cifar10 \
    --norm Linf \
    --ckpt ./CIFAR10-LT-stage2.pth
```

For Cifar10 and norm L2
```
python test.py --gpu 0  --dout cifar --ckpt_path ./results/ --drp ./datasets/\
    --ds cifar10 \
    --norm L2 \
    --ckpt ./CIFAR10-LT-stage2.pth
```

For Cifar100 and norm Linf
```
python test.py --gpu 0  --dout cifar --ckpt_path ./results/ --drp ./datasets/\
    --ds cifar100 \
    --norm Linf \
    --ckpt ./CIFAR100-LT-stage2.pth
```

For Cifar100 and norm L2
```
python test.py --gpu 0  --dout cifar --ckpt_path ./results/ --drp ./datasets/\
    --ds cifar100 \
    --norm L2 \
    --ckpt ./CIFAR100-LT-stage2.pth
```

## Detecting Adversarial Examples from Inconsistency of Spatial-Transform Domain


