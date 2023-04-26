# Adversarial Sample Detection Common Project

## Training Resnet18 on Cifar 10

cd into the train_resnet folder and run train_cifar10.py
```
python train_cifar10.py
```
This will save checkpoint files into the folder ./train_resnet/checkpoints/model_ResNet18_cifar10, look for the highest numbered one that is labeled best.

Run the test_cifar10.py to test the trained model on test set. By default it loads the ./checkpoints/cifar10.pth checkpoint which is the best performing checkpoint in our experiments wtih 93.10% test accuracy.
```
python test_cifar10.py
```

## Training Resnet18 on Cifar 100

cd into the train_resnet folder and run train_cifar10.py
```
python train_cifar100.py
```
This will save checkpoint files into the folder ./train_resnet/checkpoints/model_ResNet18_cifar100, look for the highest numbered one that is labeled best

Run the test_cifar100.py file to test the trained model on test set. By default it loads the ./checkpoints/cifar100.pth checkpoint which is the best performing checkpoint in our experiments wtih 76.62% test accuracy.
```
python test_cifar100.py
```
*The best checkpoints for both Cifar10 and Cifar100 datasets are already provided in this repo.*
*The train and test scripts for both Cifar10 and Cifar100 are configured to automatically download the datasets (if not present) and place them in the ./data/ directory.*

## Running Autoattack and generating adversarial samples

1.Move the trained Resnet18 .pth checkpoint files into the directory folder AArate
Note there are already two Pre-Trained Resnet18 Model checkpoints in the ./checkpoints/ dir for cifar10 and cifar100 named cifar10.pth and cifar100.pth respectively that were used for the actual adversarial sample generation done in the report

2.cd into the AArate directory

3.Run the following sets of commands for the desired norm and dataset

For Cifar10 and norm of Linf
```
python AArate.py --dataset cifar10 --norm Linf\
    --model ../checkpoints/cifar10.pth
```

For Cifar10 and norm of L2
```
python AArate.py --dataset cifar10 --norm L2\
    --model ../checkpoints/cifar10.pth
```

For Cifar100 and norm of Linf
```
python AArate.py --dataset cifar100 --norm Linf\
    --model ../checkpoints/cifar100.pth
```

For Cifar100 and norm of L2
```
python AArate.py --dataset cifar100 --norm L2\
    --model ../checkpoints/cifar100.pth
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
python test.py --gpu 0  --dout cifar --ckpt_path ./results/ --drp ../data/\
    --ds cifar10 \
    --norm Linf \
    --ckpt ./CIFAR10-LT-stage2.pth
```

For Cifar10 and norm L2
```
python test.py --gpu 0  --dout cifar --ckpt_path ./results/ --drp ../data/\
    --ds cifar10 \
    --norm L2 \
    --ckpt ./CIFAR10-LT-stage2.pth
```

For Cifar100 and norm Linf
```
python test.py --gpu 0  --dout cifar --ckpt_path ./results/ --drp ../data/\
    --ds cifar100 \
    --norm Linf \
    --ckpt ./CIFAR100-LT-stage2.pth
```

For Cifar100 and norm L2
```
python test.py --gpu 0  --dout cifar --ckpt_path ./results/ --drp ../data/\
    --ds cifar100 \
    --norm L2 \
    --ckpt ./CIFAR100-LT-stage2.pth
```

## Detecting Adversarial Examples from Inconsistency of Spatial-Transform Domain


