# Adversarial Sample Detection Common Project
## Environment
We recomment `conda` package manager to create a virtual environment for this code. Run the following commands one by one to create, activate and install all dependencies in a conda env.
```
conda env create -f environment.yml
conda activate anomalydet
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

## Training Resnet18 on Cifar 10

cd into the `./train_resnet` folder and run `train_cifar10.py`.
```
python train_cifar10.py
```
This will save checkpoint files into the folder `./train_resnet/checkpoints/model_ResNet18_cifar10`, look for the highest numbered one that is labeled best.

Run the `test_cifar10.py` to test the trained model on test set. By default it loads the `./checkpoints/cifar10.pth` checkpoint which is the best performing checkpoint in our experiments wtih 93.10% test accuracy.
```
python test_cifar10.py
```

## Training Resnet18 on Cifar 100

cd into the `./train_resnet` folder and run `train_cifar100.py`.
```
python train_cifar100.py
```
This will save checkpoint files into the folder `./train_resnet/checkpoints/model_ResNet18_cifar100`, look for the highest numbered one that is labeled best.

Run the `test_cifar100.py` file to test the trained model on test set. By default it loads the `./checkpoints/cifar100.pth` checkpoint which is the best performing checkpoint in our experiments wtih 76.62% test accuracy.
```
python test_cifar100.py
```
*The best checkpoints for both Cifar10 and Cifar100 datasets are already provided in this repo.*
*The train and test scripts for both Cifar10 and Cifar100 are configured to automatically download the datasets (if not present) and place them in the `./data/` directory.*


## Running Autoattack and generating adversarial samples

1. Move the trained Resnet18 `.pth` checkpoint files into the directory folder AArate.
**Note** there are already two Pre-Trained Resnet18 Model checkpoints in the `./checkpoints/` dir for cifar10 and cifar100 named `cifar10.pth` and `cifar100.pth` respectively that were used for the actual adversarial sample generation done in the report.

2. cd into the `./AArate` directory.

3. Run the following sets of commands for the desired norm and dataset

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

Running the autoattack code will generate a `.pth` file in the results folder of adversarial samples that has the dataset and norm used contained in the name
for example for cifar10 dataset and linf norm the file would be called `aa_standard_1_1000_eps_0.03137_Linf_cifar10.pth`.

## Partial and Asymmetric Contrastive Learning for Out-Of-Distribution Detection in Long-Tailed Recognition

1. Move the `.pth` files generated from autoattack into the OOD_detect_ICML folder, 
**Important**: make sure not to change the name of the .pth files from the autoattack adversarial sample generation otherwise the code will not work
**Note** there are already premade adversarial sample .pth files in the folder.

2. cd into the `OOD_detect_ICML` folder.

3. Run the following commands based on which dataset and norm is in use

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

### Preparation

0. cd into the `./SID` folder

1. Then download the `results.zip` for this gdrive [link](https://drive.google.com/file/d/1XEKEix9eKRDov1b7ReNYqBLQ6S4P8lTG/view?usp=sharing), and extract it using the comman below:
```
gdown 1XEKEix9eKRDov1b7ReNYqBLQ6S4P8lTG
unzip results.zip
```
*This will download all the trained models and generated adversarial samples to be used in this experiment.*

### Running Adversarial Sample Detection

0. cd into the `./SID/scripts/` folder.

1. [**Skippable**] First step is to train the DWT Resnet18 models on both the CIFAR10 and CIFAR100 datasets. The trained checkpoints of both these models are already provided in the repo, so you can skip this step. However, if you want to train the models run the following commands:
```
bash run_classifier.sh resnet18 CIFAR10
```
Or
```
bash run_classifier.sh resnet18 CIFAR100
```

2. [**Skippable**] Second step is to generate Adversarial Samples using three Adversarial Attacks, AutoAttack, DeepFool and FGSM. We have already generated these sample and included it in the `results.zip` file downloaded earlier. However, if you want to generate samples yourself, run the following command depending on which dataset and adversarial attack to use:
```
bash save_adv_samples.sh resnet18 $adv_method $adv_expname $dataname
```
*where `$adv_method` can be one of [AutoAttack, DeepFool, FGSM], `$adv_expname` can be one of [AutoAttack, AutoAttack_Linf, DeepFool, FGSM] and `$dataset` can be one of [CIFAR10, CIFAR1000]*

3. [**Skippable**] Next step is to train the SID adversarial detection model on different adversarial attack samples and test the adversarial detection in a white box setting. To perform this step, run the following command:
```
bash known_attack.sh resnet18 $adv_expname $dataname
```
*where `$adv_expname` can be one of [AutoAttack, AutoAttack_Linf, DeepFool, FGSM] and `$dataset` can be one of [CIFAR10, CIFAR1000]*

This will save results of the specified attack and dataset in the `./SID/results/$dataset/known_attack_results/resnet18/$adv_expname/result.json`

> **Note**: If you wish to run step 1,2 and 3, please remove the `./SID/results` folder first.

4. Finally, we will can perform transfer tests, where we test SID on one adservarial attack using the trained model on a different adversarial attack. For this, run the following command:
```
bash run_transfer_attack.sh resnet18 CIFAR10
```
OR
```
bash run_transfer_attack.sh resnet18 CIFAR100
```
This will save the results, showing AUROC, in a table format, as shown below, in the `./SID/results/$(CIFAR10,CIFAR100)/transfer_attack_results/resnet18/transfer_results.csv` file.

The rows of the table indicate the source attack method which were used in training, and the columns indicate the target attack methods which were tested. Sample table for CIFAR10 is shown below. The result in bold is from the SID trained on FGPS and tested on AutoAttack with Linf norm.
|                 | DeepFool | AutoAttack_Linf | FGSM  | AutoAttack |
|-----------------|----------|-----------------|-------|------------|
| DeepFool        | 87.76    | 57.34           | 63.8  | 52.61      |
| AutoAttack_Linf | 22.67    | 99.93           | 22.32 | 27.97      |
| FGSM            | 56.2     | **93.94**           | 99.99 | 12.49      |
| AutoAttack      | 72.21    | 39.45           | 45.04 | 63.15      |


