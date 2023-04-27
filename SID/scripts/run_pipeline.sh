modelname="resnet18"
adv_method_list=("AutoAttack" "AutoAttack" "DeepFool" "FGSM")
adv_expname_list=("AutoAttack" "AutoAttack_Linf" "DeepFool" "FGSM")
dataname_list="CIFAR10 CIFAR100"

for dataname in $dataname_list
do
    # 1. train classifier
    bash run_classifier.sh $modelname $dataname 

    # 2. make adversarial examples
    for i in ${!adv_method_list[*]}
    do
        bash save_adv_samples.sh $modelname ${adv_method_list[$i]} ${adv_expname_list[$i]} $dataname
    done

    # 3. known attack
    for i in ${!adv_expname_list[*]}
    do
        bash known_attack.sh $modelname ${adv_expname_list[$i]} $dataname
    done

    # # 4. transfer attack
    bash run_transfer_attack.sh $modelname $dataname 

done