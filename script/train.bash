#!/usr/bin/bash
use_wandb=True
epochs=40
# fold_idx=0
# base_path=/home/hyunseoki/ssd1/02_src/kaggle_rsna_breast_cancer_detection/data/train/1536x960
# base_path=/home/hyunseoki/ssd1/02_src/kaggle_rsna_breast_cancer_detection/data/train/1536x960_voi-lut
base_path=/home/hyunseoki/ssd1/02_src/kaggle_rsna_breast_cancer_detection/data/train/1024_crop
# base_path=/home/hyunseoki/ssd1/02_src/kaggle_rsna_breast_cancer_detection/data/train/1024_voi-lut
# base_path=/home/hyunseoki/ssd1/02_src/kaggle_rsna_breast_cancer_detection/data/train/1024_voi-lut_keep_ratio


# for fold_idx in 0 1 2 3 4
# do
#     python ./src/main.py --use_wandb $use_wandb \
#                          --base_path $base_path \
#                          --epochs $epochs \
#                          --kfold $fold_idx \
#                          --backbone resnet \
#                          --batch_size 8 \
#                          --loss_weight 20 \
#                          --train_oversample True \
#                          --loss_type bce \
#                          --comments 3090_resnet/fold$fold_idx
# done


for fold_idx in 0 1 2 3 4
do
    python ./src/main.py --use_wandb $use_wandb \
                         --base_path $base_path \
                         --seed 777 \
                         --batch_size 8 \
                         --epochs $epochs \
                         --kfold $fold_idx \
                         --loss_weight 20 \
		            	 --train_oversample True \
                         --loss_type bce \
                         --label_smoothing 0.1 \
                         --comments 3090_seed777_bce_label-smoothing/fold$fold_idx
done


for fold_idx in 0 1 2 3 4
do
    python ./src/main.py --use_wandb $use_wandb \
                         --base_path $base_path \
                         --seed 777 \
                         --batch_size 8 \
                         --epochs $epochs \
                         --kfold $fold_idx \
                         --use_gem True \
                         --loss_weight 20 \
		            	 --train_oversample True \
                         --loss_type bce \
                         --label_smoothing 0.1 \
                         --comments 3090_seed777_bce_gem_label-smoothing/fold$fold_idx
done
# for fold_idx in 1 2 3 4
# do
#     python ./src/main.py --use_wandb $use_wandb \
#                          --base_path $base_path \
#                          --kfold $fold_idx \
#                          --backbone convnext \
#                          --epochs $epochs \
#                          --batch_size 8 \
#                          --loss_weight 20 \
#                          --train_oversample True \
#                          --loss_type bce \
#                          --comments 3090_convnext_lr-test/lr3e-5/fold$fold_idx
# done


# base_path=/home/hyunseoki/ssd1/02_src/kaggle_rsna_breast_cancer_detection/data/train/1536x960

# lr=1e-4
# python ./src/main.py --use_wandb $use_wandb \
#                         --base_path $base_path \
#                         --lr $lr \
#                         --epochs $epochs \
#                         --batch_size 6 \
#                         --backbone convnext \
#                         --loss_weight 20 \
#                         --train_oversample True \
#                         --loss_type bce \
#                         --comments 3090_convnext_newsz/lr$lr
