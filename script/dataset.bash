#!/usr/bin/bash

src_path=~/ssd1/01_dataset/kaggle/rsna-breast-cancer-detection/train_images

python ./src/dicom2png.py --crop True \
                          --apply_window True \
                          --dst_sz 1024 \
                          --src_path $src_path \
                          --dst_path ./data/train/1024_window

# python ./src/dicom2png.py --crop True \
#                           --keep_ratio False \
#                           --voi_lut True \
#                           --dst_sz 1024 \
#                           --src_path $src_path \
#                           --dst_path ./data/train/1536x960_voi-lut

# python ./src/dicom2png.py --crop True \
#                           --dst_sz 0 \
#                           --src_path ~/ssd1/01_dataset/kaggle/rsna-breast-cancer-detection/train_images \
#                           --dst_path ./data/train/crop