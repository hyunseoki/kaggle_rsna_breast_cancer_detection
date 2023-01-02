import os
import cv2
from pathlib import Path
import numpy as np
import multiprocessing as mp
import argparse
import dicomsdl as dicoml
from itertools import repeat
from util import str2bool


'''
    https://www.kaggle.com/code/snnclsr/roi-extraction-using-opencv/notebook
'''
def crop_coords(img):
    """
    Crop ROI from image.
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # _, breast_mask = cv2.threshold(blur,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + 16)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)


def dicom_file_to_ary(path, sz=0, crop=True):
    dicom = dicoml.open(str(path))
    data = dicom.pixelData()
    data = (data - data.min()) / (data.max() - data.min()+1e-6)  #this cast to float32
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1 - data

    if crop:
        (x, y, w, h) = crop_coords((data * 255).astype(np.uint8))
        data = data[y:y+h, x:x+w]

    if sz != 0:
        data = cv2.resize(data, (sz, sz), interpolation=cv2.INTER_LINEAR)

    data = (data * 65535).astype(np.uint16)
    return data


def process_path(path, args):
    parent_path = str(path).replace('\\', '/').split('/')[-1]
    dst_dir = os.path.join(args.dst_path, str(args.dst_sz), parent_path)
    os.makedirs(dst_dir, exist_ok=True)

    for image_path in path.iterdir():
        save_fn = f'{dst_dir}/{image_path.stem}.png'
        processed_ary = dicom_file_to_ary(image_path, args.dst_sz)
        cv2.imwrite(
            save_fn,
            processed_ary
        )
        # im = Image.fromarray(processed_ary).resize((args.dst_sz, args.dst_sz))
        # im.save(save_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst_sz', type=int, default=1024)
    parser.add_argument('--crop', type=str2bool, default=False)
    parser.add_argument('--src_path', type=str, default=r'G:\hyunseoki\data\kaggle\rsna-breast-cancer-detection\train_images')
    parser.add_argument('--dst_path', type=str, default='./data/train')
    args = parser.parse_args()

    assert os.path.isdir(args.src_path), args.src_path
    os.makedirs(os.path.join(args.dst_path, str(args.dst_sz)), exist_ok=True)

    paths = list(Path(args.src_path).iterdir())

    import time
    start = int(time.time())
    with mp.Pool(32) as p:
        p.starmap(process_path, zip(paths, repeat(args)))
    print(f'{int(time.time() - start)}s')

if __name__ == '__main__':
    main()