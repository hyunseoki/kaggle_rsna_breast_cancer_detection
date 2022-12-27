import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import multiprocessing as mp
import argparse
import dicomsdl as dicoml
from itertools import repeat


# def dicom_file_to_ary(path):
#     dicom = pydicom.read_file(path)
#     data = dicom.pixel_array
#     if dicom.PhotometricInterpretation == "MONOCHROME1":
#         data = np.amax(data) - data
#     data = data - np.min(data)
#     data = data / np.max(data)
#     data = (data * 255).astype(np.uint8)
#     return data


def dicom_file_to_ary(path, sz):
    dicom = dicoml.open(str(path))
    data = dicom.pixelData()
    # dicom = pydicom.dcmread(path)
    # data = apply_voi_lut(dicom.pixel_array, dicom)
    # data = (data - data.min()) / (data.max() - data.min())
    data = (data - data.min()) / (data.max() - data.min()+1e-6)  #this cast to float32
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1 - data
    # data = cv2.resize(data, (sz, sz))
    # data = (data * 255).astype(np.uint8)
    data = cv2.resize(data, (sz, sz), interpolation=cv2.INTER_LINEAR)
    data = (data * 65535).astype(np.uint16)
    return data


def img2roi(img):
    # Binarize the image
    bin_img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]

    # Make contours around the binarized image, keep only the largest contour
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # Find ROI from largest contour
    ys = contour.squeeze()[:, 0]
    xs = contour.squeeze()[:, 1]
    roi =  img[np.min(xs):np.max(xs), np.min(ys):np.max(ys)]

    return roi


def preprocess(src):

    return


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