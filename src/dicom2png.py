import os
import cv2
from pathlib import Path
import numpy as np
import multiprocessing as mp
import argparse
import dicomsdl as dicoml
from pydicom.pixel_data_handlers.util import apply_voi_lut
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


'''
    https://www.kaggle.com/code/bobdegraaf/dicomsdl-voi-lut
'''
def apply_voi_lut(dicom, image):
    # Load only the variables we need
    center = dicom["WindowCenter"]
    width = dicom["WindowWidth"]
    bits_stored = dicom["BitsStored"]
    voi_lut_function = dicom["VOILUTFunction"]

    # For sigmoid it's a list, otherwise a single value
    if isinstance(center, list):
        center = center[0]
    if isinstance(width, list):
        width = width[0]

    # Set y_min, max & range
    y_min = 0
    y_max = float(2**bits_stored - 1)
    y_range = y_max

    # Function with default LINEAR (so for Nan, it will use linear)
    if voi_lut_function == "SIGMOID":
        image = y_range / (1 + np.exp(-4 * (image - center) / width)) + y_min
    else:
        # Checks width for < 1 (in our case not necessary, always >= 750)
        center -= 0.5
        width -= 1

        below = image <= (center - width / 2)
        above = image > (center + width / 2)
        between = np.logical_and(~below, ~above)

        image[below] = y_min
        image[above] = y_max
        if between.any():
            image[between] = (
                ((image[between] - center) / width + 0.5) * y_range + y_min
            )

    return image


def image_resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized


'''
    https://www.kaggle.com/code/radek1/how-to-process-dicom-images-to-pngs
'''
def dicom_file_to_ary(path, sz=0, keep_ratio=True, crop=True, voi_lut=True):
    dicom = dicoml.open(str(path))
    data = dicom.pixelData()

    if voi_lut:
        data = apply_voi_lut(dicom, data)

    data = (data - data.min()) / (data.max() - data.min()+1e-6)  #this cast to float32
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1 - data

    if crop:
        (x, y, w, h) = crop_coords((data * 255).astype(np.uint8))
        data = data[y:y+h, x:x+w]

    ## maintaining the aspect ratio, longer side resized to 1024
    if sz!=0:
        if keep_ratio:
            h, w = data.shape
            if w > h:
                data = image_resize(data, width=sz)
            else:
                data = image_resize(data, height=sz)
        else:
            data = cv2.resize(data, (sz, sz), interpolation=cv2.INTER_LINEAR)

    data = (data * 65535).astype(np.uint16)
    return data


def process_path(path, args):
    parent_path = str(path).replace('\\', '/').split('/')[-1]
    dst_dir = os.path.join(args.dst_path, parent_path)
    os.makedirs(dst_dir, exist_ok=True)

    for image_path in path.iterdir():
        save_fn = f'{dst_dir}/{image_path.stem}.png'
        processed_ary = dicom_file_to_ary(
            path=image_path,
            sz=args.dst_sz,
            keep_ratio=args.keep_ratio,
            crop=args.crop,
            voi_lut=args.voi_lut,
        )
        cv2.imwrite(
            save_fn,
            processed_ary
        )
        # im = Image.fromarray(processed_ary).resize((args.dst_sz, args.dst_sz))
        # im.save(save_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default=r'G:\hyunseoki\data\kaggle\rsna-breast-cancer-detection\train_images')
    parser.add_argument('--dst_path', type=str, default='./data/train')
    parser.add_argument('--dst_sz', type=int, default=1024)
    parser.add_argument('--crop', type=str2bool, default=False)
    parser.add_argument('--keep_ratio', type=str2bool, default=False)
    parser.add_argument('--voi_lut', type=str2bool, default=False)
    args = parser.parse_args()

    assert os.path.isdir(args.src_path), args.src_path
    os.makedirs(os.path.join(args.dst_path), exist_ok=True)

    paths = list(Path(args.src_path).iterdir())

    import time
    start = int(time.time())
    with mp.Pool(32) as p:
        p.starmap(process_path, zip(paths, repeat(args)))
    print(f'{int(time.time() - start)}s')

if __name__ == '__main__':
    main()