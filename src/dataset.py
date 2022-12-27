import torch
import cv2
import os
import numpy as np


class RSNADataset(torch.utils.data.Dataset):
    def __init__(self, label_df, base_path, transforms, is_test=False):
        self.label_df = label_df
        self.base_path = base_path
        self.transforms = transforms
        self.is_test = is_test

    def __getitem__(self, idx):
        img_fn = os.path.join(
            self.base_path,
            f"{self.label_df.loc[idx, 'patient_id']}/{self.label_df.loc[idx, 'image_id']}.png"
        ) 
        assert os.path.isfile(img_fn), img_fn

        img = cv2.imread(img_fn, cv2.IMREAD_ANYDEPTH)
        img = img.astype(np.float32)

        if self.transforms:
            img = self.transforms(image=img)['image']

        img = torch.as_tensor(img/img.max(), dtype=torch.float)

        sample = {'input': img}
        if not self.is_test:
            label = self.label_df.loc[idx, "cancer"]
            label = torch.tensor(label, dtype=torch.float).unsqueeze(0)
            sample['target'] = label

        return sample

    def __len__(self):
        return len(self.label_df)


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    from transforms import get_train_transforms

    label_df = pd.read_csv(r'F:\hyunseoki\kaggle_mammography\data\train.csv')
    dataset = RSNADataset(
        label_df=label_df,
        base_path=r'F:\hyunseoki\kaggle_mammography\data\512',
        transforms=get_train_transforms()
    )

    sample = dataset[0]

    plt.title(sample['target'])
    plt.imshow(sample['input'].squeeze(), 'gray')
    plt.show()