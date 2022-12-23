import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms():
    return A.Compose(
        [
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(5),
                translate_percent=(0.05, 0.05),
                cval=0,
                cval_mask=0,
                p=0.3,
            ),
            A.Flip(p=0.5),
            ToTensorV2(always_apply=True),
        ],
    )


def get_valid_transforms():
    return A.Compose(
        [
            ToTensorV2(always_apply=True),
        ],
    )


def get_test_transforms():
    return A.Compose(
        [
            ToTensorV2(p=1.0),
        ],
    )