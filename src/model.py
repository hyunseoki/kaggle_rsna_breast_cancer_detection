
import torch
import torch.nn as nn
import timm
from nextvit import nextvit_base


class Classifier(nn.Module):
    def __init__(self, backbone='tf_efficientnetv2_s'):
        super().__init__()
        self.model = timm.create_model(
            model_name=backbone,
            pretrained=True,
            in_chans=1,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.model(x)


class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            model_name='seresnext50_32x4d',
            pretrained=False,
            in_chans=1,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )
        self.model.fc = nn.Linear(2048, 1)
        
    def forward(self, dcm):
        out = self.model(dcm)
        return out


class NextViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.model = nextvit_base(pretrained=False, num_classes=1)

    def forward(self, dcm):
        x = (dcm - self.mean) / self.std
        out = self.model(x)
        return out

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = NextViT()
    # model = Classifier()

    print(model)
    print(model(x).shape)