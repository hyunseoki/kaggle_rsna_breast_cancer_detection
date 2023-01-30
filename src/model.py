
import torch
import torch.nn as nn
import timm
from timm.models.layers import trunc_normal_
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


class ConvNeXt(nn.Module):
    def __init__(self, backbone='convnextv2_tiny'):
        super().__init__()
        self.model = timm.create_model(
            model_name=backbone,
            pretrained=True,
            in_chans=1,
            drop_rate=0.3,
            drop_path_rate=0.2,
            num_classes=1,
        )

    def forward(self, x):
        return self.model(x)


class NextViT(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = nextvit_base()
        if pretrained:
            # state_dict = torch.hub.load_state_dict_from_url(resnet34_default_cfg['url'])['model']
            state_dict = torch.load('/home/hyunseoki/ssd1/02_src/kaggle_rsna_breast_cancer_detection/checkpoint/nextvit_imagenet/nextvit_base_in1k_224.pth')['model']
            state_dict['stem.0.conv.weight'] = state_dict['stem.0.conv.weight'].sum(dim=1, keepdim=True)
            self.model.load_state_dict(state_dict)
        
        self.model.proj_head[0] = nn.Linear(in_features=1024, out_features=1, bias=True)
        trunc_normal_(self.model.proj_head[0].weight, std=.02)
        nn.init.constant_(self.model.proj_head[0].bias, 0)

    def forward(self, dcm):
        out = self.model(dcm)
        return out

if __name__ == '__main__':
    x = torch.randn(2, 1, 224, 224)
    model = NextViT()
    # model = Classifier()

    print(model)
    print(model(x).shape)