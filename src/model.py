import torch
import torch.nn as nn
import timm
from timm.models.layers import trunc_normal_
from nextvit import nextvit_base



'''
    https://www.kaggle.com/code/christofhenkel/se-resnext50-full-gpu-decoding
'''
class GeM(nn.Module):
    def __init__(self, p=1, eps=1e-6, p_trainable=True):
        super().__init__()
        if p_trainable:
            self.p = torch.nn.parameter.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def gem(self, x, p=3, eps=1e-6):
        return torch.nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def forward(self, x):
        ret = self.gem(x, p=self.p, eps=self.eps).squeeze()
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


class Classifier(nn.Module):
    def __init__(self, backbone='tf_efficientnetv2_s', gem=False):
        super().__init__()
        self.model = timm.create_model(
            model_name=backbone,
            pretrained=True,
            in_chans=1,
            drop_rate=0.3,
            drop_path_rate=0.2,
            num_classes=1,
        )
        if gem:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, 1)
            self.model.global_pool = GeM()

        # n_features = self.model.classifier.in_features
        # self.model.classifier = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.model(x)


class ResNetModel(nn.Module):
    def __init__(self, gem=False):
        super().__init__()
        self.model = timm.create_model(
            model_name='seresnext50_32x4d',
            pretrained=True,
            in_chans=1,
            drop_rate=0.3,
            drop_path_rate=0.2,
            num_classes=1,
        )
        if gem:
            n_features = self.model.feature_info[-1]['num_chs']
            self.model.fc = nn.Linear(n_features, 1)
            self.model.global_pool = GeM()
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
    # model = NextViT()
    model = ResNetModel()

    print(model)
    print(model(x).shape)