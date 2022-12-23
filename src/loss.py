import torch
import torch.nn as nn


class BCEWithLogitLoss(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight=weight

    def forward(self, input, target):
        # p = torch.clip(input,1e-6,1-1e-6)
        p = input.sigmoid()
        y = target
        pos_loss = -y*torch.log(p)
        neg_loss = -(1-y)*torch.log(1-p)
        loss = neg_loss + self.weight * pos_loss
        loss = loss.mean()

        return loss


class FocalWithLogitLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma=gamma

    def forward(self, input, target):
        # p = torch.clip(input,1e-6,1-1e-6)
        p = input.sigmoid()
        y = target

        pos_loss = -y*torch.log(p)*(1-p)**self.gamma
        neg_loss = -(1-y)*torch.log(1-p)*(p)**self.gamma
        loss = neg_loss+pos_loss
        loss = loss.mean()
        return loss


# '''
# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/14
# '''
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = 1e-12  # prevent training from Nan-loss error
#         self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, input, target):
#         bce_loss = self.bce_loss(input=input, target=target)
#         p_t = torch.exp(-bce_loss)
#         alpha_tensor = (1 - self.alpha) + target * (2 * self.alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
#         f_loss = alpha_tensor * (1 - p_t) ** self.gamma * bce_loss
#         return f_loss.mean()


if __name__ == '__main__':
    torch_bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.as_tensor([5]))
    bce_loss = BCEWithLogitLoss(5)

    pred = torch.as_tensor([[0.7,  0.3,  0.8]], dtype=float)
    true = torch.as_tensor([[1., 1., 1.]], dtype=float)

    print(torch_bce_loss(pred, true))
    print(bce_loss(pred, true))