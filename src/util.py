import os
import random
import numpy as np
import torch
import argparse


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_model_weights(model, weight_fn):
    device = next(model.parameters()).device
    model_state_dict = torch.load(weight_fn, map_location=device)
    try:
        model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError:
        # if model was trained in parallel
        from collections import OrderedDict
        new_model_state_dict = OrderedDict()

        for k, v in model_state_dict.items():
            k = k.replace('module.', '')
            new_model_state_dict[k] = v

        model.load_state_dict(new_model_state_dict, strict=True)

    return model



def get_sampler(df):
    labels = df['cancer'].to_list()
    n_class, class_counts = np.unique(labels, return_counts=True)
    n_class = len(n_class)

    total_num = sum(class_counts) 
    class_weights = [total_num/class_counts[i] for i in range(len(class_counts))]

    weights = [class_weights[label] for label in labels]
    sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), int(total_num), replacement=True)

    return sampler