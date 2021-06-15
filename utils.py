import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torchvision.transforms as transforms

def log_Logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

    if reduce:
        if average:
            return torch.mean(log_logist_256, dim)
        else:
            return torch.sum(log_logist_256, dim)
    else:
        return log_logist_256



def map_bag_embeddings(zx_q, zy_q, bag_idx, list_g):
    bag_latent_embeddings = torch.empty(zx_q.shape[0], zy_q.shape[1])
    for _, g in enumerate(list_g):
        group_label = g
        samples_group = bag_idx.eq(group_label).nonzero().squeeze()
        if samples_group.numel() >1 :
            for index in samples_group:
                # print("index: ", index)
                bag_latent_embeddings[index] = zy_q[list_g.index(group_label)]
        else:
            bag_latent_embeddings[samples_group] = zy_q[list_g.index(group_label)]
    return bag_latent_embeddings

def reorder_y(bag_label, bag_idx, list_g):
    def unique_keeporder(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]
    bag_idx = bag_idx.tolist()
    index = unique_keeporder(bag_idx)
    y_reordered = torch.empty(bag_label.shape)
    for i in range(len(list_g)):
        y_reordered[i] = bag_label[index.index(list_g[i])]
    return y_reordered


def accumulate_group_evidence_sum(class_mu, class_logvar, bag_idx, is_cuda):
    # convert logvar to variance for calculations
    bag_mu = []
    bag_logvar = []
    list_bags_labels = []
    sizes_bag = []
    bags = (bag_idx).unique()
    # calculate var inverse for each group using group vars
    for _, g in enumerate(bags):
        bag_label = g.item()
        samples_bag = bag_idx.eq(bag_label).nonzero().squeeze()

        if samples_bag.numel()>0:
            group_logvar =  class_logvar[samples_bag,:]
            group_mu = class_mu[samples_bag,:] 
            if samples_bag.numel()>1:
                group_mu = group_mu.mean(0,keepdim=True)
                group_logvar = group_logvar.pow(2).mean(0, keepdim=True).sqrt()
            else:
                group_mu = group_mu[None,:]
                group_logvar = group_logvar[None,:].pow(2).sqrt()

            bag_mu.append(group_mu)
            bag_logvar.append(group_logvar)
            list_bags_labels.append(bag_label)
            sizes_bag.append(samples_bag.numel())

    bag_mu = torch.cat(bag_mu,dim=0)
    bag_logvar = torch.cat(bag_logvar, dim=0)
    sizes_bag = torch.FloatTensor(sizes_bag)
    return bag_mu, bag_logvar, list_bags_labels, sizes_bag


def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def weights_init(layer):
    r"""Apparently in Chainer Lecun normal initialisation was the default one
    """
    if isinstance(layer, nn.Linear):
        layer.bias.data.zero_()
        kaiming_uniform_(layer.weight)
        # torch.nn.init.kaiming_uniform_(layer.bias)
        # torch.nn.init.kaiming_uniform_(layer.weight)


def kaiming_uniform_(tensor, gain=1):

    import math
    r"""Adapted from https://pytorch.org/docs/0.4.1/_modules/torch/nn/init.html#xavier_normal_
    """
    dimensions = tensor.size()
    if len(dimensions) == 1:  # bias
        fan_in = tensor.size(0)
    elif len(dimensions) == 2:  # Linear
        fan_in = tensor.size(1)
    else:
        num_input_fmaps = tensor.size(1)
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
    std = gain/ math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std 
    with torch.no_grad():
        return tensor.uniform_( -bound, bound )
    

def lecun_normal_(tensor, gain=1):

    import math
    r"""Adapted from https://pytorch.org/docs/0.4.1/_modules/torch/nn/init.html#xavier_normal_
    """
    dimensions = tensor.size()
    if len(dimensions) == 1:  # bias
        fan_in = tensor.size(0)
    elif len(dimensions) == 2:  # Linear
        fan_in = tensor.size(1)
    else:
        num_input_fmaps = tensor.size(1)
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size

    std = gain * math.sqrt(1.0 / (fan_in))
    with torch.no_grad():
        return tensor.normal_(0, std)
