import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

from torchvision import transforms, utils
from scipy.io import loadmat
from sklearn import preprocessing
import glob
import imageio
from sklearn.model_selection import  KFold
import matplotlib
def instance_label2bag(instance_label, bags):
    count = 0
    bag_label = np.zeros(bags.size)
    for i in range(bag_label.size):
        num_instances = bags[i].shape[0]
        label = 0
        for j in range(num_instances):
            label = label + instance_label[count+j]
        label_threshold = int(label / num_instances > 0.5)
        bag_label[i] = label_threshold
        count = count + num_instances
    return bag_label


def mi_collate_img(batch):
    # collate_fn for pytorch DataLoader
    bag = [item[0] for item in batch]
    bag = torch.tensor(np.concatenate(bag, axis = 0))
    
    bag_idx = [item[1] for item in batch]
    bag_idx = torch.tensor(np.concatenate(bag_idx, axis = 0))
    
    bag_label = [item[2] for item in batch]
    bag_label = torch.tensor(bag_label)

    instance_label = [item[3] for item in batch]
    instance_label = torch.tensor(np.concatenate(instance_label, axis = 0))
    return bag, bag_idx, bag_label, instance_label

def bag2instances(bags, bag_labels):
    """
    Covert bag list, bag label into instances pool, 
    provide bag label as instance label
    provide bag index of instances
    """
    instances = np.concatenate(bags, axis = 0)
    instances_label = np.zeros(len(instances))
    bag_index = np.zeros(len(instances))
    count = 0
    for i in range(len(bags)):
        num_instances_in_bag = bags[i].shape[0]
        instances_label[count:count+num_instances_in_bag] = bag_labels[i]
        bag_index[count:count+num_instances_in_bag] = i
        count = count + num_instances_in_bag
    return instances, instances_label, bag_index


def load_dataset(dataset_path, n_folds):
    # load datapath from path
    pos_path = glob.glob(dataset_path+'1//img*')
    neg_path = glob.glob(dataset_path+'0//img*')

    pos_num = len(pos_path)
    neg_num = len(neg_path)

    all_path = pos_path + neg_path

    #num_bag = len(all_path)
    kf = KFold(n_splits=n_folds, shuffle=True)
    datasets = []
    for train_idx, test_idx in kf.split(all_path):
        dataset = {}
        dataset['train'] = [all_path[ibag] for ibag in train_idx]
        dataset['test'] = [all_path[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets  

def show_img(img):
    plt.figure(figsize=(18,15))

    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def generate_batch(path):
    bags = []
    for each_path in path:
        name_img = []
        img = []
        img_path = glob.glob(each_path + '/*.bmp')
        num_ins = len(img_path)

        instance_label = [int('epithelial' in temp) for temp in img_path]
        label = int(each_path.split('\\')[-2])
        
        # if label == 1:
        #     curr_label = np.ones(num_ins,dtype=np.uint8)
        # else:
        #     curr_label = np.zeros(num_ins, dtype=np.uint8)
        for each_img in img_path:
            img_data = np.asarray( imageio.imread(each_img), dtype = np.uint8)
            # img_data[:, :, 0] -= 123
            # img_data[:, :, 1] -= 116
            # img_data[:, :, 2] -= 103
            # img_data /= 255
            # img_data = np.asarray(img_data, dtype = np.uint8)
            # matplotlib.pyplot.imshow(img_data)
            img.append(np.expand_dims(img_data,0))
            name_img.append(each_img.split('/')[-1])
        stack_img = np.concatenate(img, axis=0)
        bags.append((stack_img, instance_label, name_img))

    return bags

class mi_imagedata(Dataset):
    def __init__(self, data, cuda, transformations = None, batch_size=32, shuffle=True):
        self.device = torch.device('cuda') 
        self.cuda = cuda
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.transforms = transformations
        self.bags = [bag[0] for bag in data]
        self.bag_label =  [max(bag[1]) for bag in data]
        self.instance_label =  [bag[1] for bag in data]

    def __len__(self):
        return len(self.bag_label)

    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        bag = self.bags[idx]
        if self.transforms is not None:
            temp = [self.transforms(item) for item in bag]
            bag = torch.stack(temp)
        bag_label = self.bag_label[idx]
        bag_idx = np.repeat(idx, bag.shape[0])
        instance_label = self.instance_label[idx]

        return bag, bag_idx, bag_label, instance_label
        # return bag, bag_idx, bag_label, instance_label
