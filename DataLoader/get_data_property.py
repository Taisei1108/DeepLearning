import random
import matplotlib.pyplot as plt
import collections


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import ImageDataset

torch.manual_seed(0)
random.seed(0)

DATA_ROOT = "/home/takahashi/datasets/CASIA_new"

train_data = ImageDataset(DATA_ROOT,transform=transforms.ToTensor())
val_data = ImageDataset(DATA_ROOT,'val',transform=transforms.ToTensor())
test_data = ImageDataset(DATA_ROOT,'test',transform=transforms.ToTensor(),)

dataset = [train_data,val_data,test_data]

for d in dataset:
    count = 0
    data_list_count = []
    data_list_hei = []
    data_list_wid = []
    for i in range(len(d)):
        count += 1
        data_list_count.append(str(list(d[i]['image'].shape)))
        data_list_hei.append(d[i]['image'].shape[1])
        data_list_wid.append(d[i]['image'].shape[2])
    print(count)

    #頻度確認
    #c = collections.Counter(data_list_count)
    #print(c)

    #data_list_hei.sort()
    #print(data_list_hei)
    print("高さ最大値:",max(data_list_hei),"高さ最小値",min(data_list_hei))
    print("横幅最大値:",max(data_list_wid),"横幅最小値",min(data_list_wid))