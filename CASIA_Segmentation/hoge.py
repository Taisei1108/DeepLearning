from torchvision.models import resnet50,vgg16
import importlib
import torch.nn as nn
from gradcam import GradCAM, GradCAMpp
net = vgg16(pretrained=True)

#net = model = getattr(importlib.import_module('resnet50_cam'), 'Net')()
#net.fc = nn.Linear(2048,2)


#print(net)
#print(net.features)

configs = [
    dict(model_type='vgg', arch=net, layer_name='features_29')
]

for config in configs:
    config['arch'].eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]

print(config)
print(cams)

import numpy as np
def normalize(v, axis=None, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

a = np.array([[-1,-2,-3],[1,2,3]])
#a = np.array([1,2,3,4])
print(a.shape)
b = min_max(a)

print(b)
print(b.shape)

from sklearn import preprocessing
print(preprocessing.minmax_scale(a))