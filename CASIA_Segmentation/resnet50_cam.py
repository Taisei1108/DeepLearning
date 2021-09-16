import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


#class Net(nn.module):
#    def __init__(self):



def Net():
    return resnet50(pretrained=True)
