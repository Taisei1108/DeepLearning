from torchvision.models import resnet50
import importlib
import torch.nn as nn
net = resnet50(pretrained=True)

#net = model = getattr(importlib.import_module('resnet50_cam'), 'Net')()
net.fc = nn.Linear(2048,2)


print(net)