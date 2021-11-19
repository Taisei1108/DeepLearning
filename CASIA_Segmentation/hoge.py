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