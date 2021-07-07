import torch

import torchvision

ROOT = "../../datasets/VOCDetection/2012"


voc_dataset=torchvision.datasets.VOCDetection(root=ROOT,year="2012",image_set="train",download=True)