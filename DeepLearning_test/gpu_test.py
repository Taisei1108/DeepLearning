import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

import gpu_test_func1

model = resnet50(pretrained=True)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

gpu_test_func1.print_hoge()
"""
＋nvidia-smiでポート数、実行中かどうかを確かめれる

GPUのポート指定は、cuda(device = "cuda:0")のようにできるけど
初手にos.environ['CUDA_VISIBLE_DEVICES'] = '1,2'のようにすると良さそう

ちなみにimport torchの前にやらないと反映されないらしい

"""
