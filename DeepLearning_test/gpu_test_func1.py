import os
import torch

def print_hoge():
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())