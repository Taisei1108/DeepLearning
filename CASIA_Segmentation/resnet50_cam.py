import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torch
import torch.nn as nn

#class Net(nn.module):
#    def __init__(self):


"""
def Net():
    return resnet50(pretrained=True)
"""

CLASS_NUM = 2

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #self.resnet50 = resnet50(pretrained=True, strides=(2, 2, 2, 1))
        self.resnet50 = resnet50(pretrained=True)
        self.layer1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.layer2 = nn.Sequential(self.resnet50.layer2)
        self.layer3 = nn.Sequential(self.resnet50.layer3)
        self.layer4 = nn.Sequential(self.resnet50.layer4)
        self.Self_Attn1 = Self_Attn(1024)
        self.classifier = nn.Conv2d(2048, CLASS_NUM, 1, bias=False)

        self.backbone = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x).detach()

        x = self.layer3(x)
        print(x.shape)
        x = self.Self_Attn1(x)
        x = self.layer4(x)
        
        x = gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, CLASS_NUM)

        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))



class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
            inputs :
                input : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batchsize, C, width, height = input.size()
        print("debag",batchsize,C,width,height)
        proj_query = self.query_conv(input).view(batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(input).view(batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(input).view(batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, width, height)

        out = self.gamma * out + input
        return out

def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out
