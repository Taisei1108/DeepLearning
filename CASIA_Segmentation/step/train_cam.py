import os
import sys
import random
import matplotlib.pyplot as plt
import argparse
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet101
import mlflow

import importlib

from data_manage.data_loader import ImageDataset

from torchvision.models import resnet50

import utils

#乱数の固定
torch.manual_seed(0)
random.seed(0)


def run(args):
    #モデルの読み込みと最終層の変更
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.fc = nn.Linear(2048,args.cam_output_class)
    images_path = args.dataset_root + "images/"
    #データセット読み込み部
    #9/16この辺はクラスのメソッドにしたい,scaleもマルチスケールに対応させる。transformを使うかどうかも
    train_dataset = ImageDataset(images_path,args.train_list,width=args.cam_crop_size, height=args.cam_crop_size, transform=transforms.Compose([
                    #transforms.RandomCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomAffine(degrees=[-1*args.cam_affine_degree, args.cam_affine_degree], scale=args.cam_scale),
                    #transforms.CenterCrop(args.crop),
                    transforms.Resize((args.cam_crop_size,args.cam_crop_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]))
    val_dataset = ImageDataset(images_path,args.val_list, width=args.cam_crop_size, height=args.cam_crop_size, transform=transforms.Compose([
                    transforms.Resize((args.cam_crop_size,args.cam_crop_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]))
 
    train_loader = DataLoader(train_dataset, batch_size=args.cam_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.cam_batch_size)
    
    #学習の設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.cam_learning_rate, momentum=args.cam_momentum) #weight decay　未設定
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    model = torch.nn.DataParallel(model).cuda()
    model.train() 

    train_acc = []
    val_acc = []
    max_val = 0.
      
    print(model) #確認用
    
    for epoch in range(args.cam_num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.cam_num_epochs))
        print('-------------')

        #学習と検証を交互に繰り返す
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # training
            else:
                model.eval()  # inference

            epoch_loss = 0.0  # loss sum per epoch
            epoch_corrects = 0  # number of correct answers

            if (epoch == 0) and (phase == 'train'):
                train_acc.append(0.)
                continue

            if phase == 'train':
                
                for iter, batch in enumerate(train_loader):
                    inputs = batch["image"].cuda()
                    labels = batch["target"].cuda()

                    # optimizer init
                    optimizer.zero_grad()

                    # forward
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)  #
                        _, preds = torch.max(outputs, 1) #2Dテンソルなので、第２引数axisが必要
                        loss.backward()
                        optimizer.step()

                        # ロスとAccの計算(イテレータごと)
                        epoch_loss += loss.item() * inputs.size(0)  # update loss sum
                        epoch_corrects += torch.sum(preds == labels.data)
                #ロスの計算(エポックごと)
                train_loss = epoch_loss / len(train_loader.dataset)
                epoch_acc = epoch_corrects.double() / len(train_loader.dataset)
                
                #結果の表示系
                train_acc.append(epoch_acc.item())
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, train_loss, epoch_acc))
                utils.log_scalar("train_loss",train_loss,epoch)
                utils.log_scalar("train_acc",epoch_acc.item(),epoch)
                scheduler.step()

            elif phase == 'val':

                for iter, batch in enumerate(val_loader):

                    inputs = batch["image"].cuda()
                    labels = batch["target"].cuda()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    #ロスの計算(イテレータごと)
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)
                #ロスの計算(エポックごと)
                val_loss = epoch_loss / len(val_loader.dataset)
                epoch_acc = epoch_corrects.double() / len(val_loader.dataset)
                
                #最大値のときに
                if max_val < epoch_acc.item():
                    max_val = epoch_acc.item()
                    max_val_state = model.module.state_dict()
                
                #表示系
                val_acc.append(epoch_acc.item())
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss, epoch_acc))
                utils.log_scalar("val_loss",val_loss,epoch)
                utils.log_scalar("val_acc",epoch_acc.item(),epoch)
    utils.log_scalar("max_val",max_val,0)
    print("max_val=",max_val)
   
    #学習曲線を描く
    fig = plt.figure()
    plt.title('Training Process')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    l1, = plt.plot(range(args.cam_num_epochs), train_acc, c='red')
    l2, = plt.plot(range(args.cam_num_epochs), val_acc, c='blue')
    plt.legend(handles=[l1, l2], labels=['Tra_acc', 'Val_acc'], loc='best')
    #plt.savefig('./Training Process for lr-{}'+DATASET+str(data_divide_ratio)+model_name+'.png'.format(lr), dpi=600)
    
    torch.save(max_val_state, args.cam_weights_name)
"""


image_numb = 9 # 3の倍数を指定してください
for i in range(0, image_numb):
  ax = plt.subplot(int(image_numb / 3), 3, i + 1)
  plt.tight_layout()
  ax.set_title(str(i))
  plt.imshow(train_data[i]['image'].transpose(0,1).transpose(1,2))#CHW  -> HWC
  print(train_data[i]["path"])
plt.savefig('./train_sample.png')

    
        # Create a SummaryWriter to write TensorBoard events locally
        output_dir = dirpath = tempfile.mkdtemp()
        # Perform the training and test
        max_val_state = train()
        #save_path = './model'+DATASET+str(data_divide_ratio)+model_name+'.pth'
        #torch.save(model.state_dict(), save_path)
        test()
        with tempfile.TemporaryDirectory()  as tmp:
            filename = os.path.join(tmp, "model.pth")
            torch.save(max_val_state, filename)
            mlflow.log_artifacts(tmp, artifact_path="results") #mlrun内のartifacts/resultにモデル生成

"""