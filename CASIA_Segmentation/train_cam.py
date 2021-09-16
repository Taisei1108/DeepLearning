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

from data_loader import ImageDataset


torch.manual_seed(0)
random.seed(0)

def run(args):
    print(args.cam_crop_size)

"""
model.train()
#print(model)
#ResNetの最終出力をデータセットのクラス数と同じにする
model.fc = nn.Linear(2048,args.n_class)#(fc): Linear(in_features=2048, out_features=1000, bias=True) ResNet50のもともと
#データ読み込み部分 これが帰ってくるsample = {'image': image, 'target': self.targets[index], "path": self.images[index], "downscaled": downscaled}
train_data = ImageDataset(DATA_ROOT, width=args.width, height=args.height, transform=transforms.Compose([
            #transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=[-1*args.affine_degree, args.affine_degree], scale=(1.0, args.scale)),
            #transforms.CenterCrop(args.crop),
            transforms.Resize((args.width,args.height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
val_data = ImageDataset(DATA_ROOT, 'val', width=args.width, height=args.height, transform=transforms.Compose([
            transforms.Resize((args.width,args.height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
test_data = ImageDataset(DATA_ROOT, 'test', width=args.width, height=args.height, transform=transforms.Compose([
            transforms.Resize((args.width,args.height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0)
test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)

#print(len(train_data)) # ->8883 /12614
#print(len(val_data)) # ->1274 /12614
#print(len(test_data))#-> 2457 (8:2=train:test,9:1=train:val)


image_numb = 9 # 3の倍数を指定してください
for i in range(0, image_numb):
  ax = plt.subplot(int(image_numb / 3), 3, i + 1)
  plt.tight_layout()
  ax.set_title(str(i))
  plt.imshow(train_data[i]['image'].transpose(0,1).transpose(1,2))#CHW  -> HWC
  print(train_data[i]["path"])
plt.savefig('./train_sample.png')


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) #weight decay　未設定
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
train_acc = []
val_acc = []
#os._exit()
def train():
    max_val = 0.
    device = torch.device(args.cuda_port if torch.cuda.is_available() else "cpu")
    model.to(device)    
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-------------')

        # train and validation per epoch
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
                    inputs = batch["image"].to(device)
                    labels = batch["target"].to(device)

                    # optimizer init
                    optimizer.zero_grad()

                    # forward
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)  #
                        _, preds = torch.max(outputs, 1) #2Dテンソルなので、第２引数axisが必要
                        #print(preds)
                        loss.backward()
                        optimizer.step()

                        # result calculation
                        epoch_loss += loss.item() * inputs.size(0)  # update loss sum
                        # update number of correct answers
                        epoch_corrects += torch.sum(preds == labels.data)

                        # show loss and accuracy per epoch
                        train_loss = epoch_loss / len(train_loader.dataset)
                        epoch_acc = epoch_corrects.double(
                        ) / len(train_loader.dataset)
                train_acc.append(epoch_acc.item())
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, train_loss, epoch_acc))
                log_scalar("train_loss",train_loss,epoch)
                log_scalar("train_acc",epoch_acc.item(),epoch)
                scheduler.step()

            elif phase == 'val':

                for iter, batch in enumerate(val_loader):

                    inputs = batch["image"].to(device)
                    labels = batch["target"].to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

                    val_loss = epoch_loss / len(val_loader.dataset)
                    epoch_acc = epoch_corrects.double(
                    ) / len(val_loader.dataset)
                val_acc.append(epoch_acc.item())
                if max_val < epoch_acc.item():
                    max_val = epoch_acc.item()
                    max_val_state = model.state_dict()
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, val_loss, epoch_acc))
                log_scalar("val_loss",val_loss,epoch)
                log_scalar("val_acc",epoch_acc.item(),epoch)
    log_scalar("max_val",max_val,0)
    print("max_val=",max_val)
    #学習曲線を描く
    fig = plt.figure()
    plt.title('Training Process')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    l1, = plt.plot(range(args.epochs), train_acc, c='red')
    l2, = plt.plot(range(args.epochs), val_acc, c='blue')
    plt.legend(handles=[l1, l2], labels=['Tra_acc', 'Val_acc'], loc='best')
    #plt.savefig('./Training Process for lr-{}'+DATASET+str(data_divide_ratio)+model_name+'.png'.format(lr), dpi=600)
    return max_val_state
def test():
    device = torch.device(args.cuda_port if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # inference
    epoch_loss = 0.0  # loss sum per epoch
    epoch_corrects = 0  # number of correct answers
    for iter, batch in enumerate(test_loader):

        inputs = batch["image"].to(device)
        labels = batch["target"].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)

        test_loss = epoch_loss / len(test_loader.dataset)
        epoch_acc = epoch_corrects.double(
        ) / len(test_loader.dataset)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        "test", test_loss, epoch_acc))
    log_scalar("test_loss",test_loss,1)
    log_scalar("test_acc",epoch_acc.item(),1)
    
def log_scalar(name, value, step):
    Log a scalar value to both MLflow and TensorBoard
    mlflow.log_metric(name, value)
if __name__ == '__main__':
    with mlflow.start_run():
        mlflow.log_artifact('train_sample.png')
        # Log our parameters into mlflow
        for key, value in vars(args).items():
            print(key,value)
            mlflow.log_param(key, value)
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