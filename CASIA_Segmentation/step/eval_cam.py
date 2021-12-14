import importlib

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from data_manage_CASIA.data_loader import ImageDataset
#from data_manage.data_loader import ImageDataset

import utils
#テストデータでどれくらい２値分類できるか、eval_camという名前ややこしいかも
def run(args):
    #モデルの読み込み 恐らくGPU使う前提
    torch.manual_seed(0)
    
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.fc = nn.Linear(2048,args.cam_output_class)
    model.load_state_dict(torch.load(args.cam_weights_name),strict=True)
    images_path = args.dataset_root + "images/"
    test_data = ImageDataset(images_path,args.test_list, width=args.cam_crop_size, height=args.cam_crop_size, transform=transforms.Compose([
        transforms.Resize((args.cam_crop_size,args.cam_crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    test_loader = DataLoader(test_data, batch_size=args.cam_batch_size)

    #学習の設定
    model = torch.nn.DataParallel(model).cuda()
    model.eval()  # inference
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0.0  # loss sum per epoch
    epoch_corrects = 0  # number of correct answers

    for iter, batch in enumerate(test_loader):

        inputs = batch["image"].cuda()
        labels = batch["target"].cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        epoch_loss += loss.item() * inputs.size(0)
        epoch_corrects += torch.sum(preds == labels.data)

        test_loss = epoch_loss / len(test_loader.dataset)
        epoch_acc = epoch_corrects.double(
        ) / len(test_loader.dataset)
    #テスト結果の表示
    print('{} Loss: {:.4f} Acc: {:.4f}'.format("test", test_loss, epoch_acc))
    utils.log_scalar("test_loss",test_loss,1)
    utils.log_scalar("test_acc",epoch_acc.item(),1)