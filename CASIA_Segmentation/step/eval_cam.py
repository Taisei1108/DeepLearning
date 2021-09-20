import importlib

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from data_loader import ImageDataset

def run(args):
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.fc = nn.Linear(2048,args.output_class)
    model.load_state_dict(torch.load(args.cam_weights_name, map_location={'cuda:0': 'cpu'}))

    device = torch.device(args.cuda_port if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_data = ImageDataset(args.dataset_root, 'test', width=args.cam_crop_size, height=args.cam_crop_size, transform=transforms.Compose([
        transforms.Resize((args.cam_crop_size,args.cam_crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    test_loader = DataLoader(test_data, batch_size=args.cam_batch_size)


    model.eval()  # inference
    criterion = nn.CrossEntropyLoss()
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