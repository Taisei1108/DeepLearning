import importlib
import random
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from data_loader import ImageDataset

# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

def save_segmentation_image(args,heatmap,path_name,ConfM):
    #ヒートマップをセグメンテーション(２値)に変換して保存する
    
    #PILで操作
    image0 = transforms.functional.to_pil_image(heatmap)
    pixelSizeTuple = image0.size
    new_image0 = Image.new('RGB', image0.size)

    thred_r = 120
    for i in range(pixelSizeTuple[0]):
        for j in range(pixelSizeTuple[1]):
            r,g,b = image0.getpixel((i,j))
            if r > thred_r:  #R100~120くらいがよさそう、30くらいまで下げると黄色も含む、10くらいまで下げると緑とかも
                new_image0.putpixel((i,j), (255,255,255)) 
        else:
            new_image0.putpixel((i,j), (0,0,0)) 
    
    new_image0.save(args.cam_out_dir+path_name+'_'+ConfM+'_seg.png',quality=100)

def run(args):
    #乱数の初期設定
    torch.manual_seed(0)
    random.seed(0)

    #モデルの読み込み
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.fc = nn.Linear(2048,args.cam_output_class)
    model.load_state_dict(torch.load(args.cam_weights_name),strict=True)

    #データの読み込み
    model = torch.nn.DataParallel(model).cuda()
    test_data = ImageDataset(args.dataset_root, 'test', width=args.cam_crop_size, height=args.cam_crop_size, transform=transforms.Compose([
        transforms.Resize((args.cam_crop_size,args.cam_crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    test_loader = DataLoader(test_data, batch_size=args.cam_batch_size)

    # Grad-CAMを使えるようにする
    target_layer = model.module.layer2
    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMpp(model, target_layer)

    #モデルの設定
    model.eval()  # inference
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0.0  # loss sum per epoch
    epoch_corrects = 0  # number of correct answers

    #2021/09/22ここの書き方分かりづらい。 バッチごとにできるようにする?
    for iter, batch in enumerate(test_loader):
        inputs = batch["image"].cuda()
        labels = batch["target"].cuda()
        im_paths = batch["path"]
        print(torch.unsqueeze(inputs[0],0).shape) #->torch.Size([1, 3, 256, 256])
        print("labels",labels[0].item())
        #grad-cam参考https://www.yurui-deep-learning.com/2021/02/08/grad-campytorch-google-colab/
        for i in range(inputs.shape[0]): #batchsize分回す
            img = torch.unsqueeze(inputs[i],0)
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            epoch_corrects += torch.sum(preds == labels[i].data)
            print(preds.item())
            ConfM = "Null"
            # out 1 label 1 -> TP ,out 1 label0 ->FP, out 0 label -> 1 FN, out 0 label 0 -> TN
            #ここ絶対治す
            if preds.item() == 1 and labels[i].item() == 1:
                print("TP")
                ConfM = "TP"
            elif preds.item() == 1 and labels[i].item() == 0:
                print("FP")
                ConfM = "FP"
            elif preds.item() == 0 and labels[i].item() == 1:
                print("FN")
                ConfM = "FN"
            else:
                print("TN")
                ConfM = "TN"
            
            #grad-camの部分
            mask, _ = gradcam(img)
            heatmap, result = visualize_cam(mask, img)
            mask_pp, _ = gradcam_pp(img)
            heatmap_pp, result_pp = visualize_cam(mask_pp, img)
            path_name = im_paths[i].split('/')[-1].split('.')[0]
            print(path_name)

            #画像保存部
            save_segmentation_image(args,heatmap_pp,path_name,ConfM) #評価に使います
            save_image(result_pp,args.cam_out_dir+path_name+ConfM+"_result.png") #確認用
            save_image(heatmap_pp,args.cam_out_dir+path_name+ConfM+"_heatmap.png") #確認用
    #一応２値分類結果も表示        
    epoch_acc = epoch_corrects.double() / len(test_loader.dataset)
    print("test_acc=",epoch_acc)