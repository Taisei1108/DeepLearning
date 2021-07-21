import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
from torchvision.models import resnet101
from torchvision.utils import make_grid, save_image

from data_loader import ImageDataset

# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
#https://deepblue-ts.co.jp/image-processing/pytorch_guided_grad_cam/
# directoryの./out_segを設定する必要がある ()
# CAMの結果は./outputに入る

DATA_ROOT = "/home/takahashi/datasets/ColumbiaUncompressedImageSplicingDetection/data"
width = 256
height = 256
n_class = 2
batch_size = 1

torch.manual_seed(0)
random.seed(0)

model = resnet101()
#ResNetの最終出力をデータセットのクラス数と同じにする
model.fc = nn.Linear(2048,n_class)
# weight path 
model_path = "mlruns/0/cb0357f7e5ec46a0abae186dd85ac824/artifacts/results/model.pth"
model.load_state_dict(torch.load(model_path, map_location={'cuda:0': 'cpu'}))
model.eval()

test_data = ImageDataset(DATA_ROOT, 'test', width=width, height=height, transform=transforms.Compose([
            transforms.Resize((width,height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)
#print(model)

# Grad-CAM
target_layer = model.layer2
gradcam = GradCAM(model, target_layer)
gradcam_pp = GradCAMpp(model, target_layer)
#print(next(model.parameters()).is_cuda) #cudaチェック


def makedir():
    path1 = "./output"
    path2 = "./output/Out1_Ans1/"
    path3 = "./output/Out1_Ans0/"
    path4 = "./output/Out0_Ans1/"
    path5 = "./output/Out0_Ans0/"
    if not os.path.isdir(path1):
        os.mkdir(path1)
    if not os.path.isdir(path2):
        os.mkdir(path2)
    if not os.path.isdir(path3):
        os.mkdir(path3)
    if not os.path.isdir(path4):
        os.mkdir(path4)
    if not os.path.isdir(path5):
        os.mkdir(path5)
makedir()
print("test_data_len=",len(test_data))

def toHeatmap(x):
    x = (x*255).reshape(-1)
    cm = plt.get_cmap('jet')
    x = np.array([cm(int(np.round(xi)))[:3] for xi in x])
    return x.reshape(width,height,3) #もともと224,224だった
def cam_calc(feature,input):
    alpha = torch.mean(feature.grad.view(16,5*5),1)
    feature = feature.view(16,5,5)
    L = F.relu(torch.sum(feature*alpha.view(-1,1,1),0)).cpu().detach().numpy()
    L_min = np.min(L)
    L_max = np.max(L - L_min)
    L = (L - L_min)/L_max
    # 得られた注目度をヒートマップに変換
    L = toHeatmap(cv2.resize(L,(128,128)))

    img1 = torch.squeeze(input).permute(1,2,0).to('cpu').detach().numpy() 
    #print("img1=",img1.shape) img1= (128, 128, 3)
    img2 = L
    alpha = 0.3
    

    blended = img1*alpha + img2*(1-alpha)
    return blended

def cam(input,label):
    feature_extractor = net.features
    classifier = net.classifier
    feature_extractor = feature_extractor.eval()
    classifier = classifier.eval()
    #print(inputs.shape) = torch.Size([32, 3, 128, 128])
    with torch.no_grad():
        output = net(input)
    _, preds = torch.max(output, 1)
    feature = feature_extractor(input) #特徴マップを計算
    
    #print('特徴マップのサイズは　{}'.format(feature.shape)) 特徴マップのサイズは　torch.Size([1, 16, 5, 5])
    feature = feature.clone().detach().requires_grad_(True) #勾配を計算するようにコピー
    y_pred = classifier(feature.view(-1,16*5*5)) #予測を行う
    #print(y_pred.shape) =torch.Size([32, 2])
    #print(y_pred) tensor([[0.9921, 0.0079]], grad_fn=<SoftmaxBackward>)
    #print(torch.argmax(y_pred))tensor(0)
    y_pred[0][torch.argmax(y_pred)].backward() # 予測でもっとも高い値をとったクラスの勾配を計算
    blended = cam_calc(feature,input)
    return blended,preds

def save_dir_chk(label,pred):
    label_data = label.item()
    pred_data = pred.item()
    save_dir =""
    if label_data == 1 and pred_data == 1:
        save_dir = "./output/Out1_Ans1/"
    elif label_data == 1 and pred_data == 0:
        save_dir = "./output/Out0_Ans1/"
    elif label_data == 0 and pred_data == 1:
        save_dir = "./output/Out1_Ans0/"
    elif label_data == 0 and pred_data == 0:
        save_dir = "./output/Out0_Ans0/"
    return save_dir

def cam_save_image(blended,path,label,pred):
    #問1 blendedとimageとmaskを保存するプログラムを書け 各20点
    #print(path)../../datasets/CASIA/test/1/Tp_S_NNN_S_N_arc00035_arc00035_01089.jpg
    save_dir = save_dir_chk(label,pred)
    im_name = path.split('/')[-1]
    im_mask_name = im_name.split('.')[0]+"_mask.png"
    # print(im_mask_name) Tp_D_CNN_M_N_arc00086_xxx00000_00306_mask.png
    image = Image.open(path).convert('RGB')
    if label.item() == 1: #編集画像だったらマスクを保存
        #maskを読み込んで保存する処理
        im_mask_path = "../../datasets/CASIAv2/mask/"+im_mask_name 
        mask = Image.open(im_mask_path).convert('RGB')
        #plt.figure(figsize=(10,10))
        plt.imshow(mask)
        plt.axis('off')
        plt.savefig(save_dir+im_mask_name)

    #plt.figure(figsize=(10,10))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig(save_dir+im_name.split('.')[0]+'_cam.png')
    
    #plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(save_dir+im_name)

def heatmap_seg(heatmap,path_name,ConfM):
    #print(heatmap)
    image0 = transforms.functional.to_pil_image(heatmap)
    pixelSizeTuple = image0.size
    new_image0 = Image.new('RGB', image0.size)

    for i in range(pixelSizeTuple[0]):
        for j in range(pixelSizeTuple[1]):
            r,g,b = image0.getpixel((i,j))
            if r > 120:  #R100~120くらいがよさそう、30くらいまで下げると黄色も含む、10くらいまで下げると緑とかも
                new_image0.putpixel((i,j), (255,255,255)) 
        else:
            new_image0.putpixel((i,j), (0,0,0)) 
    
    new_image0.save('./out_seg/'+path_name+'_'+ConfM+'.png',quality=100)

def test():
    epoch_corrects = 0  # number of correct answers
    images = []
    for iter, batch in enumerate(test_loader):
        inputs = batch["image"]
        labels = batch["target"]
        im_paths = batch["path"]
        # print(torch.unsqueeze(inputs[0],0).shape) ->torch.Size([1, 3, 256, 256])
        print("labels",labels[0].item())
        #grad-cam参考https://www.yurui-deep-learning.com/2021/02/08/grad-campytorch-google-colab/
        for i in range(inputs.shape[0]): #batchsize分回す
            img = torch.unsqueeze(inputs[i],0)
            outputs = model(img)
            _, preds = torch.max(outputs, 1)
            epoch_corrects += torch.sum(preds == labels[i].data)
            
            ConfM = "Null"
            # out 1 label 1 -> TP ,out 1 label0 ->FP, out 0 label -> 1 FN, out 0 label 0 -> TN
            if preds[i].item() == 1 and labels[i].item() == 1:
                print("TP")
                ConfM = "TP"
            elif preds[i].item() == 1 and labels[i].item() == 0:
                print("FP")
                ConfM = "FP"
            elif preds[i].item() == 0 and labels[i].item() == 1:
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
            images.extend([img.cpu(), heatmap, heatmap_pp, result, result_pp])
            path_name = im_paths[i].split('/')[-1].split('.')[0]
            print(path_name)
            heatmap_seg(heatmap_pp,path_name,ConfM)
            save_image(result_pp,"./output/"+path_name+ConfM+".png")
            save_image(heatmap_pp,"./output/"+path_name+ConfM+"_heatmap.png")
            #blended,preds = cam(torch.unsqueeze(inputs[i],0),labels[i])
            #print("label&preds",labels[0].item(),preds[0].item()) #preds= tensor([1])
            #cam_save_image(blended,im_paths[i],labels[i],preds)
    epoch_acc = epoch_corrects.double() / len(test_loader.dataset)
    print("test_acc=",epoch_acc)
    #print(images)
    #grid_image = make_grid(images, nrow=4)
    #結果の保存
    #save_image(grid_image,"grid_images")
    #transforms.ToPILImage()(grid_image)

if __name__ == '__main__':
    test()