import importlib
import random
from PIL import Image
import cv2

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from data_loader import ImageDataset

# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
#CRF
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
def CAM_image2binary(args,heatmap,path_name,pred_mani):
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
    
    return new_image0

def get_prediction_manipulation(pred,label): #item()して入力されることを仮定
    if pred == 1 and label == 1:
        pred_mani = "TP"
    elif pred == 1 and label == 0:
        pred_mani = "FP"
    elif pred == 0 and label == 1:
        pred_mani = "FN"
    else:
        pred_mani = "TN"
    return pred_mani

def CRF(args,img,CAM_binary): #両引数numpyとして渡したい predictionがanno_rgb
    anno_rgb=np.array(CAM_binary,dtype=np.uint32)
    img = img.to('cpu').detach().numpy().copy() #tensorからnumpyへ
    img = np.squeeze(img).transpose(1,2,0)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    colors, labels = np.unique(anno_lbl, return_inverse=True) #color [  0 16777215] labels[1 1 1 ... 0 0 0] (.shape = 65536,)

    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    n_labels = len(set(labels.flat))
    #n_labels = len(set(labels.flat)) - int(HAS_UNK)
    use_2d = False     

    if use_2d:                   
                 # Use densecrf2d class
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

                 # Get a dollar potential (negative log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
                 #U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## If there is an indeterminate area, replace the previous line with this line of code
        d.setUnaryEnergy(U)

                 # Added color-independent terms, the function is just the location
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

                 # Added color related terms, ie the feature is (x, y, r, g, b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
                 # Use densecrf class
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

                 # Get a dollar potential (negative log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=None)  
                 #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## If there is an indeterminate area, replace the previous line with this line of code
        d.setUnaryEnergy(U)

                 # This will create color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

                 # This will create color-related features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        

        # 5 times reasoning
        Q = d.inference(5)

        # Find the most likely class for each pixel
        MAP = np.argmax(Q, axis=0)

        # Convert predicted_image back to the appropriate color and save the image
        MAP = colorize[MAP,:]
        print("CRF Done!")

        return MAP
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
        #grad-cam参考https://www.yurui-deep-learning.com/2021/02/08/grad-campytorch-google-colab/
        for i in range(inputs.shape[0]): #batchsize分回す,単一のデータごとに処理
            img = torch.unsqueeze(inputs[i],0)
            outputs = model(img)
            _, pred = torch.max(outputs, 1) #ここargmaxじゃなくていいのか？
            epoch_corrects += torch.sum(pred == labels[i].data)
            pred_mani = get_prediction_manipulation(pred.item(),labels[i].item())
            # pred 1 label 1 -> TP ,pred 1 label0 ->FP, pred 0 label -> 1 FN, pred 0 label 0 -> TN
                
            #grad-camの部分
            mask, _ = gradcam(img)
            heatmap, result = visualize_cam(mask, img)
            mask_pp, _ = gradcam_pp(img)
            heatmap_pp, result_pp = visualize_cam(mask_pp, img)
            
            #パス保存用
            path_name = im_paths[i].split('/')[-1].split('.')[0]

            CAM_binary =CAM_image2binary(args,heatmap_pp,path_name,pred_mani) #評価に使います
            CAM_binary_torch = transforms.functional.to_tensor(CAM_binary)
            CRF_result = CRF(args,img,CAM_binary) #この時CAM_binaryはPIL
            CRF_result_torch=torch.from_numpy(CRF_result.astype(np.float32)).clone()
            print(CAM_binary_torch)
            #画像保存部
            save_image(CAM_binary_torch,args.segmentation_out_dir_CAM+path_name+'_'+pred_mani+'_binary_CAM.png')
            save_image(CRF_result_torch,args.segmentation_out_dir_CRF+path_name+'_'+pred_mani+'_binary_CRF.png')
            save_image(result_pp,args.cam_out_dir+path_name+pred_mani+"_result.png") #確認用
            save_image(heatmap_pp,args.cam_out_dir+path_name+pred_mani+"_heatmap.png") #確認用
            print(path_name,":",pred_mani,"(",pred.item(),",",labels[i].item(),")")
    #一応２値分類結果も表示        
    epoch_acc = epoch_corrects.double() / len(test_loader.dataset)
    print("test_acc=",epoch_acc)