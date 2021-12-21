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

#from data_manage_CASIA.data_loader import ImageDataset
from data_manage.data_loader import ImageDataset
# Grad-CAM
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
#CRF
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

from cv2 import imread, imwrite
from utils import np_img_HWC_debug
from matplotlib import pyplot as plt

MASK_EX = '_edgemask_3.jpg'
#MASK_EX = '_mask.png'

def normalize(v, axis=None, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2
def min_max(x, axis=None):  #確か配列を0~1にしてくれるとかだと思う　ーがあっても行けた気がする。
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def CAM_image2binary_save(args,heatmap,path_name,pred_mani):
    #ヒートマップをセグメンテーション(２値)に変換して保存する
    
    #PILで操作
    image0 = transforms.functional.to_pil_image(heatmap)
    pixelSizeTuple = image0.size
    new_image0 = Image.new('RGB', image0.size)

    thred_r = 45 #元々120にしてたけど低いほうがいい
    for i in range(pixelSizeTuple[0]):
        for j in range(pixelSizeTuple[1]):
            r,g,b = image0.getpixel((i,j))
            if r > thred_r:  #R100~120くらいがよさそう、30くらいまで下げると黄色も含む、10くらいまで下げると緑とかも 前45
                new_image0.putpixel((i,j), (255,255,255)) 
            else:
                new_image0.putpixel((i,j), (0,0,0)) 
    new_image0.save(args.segmentation_out_dir_CAM+path_name+'_'+pred_mani+'_binary_CAM.png',quality=100)

def SA2binary(args,SA_output,path_name,pred_mani,thr=115):
    """
    input Self-Attention(256,256) 0~255 numpyだった
    output 閾値処理をして(256,256,3),画像を保存
    """
    print(SA_output.shape)
    image0=transforms.functional.to_pil_image(SA_output)
    pixelSizeTuple = image0.size
    new_image0 = Image.new('RGB', image0.size)
    for i in range(pixelSizeTuple[0]):
        for j in range(pixelSizeTuple[1]):
            r = image0.getpixel((i,j))
         
            if r > thr:  #R100~120くらいがよさそう、30くらいまで下げると黄色も含む、10くらいまで下げると緑とかも
                new_image0.putpixel((i,j), (255,255,255)) 
            else:
                new_image0.putpixel((i,j), (0,0,0)) 
    new_image0.save(args.segmentation_out_dir_SA_CAM+path_name+'_'+pred_mani+'_binary_SA_CAM.png',quality=100)
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
    img = np.squeeze(img).transpose(1,2,0) #256,256,3に合わせる
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    colors, labels = np.unique(anno_lbl, return_inverse=True) #color [  0 16777215] labels[1 1 1 ... 0 0 0] (.shape = 65536,)
   
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    #np_img_HWC_debug(CAM_binary,np_img_str="CAMbinary")
    #np_img_HWC_debug(img,np_img_str="img")
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
        Q = d.inference(10)

        # Find the most likely class for each pixel
        Q_np = np.array(Q)
        MAP = np.argmax(Q_np, axis=0)

        # Convert predicted_image back to the appropriate color and save the image
        MAP = colorize[MAP,:]
        print("CRF Done!")
        
        return MAP

def save_SA_CAM(args,SA_output,path_name,model,pred):

    C = SA_output.shape[0]
    width = SA_output.shape[1]
    height = SA_output.shape[2]

    SA_output_np = SA_output.to('cpu').detach().numpy().copy()
    SA_output_hwc= np.transpose(SA_output_np, (1, 2, 0))
    SA_sum = 0
    for i in range(C):
        SA_sum += min_max(SA_output_hwc[:,:,i])*model.module.fc.weight[pred][i].to('cpu').detach().numpy().copy()
    SA_mean  = SA_sum/C
    
    SA_mean_minmax = min_max(SA_mean)
    SA_mean_minmax_resize = cv2.resize(SA_mean_minmax,(256,256)) #ベタ打ちだから変える
    
    
    imwrite(args.cam_out_dir+path_name+'_'+str(pred)+'_SA_CAM.png',SA_mean_minmax_resize*255)
    return SA_mean_minmax_resize*255

def save_SA(args,attention,CAM,path_name,pred):
    attention = attention.reshape(64,8,8).to('cpu').detach().numpy().copy() #ベタ打ち
    SA_output =np.zeros((256,256))
    SA_output_mm =np.zeros((256,256))
    #この辺のプログラムは入力画像256*256、SAの出力[64*64]を想定
    count = 0
    for i in range(CAM.shape[0]):
        for k in range(CAM.shape[1]):
            if CAM[i][k][0] == 255:
                #print((i//32*8)+k//32,":",attention[i//32*8+k//32])  #256/8=32
                attention_resize = cv2.resize(attention[i//32*8+k//32],(256,256))
                #print(normalize(attention_resize))
                
                SA_output += attention_resize
                SA_output_mm += min_max(attention_resize) #min_maxすると0~1になる。もともと0~1だからしなくてもいい説。sumの値もそんな変わらん
                count += 1
    th, im_th = cv2.threshold(SA_output_mm/count*255, 10, 255, cv2.THRESH_BINARY)
    #imwrite(args.cam_out_dir+path_name+'_'+str(pred)+'_SA_out.png',SA_output/count*255)
    imwrite(args.segmentation_out_dir_SA+path_name+'_'+str(pred)+'_binary_SA.png',im_th) #バイナリで保存される。
    #attention_resize = cv2.resize(attention,(64,256,256))
    #return im_th

def save_image_func(args,path_name,pred_mani,CRF_result_CAM,CRF_result_SA_CAM,CRF_result_SA,result_pp,heatmap_pp):

    imwrite(args.segmentation_out_dir_CRF+path_name+'_'+pred_mani+'_binary_CRF.png',CRF_result_CAM)
    imwrite(args.segmentation_out_dir_SA_CAM_CRF+path_name+'_'+pred_mani+'_binary_SA_CAM_CRF.png',CRF_result_SA_CAM)
    imwrite(args.segmentation_out_dir_SA_CRF+path_name+'_'+pred_mani+'_binary_SA_CRF.png',CRF_result_SA)
    save_image(result_pp,args.cam_out_dir+path_name+pred_mani+"_result.png") #確認用 #torch.Size([3, 256, 256])　#リザルトというのはheatmapと実画像を重ね合わせているということ
    save_image(heatmap_pp,args.cam_out_dir+path_name+pred_mani+"_heatmap.png") #確認用

def images_prot(args,path_name,MASK_ROOT,img,CAM_binary,CRF_result_CAM,SA_CAM_binary,CRF_result_SA_CAM,SA_binary,CRF_result_SA):
    
    img_mask = imread(MASK_ROOT+path_name+MASK_EX)
    img_mask = cv2.resize(img_mask, dsize=(args.cam_crop_size, args.cam_crop_size)).astype(np.uint32)
    
    img_numpy = torch.squeeze(img).to('cpu').detach().numpy().copy()
    img_numpy_hwc =np_img = np.transpose(img_numpy, (1, 2, 0))
    
    plt.figure(figsize=(5,5))
    plt.title(path_name)
    plt.subplot(4, 2, 1)
    plt.imshow(img_numpy_hwc)
    plt.axis('off')
    plt.subplot(4, 2, 2)
    plt.imshow(img_mask)
    plt.axis('off')
    plt.subplot(4, 2, 3)    
    plt.imshow(CAM_binary)
    plt.axis('off')
    plt.subplot(4, 2, 4)
    plt.imshow(CRF_result_CAM)
    plt.axis('off')
    plt.subplot(4, 2, 5)
    plt.imshow(SA_CAM_binary)
    plt.axis('off')
    plt.subplot(4, 2, 6)
    plt.imshow(CRF_result_SA_CAM)
    plt.axis('off')
    plt.subplot(4, 2, 7)
    plt.imshow(SA_binary)
    plt.axis('off')
    plt.subplot(4, 2, 8)
    plt.imshow(CRF_result_SA)
    plt.axis('off')
    plt.savefig(args.segmentation_out_dir_CRF+path_name+"_plot.png")
    plt.close()

def run(args):
    #乱数の初期設定
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(seed=0)
    MASK_ROOT = args.dataset_root + "mask_binary/"
    #モデルの読み込み
    model = getattr(importlib.import_module(args.cam_network), 'Net')()
    model.fc = nn.Linear(2048,args.cam_output_class)
    model.load_state_dict(torch.load(args.cam_weights_name),strict=True)
    model = torch.nn.DataParallel(model).cuda()

    model_s = getattr(importlib.import_module(args.cam_network), 'out_selfA')()
    model_s.fc = nn.Linear(2048,args.cam_output_class)
    model_s.load_state_dict(torch.load(args.cam_weights_name),strict=True)
    model_s = torch.nn.DataParallel(model_s).cuda()

    #model = model.cuda()
    #データの読み込み
 
    images_path = args.dataset_root + "images/"

    test_data = ImageDataset(images_path,args.test_list, width=args.cam_crop_size, height=args.cam_crop_size, transform=transforms.Compose([
        transforms.Resize((args.cam_crop_size,args.cam_crop_size)),
        transforms.ToTensor()
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    test_loader = DataLoader(test_data, batch_size=args.cam_batch_size)
    
    # Grad-CAMを使えるようにする
    """
    VGGの場合こうする
    configs = [
    dict(model_type='vgg', arch=model, layer_name='features_29')
    ]

    for config in configs:
        config['arch'].cuda().eval()

    cams = [
        [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
        for config in configs
    ]
    gradcam,gradcam_pp = cams[0]
    """
    target_layer = model.module.layer4
    
    gradcam = GradCAM(model, target_layer)
    gradcam_pp = GradCAMpp(model, target_layer)
    
    #モデルの設定
    model.eval()  # inference
    model_s.eval()
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0.0  # loss sum per epoch
    epoch_corrects = 0  # number of correct answers

    #2021/09/22ここの書き方分かりづらい。 バッチごとにできるようにする?
    for iter, batch in enumerate(test_loader):
        inputs = batch["image"].cuda()
        labels = batch["target"].cuda()
        im_paths = batch["path"]

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        x,attention = model_s(inputs)
        
        #grad-cam参考https://www.yurui-deep-learning.com/2021/02/08/grad-campytorch-google-colab/
        
        for i in range(inputs.shape[0]):
              #パス保存用
            path_name = im_paths[i].split('/')[-1].split('.')[0]

            epoch_corrects += torch.sum(preds[i] == labels[i])
            pred_mani = get_prediction_manipulation(preds[i].item(),labels[i].item())
            # pred 1 label 1 -> TP ,pred 1 label0 ->FP, pred 0 label -> 1 FN, pred 0 label 0 -> TN

            #grad-camの部分
            img = torch.unsqueeze(inputs[i],0)
            mask, _ = gradcam(img)
            heatmap, result = visualize_cam(mask, img)
            mask_pp, _ = gradcam_pp(img)
            heatmap_pp, result_pp = visualize_cam(mask_pp, img)
         
            #SAのCAM
            SA_CAM_kasika = save_SA_CAM(args,x[i],path_name,model_s,preds[i].item())
            SA2binary(args,SA_CAM_kasika,path_name,pred_mani)

           
            CAM_image2binary_save(args,heatmap_pp,path_name,pred_mani) #評価に使います
            

            #バイナリCAMを読み込んで、SAの可視化やCRFなどをしていく
            CAM_binary = imread(args.segmentation_out_dir_CAM+path_name+'_'+pred_mani+'_binary_CAM.png').astype(np.uint32)

            #SAの可視化する(attentionベース)
            save_SA(args,attention[i],CAM_binary,path_name,pred_mani)

            #保存したやつらを読み込んで、CRFかけたりplotしたりする。
            SA_CAM_binary = imread(args.segmentation_out_dir_SA_CAM+path_name+'_'+pred_mani+'_binary_SA_CAM.png').astype(np.uint32)
            SA_binary = imread(args.segmentation_out_dir_SA+path_name+'_'+pred_mani+'_binary_SA.png').astype(np.uint32)
            
            
            """
            CAMforCRF = np.squeeze(mask_pp.to('cpu').detach().numpy().copy())
            print(CAMforCRF.shape)
            CAMforCRF_gray = np.stack([CAMforCRF,CAMforCRF,CAMforCRF], -1) #[256,256,1]を重ねて３チャンにしたい
            print(CAMforCRF_gray.shape)
            #CAMforCRF = np.transpose(CAMforCRF, (1, 2, 0))
            """
            

            CRF_result_CAM = CRF(args,img*255,CAM_binary).reshape(img.shape[2],img.shape[3],img.shape[1])
         
            #CRF_result_CAM = CRF(args,img*255,CAM_binary).reshape(img.shape[2],img.shape[3],img.shape[1])
            CRF_result_SA_CAM = CRF(args,img*255,SA_CAM_binary).reshape(img.shape[2],img.shape[3],img.shape[1])
            CRF_result_SA = CRF(args,img*255,SA_binary).reshape(img.shape[2],img.shape[3],img.shape[1])
            

            save_image_func(args,path_name,pred_mani,CRF_result_CAM,CRF_result_SA_CAM,CRF_result_SA,result_pp,heatmap_pp)
            """
            CRF_result_CAM numpy
            CRF_result_SA_CAM numpy
            CRF_result_SA numpy
            result_pp torch
            heatmap_pp torch
            """
           
            print(path_name,":",pred_mani,"(",preds[i].item(),",",labels[i].item(),")")
            
            if pred_mani == "TP":
                images_prot(args,path_name,MASK_ROOT,img,CAM_binary,CRF_result_CAM,SA_CAM_binary,CRF_result_SA_CAM,SA_binary,CRF_result_SA)
               
    #一応２値分類結果も表示        
    epoch_acc = epoch_corrects.double() / len(test_loader.dataset)
    print("test_acc=",epoch_acc)