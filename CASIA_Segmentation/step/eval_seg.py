import os
import glob
from PIL import Image
import numpy as np
from matplotlib import pylab as plt
from torchvision import transforms

"""
マスク画像を読み込んで、make_cam.pyで作成したセグメンテーション画像と比較
iouとF値を計算して評価を行う
"""

def calc_iou(cam_image,mask_image):
    pixelSizeTuple = cam_image.size
    intersection = 0 #積集合
    union = 0 #和集合
    FP=0
    FN=0
    TP=0
    for i in range(pixelSizeTuple[0]):
            for j in range(pixelSizeTuple[1]):
                
                cam_pix = cam_image.getpixel((i,j))
                mask_pix = mask_image.getpixel((i,j))
                if cam_pix == 255 or mask_pix == 255:
                    union += 1
                if cam_pix  == 255 and mask_pix == 0: #FP  予測はしたけど正解ではない
                    FP += 1
                if mask_pix == 255 and cam_pix  == 0: #FN 正解なんだけど予測できていない
                    FN += 1
                if cam_pix == 255 and mask_pix ==255: #TP
                    intersection += 1
                    TP += 1
    print("interesection/union",intersection,"/",union,"=",intersection/union)
    print("TP/(TP+FN+FP)",TP,"/",TP,"+",FN,"+",FP,"=",TP/(TP+FN+FP)) #一応TP,FPの概念で確かめ算
    precision = TP /(TP+FP)
    recall = TP /(TP+FN)
    F_measure = 2*recall*precision/(recall+precision)
    print("precision=",precision)
    print("recall=",recall)
    print("F-measure",F_measure)
    return intersection/union


def run(args):
    MASK_ROOT = args.dataset_root + "edgemask/"
    seg_data_files = glob.glob(args.segmentation_out_dir+"*")
    print(len(seg_data_files))

    transform=transforms.Compose([
            transforms.Resize((args.cam_crop_size,args.cam_crop_size))
    ])
    iou = 0
    count = 0
    
    for f in seg_data_files:
        print(f)
        path_name = f.split('/')[-1].split('.')[0]
        print(path_name)
        seg_image = Image.open(f).convert('L') #恐らく、２値化した状態で読み込めているはず(チャンネル１)
        if path_name.split('_')[-1] == "TP":# or path_name.split('_')[-1] == "FN":
            count+=1
            mask_path = path_name[:-3]
            print(MASK_ROOT+mask_path+'_edgemask_3.jpg')
            mask_image_gray = Image.open(MASK_ROOT+mask_path+'_edgemask_3.jpg').convert('L')
            mask_image_crop = transform(mask_image_gray)
            mask_image_binary=mask_image_crop.point(lambda x: 0 if x < 90 else 255)#マスクの閾値はこれかな ~70,130~だと漏れるので
            
            #Mask化したnew_imageとおそらく2値化されてるcam_imageと比較してIoU計算(convert(L)でいいのか)
            iou += calc_iou(seg_image,mask_image_binary)
            #seg_image.save('./out_mask_seg/'+path_name+'_cam.png')
            #mask_image_binary.save('./out_mask_seg/'+path_name+'_mask.png')
        
    print("IoU:",iou,"/",count,"=",iou/count)