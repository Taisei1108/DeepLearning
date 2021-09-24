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

def calc_iou_F_measure(cam_image,mask_image): 
    #cam_imageとmask_imageはバイナリの必要がある 0 or 255
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
    #print("interesection/union",intersection,"/",union,"=",intersection/union)
    #print("TP/(TP+FN+FP)",TP,"/",TP,"+",FN,"+",FP,"=",TP/(TP+FN+FP)) #一応TP,FPの概念で確かめ算
    precision = TP /(TP+FP)
    recall = TP /(TP+FN)
    F_measure = 2*recall*precision/(recall+precision)
    IoU = intersection/union

    return IoU, F_measure


def run(args,seg_dir_path):

    MASK_ROOT = args.dataset_root + "edgemask/"
    seg_data_files = glob.glob("./"+seg_dir_path+"*")
    print(len(seg_data_files))

    transform=transforms.Compose([
            transforms.Resize((args.cam_crop_size,args.cam_crop_size))
    ])
    iou = 0
    count = 0
    F_measure = 0
    for f in seg_data_files:
        seg_name = f.split('/')[-1].split('.')[0] #pathやpingを取り外す
        seg_image = Image.open(f).convert('L') #恐らく、２値化した状態で読み込めているはず(チャンネル１)
        if seg_name.split('_')[-2] == "TP":# or path_name.split('_')[-1] == "FN":
            count+=1
            mask_path = seg_name[:-7] #_TP_segのところを削る
           
            mask_image_gray = Image.open(MASK_ROOT+mask_path+'_edgemask_3.jpg').convert('L')
            mask_image_crop = transform(mask_image_gray)
            mask_image_binary=mask_image_crop.point(lambda x: 0 if x < 90 else 255)#マスクの閾値はこれかな ~70,130~だと漏れるので
            
            #Mask化したnew_imageとおそらく2値化されてるcam_imageと比較してIoU計算(convert(L)でいいのか)
            print("debag:",mask_path)
            iou_tmp , F_measure_tmp = calc_iou_F_measure(seg_image,mask_image_binary)
            iou += iou_tmp
            F_measure += F_measure_tmp
            print("img_name:",mask_path,"IoU:",iou_tmp,"F-measure:",F_measure_tmp)
    print("Result IoU:",iou,"/",count,"=",iou/count)
    print("Result F_measure:",F_measure,"/",count,"=",F_measure/count)