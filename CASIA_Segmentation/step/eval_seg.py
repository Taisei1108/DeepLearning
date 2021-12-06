import os
import glob
from PIL import Image
import numpy as np
from matplotlib import pylab as plt
from torchvision import transforms

"""
マスク画像を読み込んで、make_cam.pyで作成したセグメンテーション画像と比較
iouとF値を計算して評価を行う

画像を保存した名前によってプログラムの定数を変えたり、汎化性がなかったりするので
変えたい2021/09/25

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
    if TP!=0 and FP!=0:
        precision = TP /(TP+FP)
        recall = TP /(TP+FN)
        F_measure = 2*recall*precision/(recall+precision)
        IoU = intersection/union

        return IoU, F_measure

    return 0,0
def run(args,seg_dir_path):

    np.random.seed(seed=0)

    MASK_ROOT = args.dataset_root + "mask_binary/"
    seg_data_files = glob.glob("./"+seg_dir_path+"*")
    print(len(seg_data_files))

    transform=transforms.Compose([
            transforms.Resize((args.cam_crop_size,args.cam_crop_size))
    ])
    iou = 0
    count = 0
    F_measure = 0
    for f in seg_data_files:
        #2021年9月25日現在　fは./result/seg/CAM/canong3_canonxt_sub_03_FN_binary_CAM.pngみたいな感じ
        seg_name = f.split('/')[-1][:-15] #pathやpingを取り外す(保存の名前変わると変えなきゃいけないの変えないとだな)
        seg_image = Image.open(f).convert('L') #恐らく、２値化した状態で読み込めているはず(チャンネル１)
       
        #普通にimage openしただけなのに (3,65536)になるの意味不だな
        #ここの条件分岐エラーでやすい
   
        if seg_name.split('_')[-1] == "TP":# or path_name.split('_')[-1] == "FN":
            count+=1
            
            # mask_path = seg_name[:-7] #_TP_segのところを削る
            mask_path = seg_name[:-3]
            mask_image_gray = Image.open(MASK_ROOT+mask_path+'_edgemask_3.jpg').convert('L')
            mask_image_crop = transform(mask_image_gray)
            mask_image_binary=mask_image_crop.point(lambda x: 0 if x < 90 else 255)#マスクの閾値はこれかな ~70,130~だと漏れるので
            
            #Mask化したnew_imageとおそらく2値化されてるcam_imageと比較してIoU計算(convert(L)でいいのか)
            
            iou_tmp , F_measure_tmp = calc_iou_F_measure(seg_image,mask_image_binary)
                                                #データサイズ　seg_image(256,256) image_binary(256,256)
            iou += iou_tmp

            print(mask_path,":iou:",iou_tmp,"F_measure:",F_measure_tmp)
            F_measure += F_measure_tmp
            
    print("Result IoU:",iou,"/",count,"=",iou/count)
    print("Result F_measure:",F_measure,"/",count,"=",F_measure/count)