import os
import glob
from PIL import Image
import numpy as np
from matplotlib import pylab as plt
from torchvision import transforms

#07/21 コロンビアデータセットのセグメンテーションテストをする
#必要なもの：CAMの結果(r = n1 , g = n2 の閾値で2値化)のディレクトリとコロンビアのマスクとそのパス
#CAMは学習時にリサイズされているので、Maskにも同じようにリサイズを施す
#./out_mask_segに マスクとCAMの結果が入る、 CAMのsegはout_segにも同じものが入っているが、マスクと一緒に入っていて見やすい
CAM_ROOT ="./out_seg_r120/"
MASK_ROOT = "../../datasets/ColumbiaUncompressedImageSplicingDetection/edgemask/"
width = 256
height = 256
print(CAM_ROOT)


files = glob.glob(CAM_ROOT+"*")
print(len(files))

transform=transforms.Compose([
            transforms.Resize((width,height))
            #transforms.ToTensor(),
            #ransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

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
    return intersection/union

if __name__ == '__main__':
    iou = 0
    count = 0
    for f in files:
        print(f)
        path_name = f.split('/')[-1].split('.')[0]
        print(path_name)
        cam_image = Image.open(f).convert('L')
        if path_name.split('_')[-1] == "TP":# or path_name.split('_')[-1] == "FN":
            count+=1
            mask_path = path_name[:-3]
            print(MASK_ROOT+mask_path+'_edgemask_3.jpg')
            mask_image_gray = transform(Image.open(MASK_ROOT+mask_path+'_edgemask_3.jpg').convert('L'))
            mask_image=mask_image_gray.point(lambda x: 0 if x < 90 else 255)#マスクの閾値はこれかな ~70,130~だと漏れるので
            
            #Mask化したnew_imageとおそらく2値化されてるcam_imageと比較してIoU計算(convert(L)でいいのか)
            iou += calc_iou(cam_image,mask_image)
            cam_image.save('./out_mask_seg/'+path_name+'_cam.png')
            mask_image.save('./out_mask_seg/'+path_name+'_mask.png')
        
    print("IoU:",iou,"/",count,"=",iou/count)