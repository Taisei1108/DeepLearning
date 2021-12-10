"""
コロンビアのマスクを事前にバイナリ化するためのファイル
image2segmentationとほぼ同じだが、なるべく汎用的にした

"""

import argparse
import glob
import os
from PIL import Image

from torchvision import transforms

CROP_SIZE = 256
transform=transforms.Compose([
        transforms.Resize((CROP_SIZE,CROP_SIZE))
])

#python Columbia2binary.py  --Columbia_root ../../datasets/Columbia/data/

if __name__ == '__main__':

    
    parser = argparse.ArgumentParser()

    parser.add_argument('--Columbia_root',required=True,type=str)
    parser.add_argument('--mask_dir',default="edgemask/",type=str)
    parser.add_argument('--out',default="mask_binary/",type=str)
    args = parser.parse_args()
    print(vars(args))

    os.makedirs(args.Columbia_root+args.out, exist_ok=True)

    files = glob.glob(args.Columbia_root+args.mask_dir+"/*")
    print(files)


    for path in files:
        if path.split(".")[-2][-10:]=="edgemask_3": #想定の入力'../../datasets/Columbia/data/edgemask/canong3_nikond70_sub_11_edgemask.jpg'
            mask_image_gray = Image.open(path).convert('L')
            #mask_image_crop = transform(mask_image_gray)
            #mask_image_binary=mask_image_crop.point(lambda x: 0 if x < 90 else 255)#マスクの閾値はこれかな ~70,130~だと漏れるので
            mask_image_binary=mask_image_gray.point(lambda x: 0 if x < 90 else 255)#マスクの閾値はこれかな ~70,130~だと漏れるので
            
            path_name = path.split('/')[-1]

            #mask_image_binary.save(args.Columbia_root+args.out+path_name, quality=100)
            mask_image_binary.save("./"+args.out+path_name, quality=100)