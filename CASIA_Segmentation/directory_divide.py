import os
import glob

#データセットを指定した確率でランダムに移すプログラム
#データセットのところでいつも動かす
"""
dataset/train/0
             -1
        |
        -val/0
            -1
        |
        -test/0
             -1    
のようにすると動く(0,1はラベルごとにフォルダにする)
"""
#root = "./CASIA/CASIAv2/tam/"
root = "./CASIA/AU_origin/"

print(root)
files = glob.glob(root+"*")
print(files)
print(len(files))

import random
random.seed(0)
import shutil
for f in files:
    print(f)
    rand_num =random.random()
    if rand_num >= 0.9: # 10%
        # valへ移動
        print('To val',f.split('/')[-1])
        #shutil.copyfile(f,"./CASIA/val/tam/"+f.split('/')[-1])
        shutil.copyfile(f,"./CASIA/val/AU/"+f.split('/')[-1])
    elif rand_num >= 0.2: #60%
        # trainへ移動
        print('To train',f.split('/')[-1])
        #shutil.copyfile(f,"./CASIA/train/tam/"+f.split('/')[-1])
        shutil.copyfile(f,"./CASIA/train/AU/"+f.split('/')[-1])
    else:                      # 30%
        # testへ移動
        print('To test',f.split('/')[-1])
        #shutil.copyfile(f,"./CASIA/test/tam/"+f.split('/')[-1])
        shutil.copyfile(f,"./CASIA/test/AU/"+f.split('/')[-1])


"""
train/tam 3592
test/tam 1003
val/tam 528
train/AU 5291
test/AU 1454
val/AU 746


"""