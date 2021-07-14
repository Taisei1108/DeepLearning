import os
import glob
import shutil
#データセットを指定した確率でランダムに移すプログラム
#データセットのところでいつも動かす
#0715変更、学習プログラムを動かす前に動かしてデータセットを柔軟に変えれるようにする
#ColumbiaとCASIAをpathを変えるだけで実装したい
#data /0,1,train,test,val のようなデータ構成にする
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

#root = "/home/takahashi/datasets/CASIAv2_data/"
root ="/home/takahashi/datasets/ColumbiaUncompressedImageSplicingDetection/data/"
print(root)

def make_dir():
    print("train/test/valを消しますか？[y/n]")
    str = input()
    if str == "y":
        if os.path.exists(root+"train"):
            shutil.rmtree(root+"train")
        if os.path.exists(root+"test"):
            shutil.rmtree(root+"test")
        if os.path.exists(root+"val"):
            shutil.rmtree(root+"val")
        os.makedirs(root+"train/0", exist_ok=True)
        os.makedirs(root+"train/1", exist_ok=True)
        os.makedirs(root+"test/0", exist_ok=True)
        os.makedirs(root+"test/1", exist_ok=True)
        os.makedirs(root+"val/0", exist_ok=True)
        os.makedirs(root+"val/1", exist_ok=True)
    else:
        print("削除しませんでした！")
make_dir()

files = glob.glob(root+"*")
print(len(files))


label = [0,1]
import random
random.seed(0)

count_train = 0
count_test = 0
count_val = 0

for i in label:
    files = glob.glob(root+str(i)+"/*")
    print(len(files))   
    for f in files:
        print(f)
        rand_num =random.random()
        if rand_num >= 0.9: # 10%
            # valへ移動
            print('To val',f.split('/')[-1])
            shutil.copyfile(f,root+"val/"+str(i)+"/"+f.split('/')[-1])
            count_val +=1
        elif rand_num >= 0.2: #60%
            # trainへ移動
            print('To train',f.split('/')[-1])
            shutil.copyfile(f,root+"train/"+str(i)+"/"+f.split('/')[-1])
            count_train += 1
        else:                      # 30%
            # testへ移動
            print('To test',f.split('/')[-1])
            shutil.copyfile(f,root+"test/"+str(i)+"/"+f.split('/')[-1])
            count_test += 1
print("count_train:",count_train)
print("count_val",count_val)
print("count_test",count_test)

