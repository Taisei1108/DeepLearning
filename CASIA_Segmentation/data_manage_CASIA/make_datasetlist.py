"""
データセットをすべて読み込んで、指定した数で分割 -> パス名をこのディレクトリにtxtデータで保存するプログラム
・train_list.txt
・test_list.txt
・val_list.txt
が作られる予定

実験でデータセットを使う場合はリストから読み込むようにする
データセットROOT/サブディレクトリ(ラベルごとに0,1,...Nのように並んでいてほしい)
"""
#python make_datasetlist.py --dataset_root ../../../datasets/Columbia/data/ --class_num 2
#python  make_datasetlist.py --dataset_root ../../../datasets/CASIAv2_data/data_label/ --class_num 2
import argparse
import glob
import os
import random

random.seed(3)

if __name__ == '__main__':

    #引数の読み込み、データセットのルートとデータセットのサブディレクトリを指定する

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root',required=True,type=str)
    parser.add_argument('--class_num',required=True,type=int)
    #デフォルトで8:1:1に分割する
    parser.add_argument('--divide_ratio',default=(8,1,1),help='データセットの割合を決める、いい案が浮かばないのでタプルにする')

    args = parser.parse_args()
    print(vars(args))

    all_files =[]

    train_ratio = args.divide_ratio[0]/10
    val_ratio = args.divide_ratio[1]/10
    test_ratio =  args.divide_ratio[2]/10

    """
    #データをすべて取得する
    for i in range(args.class_num):
        files = glob.glob(args.dataset_root+str(i)+"/*")
        all_files.append(files)
    """

    #取得するデータを選別する。
    files_0 = glob.glob(args.dataset_root+"0"+"/*")
    files_1 = glob.glob(args.dataset_root+"1"+"/Tp_D*")
    all_files.append(files_0)
    all_files.append(files_1)
    
    dataset = dict(train=[],val=[],test=[]) #0から train,val,testの順で打ち込む予定
    
    dataset_name =["train","val","test"]

    for i in range(args.class_num):
        shuffled_list = random.sample(all_files[i],len(all_files[i]))
        dataset["train"] += shuffled_list[:int(len(shuffled_list)*train_ratio)] 
        dataset["val"] += shuffled_list[int(len(shuffled_list)*train_ratio):int(-1*len(shuffled_list)*test_ratio)]
        dataset["test"]  += shuffled_list[int(-1*len(shuffled_list)*test_ratio):]
        #print(shuffled_list)
        #print(shuffled_list[:int(len(shuffled_list)*train_ratio)] )
    print(len(dataset["train"]))
    print(len(dataset["val"]))
    print(len(dataset["test"]))
    random.shuffle(dataset["train"])
    random.shuffle(dataset["val"])
    random.shuffle(dataset["test"])

    with open('train_CASIA_2.txt', 'w') as f:
        for d in dataset["train"]:
            f.write("%s\n" % d.split('/')[-1].split('.')[0]) 

    with open('val_CASIA_2.txt', 'w') as f:
        for d in dataset["val"]:
            f.write("%s\n" % d.split('/')[-1].split('.')[0])

    with open('test_CASIA_2.txt', 'w') as f:
        for d in dataset["test"]:
            f.write("%s\n" % d.split('/')[-1].split('.')[0])