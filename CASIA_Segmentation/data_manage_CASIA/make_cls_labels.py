"""
[データセットのラベルがファイル名からわからなくて、ディレクトリ自体に
ついている場合]
パス名をキーにラベルを対応させて辞書ファイルをnpy方式で保存することで回避する

2021年9月27日もしかしたら、labelを0,1,2...じゃなくて [1,0,0,0,...],[0,1,0.....]にしないとかも

CASIAのやりかた
python make_cls_labels.py --dataset_root /home/takahashi/datasets/CASIAv2_data/data_label/ --class_num 2
"""



import argparse
import glob
import os

import numpy as np

if __name__ == '__main__':

    #引数の読み込み、データセットのルートとデータセットのサブディレクトリを指定する

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root',required=True,type=str)
    parser.add_argument('--class_num',required=True,type=int)
    parser.add_argument('--out',default="cls_labels.npy",type=str)
    args = parser.parse_args()
    print(vars(args))

    all_files =[]

    for i in range(args.class_num):
        files = glob.glob(args.dataset_root+str(i)+"/*")
        all_files.append(files)

    dataset = dict()
    for class_label in range(len(all_files)):
        for path_name in all_files[class_label]:
            path_name_split = path_name.split("/")[-1].split(".")[0]
            print(path_name_split,":",class_label)
            dataset[path_name_split] = class_label #"data_path_name":class_labelで保存される
    

    np.save(args.out, dataset)