from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

cls_labels_dict = np.load('data_manage/cls_labels.npy', allow_pickle=True).item()

def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

def load_img_name_list(dataset_path):

    img_name_list = np.loadtxt(dataset_path, dtype=str)

    return img_name_list


class ImageDataset(Dataset):

    def __init__(self, root,data_list_path, width=320, height=240, transform=None):
        self.root = root
        #self.images_dir = os.path.join(self.root, split)  # train or val
        self.images = []  # image list
        self.targets = []  # label list
        self.width = width
        self.height = height
        self.transform = transform
        print(data_list_path)
        self.img_name_list = load_img_name_list(data_list_path)  #従来はint32だが、strで取得できた
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        print(len(self.img_name_list))
        print(len(self.label_list))
        if len(self.img_name_list) != len(self.label_list):
                print("img_listとlabel_listの長さが違います")
        for i in range(len(self.img_name_list)):
            img_dir = os.path.join(self.root, self.img_name_list[i]+".tif") #2021年9月29日、拡張子ベタ打ちになっている
            # print(img_dir,int(self.label_list[i])) #../../datasets/Columbia/all_data/canonxt_32_sub_08 0　と表示
            self.images.append(img_dir)
            self.targets.append(int(self.label_list[i]))
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        #image = image.resize((self.width, self.height)) //0708 transformで実装することに
        downscaled = np.asarray((image.resize((self.width // 8, self.height // 8))))

        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'target': self.targets[index], "path": self.images[index], "downscaled": downscaled}
        return sample

