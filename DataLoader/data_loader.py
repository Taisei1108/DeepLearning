from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class ImageDataset(Dataset):

    def __init__(self, root, split='train',transform=None):
        self.root = root
        self.images_dir = os.path.join(self.root, split)  # train or val
        self.images = []  # image list
        self.targets = []  # label list
        self.transform = transform

        for label in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, label)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(int(label))
                
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        #image = image.resize((self.width, self.height)) //0708 transformで実装することに
        #downscaled = np.asarray((image.resize((self.width // 8, self.height // 8))))

        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'target': self.targets[index], "path": self.images[index]}
        return sample

