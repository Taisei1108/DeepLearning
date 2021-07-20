import os
import glob
from PIL import Image
#ヒートマップやコロンビアのマスクがIoUを測るにはまだ不十分なため

root ="./data/"
print(root)

files = glob.glob(root+"*")
index = 0
print(files[index])

image = Image.open(files[index]).convert('RGB')


print(image.size, image.mode)