import os
import glob
from PIL import Image
import numpy as np
from matplotlib import pylab as plt
from torchvision import transforms
#ヒートマップやコロンビアのマスクがIoUを測るにはまだ不十分なため,2値化させる
#メモ　maskはedgemask_3の方を使う

root ="./data/"
width = 256
height = 256
print(root)

files = glob.glob(root+"*")
index = 0
print(files)
print(files[index])

transform=transforms.Compose([
            transforms.Resize((width,height))
            #transforms.ToTensor(),
            #ransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image = Image.open(files[index]).convert('RGB')

image0 = transform(Image.open(files[0]).convert('RGB'))
image1 = transform(Image.open(files[1]).convert('RGB'))
image2 = transform(Image.open(files[2]).convert('RGB'))

new_image1 =  image1.convert('L')

pixelSizeTuple = image0.size
new_image0 = Image.new('RGB', image0.size)

for i in range(pixelSizeTuple[0]):
    for j in range(pixelSizeTuple[1]):
        r,g,b = image0.getpixel((i,j))
        if r > 120 :  #100~120くらいがよさそう、30くらいまで下げると黄色も含む、10くらいまで下げると緑とかも
            new_image0.putpixel((i,j), (255,255,255)) 
    else:
        new_image0.putpixel((i,j), (0,0,0)) 

new_image0.save('test0.png',quality=100)

new_image1=new_image1.point(lambda x: 0 if x < 90 else 255)#マスクの閾値はこれかな ~70,130~だと漏れるので
print(new_image1.mode)
new_image1.save('test1_2.png',quality=100)

#print(image.size, image.mode,image.getextrema())