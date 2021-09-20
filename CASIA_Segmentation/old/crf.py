import numpy as np
import glob
import pydensecrf.densecrf as dcrf
import cv2
try:
    from cv2 import imread, imwrite
except ImportError:
    # If you don't have OpenCV installed, use skimage
    from skimage.io import imread, imsave
    imwrite = imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from torchvision import transforms

from data_loader import ImageDataset

#参考サイト　https://www.programmersought.com/article/45791248551/
width = 256
height = 256

predict_image_path = "./out_seg_r120/"
original_image_path = "/home/takahashi/datasets/ColumbiaUncompressedImageSplicingDetection/data/test/1/"

transform=transforms.Compose([
            transforms.Resize((width,height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #これいるのかな
])

files_p = glob.glob(predict_image_path+"*")
files_o = glob.glob(original_image_path+"*")



#メモ predictしたデータの中でTPになった結果をCRFに入れる
"""
print(len(files)) -> 66
print(len(original_data)) -> 66 
同じテストデータだから。ここからTPの画像のみ抜き出してCRF関数にぶちこむ
"""

def CRFs(original,predict):
    img = imread(original)
    img = cv2.resize(img, dsize=(width, height)) #(256, 256, 3)
    # Convert the RGB color of predicted_image to uint32 color 0xbbggrr
    anno_rgb = imread(predict).astype(np.uint32)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    # Convert uint32 color to 1, 2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True) #color [  0 16777215] labels[1 1 1 ... 0 0 0] (.shape = 65536,)
    """
    HAS_UNK = 0 in colors #コメントアウトするかもしれない
    if HAS_UNK:
        colors = colors[1:]
    """
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    n_labels = len(set(labels.flat))
    #n_labels = len(set(labels.flat)) - int(HAS_UNK)
    use_2d = False     
    if use_2d:                   
                 # Use densecrf2d class
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

                 # Get a dollar potential (negative log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
                 #U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## If there is an indeterminate area, replace the previous line with this line of code
        d.setUnaryEnergy(U)

                 # Added color-independent terms, the function is just the location
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

                 # Added color related terms, ie the feature is (x, y, r, g, b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
                 # Use densecrf class
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

                 # Get a dollar potential (negative log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=None)  
                 #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## If there is an indeterminate area, replace the previous line with this line of code
        d.setUnaryEnergy(U)

                 # This will create color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

                 # This will create color-related features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    # 5 times reasoning
    Q = d.inference(5)

    # Find the most likely class for each pixel
    MAP = np.argmax(Q, axis=0)

    # Convert predicted_image back to the appropriate color and save the image
    MAP = colorize[MAP,:]
    output_path = predict.split('/')[2].split('.')[0]
    imwrite("./crf/"+output_path+".png", MAP.reshape(img.shape))
    print(predict)
    print("CRF image saved in ./crf/"+output_path+".png!")

    return 0

for f in files_p:
    path_name = f.split('/')[-1]
    if path_name.split('_')[-1][0:2] == "TP":
        original_image_name = path_name.split('.')[0][:-3]+".tif"
        print(path_name)
        print(original_image_name)
        original =  original_image_path + original_image_name
        predict = predict_image_path + path_name
        CRFs(original, predict)
    