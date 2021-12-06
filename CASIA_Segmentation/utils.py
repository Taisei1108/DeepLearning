import mlflow
import numpy as np

np.random.seed(seed=0)

def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard """
    mlflow.log_metric(name, value)


def np_img_HWC_debug(np_img,np_img_str=None,transpose=False):
    
    if len(np_img.shape) == 4:
        np_img = np.squeeze(np_img) 

    if transpose ==True: #CHW to HWC
        np_img = np.transpose(np_img.cpu(), (1, 2, 0))

    if len(np_img.shape) == 2:
        np_img = np_img[:,:,np.newaxis] 
    
    
    min = np_img[0][0][0]
    max = np_img[0][0][0]
    sum = 0

    if np_img.shape[0] == 3 or np_img.shape[0]  == 1:
        print("もしかしてCHWになってませんか")
        return 0
    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            sum += np_img[i][j][0]
            if min > np_img[i][j][0]:
                min = np_img[i][j][0]
            if max < np_img[i][j][0]:
                max = np_img[i][j][0]
    if np_img_str !=None:
        print("===",np_img_str,"===")
    print("-image_size",np_img.shape)
    print("-image_array_size",len(np_img.shape))
    print("-SUM:",sum)
    print("-MAX:",max)
    print("-MIN:",min)
    return 0