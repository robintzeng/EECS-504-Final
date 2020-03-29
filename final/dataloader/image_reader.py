from skimage import io
import numpy as np
import torchvision.transforms as transforms

def image_transforms():
    return transforms.Compose([
            transforms.ToTensor()
        ])

def read_rgb(path):
    """Read rgb image as np array

    Returns:
    img: numpy array with shape (h, w, c) = (375 x 1242 x 3)
    """
    img = io.imread(path)
    return img.astype('float32')
    
def read_lidar(path):
    """Read lidar image and generate mask

    Returns:
    lidar: np array with shape (h, w, 1) = (375 x 1242 x 1)
    mask: np array with shape (h, w, 1)
    """
    lidar = io.imread(path) # with shape (h, w)
    lidar = lidar * 1.0 / 256.0
    mask = np.where(lidar > 0.0, 1.0, 0.0) # with shape (h, w)

    lidar = lidar[:, :, np.newaxis].astype('float32')
    mask = mask[:, :, np.newaxis].astype('float32')
    return lidar, mask    

def read_gt(path):
    """Read gt image.

    Returns:
    dense: np array with shape (h, w, 1) = (375 x 1242 x 1)
    """
    dense = io.imread(path)
    dense = dense * 1.0 / 256.0
    dense = dense[:, :, np.newaxis].astype('float32')

    return dense