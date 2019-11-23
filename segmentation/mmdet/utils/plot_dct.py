import math
import numpy as np
import matplotlib.pyplot as plt

filelist = ['000000120420.jpg', '000000166287.jpg', '000000166391.jpg', '000000212559.jpg', '000000286994.jpg',
            '000000300659.jpg', '000000438862.jpg', '000000460347.jpg', '000000509735.jpg']

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def dct_flatten_2d(img):
    height, width, channel = img.shape
    N = int(math.sqrt(channel))
    height_resized, width_resized = height * N, width * N

    # Do 8x8 DCT on image (in-place)
    img = img.reshape((height, width, N, N)).reshape(-1, N, N).astype(dtype='float32')
    img_resized = unblockshaped(img, height_resized, width_resized)
    return img_resized


def plot_dct(img, filename):
    if filename in filelist:
        dct = dct_flatten_2d(img)
        plt.figure()
        plt.imshow(dct,cmap='gray',vmax = np.max(dct)*0.01,vmin = 0)
        plt.title( "8x8 DCTs of the image")
        plt.savefig(filename)