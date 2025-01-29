import numpy as np
from PIL import Image, ImageSequence

# Script for genereting numpy timeseries from gif-file

def get_anime_timeseries(rgb=False):
    with Image.open('anime_10f.gif') as im:
        array = []
        for frame in ImageSequence.Iterator(im):
            if rgb:
                im_data = frame.copy().convert('RGB').getdata()
                im_array = np.array(im_data).reshape(frame.size[1], frame.size[0], 3)
            else:
                im_data = frame.copy().convert('L').getdata()
                im_array = np.array(im_data).reshape(frame.size[1], frame.size[0], 1)
            array.append(im_array)
        array = np.array(array)
        array = array/255
    return array


def get_cycled_data(array, cycles_num):
    arr = []
    for i in range(cycles_num):
        arr.append(array)
    arr = np.array(arr)
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4])
    return arr