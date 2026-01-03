import numpy as np
from PIL import Image, ImageSequence

# Script for genereting numpy timeseries from gif-file


def get_anime_timeseries(rgb=False):
    """
    Gets a time series of frames from an animated GIF.

        Args:
            rgb: If True, returns the time series as RGB images.
                 If False, returns the time series as grayscale images.

        Returns:
            numpy.ndarray: A NumPy array representing the time series of image frames,
                           normalized to the range [0, 1]. The shape will be
                           (number_of_frames, height, width, channels), where channels is 3 for RGB and 1 for grayscale.
    """
    with Image.open("anime_10f.gif") as im:
        array = []
        for frame in ImageSequence.Iterator(im):
            if rgb:
                im_data = frame.copy().convert("RGB").getdata()
                im_array = np.array(im_data).reshape(frame.size[1], frame.size[0], 3)
            else:
                im_data = frame.copy().convert("L").getdata()
                im_array = np.array(im_data).reshape(frame.size[1], frame.size[0], 1)
            array.append(im_array)
        array = np.array(array)
        array = array / 255
    return array


def get_cycled_data(array, cycles_num):
    """
    Cycles the input array a specified number of times.

        This method repeats the input array 'cycles_num' times and then reshapes it
        into a new array with an increased first dimension.

        Args:
            array: The input NumPy array to be cycled.
            cycles_num: The number of times to cycle (repeat) the array.

        Returns:
            numpy.ndarray: A new NumPy array that is the result of cycling and reshaping
                           the original array.  The first dimension will be cycles_num
                           times the original array's first dimension, while the remaining
                           dimensions remain unchanged.
    """
    arr = []
    for i in range(cycles_num):
        arr.append(array)
    arr = np.array(arr)
    arr = arr.reshape(
        arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4]
    )
    return arr
