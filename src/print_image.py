import numpy as np
from PIL import Image


def fill_red(x, i, rgbArray):
    rgbArray[i, :, 0] = np.floor((x[(32 * i):(32 * (i + 1))] * 255))


def fill_green(x, i, rgbArray):
    rgbArray[i, :, 0] = np.floor((x[(32 * i):(32 * (i + 1))] * 255))


def fill_blue(x, i, rgbArray):
    rgbArray[i, :, 0] = np.floor((x[(32 * i):(32 * (i + 1))] * 255))


# Show and/or save an image
# img_src is a line from the data (panda DataFrame)
def print_img(img_src, show=True, save=False):
    x = img_src.values + 0.5

    rgbarray = np.zeros((32, 32, 3), 'uint8')

    for i in range(32):
        fill_red(x, i, rgbarray)
        fill_green(x, i, rgbarray)
        fill_blue(x, i, rgbarray)

    img = Image.fromarray(rgbarray, 'RGB')

    if show:
        img.show()
    if save:
        img.save('myimg2.jpeg')

