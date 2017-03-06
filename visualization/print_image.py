import numpy as np
import pandas as pd
from PIL import Image
from math import floor


# Show and/or save an image
# img_src is a line from the data (panda DataFrame)
def print_img(img_src, show = True,save = False):
	x = img_src.values + 0.5
	print("Hello")

	print("2")
	rgbArray = np.zeros((32,32,3), 'uint8')

	for i in range(32):
	    rgbArray[i,:,0] = (x[(32*i):(32*(i+1))]*255).map(floor)
	    rgbArray[i,:,1] = (x[(1024+32*i):(1024+32*(i+1))]*255).map(floor)
	    rgbArray[i,:,2] = (x[(2048+32*i):(2048+32*(i+1))]*255).map(floor)
	img = Image.fromarray(rgbArray,'RGB')

	if show:
		img.show()
	if save:
		img.save('myimg2.jpeg')