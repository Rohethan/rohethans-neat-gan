from glob import glob
from PIL import Image
import numpy
import random
from tqdm import tqdm

path = "MNIST Dataset JPG format\MNIST - JPG - training/"
digit = "8"
n_elements = 5000

i = 0
megaarray = numpy.ndarray((n_elements, 784))
filepaths = glob(path +digit+"\*.jpg")
random.shuffle(filepaths)
for element in filepaths[:n_elements]:
	megaarray[i] = (numpy.asarray(Image.open(element))/255).reshape(784)
	i += 1

numpy.save("dataset.npy", megaarray)