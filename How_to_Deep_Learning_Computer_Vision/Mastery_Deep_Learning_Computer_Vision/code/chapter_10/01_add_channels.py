# example of expanding dimensions
from numpy import expand_dims
from numpy import asarray
from PIL import Image
# load the image
img = Image.open('penguin_parade.jpg')
# convert the image to grayscale
img = img.convert(mode='L')
# convert to numpy array
data = asarray(img)
print(data.shape)
# add channels first
data_first = expand_dims(data, axis=0)
print(data_first.shape)
# add channels last
data_last = expand_dims(data, axis=2)
print(data_last.shape)