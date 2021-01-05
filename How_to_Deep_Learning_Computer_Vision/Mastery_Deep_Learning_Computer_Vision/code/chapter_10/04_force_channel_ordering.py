# force a channel ordering
from keras import backend
# force channels-first ordering
backend.set_image_data_format('channels_first')
print(backend.image_data_format())
# force channels-last ordering
backend.set_image_data_format('channels_last')
print(backend.image_data_format())