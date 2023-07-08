# Transfer learning

import os

from keras import layers
from keras import Model

from keras.applications.inception_v3 import InceptionV3

local_weights = ''

pretrained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False, # get straight to convolutions
    weights=None
)

pretrained_model.load_weights(local_weights)

for layer in pretrained_model.layers:
    layer.trainable = False

# See the architectureo of model
pretrained_model.summary()


# Get layers with layer names
last_layer = pretrained_model.get_layer('mixed7')
last_output = last_layer.output

