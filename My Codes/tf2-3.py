# Transfer learning

import os
import tensorflow as tf
from keras import layers
from keras import Model

from keras.applications.inception_v3 import InceptionV3

local_weights = ''

pretrained_model = InceptionV3(input_shape=(150, 150, 3),
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


# Now lets define our model
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pretrained_model.input, x)
model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=0.0001),
              loss= tf.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


# ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.2,
                                     shear_range=0.2,
                                     horizontal_flip=True)


train_generator = train_datagen.flow_from_directory('training-dir',
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = train_datagen.flow_from_directory('validation-dir',
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=100,
                    validation_steps=50,
                    verbose=2)