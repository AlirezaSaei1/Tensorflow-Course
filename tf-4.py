import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'Images'
train_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size = (300, 300),
    batch_size = 128,
    class_mode = 'binary'
)

valid_dir = 'Images'
valid_data_gen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_data_gen.flow_from_directory(
    valid_dir,
    target_size = (300, 300),
    batch_size = 32,
    class_mode = 'binary'
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
    input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    validation_data=valid_generator,
    validation_steps=8,
    verbose=2
)