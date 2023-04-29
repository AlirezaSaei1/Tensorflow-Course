import tensorflow
import keras

# Simple NN with 1 neuron
model = keras.Sequential([
    keras.layers.Dense(units=1 ,input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit([-1, 0, 1, 2, 3, 4, 5], [-3, -1, 1, 3, 5, 7, 9], epochs=50)

print(model.predict([10]))