"""
https://apmonitor.com/pdc/index.php/Main/SolveDifferentialEquations
https://lazyprogrammer.me/neural-ordinary-differential-equations/

https://blog.acolyer.org/2019/01/09/neural-ordinary-differential-equations/
https://towardsdatascience.com/paper-summary-neural-ordinary-differential-equations-37c4e52df128
https://rkevingibson.github.io/blog/neural-networks-as-ordinary-differential-equations/
https://github.com/jason71995/Keras_ODENet/blob/master/mnist_odenet.py
https://www.reddit.com/r/MachineLearning/comments/a65v5r/neural_ordinary_differential_equations_pdf/
"""

import tensorflow as tf
import keras.backend as K
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D

def set_gpu_config(device = "0",fraction=0.25):
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = fraction
	config.gpu_options.visible_device_list = device
	K.set_session(tf.Session(config=config))

class ODEBlock(Model):
	def __init__(self, filters, kernel_size, **kwargs):
		self.filters = filters
		self.kernel_size = kernel_size
		super(ODEBlock, self).__init__(**kwargs)

	def build(self, input_shape):
		self.Conv2DLayer = Conv2D(self.filters, self.kernel_size, padding="same", activation="relu")
		super(ODEBlock, self).build(input_shape)

	def block(self, x, t):
		return self.Conv2DLayer(x)

	def call(self,x):
		t = K.variable([0,1.0],dtype="float32")
		return tf.contrib.integrate.odeint(self.block, x, t, rtol=1e-3, atol=1e-3)[1]
	def compute_output_shape(self, input_shape):
		print(input_shape)
		return (input_shape[0], input_shape[1], input_shape[2], input_shape[3])

def build_model(input_shape, num_classes):
	x = Input(input_shape)
	y = Conv2D(32, (3, 3), activation='relu')(x)
	y = MaxPooling2D((2,2))(y)
	y = Conv2D(64, (3, 3), activation='relu')(y)
	y = MaxPooling2D((2,2))(y)
	y = ODEBlock(64, (3, 3))(y)
	y = ODEBlock(64, (3, 3))(y)
	y = Flatten()(y)
	y = Dense(num_classes, activation='softmax')(y)
	return Model(x,y)


set_gpu_config("0",0.25)

batch_size = 128
num_classes = 10
epochs = 10
image_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = x_train.reshape((-1,) + image_shape)
x_test = x_test.reshape((-1,) + image_shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical(y_test, num_classes)


model = build_model(image_shape, num_classes)
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=keras.optimizers.Adam(),
			  metrics=['accuracy'])

model.fit(x_train, y_train,
		  batch_size=batch_size,
		  epochs=epochs,
		  verbose=1,
		  validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
