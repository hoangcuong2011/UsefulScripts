I found this code is useful to do KL annealing - reference: https://github.com/keras-team/keras/issues/2595


alpha = K.variable(1.0)
beta = K.variable(1.0)
gamma = K.variable(1.0)


optimizer = Adam()
model.compile(loss={'loss1': 'loss1', 'loss2': 'loss2', 'loss2': 'loss3'},
				  loss_weights=[alpha, beta, gamma],
				  optimizer=optimizer)



class MyCallback(keras.callbacks.Callback):
	def __init__(self, alpha, beta, gamma):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
	def on_epoch_begin(self, epoch, logs={}):
		K.set_value(self.alpha, K.get_value(self.alpha))
		K.set_value(self.beta, K.get_value(self.beta))
		K.set_value(self.gamma, K.get_value(self.gamma+epoch*10)) #random *10

model.fit(x, [y1, y2, y3], epochs=number_of_epoch,
			  verbose=2,
			  callbacks=[MyCallback(alpha, beta, gamma)])
