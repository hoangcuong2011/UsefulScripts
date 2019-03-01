import numpy as np
import keras
from keras import backend as K

model = keras.Sequential()
model.add(keras.layers.Dense(20, input_shape = (10, )))
model.add(keras.layers.Dense(5))
model.compile('adam')

dummy_in = np.ones((4, 10))
dummy_out = np.ones((4, 5))
dummy_loss = model.train_on_batch(dummy_in, dummy_out)

def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


def get_layer_output_grad(model, inputs, outputs, layer=-1):
    """ Gets gradient a layer output for given inputs and outputs"""
    grads = model.optimizer.get_gradients(model.total_loss, model.layers[layer].output)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


weight_grads = get_weight_grad(model, dummy_in, dummy_out)
output_grad = get_layer_output_grad(model, dummy_in, dummy_out)

print(weight_grads)
print(output_grad)
