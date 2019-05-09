For a shared layer style in two neural networks, Dropout can be broken. We need to sync droping units for two networks.

This code can help you figure out the problem clearly:


    from keras.preprocessing.text import Tokenizer
    from keras.utils import to_categorical
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import LSTM, GRU
    from keras.layers import Embedding
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping
    from keras.layers import concatenate
    import os
    import tensorflow as tf
    import random
    from sklearn.metrics import accuracy_score
    from keras.engine import Layer
    from keras import backend as K
    import numpy as np
    import nltk
    import os
    import codecs
    import json
    import pickle
    import sys
    max_length_for_paddle = 100
    EmbeddingSize = 50
    from keras import optimizers
    my_batch_size = 256
    from keras.layers.core import Activation
    from keras.layers.wrappers import TimeDistributed
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras import backend as K
    import keras
    import warnings
    import numpy as np
    import tensorflow as tf

    import keras
    from keras.layers import Lambda
    from keras.layers.recurrent import Recurrent, GRU, LSTM
    from keras.layers.core import Dense
    from keras.initializers import RandomNormal, Orthogonal, Zeros, Constant
    from keras import backend as K
    from keras.engine.topology import InputSpec
    from keras.activations import get as get_activations
    from keras.activations import softmax, tanh, sigmoid, hard_sigmoid
    from keras.layers import Dropout


    from keras.layers import Multiply, RepeatVector
    from keras.layers.core import Reshape


    class L1Distance(Layer):
      def __init__(self, **kwargs):
        super(L1Distance, self).__init__(**kwargs)

      def build(self, input_shape):
        super(L1Distance, self).build(input_shape)

      def call(self, inputs):
        if len(inputs) != 2:
          raise ValueError('A `Subtract` layer should be called '
                   'on exactly 2 inputs')
        return K.sum(K.abs(inputs[0] - inputs[1]))

      def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


    def NeuralNetwork():
      x_input_left = Input(shape=(4,), name='x_input_left')
      x_input_right = Input(shape=(4,), name='x_input_right')
      shared_dropout = Dropout(0.0)
      x_left = shared_dropout(x_input_left)
      x_right = shared_dropout(x_input_left)
      y = L1Distance()([x_left, x_right])
      return Model(inputs=[x_input_left,
                 x_input_right], outputs=y)


    if __name__=="__main__":
      model = NeuralNetwork()
      print(model.summary())
      adam = optimizers.Adam(lr=0.001, clipvalue=0.5)
      model.compile(loss='mse', optimizer=adam)
      list = [1, 2, 3, 4]
      list_verticle = []
      list_verticle.append(np.asarray(list))
      list_verticle.append(np.asarray(list))
      list_verticle.append(np.asarray(list))
      list_verticle.append(np.asarray(list))
      list_verticle.append(np.asarray(list))

      x0 = np.asarray(list_verticle)
      x1 = np.asarray(list_verticle)
      x = [x0, x1]
      y = np.array([0, 0, 0, 0, 0])
      print(x)

      model.fit({'x_input_left': x0, 'x_input_right': x1}, y, verbose=1, epochs=10, batch_size=2)

