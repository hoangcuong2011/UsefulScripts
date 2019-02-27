add custom loss is something very complicated.
It is trivial if your custom loss is as what it is a function with two inputs: ground_truth and prediction output. Nonetheless, sometimes we need a rather completely different kind of losses.

I found a good way to do that is uses a custom layers. We declare a loss function with whatever inputs you consider for your own loss. Then we can call it from the call function:

    def call(self ,x ,mask=None):
      loss_value = your loss function (whatever inputs)
      self.add_loss(your loss_function, x)

The only thing I don't like is: we need to return a dumpy output for that layer. To besides, we need to assign a dummpy loss value for the network.


Some examples are as follows:


First example can be found here: https://github.com/keras-team/keras/issues/5563


    import numpy as np

    from keras.models import Model
    from keras.layers import Input

    import keras.backend as K
    from keras.engine.topology import Layer
    from keras.layers.core import  Dense

    from keras import objectives

    def zero_loss(y_true, y_pred):
        return K.zeros_like(y_pred)


    class CustomRegularization(Layer):
        def __init__(self, **kwargs):
            super(CustomRegularization, self).__init__(**kwargs)

        def call(self ,x ,mask=None):
            ld=x[0]
            rd=x[1]
            bce = objectives.binary_crossentropy(ld, rd)
            loss2 = K.sum(bce)
            self.add_loss(loss2,x)
            #you can output whatever you need, just update output_shape adequately
            #But this is probably useful
            return bce

        def get_output_shape_for(self, input_shape):
            return (input_shape[0][0],1)

    input_size= 100
    output_dim = 1
    x1 = Input(shape=(input_size,))
    ld = Dense(128, activation='relu')(x1)
    out1 = Dense(output_dim, activation='sigmoid')(ld)

    x2 = Input(shape=(input_size,))
    rd = Dense(128, activation='relu')(x2)
    out2 = Dense(output_dim, activation='sigmoid')(rd)

    cr = CustomRegularization()([ld,rd])

    m = Model( [x1,x2], [out1,out2,cr])
    m.compile( loss=[K.binary_crossentropy,K.binary_crossentropy,zero_loss], optimizer="adam")

    nb_examples = 32
    print m.predict( [np.random.randn(nb_examples,input_size),np.random.randn(nb_examples,input_size)] )
    print m.fit(  [np.random.randn(nb_examples,input_size),np.random.randn(nb_examples,input_size)], [np.random.randn(nb_examples,output_dim),np.random.randn(nb_examples,output_dim), np.random.randn(nb_examples,1) ]  )



Another example (from here https://github.com/hoangcuong2011/text_VAE/blob/master/vae_text_s2sloss_kl_weight_V18_github.py)


    # -*- coding: utf-8 -*-
    """
    Created on Sun Dec 17 14:59:07 2017

    @author: Giancarlo
    """


    from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers.advanced_activations import ELU
    from keras.preprocessing.text import Tokenizer
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam
    from keras import backend as K
    from keras.models import Model
    from scipy import spatial
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import codecs
    import csv
    import os



    BASE_DIR = 'C:/Users/gianc/Desktop/PhD/Progetti/vae/'
    TRAIN_DATA_FILE = BASE_DIR + 'train.csv'#'train_micro.csv'
    GLOVE_EMBEDDING = BASE_DIR + 'glove.6B.300d.txt'
    VALIDATION_SPLIT = 0.2
    MAX_SEQUENCE_LENGTH = 25
    MAX_NB_WORDS = 20000
    EMBEDDING_DIM = 300



    texts = [] 
    with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        for values in reader:
            if len(values[3].split()) <= MAX_SEQUENCE_LENGTH:
                texts.append(values[3])
            if len(values[4].split()) <= MAX_SEQUENCE_LENGTH:
                texts.append(values[4])
    print('Found %s texts in train.csv' % len(texts))
    n_sents = len(texts)


    #======================== Tokenize and pad texts lists ===================#
    tokenizer = Tokenizer(MAX_NB_WORDS+1, oov_token='unk') #+1 for 'unk' token
    tokenizer.fit_on_texts(texts)
    print('Found %s unique tokens' % len(tokenizer.word_index))
    ## **Key Step** to make it work correctly otherwise drops OOV tokens anyway!
    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_NB_WORDS} # <= because tokenizer is 1 indexed
    tokenizer.word_index[tokenizer.oov_token] = MAX_NB_WORDS + 1
    word_index = tokenizer.word_index #the dict values start from 1 so this is fine with zeropadding
    index2word = {v: k for k, v in word_index.items()}
    sequences = tokenizer.texts_to_sequences(texts)
    data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', data_1.shape)
    NB_WORDS = (min(tokenizer.num_words, len(word_index))+1) #+1 for zero padding 



    #==================== sample train/validation data =====================#
    data_val = data_1[775000:783000]
    data_train = data_1[:775000]

    #def sent_generator(TRAIN_DATA_FILE, chunksize):
    #    reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)
    #    for df in reader:
    #        #print(df.shape)
    #        #df=pd.read_csv(TRAIN_DATA_FILE, iterator=False)
    #        val3 = df.iloc[:,3:4].values.tolist()
    #        val4 = df.iloc[:,4:5].values.tolist()
    #        flat3 = [item for sublist in val3 for item in sublist]
    #        flat4 = [str(item) for sublist in val4 for item in sublist]
    #        texts = [] 
    #        texts.extend(flat3[:])
    #        texts.extend(flat4[:])
    #        
    #        sequences = tokenizer.texts_to_sequences(texts)
    #        data_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    #        yield [data_train, data_train]





    #======================== prepare GLOVE embeddings =============================#
    embeddings_index = {}
    f = open(GLOVE_EMBEDDING, encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    glove_embedding_matrix = np.zeros((NB_WORDS, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i < NB_WORDS+1: #+1 for 'unk' oov token
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                glove_embedding_matrix[i] = embedding_vector
            else:
                # words not found in embedding index will the word embedding of unk
                glove_embedding_matrix[i] = embeddings_index.get('unk')
    print('Null word embeddings: %d' % np.sum(np.sum(glove_embedding_matrix, axis=1) == 0))



    #====================== VAE model ============================================#
    batch_size = 100
    max_len = MAX_SEQUENCE_LENGTH
    emb_dim = EMBEDDING_DIM
    latent_dim = 64
    intermediate_dim = 256
    epsilon_std = 1.0
    kl_weight = 0.01
    num_sampled=500
    act = ELU()


    x = Input(shape=(max_len,))
    x_embed = Embedding(NB_WORDS, emb_dim, weights=[glove_embedding_matrix],
                                input_length=max_len, trainable=False)(x)
    h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
    #h = Bidirectional(LSTM(intermediate_dim, return_sequences=False), merge_mode='concat')(h)
    #h = Dropout(0.2)(h)
    #h = Dense(intermediate_dim, activation='linear')(h)
    #h = act(h)
    #h = Dropout(0.2)(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    # we instantiate these layers separately so as to reuse them later
    repeated_context = RepeatVector(max_len)
    decoder_h = LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
    decoder_mean = Dense(NB_WORDS, activation='linear')#softmax is applied in the seq2seqloss by tf #TimeDistributed()
    h_decoded = decoder_h(repeated_context(z))
    x_decoded_mean = decoder_mean(h_decoded)


    # placeholder loss
    def zero_loss(y_true, y_pred):
        return K.zeros_like(y_pred)

    #Sampled softmax
    #logits = tf.constant(np.random.randn(batch_size, max_len, NB_WORDS), tf.float32)
    #targets = tf.constant(np.random.randint(NB_WORDS, size=(batch_size, max_len)), tf.int32)
    #proj_w = tf.constant(np.random.randn(NB_WORDS, NB_WORDS), tf.float32)
    #proj_b = tf.constant(np.zeros(NB_WORDS), tf.float32)
    #
    #def _sampled_loss(labels, logits):
    #    labels = tf.cast(labels, tf.int64)
    #    labels = tf.reshape(labels, [-1, 1])
    #    logits = tf.cast(logits, tf.float32)
    #    return tf.cast(
    #                    tf.nn.sampled_softmax_loss(
    #                        proj_w,
    #                        proj_b,
    #                        labels,
    #                        logits,
    #                        num_sampled=num_sampled,
    #                        num_classes=NB_WORDS),
    #                    tf.float32)
    #softmax_loss_f = _sampled_loss


    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)
            self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)

        def vae_loss(self, x, x_decoded_mean):
            #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
            labels = tf.cast(x, tf.int32)
            xent_loss = K.sum(tf.contrib.seq2seq.sequence_loss(x_decoded_mean, labels, 
                                                         weights=self.target_weights,
                                                         average_across_timesteps=False,
                                                         average_across_batch=False), axis=-1)#,
                                                         #softmax_loss_function=softmax_loss_f), axis=-1)#,
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            xent_loss = K.mean(xent_loss)
            kl_loss = K.mean(kl_loss)
            return K.mean(xent_loss + kl_weight * kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            print(x.shape, x_decoded_mean.shape)
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # we don't use this output, but it has to have the correct shape:
            return K.ones_like(x)

    def kl_loss(x, x_decoded_mean):
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss = kl_weight * kl_loss
        return kl_loss

    loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, [loss_layer])
    opt = Adam(lr=0.01) 
    vae.compile(optimizer='adam', loss=[zero_loss], metrics=[kl_loss])
    vae.summary()


    #======================= Model training ==============================#
    def create_model_checkpoint(dir, model_name):
        filepath = dir + '/' + model_name + ".h5" 
        directory = os.path.dirname(filepath)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
        return checkpointer

    checkpointer = create_model_checkpoint('models', 'vae_seq2seq_test_very_high_std')



    vae.fit(data_train, data_train,
         shuffle=True,
         epochs=100,
         batch_size=batch_size,
         validation_data=(data_val, data_val), callbacks=[checkpointer])

    print(K.eval(vae.optimizer.lr))
    K.set_value(vae.optimizer.lr, 0.01)

    #batch_size=512
    #nb_epoch=100
    #n_steps = 400000/batch_size#404000/batch_size
    #for counter in range(nb_epoch):
    #    print('-------epoch: ',counter,'--------')
    #    vae.fit_generator(sent_generator(TRAIN_DATA_FILE, batch_size/2),
    #                          steps_per_epoch=n_steps, epochs=1, callbacks=[checkpointer],
    #                          validation_data=(data_1_val, data_1_val))


    vae.save('models/vae_lstm.h5')
    #vae.load_weights('models/vae_seq2seq_test.h5')
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    #encoder.save('models/encoder32dim512hid30kvocab_loss29_val34.h5')

    # build a generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(repeated_context(decoder_input))
    _x_decoded_mean = decoder_mean(_h_decoded)
    _x_decoded_mean = Activation('softmax')(_x_decoded_mean)
    generator = Model(decoder_input, _x_decoded_mean)


    index2word = {v: k for k, v in word_index.items()}
    index2word[0] = 'pad'

    #test on a validation sentence
    sent_idx = 100
    sent_encoded = encoder.predict(data_val[sent_idx:sent_idx+2,:])
    x_test_reconstructed = generator.predict(sent_encoded, batch_size = 1)
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, x_test_reconstructed[0])
    np.apply_along_axis(np.max, 1, x_test_reconstructed[0])
    np.max(np.apply_along_axis(np.max, 1, x_test_reconstructed[0]))
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    print(' '.join(word_list))
    original_sent = list(np.vectorize(index2word.get)(data_val[sent_idx]))
    print(' '.join(original_sent))



    #=================== Sentence processing and interpolation ======================#
    # function to parse a sentence
    def sent_parse(sentence, mat_shape):
        sequence = tokenizer.texts_to_sequences(sentence)
        padded_sent = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
        return padded_sent#[padded_sent, sent_one_hot]


    # input: encoded sentence vector
    # output: encoded sentence vector in dataset with highest cosine similarity
    def find_similar_encoding(sent_vect):
        all_cosine = []
        for sent in sent_encoded:
            result = 1 - spatial.distance.cosine(sent_vect, sent)
            all_cosine.append(result)
        data_array = np.array(all_cosine)
        maximum = data_array.argsort()[-3:][::-1][1]
        new_vec = sent_encoded[maximum]
        return new_vec


    # input: two points, integer n
    # output: n equidistant points on the line between the input points (inclusive)
    def shortest_homology(point_one, point_two, num):
        dist_vec = point_two - point_one
        sample = np.linspace(0, 1, num, endpoint = True)
        hom_sample = []
        for s in sample:
            hom_sample.append(point_one + s * dist_vec)
        return hom_sample



    # input: original dimension sentence vector
    # output: sentence text
    def print_latent_sentence(sent_vect):
        sent_vect = np.reshape(sent_vect,[1,latent_dim])
        sent_reconstructed = generator.predict(sent_vect)
        sent_reconstructed = np.reshape(sent_reconstructed,[max_len,NB_WORDS])
        reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
        word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
        w_list = [w for w in word_list if w not in ['pad']]
        print(' '.join(w_list))
        #print(word_list)



    def new_sents_interp(sent1, sent2, n):
        tok_sent1 = sent_parse(sent1, [27])
        tok_sent2 = sent_parse(sent2, [27])
        enc_sent1 = encoder.predict(tok_sent1, batch_size = 16)
        enc_sent2 = encoder.predict(tok_sent2, batch_size = 16)
        test_hom = shortest_homology(enc_sent1, enc_sent2, n)
        for point in test_hom:
            print_latent_sentence(point)



    #====================== Example ====================================#
    sentence1=['gogogo where can i find a bad restaurant endend']
    mysent = sent_parse(sentence1, [27])
    mysent_encoded = encoder.predict(mysent, batch_size = 16)
    print_latent_sentence(mysent_encoded)
    print_latent_sentence(find_similar_encoding(mysent_encoded))

    sentence2=['gogogo where can i find an extremely good restaurant endend']
    mysent2 = sent_parse(sentence2, [27])
    mysent_encoded2 = encoder.predict(mysent2, batch_size = 16)
    print_latent_sentence(mysent_encoded2)
    print_latent_sentence(find_similar_encoding(mysent_encoded2))
    print('-----------------')

    new_sents_interp(sentence1, sentence2, 5)


