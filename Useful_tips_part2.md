1. **shared embedding in keras**

Reference: https://stackoverflow.com/questions/42122168/keras-how-to-construct-a-shared-embedding-layer-for-each-input-neuron


    from keras.layers import Input, Embedding

    first_input = Input(shape = (your_shape_tuple) )
    second_input = Input(shape = (your_shape_tuple) )
    ...

    embedding_layer = Embedding(embedding_size)

    first_input_encoded = embedding_layer(first_input)
    second_input_encoded = embedding_layer(second_input)
    ...

    Rest of the model....

