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

2. **created zeros like vector from unknown shape**


        print("inputs")
        print(inputs)
        # Tensor("embedding_1/embedding_lookup/Identity:0", shape=(?, 100, 50), dtype=float32)
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        # here samples means ?
        
        
3. **set initial cell states for LSTM (which can be learnable))**

Something like this:

        shared_LSTM = LSTM(word_embedding_size, return_sequences=True)
        x_left = shared_LSTM(x_left, initial_state=[vector1, vector2])
