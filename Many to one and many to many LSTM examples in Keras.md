I found this article quite good, and then I copy it here: https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras
So:

One-to-one: you could use a Dense layer as you are not processing sequences:
    model.add(Dense(output_size, input_shape=input_shape))
2. One-to-many: this option is not supported well as chaining models is not very easy in Keras, so the following version is the easiest one:

    model.add(RepeatVector(number_of_times, input_shape=input_shape))
    model.add(LSTM(output_size, return_sequences=True))
Many-to-one: actually, your code snippet is (almost) an example of this approach:
    model = Sequential()
    model.add(LSTM(1, input_shape=(timesteps, data_dim)))
Many-to-many: This is the easiest snippet when the length of the input and output matches the number of recurrent steps:
    model = Sequential()
    model.add(LSTM(1, input_shape=(timesteps, data_dim), return_sequences=True))
Many-to-many when number of steps differ from input/output length: this is freaky hard in Keras. There are no easy code snippets to code that.
EDIT: Ad 5

In one of my recent applications, we implemented something which might be similar to many-to-many from the 4th image. In case you want to have a network with the following architecture (when an input is longer than the output):

                                        O O O
                                        | | |
                                  O O O O O O
                                  | | | | | | 
                                  O O O O O O
You could achieve this in the following manner:

    model = Sequential()
    model.add(LSTM(1, input_shape=(timesteps, data_dim), return_sequences=True))
    model.add(Lambda(lambda x: x[:, -N:, :]
Where N is the number of last steps you want to cover (on image N = 3).

From this point getting to:

                                        O O O
                                        | | |
                                  O O O O O O
                                  | | | 
                                  O O O 
is as simple as artificial padding sequence of length N using e.g. with 0 vectors, in order to adjust it to an appropriate size.
