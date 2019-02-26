Assuming we want to perform N sampling on N Gaussian distributions with mean z_mean and sigma z_log_sigma.
The problem is we don't know N in advance.


I tried different ways to do that (including keras.backend.int_shape(something)), but failed all the time. Later on I noticed using K.shape can make it work (K.shape).

The funny thing is that K.shape(something)[0] produce something like this:

    Tensor("lambda_1/strided_slice:0", shape=(), dtype=int32)

But it still works! 

Good to know.

Full code (reference: https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py)
  

    def sampling(args):
      z_mean, z_log_sigma = args
      epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                    mean=0., stddev=epsilon_std)
      return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    print(z)
