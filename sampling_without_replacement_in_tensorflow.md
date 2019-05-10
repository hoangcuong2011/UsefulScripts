    import tensorflow as tf

    def sample_without_replacement(logits, K):
      """
      Courtesy of https://github.com/tensorflow/tensorflow/issues/9260#issuecomment-437875125
      """
      z = -tf.log(-tf.log(tf.random_uniform(tf.shape(logits),0,1)))
      _, indices = tf.nn.top_k(logits + z, K)
      return indices

    number_of_flips = 10
    a = range(10)
    print(a)
    data = tf.constant(list(a), dtype=tf.int32)
    print(data)
    with tf.Session() as sess:
      data = tf.cast(data, dtype=tf.float32)
      taken = sample_without_replacement(tf.ones([10]), number_of_flips)
      print(taken)
      for i in range(10):
        a = sess.run(taken)
        print(a)
