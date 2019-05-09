    random_tensor = keep_prob
    random_tensor += tf.random_uniform(tf.shape(input), seed=1, dtype=inputs.dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = tf.floor(random_tensor)
    ret = tf.divide(input, keep_prob) * binary_tensor 
    
    
reference can be found here: https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/nn_ops.py
