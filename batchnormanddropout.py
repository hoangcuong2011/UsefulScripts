# peace of code related to batch norm and dropout in tensorflow

x = tf.placeholder("float", [None, 614])
y = tf.placeholder("float", [None, 1])
droprate = tf.placeholder("float")
train_phase = tf.placeholder_with_default(False, shape=()) 
def make_feedforward_nn(x):
    W1 = tf.get_variable("W1", shape=[614, 512], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", initializer=create_bias([512]))
    h1 = (tf.matmul(x, W1) + b1)
    h1 = tf.contrib.layers.batch_norm(h1, is_training=train_phase)
    h1 = tf.nn.relu(h1)
    h1 = tf.nn.dropout(h1, droprate)
    W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", initializer=create_bias([512]))
    #h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    h2 = (tf.matmul(h1, W2) + b2)
    h2 = tf.contrib.layers.batch_norm(h2, is_training=train_phase)
    h2 = tf.nn.relu(h2)

    h2 = tf.nn.dropout(h2, droprate)
    W3 = tf.get_variable("W3", shape=[512, 17], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", initializer=create_bias([17]))
    
    h3 = (tf.matmul(h2, W3) + b3)
    h3 = tf.contrib.layers.batch_norm(h3, is_training=train_phase)

    #h3 = tf.nn.relu(tf.matmul(h2, W3) + b3)
    h3 = tf.nn.relu(h3)
    h3 = tf.nn.dropout(h3, droprate)
    W4 = tf.get_variable("W4", shape=[17, 1], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", initializer=create_bias([1]))
    h4 = (tf.matmul(h3, W4) + b4)
    return h4
