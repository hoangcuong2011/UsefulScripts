1. **elementwise in tensorflow with eager execution**

note: tf.multiply(y, t) is equivalent to *

        from __future__ import absolute_import, division, print_function

        import tensorflow as tf

        tf.enable_eager_execution()


        z = tf.constant([[[0.11146947], [0.14126696], [0.18151596], [0.17806149], [0.19062787], [0.19705829]]])


        t = tf.constant([[[0.14725718, 0.24335186, 0.3550581], [0.23426636, 0.4415838,  0.5154056 ], [0.05640721, 0.6009872, 0.5530113 ], [0.11489896, 0.5269593,  0.7408159 ], [0.13170667, 0.64790857, 0.6709326 ], [0.3019519, 0.73433447, 0.80643225]]])


        t
        <tf.Tensor: id=13, shape=(1, 6, 3), dtype=float32, numpy=
        array([[[0.14725718, 0.24335186, 0.3550581 ],
                [0.23426636, 0.4415838 , 0.5154056 ],
                [0.05640721, 0.6009872 , 0.5530113 ],
                [0.11489896, 0.5269593 , 0.7408159 ],
                [0.13170667, 0.64790857, 0.6709326 ],
                [0.3019519 , 0.73433447, 0.80643225]]], dtype=float32)>

        z
        <tf.Tensor: id=7, shape=(1, 6, 1), dtype=float32, numpy=
        array([[[0.11146947],
                [0.14126696],
                [0.18151596],
                [0.17806149],
                [0.19062787],
                [0.19705829]]], dtype=float32)>


        z*t
        <tf.Tensor: id=72, shape=(1, 6, 3), dtype=float32, numpy=
        array([[[0.01641468, 0.0271263 , 0.03957814],
                [0.0330941 , 0.0623812 , 0.07280978],
                [0.01023881, 0.10908877, 0.10038038],
                [0.02045908, 0.09383116, 0.13191077],
                [0.02510696, 0.12350943, 0.12789845],
                [0.05950212, 0.1447067 , 0.15891416]]], dtype=float32)>

2. **tf.reduce_sum**

                z*t
                <tf.Tensor: id=72, shape=(1, 6, 3), dtype=float32, numpy=
                array([[[0.01641468, 0.0271263 , 0.03957814],
                        [0.0330941 , 0.0623812 , 0.07280978],
                        [0.01023881, 0.10908877, 0.10038038],
                        [0.02045908, 0.09383116, 0.13191077],
                        [0.02510696, 0.12350943, 0.12789845],
                        [0.05950212, 0.1447067 , 0.15891416]]], dtype=float32)>
        

                tf.reduce_sum(z*t, axis=0)
                <tf.Tensor: id=76, shape=(6, 3), dtype=float32, numpy=
                array([[0.01641468, 0.0271263 , 0.03957814],
                       [0.0330941 , 0.0623812 , 0.07280978],
                       [0.01023881, 0.10908877, 0.10038038],
                       [0.02045908, 0.09383116, 0.13191077],
                       [0.02510696, 0.12350943, 0.12789845],
                       [0.05950212, 0.1447067 , 0.15891416]], dtype=float32)>


                tf.reduce_sum(z*t, axis=1)
                <tf.Tensor: id=88, shape=(1, 3), dtype=float32, numpy=array([[0.16481575, 0.56064355, 0.63149166]], dtype=float32)>


                tf.reduce_sum(z*t, axis=2)
                <tf.Tensor: id=80, shape=(1, 6), dtype=float32, numpy=
                array([[0.08311912, 0.16828507, 0.21970797, 0.24620101, 0.27651483,
                        0.363123  ]], dtype=float32)>


3. **tf.expand_dim**

                # 't' is a tensor of shape [2]
                tf.shape(tf.expand_dims(t, 0))  # [1, 2]
                tf.shape(tf.expand_dims(t, 1))  # [2, 1]
                tf.shape(tf.expand_dims(t, -1))  # [2, 1]

                # 't2' is a tensor of shape [2, 3, 5]
                tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
                tf.shape(tf.expand_dims(t2, 2))  # [2, 3, 1, 5]
                tf.shape(tf.expand_dims(t2, 3))  # [2, 3, 5, 1]
