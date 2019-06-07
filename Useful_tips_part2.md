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
        
        
4. **get shape for unknown shape tensor**

        tf.keras.backend.int_shape

I found this function provides exactly the same goal as tf.shape().

5. **Learn to specify the shape of input in a custom layer**
reference: https://github.com/keras-team/keras/blob/master/keras/layers/normalization.py#L16
        self.input_spec = InputSpec(ndim=len(input_shape),
                                            axes={self.axis: dim})
                                 
                                 
 6. **f1 score from sklearn**

         from sklearn.metrics import f1_score
         f1_score(np.array([1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]), average='binary')  

7. **compute roc and auc in sklearn**

        print(__doc__)

        import numpy as np
        import matplotlib.pyplot as plt
        from itertools import cycle

        from sklearn import svm, datasets
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import label_binarize
        from sklearn.multiclass import OneVsRestClassifier
        from scipy import interp

        # Import some data to play with
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        # Binarize the output
        y = label_binarize(y, classes=[0, 1, 2])
        n_classes = y.shape[1]

        # Add noisy features to make the problem harder
        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape
        X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        #print(y_score)
        print(type(y_score))
        print(y_train)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


8. **created a python package to distribute it**
reference for this: https://packaging.python.org/tutorials/packaging-projects/
command: 
        
        python setup.py sdist bdist_wheel

setup.py will be something as:

        import setuptools

        with open("README.md", "r") as fh:
            long_description = fh.read()

        setuptools.setup(
            name="example-pkg-your-username",
            version="0.0.1",
            author="Example Author",
            author_email="author@example.com",
            description="A small example package",
            long_description=long_description,
            long_description_content_type="text/markdown",
            url="https://github.com/pypa/sampleproject",
            packages=setuptools.find_packages(),
            classifiers=[
                "Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
            ],
        )

reference for distributing the package within organization: https://blog.chezo.uno/simple-way-to-distribute-your-private-python-packages-within-your-organization-fb7af5dbd4c9


9. **repeat vector with tensorflow**
Let us assume we have an input with size: (batch_size, 100). And we want to create a new vector: (batch_size, 100, 50), where 50 extra dimension is repeated from each of the 100 elements. How can we do that in tensorflow?
Here is the way

        a = tf.expand_dims(input, -1)
		b = tf.keras.backend.repeat_elements(a, 50, -1)
		print(b)
        
        
        
