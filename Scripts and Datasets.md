1. **make with the number of cores**

        make all

        it can be sped up by appending -j $(($(nproc) + 1))

2. **install a new package in R**


        install.packages("h2o") (package name: h20)


3. **copy files line by line under linux**

        paste -d" " file1.txt file2.txt
output looks like this:

    /etc/port1-192.9.200.1-255.555.255.0 /etc/port1-192.90.2.1-255.555.0.0

    /etc/port2-192.9.200.1-255.555.255.0 /etc/port2-192.90.2.1-255.555.0.0

    /etc/port3-192.9.200.1-255.555.255.0 /etc/port3-192.90.2.1-255.555.0.0

    /etc/port4-192.9.200.1-255.555.255.0 /etc/port4-192.90.2.1-255.555.0.0

    /etc/port5-192.9.200.1-255.555.255.0 /etc/port5-192.90.2.1-255.555.0.0

4. **python code for loading housing datasets (kin40k)**


        dataset = np.loadtxt("kin40ktraindata.txt", delimiter=",")
    
        
        X_train = dataset[:,0:8]
    
        y_train = dataset[:,8]

        dataset = np.loadtxt("kin40ktestdata.txt", delimiter=",")
    
    
        X_test = dataset[:,0:8]
    
        y_test = dataset[:,8]
    
        X_valid, y_valid = X_test, y_test
    
 5. **find all public comments for a Github user**
 
 We can do this with an advanced search query. In the Issues search box, enter "commenter:username".

6. **Proprietary GPU Drivers**

        sudo add-apt-repository ppa:graphics-drivers/ppa
        
        sudo apt-get update
        
        sudo apt-get install nvidia-384

7. **Connecting to the Internet in the recovery mode**

        vi /etc/resolve.conf
        
        insert nameserver 8.8.8.8
        
        /etc/init.d/networking restart

8. **Install pip with a specific version**

I take an example with Keras

        pip install --upgrade protobuf==2.0

9. **Running Keras-GP**

Keras-GP is a great project. It however may run with Keras 2.0 only. Upgrading Keras would screw up everything!

10. **Enable 64bit with Octave**

Octave is a great software. The only thing I want to make complaint is about to get rid of the error:

        **memory exhausted or requested size too large for range of Octave’s index type**
        
(see this: http://calaba.tumblr.com/post/107087607479/octave-64)

Basically, what we have to do is to  switch –enable-64 when compiling from the source. It is however is not easy at all to do so. The link I found pretty helpful is the following: https://github.com/siko1056/GNU-Octave-enable-64. Mark it, and I believe one day you might need it.


Note: to follow the steps described in the link above, you should install the several packages:

        sudo apt-get lzip

        sudo apt-get install libtool-bin
        
        sudo apt-get install autoconf

To validate once you finish building the software, you can enter the following command: 

        a = zeros (1024 * 1024 * 1024 * 3, 1, 'int8'); 

The default version should give the following error

        memory exhausted or requested size too large for range of Octave's index type -- trying to return to prompt

Meanwhile, your version should work fine for this command.

11. **Check Ubuntu version**

        lsb_release -a

12. **TexLive**
Texlive and related packages

        sudo apt-get install texlive
        
        sudo apt-get install texlive-latex-extra
        
        sudo apt-get install texlive-lang-all

13. **Cut and Paste in VIM**

        press dd to cut a line of text, and then press p

14. **Time measure in Python**

        import time

        start = time.time()
        
        print("hello")
        
        end = time.time()
        
        print(end - start)


15. **kmeans in python**

       
        from scipy.cluster.vq import kmeans2
        
        Z_100 = kmeans2(np.array(X), 100, minit='points')[0]
        
        
16. **List all environments in conda**

        conda info --envs

17. **Clone conda enviroments**

        conda create --name py27_environment_tensorflow_gpus --clone py27_environment
        
18. **A normal script file to upload to a cluster**

        #!/bin/sh
        #PBS -q interactive
        #PBS -N my_first_trial
        #PBS -l select=1:ncpus=1
        #PBS -l place=free
        #PBS -V
        cd $PBS_O_WORKDIR

        sh ./script.sh
        
Note: production is the normal queue for processing your work. development is used when you are testing an application. Jobs submitted to this queue can not request more than 8 cores or use more than 1 hour of total CPU time. If the job exceeds these parameters, it will be automatically killed. Development queue has higher priority and thus jobs in this queue have shorter wait time. interactive is used for quick interactive tests. Jobs submitted into this queue allow you interactive terminal session on one of the compute nodes. They can not use more than 4 cores or use more than a total of 15 minutes of compute time.




19. **Check queue**

        qstat -u cuong.hoang
        
        qstat -f | grep -C 3 cuong.hoang
        

20. **Good conda create new enviroment command**

        conda create -n py27_env_tensor_gpu_pip_local numpy pip tensorflow-gpu python=2.7
        
21. **Find command without Permission denied**

        find / -name "octave" 2>/dev/null

22. **Install matlab.engine**

        cd $(dirname $(realpath $(which matlab)))/../extern/engines/python

        python setup.py build --build-base=$(mktemp -d) install

23. **How to fix: “UnicodeDecodeError: 'ascii' codec can't decode byte”**

https://stackoverflow.com/questions/21129020/how-to-fix-unicodedecodeerror-ascii-codec-cant-decode-byte see here
        import sys  

        reload(sys)  
        
        sys.setdefaultencoding('utf8')

24. **Replacescript python**


        filename = 'de_en_task3_train_features'
        # Read in the file
        with open(filename, 'r') as file :
        filedata = file.read()

        # Replace the target string
        filedata = filedata.replace('\t', ',')

        # Write the file out again
        with open(filename+'.replaced', 'w') as file:
        file.write(filedata)

25. **Shuffle two arrays in Numpy**

Reference: https://play.pixelblaster.ro/blog/2017/01/20/how-to-shuffle-two-arrays-to-the-same-order/

                >>> import numpy as np
                >>> x = np.arange(10)
                >>> y = np.arange(9, -1, -1)
                >>> x
                array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
                >>> y
                array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
                >>> s = np.arange(x.shape[0])
                >>> np.random.shuffle(s)
                >>> s
                array([9, 3, 5, 2, 6, 0, 8, 1, 4, 7])
                >>> x[s]
                array([9, 3, 5, 2, 6, 0, 8, 1, 4, 7])
                >>> y[s]
                array([0, 6, 4, 7, 3, 9, 1, 8, 5, 2])
                
26. **Shuffle data and train in minibach - tensorflow**
Reference: https://github.com/hoangcuong2011/DeepKernelLearning/blob/master/Classification_MLP_with_correct_Epoch_implementation.py

                for i in range(100): #100 epochs
                        shuffle = np.arange(y_train.size)
                        np.random.shuffle(shuffle)
                        print(shuffle)
                        x_train_shuffle = x_train[shuffle]
                        y_train_shuffle = y_train[shuffle]
                        data_indx = 0
                        while data_indx<y_train.size:
                                lastIndex = data_indx + minibatch_size
                                if lastIndex>=y_train.size:
                                        lastIndex = y_train.size
                                indx_array = np.mod(np.arange(data_indx, lastIndex), x_train_shuffle.shape[0])
                                #print("array", indx_array)
                                data_indx += minibatch_size
                                #print(data_indx)
                                fd = gp_model.feeds or {}
                                fd.update({
                                phs.keep_prob: 1.0,
                                phs.ximage_flat: x_train_shuffle[indx_array],
                                phs.label: y_train_shuffle[indx_array]
                                })
                                _, loss_evd = tf_session.run([minimise, -gp_model.objective], feed_dict=fd)            

27. **Convert number to one-hot vector representation**

        from sklearn.preprocessing import LabelEncoder
	from keras.utils import np_utils
        encoder = LabelEncoder()
        encoder.fit(y_test)
        encoded_Y = encoder.transform(y_test)
        # convert integers to dummy variables (i.e. one hot encoded)
        y_test = np_utils.to_categorical(encoded_Y)

28. **Read tensorflow trainable variables' value**

        variables_names =[v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k,v in zip(variables_names, values):
                print(k, v)
    
29. **tokenize with NLTK**

        >>> import nltk
        >>> sentence = """At eight o'clock on Thursday morning
        ... Arthur didn't feel very good."""
        >>> tokens = nltk.word_tokenize(sentence)
        >>> tokens
        ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning',
        'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']

30. **shuffle data in linux**

https://shapeshed.com/unix-shuf/

shuf cards.txt
4D
9D
QC
3S
6D


31. **readfile python**
        with open("feature.py") as f:	
	        for line in f:
		        line = line.strip()
		        if(len(line)>0):
			        print(line)

32. **How to turn off dropout for testing in Tensorflow?**
https://stackoverflow.com/questions/44971349/how-to-turn-off-dropout-for-testing-in-tensorflow


The easiest way is to change the keep_prob parameter using a placeholder_with_default:

	prob = tf.placeholder_with_default(1.0, shape=())
	layer = tf.nn.dropout(layer, prob)
in this way when you train you can set the parameter like this:

	sess.run(train_step, feed_dict={prob: 0.5})
and when you evaluate the default value of 1.0 is used.


33. **Read and convert words to word count**

	def read_data(raw_text):
		content = raw_text
		content = content.split() #splits the text by spaces (default split character)
		content = np.array(content)
		#print(content)
		content = np.reshape(content, [-1, ])
		#print(content)
		return content

	training_data = read_data(fable_text)

	def build_dictionaries(words):
	    count = collections.Counter(words).most_common() #creates list of word/count pairs;
	    print(count)
	    dictionary = dict()
	    for word, _ in count:
		dictionary[word] = len(dictionary) #len(dictionary) increases each iteration
		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	    return dictionary, reverse_dictionary

	dictionary, reverse_dictionary = build_dictionaries(training_data)

34. **updatedb in MACOS**

sudo /usr/libexec/locate.updatedb


35. **pom.xml - maven**

<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/maven-v4_0_0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>Agolo</groupId>
    <artifactId>hello-world-maven</artifactId>
    <packaging>jar</packaging>
    <version>0.1.0</version>

    <properties>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
    </properties>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-shade-plugin</artifactId>
                <version>2.1</version>
                <executions>
                    <execution>
                        <phase>package</phase>
                        <goals>
                            <goal>shade</goal>
                        </goals>
                        <configuration>
                            <transformers>
                                <transformer
                                    implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                                    <mainClass>hello.HelloWorld</mainClass>
                                </transformer>
                            </transformers>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>

https://spring.io/guides/gs/maven/



36. **show line number in vim**

add this line to ~/.vimrc (if not exist, create a new file)

:set nu
and save the file


37. **gradle example**

apply plugin: 'java'
apply plugin: 'eclipse'
apply plugin: 'application'

mainClassName = 'hello.HelloWorld'

// tag::repositories[]
repositories {
    mavenCentral()
}
// end::repositories[]

// tag::jar[]
jar {
    baseName = 'gs-gradle'
    version =  '0.1.0'
}
// end::jar[]

// tag::dependencies[]
sourceCompatibility = 1.8
targetCompatibility = 1.8

dependencies {
    compile "joda-time:joda-time:2.2"
    testCompile "junit:junit:4.12"
}
// end::dependencies[]

// tag::wrapper[]
// end::wrapper[]


38. **Common Jsoup commands**


        
        String html = "<p>An <a href='http://example.com/'><b>example</b></a> link.</p>";
        doc = Jsoup.parse(html);
        Element linkE = doc.select("a").first();

        String text = doc.body().text(); // "An example link"
        String linkHref = linkE.attr("href"); // "http://example.com/"
        String linkText = linkE.text(); // "example""

        String linkOuterH = linkE.outerHtml();
        // "<a href="http://example.com"><b>example</b></a>"
        String linkInnerH = linkE.html(); // "<b>example</b>"
        
        System.out.println(text);
	
	
39. **Shortcut in MacOS**

On OS X apps I can switch between apps using ⌘+Tab.


40. **Copy a file to clipboard in MacOS**

	pbcopy < ~/.ssh/id_rsa.pub

41. **Change an element in numpy array python**

	import numpy as np

	x = np.array([[ 0.42436315, 0.48558583, 0.32924763], [ 0.7439979,0.58220701,0.38213418], [ 0.5097581,0.34528799,0.1563123 ]])

	print("Original array:")

	print(x)

	print("Replace all elements of the said array with .5 which are greater than .5")

	x[x > .5] = .5

	print(x)


42. **Weighting loss function in keras**

You could simply implement the class_weight from sklearn:

		Let's import the module first

		from sklearn.utils import class_weight
In order to calculate the class weight do the following

		class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
Thirdly and lastly add it to the model fitting

		model.fit(X_train, y_train, class_weight=class_weights)
		
Attention: I edited this post and changed the variable name from class_weight to class_weights in order to not to overwrite the imported module. Adjust accordingly when copying code from the comments.

Actually, this code is not correct.

Here is a correct code

		model.fit(padded_x_train, y_train, epochs=30, verbose=0,
          class_weight={0: 1, 1: 5}, validation_data=(padded_x_dev, y_dev))
	  
43. **Get DBpedia dataset using tensorflow**

		dbpedia = tf.contrib.learn.datasets.load_dataset(
      			'dbpedia', size='large')
  		x_train = pandas.Series(dbpedia.train.data[:, 1])
  		y_train = pandas.Series(dbpedia.train.target)
  		x_test = pandas.Series(dbpedia.test.data[:, 1])
  		y_test = pandas.Series(dbpedia.test.target)
  
  It is quite weird that size has two options: 'small' and 'large'
  The dataset (dbpedia) can be downloaded here: https://drive.google.com/file/d/1GgyCU86oxhi9E1P_z7rGnbjLDI1cjCFr/view?usp=sharing
  
  The DBpedia datasets are licensed under the terms of the Creative Commons Attribution-ShareAlike License and the GNU Free Documentation License. For more information, please refer to http://dbpedia.org. For a recent overview paper about DBpedia, please refer to: Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo N. Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick van Kleef, Sören Auer, Christian Bizer: DBpedia – A Large-scale, Multilingual Knowledge Base Extracted from Wikipedia. Semantic Web Journal, Vol. 6 No. 2, pp 167–195, 2015.
  
  The DBPedia ontology classification dataset is constructed by Xiang Zhang (xiang.zhang@nyu.edu), licensed under the terms of the Creative Commons Attribution-ShareAlike License and the GNU Free Documentation License. This dataset was first used as a classification benchmark in the following technical report: Xiang Zhang, Yann LeCun, Text Understanding from Scratch, Arxiv 1502.01710.


44. **Get maximum length for different strings in an array.**

	x_text = ['This is a cat','This must be boy', 'This is a a dog']
	max_document_length = max([len(x.split(" ")) for x in x_text])

Get maximum length for different strings in an array.

45. **Padding with CNN**

If you like ascii art:

"VALID" = without padding:

   inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
                  |________________|                dropped
                                 |_________________|
"SAME" = with zero padding:

               pad|                                      |pad
   inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
               |________________|
                              |_________________|
                                             |________________|
In this example:

Input width = 13
Filter width = 6
Stride = 5
Notes:

"VALID" only ever drops the right-most columns (or bottom-most rows).
"SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right, as is the case in this example (the same logic applies vertically: there may be an extra row of zeros at the bottom).


46 **Add regularizers in Neural Network**

		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
		    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
		    tf_valid_dataset = tf.constant(valid_dataset)
		    tf_test_dataset = tf.constant(test_dataset)

		    # Variables.
		    weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]))
		    biases_1 = tf.Variable(tf.zeros([num_nodes]))
		    weights_2 = tf.Variable(tf.truncated_normal([num_nodes, num_labels]))
		    biases_2 = tf.Variable(tf.zeros([num_labels]))

		    # Training computation.
		    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
		    relu_layer= tf.nn.relu(logits_1)
		    logits_2 = tf.matmul(relu_layer, weights_2) + biases_2
		    # Normal loss function
		    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_2, tf_train_labels))
		    # Loss function with L2 Regularization with beta=0.01
		    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2)
		    loss = tf.reduce_mean(loss + beta * regularizers)

		    # Optimizer.
		    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

		    # Predictions for the training
		    train_prediction = tf.nn.softmax(logits_2)


47. **Stacking multiple LSTM using keras**

		model = Sequential()
		model.add(LSTM(512, return_sequences=True, input_shape=(10,13))
		model.add(LSTM(512))
		model.add(Dense(1201))

https://github.com/keras-team/keras/issues/3522


48. **clone only one branch with git**

Git actually allows you to clone only one branch, for example:

		git clone -b mybranch --single-branch git://sub.domain.com/repo.git


49. **Remove folders/files in git**

.gitignore will prevent untracked files from being added (without an add -f) to the set of files tracked by git, however git will continue to track any files that are already being tracked.

To stop tracking a file you need to remove it from the index. This can be achieved with this command.

git rm --cached <file>
The removal of the file from the head revision will happen on the next commit.
	

50. **Adding more information during training tensorflow estimator**

A very bad thing related to tf.estimator is that it is very hard to do basic things such as printing
accuracy from training dataset.
Here is a code that can help this (maccuracy)

	    if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
		train_op = optimizer.minimize(loss,
					      global_step=tf.train.get_global_step())
		accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
		logging_hook = tf.train.LoggingTensorHook({"mloss": loss, "maccuracy": accuracy[1]}, every_n_iter=100)
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks = [logging_hook])

	    eval_metric_ops = {
		'accuracy':
		    tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
	    }
    

An even more detail example:


	    if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
		train_op = optimizer.minimize(loss,
					      global_step=tf.train.get_global_step())
		accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)

		logging_hook = tf.train.LoggingTensorHook({"mloss": loss, "predicted classes": predicted_classes,
							   "ground truth": labels, "maccuracy": accuracy[1]}, every_n_iter=100)
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks = [logging_hook])

	
	
51. **Difference between np.array and np.asarray**

https://stackoverflow.com/questions/14415741/numpy-array-vs-asarray

The difference can be demonstrated by this example:

generate a matrix

		>>> A = numpy.matrix(np.ones((3,3)))
		>>> A
		matrix([[ 1.,  1.,  1.],
			[ 1.,  1.,  1.],
			[ 1.,  1.,  1.]])
		use numpy.array to modify A. Doesn't work because you are modifying a copy

		>>> numpy.array(A)[2]=2
		>>> A
		matrix([[ 1.,  1.,  1.],
			[ 1.,  1.,  1.],
			[ 1.,  1.,  1.]])
		use numpy.asarray to modify A. It worked because you are modifying A itself

		>>> numpy.asarray(A)[2]=2
		>>> A
		matrix([[ 1.,  1.,  1.],
			[ 1.,  1.,  1.],
			[ 2.,  2.,  2.]])
		Hope this helps!
