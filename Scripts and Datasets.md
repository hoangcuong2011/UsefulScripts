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

52. **Python Ranking Dictionary Return Rank**

https://stackoverflow.com/questions/30282600/python-ranking-dictionary-return-rank

Python Ranking Dictionary Return Rank

		I have a python dictionary:

		x = {'a':10.1,'b':2,'c':5}

		How do I go about ranking and returning the rank value? Like getting back:

		res = {'a':1,c':2,'b':3}

		Thanks

		If I understand correctly, you can simply use sorted to get the ordering, and then enumerate to number them:

		>>> x = {'a':10.1, 'b':2, 'c':5}
		>>> sorted(x, key=x.get, reverse=True)
		['a', 'c', 'b']
		>>> {key: rank for rank, key in enumerate(sorted(x, key=x.get, reverse=True), 1)}
		{'b': 3, 'c': 2, 'a': 1}
		Note that this assumes that the ranks are unambiguous. If you have ties, the rank order among the tied keys will be arbitrary. It's easy to handle that too using similar methods, for example if you wanted all the tied keys to have the same rank. We have

		>>> x = {'a':10.1, 'b':2, 'c': 5, 'd': 5}
		>>> {key: rank for rank, key in enumerate(sorted(x, key=x.get, reverse=True), 1)}
		{'a': 1, 'b': 4, 'd': 3, 'c': 2}
		but

		>>> r = {key: rank for rank, key in enumerate(sorted(set(x.values()), reverse=True), 1)}
		>>> {k: r[v] for k,v in x.items()}
		{'a': 1, 'b': 3, 'd': 2, 'c': 2}
		
		
53. **git: How do I get the latest version of my code?**



		167
		down vote
		Case 1: Don’t care about local changes

		Solution 1: Get the latest code and reset the code

		git fetch origin
		git reset --hard origin/[tag/branch/commit-id usually: master]
		Solution 2: Delete the folder and clone again :D

		rm -rf [project_folder]
		git clone [remote_repo]
		Case 2: Care about local changes

		Solution 1: no conflicts with new-online version

		git fetch origin
		git status
		will report something like:

		Your branch is behind 'origin/master' by 1 commit, and can be fast-forwarded.
		Then get the latest version

		git pull
		Solution 2: conflicts with new-online version

		git fetch origin
		git status
		will report something like:

		error: Your local changes to the following files would be overwritten by merge:
		    file_name
		Please, commit your changes or stash them before you can merge.
		Aborting
		Commit your local changes

		git add .
		git commit -m ‘Commit msg’
		Try to get the changes (will fail)

		git pull
		will report something like:

		Pull is not possible because you have unmerged files.
		Please, fix them up in the work tree, and then use 'git add/rm <file>'
		as appropriate to mark resolution, or use 'git commit -a'.
		Open the conflict file and fix the conflict. Then:

		git add .
		git commit -m ‘Fix conflicts’
		git pull
		will report something like:

		Already up-to-date.
		More info: How do I use 'git reset --hard HEAD' to revert to a previous commit?

54. **Check GPU usages**

		nvidia-smi --loop-ms=1000


55. **Using fastText with Python**
https://github.com/facebookresearch/fastText

There are many ways to use fasttext with python. A simpler way is as follows:

		$ git clone https://github.com/facebookresearch/fastText.git
		$ cd fastText
		$ pip install .

Then in the code:

		import fastText as ft
		model = ft.train_unsupervised(input='dataset_file', maxn=0, dim=100, epoch=15, neg=15)
		model.get_word_vector(a word)
		
		
56. **initialize an list with 60 zeros you do**

To initialize an list with 60 zeros you do:

List<Integer> list = new ArrayList<Integer>(Collections.nCopies(60, 0));
	
	
57. **keras - flatten a tensor**
	
		x = Flatten()(x)
		
58. **parser funtion for main function in python**

		def parse_args(parser):
			parser.add_option("-m", "--model", dest="model_name", type="string", default="best_TransE_L2")
			parser.add_option("-d", "--data", dest="data_name", type="string", default="wn18")
			parser.add_option("-r", "--relation", dest="relation", action="store_true", default=False)
			parser.add_option("-s", "--save", dest="save", action="store_true", default=False)

			options, args = parser.parse_args()
			return options, args

		def main(options):
			logger = logging.getLogger()
			logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)
			train(options.model_name, options.data_name,
				params_dict=param_space_dict[options.model_name],
				logger=logger, eval_by_rel=options.relation, if_save=options.save)

		if __name__ == "__main__":
			parser = OptionParser()
			print(parser)
			print(type(parser))
			options, args = parse_args(parser)
			main(options)


59. **yield in python**

When you call a function that contains a yield statement anywhere, you get a generator object, but no code runs. Then each time you extract an object from the generator, Python executes code in the function until it comes to a yield statement, then pauses and delivers the object. When you extract another object, Python resumes just after the yield and continues until it reaches another yield (often the same one, but one iteration later). This continues until the function runs off the end, at which point the generator is deemed exhausted


60. **defaultdict in python**

		>>> from collections import defaultdict
		>>> food_list = 'spam spam spam spam spam spam eggs spam'.split()
		>>> food_count = defaultdict(int) # default value of int is 0
		>>> for food in food_list:
		...     food_count[food] += 1 # increment element's value by 1
		...
		defaultdict(<type 'int'>, {'eggs': 1, 'spam': 7})
		>>>

61. **None in keras**

https://stackoverflow.com/questions/47240348/what-is-the-meaning-of-the-none-in-model-summary-of-keras

None means this dimension is variable.

The first dimension in a keras model is always the batch size. You don't need fixed batch sizes, unless in very specific cases (for instance, when working with stateful=True LSTM layers).

That's why this dimension is often ignored when you define your model. For instance, when you define input_shape=(100,200), actually you're ignoring the batch size and defining the shape of "each sample". Internally the shape will be (None, 100, 200), allowing a variable batch size, each sample in the batch having the shape (100,200).

The batch size will be then automatically defined in the fit or predict methods.

Other None dimensions:

Not only the batch dimension can be None, but many others as well.

For instance, in a 2D convolutional network, where the expected input is (batchSize, height, width, channels), you can have shapes like (None, None, None, 3), allowing variable image sizes.

In recurrent networks and in 1D convolutions, you can also make the length/timesteps dimension variable, with shapes like (None, None, featuresOrChannels)

62. **Different operations in Keras**


		tanh_hidden_hotel = tf.transpose(tanh_hidden_hotel, perm=[0, 2, 1])
		embedding_user = Flatten()(embedding_user)
		r = K.dot(hidden_hotel, alpha)
		alpha = Permute((1, 2))(alpha)
		from keras import backend as K
		r = K.reshape(r, (-1, 20))
		from keras.layers import Reshape
		r = Reshape((-1, 20))(r)
		combine_input = concatenate([attention_output, last_hidden])


63. **How to use lambda layer in keras?**

See here for a reference https://stackoverflow.com/questions/51963377/keras-nonetype-object-has-no-attribute-inbound-nodes


		import keras.backend as K
		def myFunc(x):
		    return x[0] * x[1]
		    
And then:
		
		cross1 = Lambda(myFunc, output_shape=....)([d1,d4])
		
		
		cross2 = Lambda(myFunc, output_shape=....)([d2,d3])

64. **Get the last cell from a LSTM sequence from Keras**

		last_timestep = Lambda(lambda x: x[:, -1, :])(lstm_layer)


65. **Lambda wrapper for keras with a function that has multiple parameters**

Lambda layer enhancements

Let's assume we have this function with two parameters hidden_hotel_sequence and w

		def attention_mechanism(hidden_hotel_sequence, w):
		
Then from the code we can use Lambda wrapper:

		attention_output = Lambda(attention_mechanism, arguments={'w': W})(hidden_hotel)
		
See this for a reference: https://github.com/keras-team/keras/pull/1911

66. **return best model with keras**

		best_weights_filepath = './my_model_best_weights.hdf5'
				earlyStopping = EarlyStopping(monitor='val_loss',
														 patience=10, verbose=1, mode='auto')
				saveBestModel = ModelCheckpoint(best_weights_filepath,
														   monitor='val_loss', verbose=1,
														   save_best_only=True, mode='auto')
				model.fit([X_train_hotel_sequence, X_train_user_ids], y_train,
						  validation_data=([X_valid_hotel_sequence, X_valid_user_ids], y_valid),
						  epochs=number_of_epoch, callbacks=[earlyStopping, saveBestModel], verbose=2)
				model.load_weights(best_weights_filepath)
			evaluate_with_rnn(model, tokenizer_hotel_sequence, test_hotel_sequence, max_sequence_hotel_len,
							  tokenizer_user_id, test_user_ids, max_sequence_user_id_len)


67. **write json file in python**

		f = open(export_path+'config', "w", encoding='utf-8')
		f.write(json.dumps({"word2id": word2id,
				    "relation2id": relation2id,
				    "word_size": word_size,
				    "fixlen": fixlen,
				    "maxlen": maxlen}))
		f.close()

68. **Play with keras- tensorflow**

Some simple example that help us debug tensor operators easier.



		tf.set_random_seed(100)
		batch = 3
		length = 2
		cell = 4
		a_original = tf.random_uniform((2, 3, 4))
		a = tf.transpose(a_original, perm=[2, 0, 1])
		b = tf.random_uniform((2, 3))
		c = tf.multiply(a, b)

		print(a_original, a, b, c)

		with tf.Session() as sess:
			a_original, a, b, c = sess.run([a_original, a, b, c])
			print(a_original)
			print("a")
			print(a)
			print("b")
			print(b)
			print("c")
			print(c)


69. **pass default argument in python**

		parser = argparse.ArgumentParser()
		parser.add_argument("-m", "--modelType", help="The kind of model you want to test, either ntm, dense or lstm", default="lstm")
		parser.add_argument("-e", "--epochs", help="The number of epochs to train", default="3", type=int)
		parser.add_argument("-c", "--ntm_controller_architecture", help="""Valid choices are: dense, double_dense or
						    lstm. Ignored if model is not ntm""", default="dense")
		parser.add_argument("-v", "--verboose", help="""Verboose training: If enabled, the model is evaluated extensively
						    after each training epoch.""", action="store_true")
		args = parser.parse_args()
		modelType = args.modelType
		epochs = args.epochs
		ntm_controller_architecture = args.ntm_controller_architecture
		verboose = args.verboose


70. **different loss functions in keras**

		def mean_squared_error(y_true, y_pred):
		    return K.mean(K.square(y_pred - y_true), axis=-1)


		def mean_absolute_error(y_true, y_pred):
		    return K.mean(K.abs(y_pred - y_true), axis=-1)


		def mean_absolute_percentage_error(y_true, y_pred):
		    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
							    K.epsilon(),
							    None))
		    return 100. * K.mean(diff, axis=-1)


		def mean_squared_logarithmic_error(y_true, y_pred):
		    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
		    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
		    return K.mean(K.square(first_log - second_log), axis=-1)


		def squared_hinge(y_true, y_pred):
		    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


		def hinge(y_true, y_pred):
		    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


		def categorical_hinge(y_true, y_pred):
		    pos = K.sum(y_true * y_pred, axis=-1)
		    neg = K.max((1. - y_true) * y_pred, axis=-1)
		    return K.maximum(0., neg - pos + 1.)


		def logcosh(y_true, y_pred):
		    """Logarithm of the hyperbolic cosine of the prediction error.
		    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
		    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
		    like the mean squared error, but will not be so strongly affected by the
		    occasional wildly incorrect prediction.
		    # Arguments
			y_true: tensor of true targets.
			y_pred: tensor of predicted targets.
		    # Returns
			Tensor with one scalar loss entry per sample.
		    """
		    def _logcosh(x):
			return x + K.softplus(-2. * x) - K.log(2.)
		    return K.mean(_logcosh(y_pred - y_true), axis=-1)


		def categorical_crossentropy(y_true, y_pred):
		    return K.categorical_crossentropy(y_true, y_pred)


		def sparse_categorical_crossentropy(y_true, y_pred):
		    return K.sparse_categorical_crossentropy(y_true, y_pred)


		def binary_crossentropy(y_true, y_pred):
		    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


		def kullback_leibler_divergence(y_true, y_pred):
		    y_true = K.clip(y_true, K.epsilon(), 1)
		    y_pred = K.clip(y_pred, K.epsilon(), 1)
		    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


		def poisson(y_true, y_pred):
		    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


		def cosine_proximity(y_true, y_pred):
		    y_true = K.l2_normalize(y_true, axis=-1)
		    y_pred = K.l2_normalize(y_pred, axis=-1)
		    return -K.sum(y_true * y_pred, axis=-1)

ranking loss (BPR: https://arxiv.org/pdf/1511.06939.pdf)

		def custom_objective(y_true, y_pred):
			abc = y_true * y_pred
			a = K.max(abc, axis=-1)
			a = keras.layers.Subtract()([a, y_pred])
			return - K.mean(K.log(sigmoid(a)),axis=-1)
			
71 **categorical_crossentropy vs sparse_categorical_crossentropy**
		
		If your targets are one-hot encoded, use categorical_crossentropy.
		Examples of one-hot encodings:
		[1,0,0]
		[0,1,0]
		[0,0,1]
		But if your targets are integers, use sparse_categorical_crossentropy.
		Examples of integer encodings (for the sake of completion):
		1
		2
		3

72. **GRU in tensorflow**

		def gru(units):
			# If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
			#  the code automatically does that.
			if tf.test.is_gpu_available():
				return tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
			else:
				return tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_activation='sigmoid', recurrent_initializer='glorot_uniform')


Check this for reference https://github.com/hoangcuong2011/ntm


73. **Read configuration file from Java and Python**

		Properties prop = new Properties();
			String propFileName = "resources/config_filtering_models.txt";
			FileReader reader = new FileReader(propFileName);
			prop.load(reader);
			input_file = prop.getProperty("input_file");
			dice_output_file_ranking = prop.getProperty("dice_output_file_ranking");
			wordembedding_output_file_ranking = prop.getProperty("dice_output_file_ranking");
			classification_output_ranking_file = prop.getProperty("classification_output_ranking_file");
			ner_output_file_ranking = prop.getProperty("ner_output_file_ranking");
			ner_output_file_ranking = prop.getProperty("ner_output_file_ranking");
			cut_off_ranking_dice = Integer.parseInt(prop.getProperty("cut_off_ranking_dice"));
			cut_off_ranking_wordembedding = Integer.parseInt(prop.getProperty("cut_off_ranking_wordembedding"));
			cut_off_ranking_classification = Integer.parseInt(prop.getProperty("cut_off_ranking_classification"));
			cut_off_ranking_ner = Integer.parseInt(prop.getProperty("cut_off_ranking_ner"));
			final_relation_extraction_produced_output = prop.getProperty("final_relation_extraction_produced_output");
			if (prop.getProperty("remove_if_not_contain_important_words").compareToIgnoreCase("yes")==0) {
			    remove_if_not_contain_important_words = true;
			}
	
	
	

		cfg = configparser.ConfigParser()
		filename = 'resources/config_filtering_models.txt'
		with open(filename) as fp:
			cfg.read_file(itertools.chain(['[global]'], fp), source=filename)
		print(cfg.items('global'))
		seed_number = int(cfg.get('global', 'seed_number'))
		batch_size = seed_number * 2  # entity numbers used each training time
		input_file = cfg.get('global', 'input_file')
		test_file = cfg.get('global', 'testFile')
		
		
74 **sample categorical distribution by python**

		Generate a uniform random sample from np.arange(5) of size 3:

		>>>
		>>> np.random.choice(5, 3)
		array([0, 3, 4])
		>>> #This is equivalent to np.random.randint(0,5,3)
		Generate a non-uniform random sample from np.arange(5) of size 3:

		>>>
		>>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
		array([3, 3, 0])
		Generate a uniform random sample from np.arange(5) of size 3 without replacement:

		>>>
		>>> np.random.choice(5, 3, replace=False)
		array([3,1,0])
		>>> #This is equivalent to np.random.permutation(np.arange(5))[:3]
		Generate a non-uniform random sample from np.arange(5) of size 3 without replacement:

		>>>
		>>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
		array([2, 3, 0])
		Any of the above can be repeated with an arbitrary array-like instead of just integers. For instance:

		>>>
		>>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
		>>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
		array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
		      dtype='|S11')
      

75. **set tensorflow to run on a specific GPU**

		import os
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
		os.environ["CUDA_VISIBLE_DEVICES"]="2"

76. **Print dependency parsing in Stanford NLP**

		package edu.stanford.nlp.examples;

		import edu.stanford.nlp.pipeline.*;
		import edu.stanford.nlp.ling.*;
		import edu.stanford.nlp.trees.*;
		import edu.stanford.nlp.semgraph.*;
		import edu.stanford.nlp.util.*;

		import java.util.*;

		public class PrintParse {

		  public static void main(String[] args) {
		    Annotation document =
			new Annotation("My dog also likes eating sausage.");
		    Properties props = new Properties();
		    props.setProperty("annotators", "tokenize,ssplit,pos,lemma,ner,parse");
		    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		    pipeline.annotate(document);
		    for (CoreMap sentence : document.get(CoreAnnotations.SentencesAnnotation.class)) {
		      Tree constituencyParse = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
		      System.out.println(constituencyParse);
		      SemanticGraph dependencyParse =
			  sentence.get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
		      System.out.println(dependencyParse.toList());
		    }
		  }

		}



77. **Generating classification dataset with sklearn**

		from sklearn.datasets import make_classification
		from sklearn import ensemble
		from sklearn.calibration import CalibratedClassifierCV
		from sklearn.metrics import log_loss
		from sklearn import cross_validation

		X, y = make_classification(n_samples=1000,
					   n_features=1000,
					   n_informative=30,
					   n_redundant=0,
					   n_repeated=0,
					   n_classes=9,
					   random_state=0,
					   shuffle=False)

		print(X)
		print(y)



78. **repeat vector in keras**
reference: https://blog.keras.io/building-autoencoders-in-keras.html

		from keras.layers import Input, LSTM, RepeatVector
		from keras.models import Model

		inputs = Input(shape=(timesteps, input_dim))
		encoded = LSTM(latent_dim)(inputs)

		decoded = RepeatVector(timesteps)(encoded)
		decoded = LSTM(input_dim, return_sequences=True)(decoded)

		sequence_autoencoder = Model(inputs, decoded)
		encoder = Model(inputs, encoded)

79. **ensemble models in keras**

reference: https://towardsdatascience.com/ensembling-convnets-using-keras-237d429157eb


		def model_1(input_length, output_class_size):
			input = Input(shape=(input_length,))
			output = Dense(output_class_size, activation='softmax')(input)
			return Model(inputs=input, outputs=output)


		def model_2(input_length, output_class_size):
			input = Input(shape=(input_length,))
			output = Dense(output_class_size, activation='softmax')(input)
			return Model(inputs=input, outputs=output)

		def ensemble(models, input_length, output_class_size):
			input = Input(shape=(input_length,))
			outputs = []
			for model in models:
				outputs.append(model(input))
			output = Average()(outputs)
			model = Model(inputs=input, outputs=output, name='ensemble')
			return model
		model1 = model_1()
		model2 = model_2()

		ensemble_model = ensemble([model1, model2], input_length, output_class_size)


80. **requirements**

		In the working virtualenv, create a file with the version of each installed Python library :

		pip freeze > requirements.txt
		In the new virtualenv, ask pip to install those libraries with the same version :

		pip install -r requirements.txt


81. **learning rate decay**

		def scheduler(epoch):
		    if epoch == 5:
			model.lr.set_value(.02)
		    return model.lr.get_value()

		change_lr = LearningRateScheduler(scheduler)

		model.fit(x_embed, y, nb_epoch=1, batch_size = batch_size, show_accuracy=True,
		       callbacks=[chage_lr])
       
       
82. **learning rate decay - another example**
reference: https://gist.github.com/hoangcuong2011/2f1e421d35db0193b513678eae31e77b

		import numpy as np
		from keras.callbacks import LearningRateScheduler

		def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
		    '''
		    Wrapper function to create a LearningRateScheduler with step decay schedule.
		    '''
		    def schedule(epoch):
			return initial_lr * (decay_factor ** np.floor(epoch/step_size))

		    return LearningRateScheduler(schedule)

		lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)

		model.fit(X_train, Y_train, callbacks=[lr_sched])


output is something like this:

		loss: 4.2416 - acc: 0.5438 - val_loss: 3.8026 - val_acc: 0.5961
		loss: 3.7341 - acc: 0.6040 - val_loss: 3.7058 - val_acc: 0.6097
		loss: 3.6075 - acc: 0.6187 - val_loss: 3.6014 - val_acc: 0.6208
		loss: 3.5884 - acc: 0.6200 - val_loss: 3.6015 - val_acc: 0.6184
		loss: 3.5298 - acc: 0.6248 - val_loss: 3.5471 - val_acc: 0.6243
		loss: 3.5165 - acc: 0.6254 - val_loss: 3.5449 - val_acc: 0.6250
		loss: 3.4762 - acc: 0.6280 - val_loss: 3.4928 - val_acc: 0.6281
		loss: 3.4687 - acc: 0.6284 - val_loss: 3.4952 - val_acc: 0.6276
		loss: 3.4399 - acc: 0.6297 - val_loss: 3.4530 - val_acc: 0.6293
		loss: 3.4321 - acc: 0.6299 - val_loss: 3.4536 - val_acc: 0.6294
		loss: 3.4088 - acc: 0.6309 - val_loss: 3.4220 - val_acc: 0.6303
		loss: 3.4027 - acc: 0.6310 - val_loss: 3.4162 - val_acc: 0.6310
		loss: 3.3828 - acc: 0.6316 - val_loss: 3.3967 - val_acc: 0.6312
		loss: 3.3780 - acc: 0.6316 - val_loss: 3.3992 - val_acc: 0.6312
		loss: 3.3631 - acc: 0.6320 - val_loss: 3.3834 - val_acc: 0.6316
		loss: 3.3566 - acc: 0.6322 - val_loss: 3.3765 - val_acc: 0.6315
		loss: 3.3447 - acc: 0.6324 - val_loss: 3.3656 - val_acc: 0.6320
		loss: 3.3410 - acc: 0.6325 - val_loss: 3.3609 - val_acc: 0.6320
		loss: 3.3311 - acc: 0.6327 - val_loss: 3.3528 - val_acc: 0.6322
		loss: 3.3301 - acc: 0.6327 - val_loss: 3.3511 - val_acc: 0.6323
		loss: 3.3211 - acc: 0.6329 - val_loss: 3.3439 - val_acc: 0.6324
		loss: 3.3169 - acc: 0.6329 - val_loss: 3.3395 - val_acc: 0.6324
		loss: 3.3101 - acc: 0.6331 - val_loss: 3.3352 - val_acc: 0.6325


83. **adding gaussian noise in gradient**
reference: https://github.com/cpury/keras_gradient_noise/blob/master/keras_gradient_noise/gradient_noise.py
paper: https://openreview.net/pdf?id=rkjZ2Pcxe


		import inspect
		import keras
		from keras import backend as K


		def _get_shape(x):
		    if hasattr(x, 'dense_shape'):
			return x.dense_shape

		    return K.shape(x)


		def add_gradient_noise(BaseOptimizer):
		    """
		    Given a Keras-compatible optimizer class, returns a modified class that
		    supports adding gradient noise as introduced in this paper:
		    https://arxiv.org/abs/1511.06807
		    The relevant parameters from equation 1 in the paper can be set via
		    noise_eta and noise_gamma, set by default to 0.3 and 0.55 respectively.
		    """
		    if not (
			inspect.isclass(BaseOptimizer) and
			issubclass(BaseOptimizer, keras.optimizers.Optimizer)
		    ):
			raise ValueError(
			    'add_gradient_noise() expects a valid Keras optimizer'
			)

		    class NoisyOptimizer(BaseOptimizer):
			def __init__(self, noise_eta=0.3, noise_gamma=0.55, **kwargs):
			    super(NoisyOptimizer, self).__init__(**kwargs)
			    with K.name_scope(self.__class__.__name__):
				self.noise_eta = K.variable(noise_eta, name='noise_eta')
				self.noise_gamma = K.variable(noise_gamma, name='noise_gamma')

			def get_gradients(self, loss, params):
			    grads = super(NoisyOptimizer, self).get_gradients(loss, params)

			    # Add decayed gaussian noise
			    t = K.cast(self.iterations, K.dtype(grads[0]))
			    variance = self.noise_eta / ((1 + t) ** self.noise_gamma)

			    grads = [
				grad + K.random_normal(
				    _get_shape(grad),
				    mean=0.0,
				    stddev=K.sqrt(variance),
				    dtype=K.dtype(grads[0])
				)
				for grad in grads
			    ]

			    return grads

			def get_config(self):
			    config = {'noise_eta': float(K.get_value(self.noise_eta)),
				      'noise_gamma': float(K.get_value(self.noise_gamma))}
			    base_config = super(NoisyOptimizer, self).get_config()
			    return dict(list(base_config.items()) + list(config.items()))

		    NoisyOptimizer.__name__ = 'Noisy{}'.format(BaseOptimizer.__name__)

		    return NoisyOptimizer


84. **convert a tensor to int32 in keras**

		labels = tf.cast(x_input, tf.int32)

85. **Install box - need this for openAIGYM**

		brew install swig

		$git clone https://github.com/pybox2d/pybox2d pybox2d_dev
		$cd pybox2d_dev
		$python setup.py build 
		$sudo python setup.py install

86. **Custom loss with Euclidean distance**

Define a custom loss function:

		import keras.backend as K


		def euclidean_distance_loss(y_true, y_pred):
		    """
		    Euclidean distance loss
		    https://en.wikipedia.org/wiki/Euclidean_distance
		    :param y_true: TensorFlow/Theano tensor
		    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
		    :return: float
		    """
		    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
Use it:

		model.compile(loss=euclidean_distance_loss, optimizer='rmsprop')


87. **Write custom keras loss - part 2 - dice coefficient**

		from keras import backend as K

		def dice_coef(y_true, y_pred, smooth=1):
		    """
		    Dice = (2*|X & Y|)/ (|X|+ |Y|)
			 =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
		    ref: https://arxiv.org/pdf/1606.04797v1.pdf
		    """
		    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
		    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

		def dice_coef_loss(y_true, y_pred):
		    return 1-dice_coef(y_true, y_pred)

88. **write a dictionary to numpy-format files**

		import numpy as np

		# Save
		dictionary = {'hello':'world'}
		np.save('my_file.npy', dictionary) 

		# Load
		read_dictionary = np.load('my_file.npy').item()
		print(read_dictionary['hello']) # displays "world"

89. **another way to write custom loss for simpler cases***

		# define model
		def l1_reg(weight_matrix):
		    return 0.01 * K.sum(K.abs(weight_matrix))

		wts = model.layers[-1].trainable_weights # -1 for last dense layer.
		reg_loss = l1_reg(wts[0]) + l1_reg(wts[1])

		def custom_loss(reg_loss):
		    def orig_loss(y_true, y_pred):
			return K.categorical_crossentropy(y_true, y_pred) + reg_loss
		    return orig_loss

		model.compile(loss=custom_loss(reg_loss),
			      optimizer=keras.optimizers.Adadelta(),
			      metrics=['accuracy'])


90. **get batch size from NONE**

		batch_size = K.shape(u_vecs)[0]
		input_num_capsule = K.shape(u_vecs)[1]
		
91. **Batch normalization in keras**

Reference:
https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras

Just to answer this question in a little more detail, and as Pavel said, Batch Normalization is just another layer, so you can use it as such to create your desired network architecture.

The general use case is to use BN between the linear and non-linear layers in your network, because it normalizes the input to your activation function, so that you're centered in the linear section of the activation function (such as Sigmoid). There's a small discussion of it here

In your case above, this might look like:

		# import BatchNormalization
		from keras.layers.normalization import BatchNormalization

		# instantiate model
		model = Sequential()

		# we can think of this chunk as the input layer
		model.add(Dense(64, input_dim=14, init='uniform'))
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))

		# we can think of this chunk as the hidden layer    
		model.add(Dense(64, init='uniform'))
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))

		# we can think of this chunk as the output layer
		model.add(Dense(2, init='uniform'))
		model.add(BatchNormalization())
		model.add(Activation('softmax'))

		# setting up the optimization of our weights 
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='binary_crossentropy', optimizer=sgd)

		# running the fitting
		model.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True, validation_split=0.2, verbose = 2)
Hope this clarifies things a bit more.

Do not use bias before batch normalization (see this) https://www.dlology.com/blog/one-simple-trick-to-train-keras-model-faster-with-batch-normalization/

		model.add(layers.Dense(64, use_bias=False))
		model.add(layers.BatchNormalization())
		model.add(Activation("relu"))

92. **Some ways of combining information in two branch of a net** 

reference: https://twitter.com/karpathy/status/1109961535630143491

		Some ways of combining information in two branch of a net A & B: 1) A+B, 2) A*B, 3) concat [A,B], 4) LSTM-style tanh(A) * sigmoid(B), 5) hypernetwork-style convolve A with weights w = f(B), 6) hypernetwork-style batch-norm A with \gamma, \beta = f(B), 7) A soft attend to B, ... ?


93. **get columns from two dimension vectors**

		 X_train_sequence_topic = X_train_sequence_topic[0:len(X_train_sequence_topic),20:30]
 
 or two specific columns:

		I assume you wanted columns 1 and 9? That's

		data[:, [1, 9]]


94. **good command to create dataset (output) for sequence to sequence or language modeling**

		def x_y_for_dataset(dataset_name):
			fat_sample = training_data_to_dense_samples(
			    dataset_name, encoder, max_seq_length)[:2]
			_x = fat_sample[:, :max_seq_length]
			_y = np.expand_dims(fat_sample[:, 1:], axis=-1)
			print("x_", _x)
			print("y_", _y)
			return _x, _y
Example of this:


		x_ [[   1   16 2279  205  135 1222  310  135   16    2    1 2279  205  135
		  1222  310  135    5  167   22    4  916 6012  758   42  521 6012  758
		     5   27   13  309    8    9 6012  758   32    4  602  946 1609    5
		  4688  809   10  921    8    4  483  809    6   30   27 3328 1189   12
		     4  217 6012  758    5  278    6 6942  181  135    6   30   89 4007
		    12   13 1220    8    7 1339   29    7   11   28   10   13 1436    8
		     7 3316 2185  269 2340   29    7 4425   28    5   10 5750   13  223
		   252  607  243  335 2084    8  742 1774    6   11  245    5    4 6012
		  2769   40 1013    5   81 1340   17 6012  758  424   17   21 5942  209
		     6  131  297 3622   11    4  951    5 3667 4248   38   40 1593   26
		     4 2334   24   91   12   13  122  115 5075 2003   73    9 2266 1941
		    99    6 2279  205  135 1222  310  135   27   13 1876 1264  129 9693
		  1081    5   10   27 1865 3290  550 6012  758  329  177    5 1428  229
		     4  225 6231    6    2    1   16   16 2009   16   16    2    1 2279
		   205  135 1222  310  135   27   13  288    9    5   25   13  495 1220
		    91   12    7 7701   29    7   11   28   10 5280   94   91   12    7
		    49    7 3316 2185  269 2340   29    7   49    7 4425   28    5  174
		     4 6012 2769 3290   11 6012  758  329  177   40 1072    7   49    7
		  1339   29    7   49]
		 [   7   11   28  271   10 7090 8251    7   49    7 2464   29    7   49
		     7 4425   28    6  200   66 1592  196 5563  540    5 6012 2769   56
		    13 1947    9   38   51 1185 1003   11  481   12 4007    5   11   13
		  1023  195    9   29    9   28    6   41   89 2775  153  343   13  122
		    24  580 6012 2769    5   45 5436  112   12  559  767    7   49    7
		   125   24 1302 2690    6    2    1    4   47 2084    8    9   27 1633
		    25   13  288    5  392 3027 4832 2142 2084    8  742 1774    6    4
		  1302   50   27    4   17    9   17    5   10   60 4668  340    9  114
		    24 1592 2607 3849   46    4   66   27    4   17  665 5567   17    5
		    38   60 7593 3533 5823    5   10   27  114   24 3318   42   85 9029
		     4 3849    6 1072    5    4  391  742  516   27    4    9    5   10
		     4  649   27    4  665 5567    6    2    1    4    9   27  876 1013
		  1014    5   25 5979   23    9    5   10 2652 1804    6    4  424 4388
		  1624   25 6012 2769   81 1480   54 5942  209    6   41 3622  191    5
		    11  245    5    4  424  544  119  639    9   27 2828   12   13 1880
		  1938    5   45    4 1938   27 2088   91   26    4 4996    8 5942  209
		     5 8374    4  424  544  119  639    6    2    1    4 5444 4156    8
		   278    6 1222  310  135   27    4  217 6012  758    5 2279  205  135
		  6942  181  135    6]]
		y_ [[[  16]
		  [2279]
		  [ 205]
		  [ 135]
		  [1222]
		  [ 310]
		  [ 135]
		  [  16]
		  [   2]
		  [   1]
		  [2279]
		  [ 205]
		  [ 135]
		  [1222]
		  [ 310]
		  [ 135]
		  [   5]
		  [ 167]
		  [  22]
		  [   4]
		  [ 916]
		  [6012]
		  [ 758]
		  [  42]
		  [ 521]
		  [6012]
		  [ 758]
		  [   5]
		  [  27]
		  [  13]
		  [ 309]
		  [   8]
		  [   9]
		  [6012]
		  [ 758]
		  [  32]
		  [   4]
		  [ 602]
		  [ 946]
		  [1609]
		  [   5]
		  [4688]
		  [ 809]
		  [  10]
		  [ 921]
		  [   8]
		  [   4]
		  [ 483]
		  [ 809]
		  [   6]
		  [  30]
		  [  27]
		  [3328]
		  [1189]
		  [  12]
		  [   4]
		  [ 217]
		  [6012]
		  [ 758]
		  [   5]
		  [ 278]
		  [   6]
		  [6942]
		  [ 181]
		  [ 135]
		  [   6]
		  [  30]
		  [  89]
		  [4007]
		  [  12]
		  [  13]
		  [1220]
		  [   8]
		  [   7]
		  [1339]
		  [  29]
		  [   7]
		  [  11]
		  [  28]
		  [  10]
		  [  13]
		  [1436]
		  [   8]
		  [   7]
		  [3316]
		  [2185]
		  [ 269]
		  [2340]
		  [  29]
		  [   7]
		  [4425]
		  [  28]
		  [   5]
		  [  10]
		  [5750]
		  [  13]
		  [ 223]
		  [ 252]
		  [ 607]
		  [ 243]
		  [ 335]
		  [2084]
		  [   8]
		  [ 742]
		  [1774]
		  [   6]
		  [  11]
		  [ 245]
		  [   5]
		  [   4]
		  [6012]
		  [2769]
		  [  40]
		  [1013]
		  [   5]
		  [  81]
		  [1340]
		  [  17]
		  [6012]
		  [ 758]
		  [ 424]
		  [  17]
		  [  21]
		  [5942]
		  [ 209]
		  [   6]
		  [ 131]
		  [ 297]
		  [3622]
		  [  11]
		  [   4]
		  [ 951]
		  [   5]
		  [3667]
		  [4248]
		  [  38]
		  [  40]
		  [1593]
		  [  26]
		  [   4]
		  [2334]
		  [  24]
		  [  91]
		  [  12]
		  [  13]
		  [ 122]
		  [ 115]
		  [5075]
		  [2003]
		  [  73]
		  [   9]
		  [2266]
		  [1941]
		  [  99]
		  [   6]
		  [2279]
		  [ 205]
		  [ 135]
		  [1222]
		  [ 310]
		  [ 135]
		  [  27]
		  [  13]
		  [1876]
		  [1264]
		  [ 129]
		  [9693]
		  [1081]
		  [   5]
		  [  10]
		  [  27]
		  [1865]
		  [3290]
		  [ 550]
		  [6012]
		  [ 758]
		  [ 329]
		  [ 177]
		  [   5]
		  [1428]
		  [ 229]
		  [   4]
		  [ 225]
		  [6231]
		  [   6]
		  [   2]
		  [   1]
		  [  16]
		  [  16]
		  [2009]
		  [  16]
		  [  16]
		  [   2]
		  [   1]
		  [2279]
		  [ 205]
		  [ 135]
		  [1222]
		  [ 310]
		  [ 135]
		  [  27]
		  [  13]
		  [ 288]
		  [   9]
		  [   5]
		  [  25]
		  [  13]
		  [ 495]
		  [1220]
		  [  91]
		  [  12]
		  [   7]
		  [7701]
		  [  29]
		  [   7]
		  [  11]
		  [  28]
		  [  10]
		  [5280]
		  [  94]
		  [  91]
		  [  12]
		  [   7]
		  [  49]
		  [   7]
		  [3316]
		  [2185]
		  [ 269]
		  [2340]
		  [  29]
		  [   7]
		  [  49]
		  [   7]
		  [4425]
		  [  28]
		  [   5]
		  [ 174]
		  [   4]
		  [6012]
		  [2769]
		  [3290]
		  [  11]
		  [6012]
		  [ 758]
		  [ 329]
		  [ 177]
		  [  40]
		  [1072]
		  [   7]
		  [  49]
		  [   7]
		  [1339]
		  [  29]
		  [   7]
		  [  49]
		  [   3]]

		 [[  11]
		  [  28]
		  [ 271]
		  [  10]
		  [7090]
		  [8251]
		  [   7]
		  [  49]
		  [   7]
		  [2464]
		  [  29]
		  [   7]
		  [  49]
		  [   7]
		  [4425]
		  [  28]
		  [   6]
		  [ 200]
		  [  66]
		  [1592]
		  [ 196]
		  [5563]
		  [ 540]
		  [   5]
		  [6012]
		  [2769]
		  [  56]
		  [  13]
		  [1947]
		  [   9]
		  [  38]
		  [  51]
		  [1185]
		  [1003]
		  [  11]
		  [ 481]
		  [  12]
		  [4007]
		  [   5]
		  [  11]
		  [  13]
		  [1023]
		  [ 195]
		  [   9]
		  [  29]
		  [   9]
		  [  28]
		  [   6]
		  [  41]
		  [  89]
		  [2775]
		  [ 153]
		  [ 343]
		  [  13]
		  [ 122]
		  [  24]
		  [ 580]
		  [6012]
		  [2769]
		  [   5]
		  [  45]
		  [5436]
		  [ 112]
		  [  12]
		  [ 559]
		  [ 767]
		  [   7]
		  [  49]
		  [   7]
		  [ 125]
		  [  24]
		  [1302]
		  [2690]
		  [   6]
		  [   2]
		  [   1]
		  [   4]
		  [  47]
		  [2084]
		  [   8]
		  [   9]
		  [  27]
		  [1633]
		  [  25]
		  [  13]
		  [ 288]
		  [   5]
		  [ 392]
		  [3027]
		  [4832]
		  [2142]
		  [2084]
		  [   8]
		  [ 742]
		  [1774]
		  [   6]
		  [   4]
		  [1302]
		  [  50]
		  [  27]
		  [   4]
		  [  17]
		  [   9]
		  [  17]
		  [   5]
		  [  10]
		  [  60]
		  [4668]
		  [ 340]
		  [   9]
		  [ 114]
		  [  24]
		  [1592]
		  [2607]
		  [3849]
		  [  46]
		  [   4]
		  [  66]
		  [  27]
		  [   4]
		  [  17]
		  [ 665]
		  [5567]
		  [  17]
		  [   5]
		  [  38]
		  [  60]
		  [7593]
		  [3533]
		  [5823]
		  [   5]
		  [  10]
		  [  27]
		  [ 114]
		  [  24]
		  [3318]
		  [  42]
		  [  85]
		  [9029]
		  [   4]
		  [3849]
		  [   6]
		  [1072]
		  [   5]
		  [   4]
		  [ 391]
		  [ 742]
		  [ 516]
		  [  27]
		  [   4]
		  [   9]
		  [   5]
		  [  10]
		  [   4]
		  [ 649]
		  [  27]
		  [   4]
		  [ 665]
		  [5567]
		  [   6]
		  [   2]
		  [   1]
		  [   4]
		  [   9]
		  [  27]
		  [ 876]
		  [1013]
		  [1014]
		  [   5]
		  [  25]
		  [5979]
		  [  23]
		  [   9]
		  [   5]
		  [  10]
		  [2652]
		  [1804]
		  [   6]
		  [   4]
		  [ 424]
		  [4388]
		  [1624]
		  [  25]
		  [6012]
		  [2769]
		  [  81]
		  [1480]
		  [  54]
		  [5942]
		  [ 209]
		  [   6]
		  [  41]
		  [3622]
		  [ 191]
		  [   5]
		  [  11]
		  [ 245]
		  [   5]
		  [   4]
		  [ 424]
		  [ 544]
		  [ 119]
		  [ 639]
		  [   9]
		  [  27]
		  [2828]
		  [  12]
		  [  13]
		  [1880]
		  [1938]
		  [   5]
		  [  45]
		  [   4]
		  [1938]
		  [  27]
		  [2088]
		  [  91]
		  [  26]
		  [   4]
		  [4996]
		  [   8]
		  [5942]
		  [ 209]
		  [   5]
		  [8374]
		  [   4]
		  [ 424]
		  [ 544]
		  [ 119]
		  [ 639]
		  [   6]
		  [   2]
		  [   1]
		  [   4]
		  [5444]
		  [4156]
		  [   8]
		  [ 278]
		  [   6]
		  [1222]
		  [ 310]
		  [ 135]
		  [  27]
		  [   4]
		  [ 217]
		  [6012]
		  [ 758]
		  [   5]
		  [2279]
		  [ 205]
		  [ 135]
		  [6942]
		  [ 181]
		  [ 135]
		  [   6]
		  [   3]]]


95. **popular common commands in tensorflow and numpy"

		1. num_sents = tf.keras.backend.expand_dims(num_sents, axis=-1) -> turn something like this: [a b c] to [[a], [b], [c]]

		2. c = tf.keras.backend.squeeze(input_numsents, axis=-1) -> turn something like this: [[a], [b], [c]] to [a, b, c]
