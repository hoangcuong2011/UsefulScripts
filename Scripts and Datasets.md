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


    encoder = LabelEncoder()
    encoder.fit(y_test)
    encoded_Y = encoder.transform(y_test)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_test = np_utils.to_categorical(encoded_Y)

