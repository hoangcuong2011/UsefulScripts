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
