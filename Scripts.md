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
    
    
