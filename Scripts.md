1. **make with the number of cores**

make all

it can be sped up by appending -j $(($(nproc) + 1))

2. **install a new package in R**


install.packages("h2o") (package name: h20)


3. **copy files line by line under linux **

paste -d" " file1.txt file2.txt
output looks like this:

/etc/port1-192.9.200.1-255.555.255.0 /etc/port1-192.90.2.1-255.555.0.0
/etc/port2-192.9.200.1-255.555.255.0 /etc/port2-192.90.2.1-255.555.0.0
/etc/port3-192.9.200.1-255.555.255.0 /etc/port3-192.90.2.1-255.555.0.0
/etc/port4-192.9.200.1-255.555.255.0 /etc/port4-192.90.2.1-255.555.0.0
/etc/port5-192.9.200.1-255.555.255.0 /etc/port5-192.90.2.1-255.555.0.0

