README 

The examples in this folder are meant to be an introduction to the use a features of libEsemble.

Some of the examples require the use of external software:

1. MUQ2 - http://muq.mit.edu/ 
For more detailed installation instructions visit http://muq.mit.edu/master-docs/muqinstall.html 
Installation: 
    1. Create a folder to install MUQ2 in 
    2. In termianl go to directory where you want to download to then
         git clone https://bitbucket.org/mituq/muq2.git
         cd muq2
         cmake -DCMAKE_INSTALL_PREFIX=/my/install/path -DMUQ_USE_PYTHON=ON 
         make -j4
         make install

     -j4 is an option specifying that make can use 4 threads for parallel compilation.
     Depending on what versions of python you have it may be nessecary to add 
        -DPYTHON_EXECUTABLE:FILEPATH=/path/to/python3
     to the cmake line

     3. Add MUQ2 to the python path by adding following line to bashrc
        export PYTHONPATH="${PYTHONPATH}:/my/install/path/lib"

2. Pandas (optional for plotting) - https://pandas.pydata.org
Installation:
    1. From terminal
       pip3 install pandas or pip3 install pandas --user