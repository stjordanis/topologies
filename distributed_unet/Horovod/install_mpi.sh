wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
tar -xzf openmpi-*.tar.gz && cd openmpi-3.0.0
./configure --prefix=$1
make -j
sudo make install
