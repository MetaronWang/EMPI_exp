#Initial Folder
mkdir -p /data/host_data/EMPI_exp
ln -s /data/host_data/EMPI_exp ~/EMPI_exp
rm -rf ~/anaconda3/pkgs/ncurses-6.4-h6a678d5_0
rm -rf ~/anaconda3/pkgs/_openmp_mutex-5.1-1_gnu
conda create -n EMPI -q -y python=3.10

# Install Packages from apt
sudo apt install cmake make gcc g++ libeigen3-dev libssl-dev swig git libasio-dev

# Install Python Package
cd $HOME/EMPI_exp
conda activate EMPI
pip install -r requirements.txt

# Download and Install CMAKE
cd ~/EMPI_exp
wget https://cmake.org/files/v3.22/cmake-3.22.4.tar.gz
tar xf cmake-3.22.4.tar.gz
cd cmake-3.22.4
./bootstrap --parallel=48
make -j 128
sudo make install

# Download and Install PyBind11
cd ~/EMPI_exp
git clone https://github.com/pybind/pybind11.git
cd  pybind11
mkdir build
cd build
cmake ..
make check -j 128
sudo make install

# Download and Install Boost
cd ~/EMPI_exp
wget https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.gz
tar xf boost_1_84_0.tar.gz
cd boost_1_84_0
./bootstrap.sh
sudo ./b2 install --prefix=/usr toolset=gcc threading=multi

# Clear Current Boost TMP
sudo rm -r /usr/lib/x86_64-linux-gnu/cmake/Boost-1.74.0

# Compile CIMP Lib
sudo ldconfig
cd ~/EMPI_exp/src/cpp/com_imp
rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake Makefile CTestTestfile.cmake _deps COMIMP
cmake -DCMAKE_BUILD_TYPE=Release && make -j 128

# Clear Source Files
sudo rm -rf ~/EMPI_exp/boost_1_84_0* ~/EMPI_exp/cmake-3.22.4* ~/EMPI_exp/pybind11

# Initial CK
ck pull repo:ck-env
ck pull repo:ck-autotuning
ck pull repo:ctuning-programs
ck pull repo:ctuning-datasets-min

# Create Mem TMP for CK
sudo mkdir /tmp_ck
sudo chown -R ${USER} /tmp_ck
sudo mount -t tmpfs -o size=32G tmpfs /tmp_ck
rm -rf /tmp_ck/*
cp -r ~/EMPI_exp/data/dataset/compiler_args/CK ~/
cp -r ~/EMPI_exp/data/dataset/compiler_args/CK-TOOLS ~/
cp -r ~/EMPI_exp/data/dataset/compiler_args/* /tmp_ck/


get https://mirrors.aliyun.com/golang/go1.24.1.linux-amd64.tar.gz