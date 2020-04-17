FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive

# essentials
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    apt-utils \
    libeigen3-dev \
    cmake \
    git \
    python3-dev \
    python3-pip \
    libboost-all-dev

# clone, build and install TEASER++
# ldconfig is needed because the installed .so files
# are in /usr/local/lib, which is not in the default Ubuntu
# linker search path (/lib)
# See: https://askubuntu.com/questions/350068/where-does-ubuntu-look-for-shared-libraries/
RUN git clone https://github.com/MIT-SPARK/TEASER-plusplus.git ~/teaser-plusplus
RUN cd ~/teaser-plusplus && mkdir build && cd build && \
      cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_PYTHON_BINDINGS=ON .. && \
      make -j$(nproc) && make install && ldconfig

# run an example
RUN cd ~/teaser-plusplus/examples/teaser_cpp_ply/ && mkdir build && cd build && \
      cmake .. && make && OMP_NUM_THREADS=$(nproc) ./teaser_cpp_ply
