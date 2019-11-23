#!/usr/bin/env bash

# Install libjpeg-turbo
sudo apt install -y nasm
sudo apt install -y cmake
sudo apt install -y libsm6 libxext6 libxrender-dev

wget https://downloads.sourceforge.net/libjpeg-turbo/libjpeg-turbo-2.0.3.tar.gz
tar xvf libjpeg-turbo-2.0.3.tar.gz &&
cd libjpeg-turbo-2.0.3 &&

mkdir build &&
cd    build &&

cmake -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_BUILD_TYPE=RELEASE  \
      -DENABLE_STATIC=FALSE       \
      -DCMAKE_INSTALL_DOCDIR=/usr/share/doc/libjpeg-turbo-2.0.3 \
      -DCMAKE_INSTALL_DEFAULT_LIBDIR=lib  \
      .. &&
make
make install

