#!/bin/bash
set -x
echo $PATH
cd $SRC_DIR
mkdir build
cd build
cmake \
    -DTEASERPP_PYTHON_VERSION=${PY_VER} \
    -DPython3_EXECUTABLE=${PYTHON} \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DLLVM_TOOLS_BINARY_DIR=${PREFIX}/bin \
    ..
make teaserpp_python
# make install does too much. We only need the regisration library.
cp pmc-build/libpmc.* ${PREFIX}/lib
cp teaser/libteaser_registration.* ${PREFIX}/lib
cd python
${PYTHON} -m pip install .
