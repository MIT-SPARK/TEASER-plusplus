.. _installation:

Installation
============

Supported Platforms
-------------------

TEASER++ has been tested on Ubuntu 18.04 with g++-7/9 and clang++-7/8/9.
You can also install it on Ubunutu 16.04, however you may need to install Eigen 3.3 manually from source.

For Python bindings, we recommend using Python 3.

Installing Dependencies
-----------------------

Building TEASER++ requires the following libraries installed:

1. A compiler that supports OpenMP. See `here <https://www.openmp.org/resources/openmp-compilers-tools/>`_ for a list.
2. CMake >= 3.10
3. Eigen3 >= 3.3
4. PCL >= 1.9 (optional)
5. Boost >= 1.58 (optional)

On Linux
^^^^^^^^

Run the following script to install all required dependencies:

.. code-block:: sh

   sudo apt install cmake libeigen3-dev libboost-all-dev

Run the following script to install PCL from source:

.. code-block:: sh

   # Compile and install PCL 1.91 from source
   PCL_PACKAGE_DIR="$HOME/pcl"
   mkdir "$PCL_PACKAGE_DIR"
   cd "$PCL_PACKAGE_DIR"
   wget "https://github.com/PointCloudLibrary/pcl/archive/pcl-1.9.1.zip"
   unzip pcl-*.zip
   rm pcl-*.zip
   cd pcl-* && mkdir build && cd build
   cmake ..
   make -j $(python3 -c 'import multiprocessing as mp; print(int(mp.cpu_count()    * 1.5))')
   sudo make install

Notice that PCL is not required for the TEASER++ registration library. Installing it merely allows you to build example tests that uses PCL's FPFH features for registration.

If you want to build Python bindings, you also need:

1. Python 2 or 3 (make sure to include the desired interpreter in your `PATH` variable)

If you want to build MATLAB bindings, you also need:

1. MATLAB
2. CMake >= 3.13

TEASER++ uses the Parallel Maximum Clique (`paper <https://arxiv.org/abs/1302.6256>`_, `code <https://github.com/ryanrossi/pmc>`_) for maximum clique calculation. It will be downloaded automatically during CMake configuration. In addition, CMake will also download Google Test and pybind11 if necessary.

On Ubuntu 16.04, you may need to install Eigen 3.3 manually. You can do so by following the official installation instructions.

On macOS
^^^^^^^^^^

First, you need to install ``homebrew`` by following instructions `here <https://brew.sh/>`_.

Run the following script to install dependencies:

.. code-block:: sh

   brew install eigen boost

On Windows
^^^^^^^^^^

Windows installation is not officially supported. However, you can try this repo `here <https://github.com/DrGabor/WinTeaser/>`_, courtesy of Di Wang at Xi'an Jiaotong University.
In addition, if you encounter errors regarding Eigen on Windows, take a look `here <https://github.com/zhongjingjogy/use-eigen-with-cmake>`_.

Compilation and Installation
----------------------------

Clone the repo to your local directory. Open a terminal in the repo root directory. Run the following commands:

.. code-block:: sh

   # Clone the repo
   git clone https://github.com/MIT-SPARK/TEASER-plusplus.git

   # Configure and compile
   cd TEASER-plusplus && mkdir build
   cd build
   cmake ..
   make

   # Generate doxygen documentation in doc/
   make doc

   # Run tests
   ctest

Installing C++ libraries and headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure you have compiled the project, then run:

.. code-block:: sh

   # Install shared libraries and headers
   sudo make install
   # Update links and cache to shared libraries
   sudo ldconfig

Installing Python bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TEASER++ uses `pybind11 <https://github.com/pybind/pybind11>`_ to allow for minimal-effort interoperability between C++ and Python. To compile Python binding, run the following in the ``build`` folder you just created:

.. code-block:: sh

   cmake -DTEASERPP_PYTHON_VERSION=3.6 ..
   make teaserpp_python

You can replace ``3.6`` with the desired Python version you want to use TEASER++ with.

Then, in the `build` folder, there should be a folder named ``python``. You can use the following commands to install the binding with ``pip``:

.. code-block:: sh

   cd python
   pip install .

If you are using virtual environments or Anaconda, make sure to activate your environment before compiling and during ``pip install``. Make sure the targeted Python interpreter is the one in your desired environment, or otherwise there might be segmentation faults.

Installing MATLAB Bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have MATLAB installed, you can optionally compile MATLAB bindings:

.. code-block:: sh

   cmake -DBUILD_MATLAB_BINDINGS=ON ..
   make

To use the compiled MATLAB bindings, just add the path to the generated mex file to your MATLAB script. Assuming your repo is located at ``/repos/TEASER-plusplus``, you can add the following to your MATLAB script:

.. code-block:: matlab

   addpath('/repos/TEASER-plusplus/build/matlab/')

Available CMake Options
-----------------------
Here are all available CMake options you can turn on/off during configuration:

+--------------------------+----------------------------------------+---------------+
| Option Name              | Description                            | Default Value |
+==========================+========================================+===============+
|`BUILD_TESTS`             | Build tests                            |  ON           |
+--------------------------+----------------------------------------+---------------+
|`BUILD_TEASER_FPFH`       | Build TEASER++ wrappers                |               |
|                          | for PCL FPFH estimation                | OFF           |
+--------------------------+----------------------------------------+---------------+
|`BUILD_MATLAB_BINDINGS`   | Build MATLAB bindings                  | OFF           |
+--------------------------+----------------------------------------+---------------+
|`BUILD_PYTHON_BINDINGS`   | Build Python bindings                  | ON            |
+--------------------------+----------------------------------------+---------------+
|`BUILD_DOC`               | Build documentation                    | ON            |
+--------------------------+----------------------------------------+---------------+
|`BUILD_WITH_MARCH_NATIVE` | Build with flag `march=native`         | OFF           |
+--------------------------+----------------------------------------+---------------+
|`ENABLE_DIAGNOSTIC_PRINT` | Enable printing of diagnostic messages | OFF           |
+--------------------------+----------------------------------------+---------------+

For example, if you want to build with the `march=native` flag (potentially faster at a loss of binary portability), run the following script for compilation:

.. code-block:: sh

   cmake -DBUILD_WITH_MARCH_NATIVE=ON ..
   make

Notice that by default the library is built in release mode. To build with debug symbols enabled, use the following commands:

.. code-block:: sh

   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make

Run Tests
---------

By default, the library is built in release mode. If you instead choose to build it in debug mode, some tests are likely to time out.

To run tests and benchmarks (for speed & accuracy tests), you can execute the following command after compilation:

.. code-block:: sh

   # Run all tests
   ctest

   # Run benchmarks
   ctest --verbose -R RegistrationBenchmark.*

The ``--verbose`` option allows you to see the output, as well as the summary tables generated by each benchmark.
