.. _known_issues:

Known Issues
============
- If you are encountering segmentation faults from PMC, try add the environmental variable ``OMP_NUM_THREADS=${MAX_THREADS}`` (replace ``${MAX_THREADS}`` with the maximum number of threads available on your machine) in your current shell. You can also just prepend ``OMP_NUM_THREADS=${MAX_THREADS}`` when running your executable.
- When using the MATLAB wrapper with MATLAB on terminal (``-nojvm`` option enabled), you might encounter errors similar to this:

   .. code-block:: sh

      /usr/local/MATLAB/R2019a/bin/glnxa64/MATLAB: symbol lookup error: /opt/intel/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin/libmkl_vml_avx2.so: undefined symbol: mkl_serv_getenv.

   One way to get around this is to run the following command in the environment where you start MATLAB:

   .. code-block:: sh

      export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_gnu_thread.so:/opt/intel/mkl/lib/intel64/libmkl_core.so

   You may need to change the paths according to your MKL installation.
- If you see errors similar to:  ``./teaser_cpp_ply: error while loading shared libraries: libpmc.so: cannot open shared object file: No such file or directory``, you need to run `sudo ldconfig` after installation.
