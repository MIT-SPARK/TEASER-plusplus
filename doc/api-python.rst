.. _api-python:

Python API
==========

The TEASER++ Python binding ``teaserpp-python`` uses `pybind11 <https://github.com/pybind/pybind11>`_ to allow for minimal-effort interoperability between C++ and Python. To use TEASER++ in Python, the following C++ constructs are exposed:

- ``teaser::RobustRegistrationSolver``: the main solver class for solving registration problems with TEASER++
- ``teaser::RobustRegistrationSolver::Params``: a struct for holding the initialization parameters for TEASER++
- ``teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM``: an enum for specifying what kind rotation estimation algorithm to use
- ``teaser::RegistrationSolution``: a struct holding the solution to a registration problem

Please refer to the C++ source code for more detailed documentation. Since Python bindings are directly bound to C++ functions, all the functionalities will be the same. For accessing the results, a general rule of thumb is to just replace all Eigen matrices with ``numpy`` matrices.
