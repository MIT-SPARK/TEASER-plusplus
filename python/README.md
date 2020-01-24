#Python Bindings of TEASER++ 
This short document will show you how to use TEASER++'s Python bindings to solver 3D registration problems.

## Introduction
The TEASER++ Python binding `teaserpp-python` uses pybind11 to allow for minimal-effort interoperability between C++ and Python. To use TEASER++ in Python, the following C++ constructs are exposed: 
- `teaser::RobustRegistrationSolver`: the main solver class for solving registration problems with TEASER++
- `teaser::RobustRegistrationSolver::Params`: a struct for holding the initialization parameters for TEASER++
- `teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM`: an enum for specifying what kind rotation estimation algorithm to use
- `teaser::RegistrationSolution`: a struct holding the solution to a registration problem

Please refer to the C++ source code for more detailed documentations.

## Installation
You can use `pip` to install `teaserpp-python`, the module that contains all TEASER++ bindings. First, make sure you have compiled `teaserpp-python`. If not, you can use the following commands:
```shell script
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DTEASERPP_PYTHON_VERSION=3.6 ..
make teaserpp_python -j4
```
You can replace `3.6` with the desired Python version you want to use TEASER++ with.

Then, in the `build` folder, there should be a folder named `python`. You can use the following commands to install the binding with `pip`:
```shell script
cd python
pip install .
```
If you are using virtual environments or Anaconda, make sure to activate your environment before compiling and during `pip install`. Make sure the targeted Python interpreter is the one in your desired environment, or otherwise there might be segmentation faults.

## An Example

```python
import numpy as np
import teaserpp_python

# Generate random data points
src = np.random.rand(3, 20)

# Apply arbitrary scale, translation and rotation
scale = 1.5
translation = np.array([[1], [0], [-1]])
rotation = np.array([[0.98370992, 0.17903344, -0.01618098],
                     [-0.04165862, 0.13947877, -0.98934839],
                     [-0.17486954, 0.9739059, 0.14466493]])
dst = scale * np.matmul(rotation, src) + translation

# Add two outliers
dst[:, 1] += 10
dst[:, 9] += 15

# Populate the parameters
solver_params = teaserpp_python.RobustRegistrationSolver.Params()
solver_params.cbar2 = 1
solver_params.noise_bound = 0.01
solver_params.estimate_scaling = True 
solver_params.rotation_estimation_algorithm = (
    teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
)
solver_params.rotation_gnc_factor = 1.4
solver_params.rotation_max_iterations = 100
solver_params.rotation_cost_threshold = 1e-12
print("TEASER++ Parameters are:", solver_params)
teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

solver = teaserpp_python.RobustRegistrationSolver(solver_params)
solver.solve(src, dst)

solution = solver.getSolution()

# Print the solution
print("Solution is:", solution)
```
