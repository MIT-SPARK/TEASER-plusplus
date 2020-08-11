.. _api-matlab:

MATLAB API
==========
The TEASER++ MATLAB binding provides a wrapper function for users to interface with the TEASER++ library:
- ``teaser_solve``: a MATLAB function that wraps around a MEX function that creates a solver and calls the ``solve()`` function.

Here we will show you how to use TEASER++'s MATLAB binding to solver 3D registration problems.

teaser_solve
------------

Input arguments:

- ``src``: a 3-by-N matrix of 3D points representing points to be transformed
- ``dst``: a 3-by-N matrix of 3D points representing points after transformation (each column needs to correspond to the same column in ``src``)

Input parameters (name-value pairs):

- ``Cbar2``: square of maximum ratio between noise and noise bound (set to 1 by default).
- ``NoiseBound``: maximum bound on noise (depends on the data, default to 0.03).
- ``EstimateScaling``: true if scale needs to be estimated, false otherwise (default to true).
- ``RotationEstimationAlgorithm``: 0 for GNC-TLS, 1 for FGR (default to 0).
- ``RotationGNCFactor``: factor for increasing/decreasing the GNC function control parameter (default to 1.4):

   - for GNC-TLS method: it's multiplied on the GNC control parameter.
   - for FGR method: it's divided on the GNC control parameter.

- ``RotationMaxIterations``: maximum iterations for the GNC-TLS/FGR loop (default to 100).
- ``RotationCostThreshold``: cost threshold for FGR termination (default to 0.005).

Outputs:

- ``s``: estimated scale (scalar)
- ``R``: estimated rotation matrix (3-by-3)
- ``t``: estimated 3D translational vector (3-by-1)
- ``time_taken``: time it takes for the TEASER++ library to compute a solution in seconds.

For more information, please refer to the comments in the source code directly.

Examples
--------

Estimate a registration problem with known scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assume we have `src` and `dst`, two 3-by-N matrices. And we know that `dst = R * src + t + e`, where `e` is bounded within 0.01. The following is a snippet of how you can use TEASER++ to solve it.

.. code-block:: matlab

   cbar2 = 1;
   noise_bound = 0.01;
   estimate_scaling = false; % we know there's no scale difference
   rot_alg = 0; % use GNC-TLS, set to 1 for FGR
   rot_gnc_factor = 1.4;
   rot_max_iters = 100;
   rot_cost_threshold = 1e-12;

   % The MEX function version
   [s, R, t, time_taken] = teaser_solve_mex(src, dst, cbar2, ...
           noise_bound, estimate_scaling, rot_alg, rot_gnc_factor,    ...
           rot_max_iters, rot_cost_threshold);

   % The MATLAB wrapper version
   [s, R, t, time_taken] = teaser_solve(src, dst, 'Cbar2', cbar2,    'NoiseBound', noise_bound, ...
                                        'EstimateScaling',    estimate_scaling, 'RotationEstimationAlgorithm', rot_alg, ...
                                     'RotationGNCFactor', rot_gnc_factor, 'RotationMaxIterations', 100, ...
                                     'RotationCostThreshold', rot_cost_threshold);

Estimate a registration problem with unknown scale
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Assume we have `src` and `dst`, two 3-by-N matrices. And we know that `dst = s * R * src + t + e`, where `e` is bounded within 0.01. The following is a snippet of how you can use TEASER++ to solve it.

.. code-block:: matlab

   cbar2 = 1;
   noise_bound = 0.01;
   estimate_scaling = true;
   rot_alg = 0; % use GNC-TLS, set to 1 to use FGR
   rot_gnc_factor = 1.4;
   rot_max_iters = 100;
   rot_cost_threshold = 1e-12;

   % The MEX function version
   [s, R, t, time_taken] = teaser_solve_mex(src, dst, cbar2, ...
           noise_bound, estimate_scaling, rot_alg, rot_gnc_factor,    ...
           rot_max_iters, rot_cost_threshold);

   % The MATLAB wrapper version
   [s, R, t, time_taken] = teaser_solve(src, dst, 'Cbar2', cbar2,    'NoiseBound', noise_bound, ...
                                        'EstimateScaling',    estimate_scaling, 'RotationEstimationAlgorithm', rot_alg, ...
                                     'RotationGNCFactor', rot_gnc_factor, 'RotationMaxIterations', 100, ...
                                     'RotationCostThreshold', rot_cost_threshold);
