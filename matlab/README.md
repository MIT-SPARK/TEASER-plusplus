# TEASER++ MATLAB Binding 
This short document will show you how to use TEASER++'s MATLAB binding to solver 3D registration problems.

## Introduction
The TEASER++ MATLAB binding provides two ways for users to interface with the TEASER++ library:
- `teaser_solve_mex`: a MEX function that calls the TEASER++ library directly.
- `teaser_solve`: a MATLAB function that wraps around the MEX function.

### teaser_solve_mex
Input arguments:
- `src`: a 3-by-N matrix of 3D points representing points to be transformed
- `dst`: a 3-by-N matrix of 3D points representing points after transformation
- `cbar2`: square of maximum allowed ratio between noise and noise bound (see [1]).
- `noise_bound`: a floating-point number indicating the bound on noise
- `estimate_scaling`: a boolean indicating whether scale needs to be estimated
- `rotation_max_iterations`: maximum iterations for the rotation estimation loop
- `rotation_cost_threshold`: cost threshold for rotation termination
- `rotation_gnc_factor`: gnc factor for rotation estimation
                       for GNC-TLS method: it's multiplied on the GNC control parameter
                       for FGR method: it's divided on the GNC control parameter
- `rotation_estimation_method`: a number indicating the rotation estimation method used;
                             if it's 0: GNC-TLS
                             if it's 1: FGR
                             
Output:
- `s_est` estimated scale (scalar)
- `R_est` estimated rotation matrix (3-by-3 matrix)
- `t_est` estimated translation vector (3-by-1 matrix)
- `time_taken` time it takes for the underlying TEASER++ library to compute a solution.
 
### teaser_solve
Input arguments:
- `src`: 3-by-N point cloud (before transformation)
- `dst`: 3-by-N point cloud (after transformation)

Input parameters (name-value pairs): 

- `Cbar2`: square of maximum ratio between noise and noise bound [1].
- `NoiseBound`: maximum bound on noise
- `EstimateScaling`: true if scale is not known, false otherwise
- `RotationEstimationAlgorithm`: 0 for GNC-TLS, 1 for FGR
- `RotationGNCFactor`: factor for increasing/decreasing the GNC function control parameter
- `RotationMaxIterations`: maximum iterations for the FGR loop
- `RotationCostThreshold`: cost threshold for FGR termination

Outputs:
- `s`: estimated scale
- `R`: estimated rotation matrix (3-by-3)
- `t`: estimated 3D translational vector (3-by-1)
- `time_taken`: time it takes for the TEASER++ library to compute a solution in milliseconds.

For more information, please refer to the comments in the source code directly.

## Estimate a registration problem with known scale 
Assume we have `src` and `dst`, two 3-by-N matrices. And we know that `dst = R * src + t + e`, where `e` is bounded within 0.01. The following is a snippet of how you can use TEASER++ to solve it.
```matlab
cbar2 = 1;
noise_bound = 0.01;
estimate_scaling = false; % we know there's no scale difference
rot_alg = 0; % use GNC-TLS, set to 1 for FGR
rot_gnc_factor = 1.4;
rot_max_iters = 100;
rot_cost_threshold = 1e-12;

% The MEX function version
[s, R, t, time_taken] = teaser_solve_mex(src, dst, cbar2, ...
        noise_bound, estimate_scaling, rot_alg, rot_gnc_factor, ...
        rot_max_iters, rot_cost_threshold);

% The MATLAB wrapper version
[s, R, t, time_taken] = teaser_solve(src, dst, 'Cbar2', cbar2, 'NoiseBound', noise_bound, ...
                                     'EstimateScaling', estimate_scaling, 'RotationEstimationAlgorithm', rot_alg, ...
                                     'RotationGNCFactor', rot_gnc_factor, 'RotationMaxIterations', 100, ...
                                     'RotationCostThreshold', rot_cost_threshold);
```

## Estimate a registration problem with unknown scale
Assume we have `src` and `dst`, two 3-by-N matrices. And we know that `dst = s * R * src + t + e`, where `e` is bounded within 0.01. The following is a snippet of how you can use TEASER++ to solve it.
```matlab
cbar2 = 1;
noise_bound = 0.01;
estimate_scaling = true; 
rot_alg = 0; % use GNC-TLS, set to 1 to use FGR
rot_gnc_factor = 1.4;
rot_max_iters = 100;
rot_cost_threshold = 1e-12;

% The MEX function version
[s, R, t, time_taken] = teaser_solve_mex(src, dst, cbar2, ...
        noise_bound, estimate_scaling, rot_alg, rot_gnc_factor, ...
        rot_max_iters, rot_cost_threshold);

% The MATLAB wrapper version
[s, R, t, time_taken] = teaser_solve(src, dst, 'Cbar2', cbar2, 'NoiseBound', noise_bound, ...
                                     'EstimateScaling', estimate_scaling, 'RotationEstimationAlgorithm', rot_alg, ...
                                     'RotationGNCFactor', rot_gnc_factor, 'RotationMaxIterations', 100, ...
                                     'RotationCostThreshold', rot_cost_threshold);
```
