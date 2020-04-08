function [s, R, t, time_taken] = teaser_solve(src, dst, varargin)
%TEASER_SOLVE MATLAB Wrapper for using C++ implementation of TEASER to
%solve point cloud registration problems.
%
%   TEASER_SOLVE provides an easy-to-use MATLAB function interface for
%   users to solve point cloud registration problems in the form of:
%                       dst = s * R * src + t
%   where dst and src are 3-by-N matrices representing 3D point clouds, s
%   is a scalar representing scale, R is a 3-by-3 rotation matrix, and t is
%   a 3-by-1 translation vector.
%   
%   If scale is fixed, then users can pass an option to the function to
%   inform TEASER that scale estimation is not needed, i.e., solve 
%                       dst = R * src + t
%   instead. Doing so will speed up the execution significantly.
%
%   Input arguments:
%   - src: 3-by-N point cloud (before transformation)
%   - dst: 3-by-N point cloud (after transformation)
%   Input parameters: 
%   - Cbar2: square of maximum ratio between noise and noise bound [1].
%   - NoiseBound: maximum bound on noise
%   - EstimateScaling: true if scale is not known, false otherwise
%   - RotationEstimationAlgorithm: 0 for GNC-TLS, 1 for FGR
%   - RotationGNCFactor: factor for increasing/decreasing the GNC function control parameter
%   - RotationMaxIterations: maximum iterations for the FGR loop
%   - RotationCostThreshold: cost threshold for FGR termination
% 
%   Outputs:
%   - s: estimated scale
%   - R: estimated rotation matrix (3-by-3)
%   - t: estimated 3D translational vector (3-by-1)
%   - time_taken: time it takes for the TEASER++ library to compute a solution in milliseconds.
% 
%   For more information, please refer to [1]
%
%  [1] H. Yang, J. Shi, and L. Carlone, “TEASER: Fast and Certifiable Point Cloud Registration,”
%  arXiv:2001.07715 [cs, math], Jan. 2020.
%
%  Copyright 2020, Massachusetts Institute of Technology,
%  Cambridge, MA 02139
%  All Rights Reserved
%  Authors: Jingnan Shi, et al. (see THANKS for the full author list)
%  See LICENSE for the license information


assert(size(src, 1) == 3, 'src must be a 3-by-N matrix.')
assert(size(dst, 1) == 3, 'src must be a 3-by-N matrix.')

params = inputParser;
params.CaseSensitive = false;
addParameter(params, 'Cbar2', 1, ...
    @(x) isnumeric(x) && isscalar(x) && x>0 && x<=1);
addParameter(params, 'NoiseBound', 0.03, ...
    @(x) isnumeric(x) && isscalar(x));
addParameter(params, 'EstimateScaling', true, ...
    @(x) islogical(x) && isscalar(x));
addParameter(params,'RotationEstimationAlgorithm', 0, ...
    @(x) isnumeric(x) && isscalar(x));
addParameter(params,'RotationGNCFactor', 1.4, ...
    @(x) isnumeric(x) && isscalar(x) && x > 1);
addParameter(params,'RotationMaxIterations', 100, ...
    @(x) isnumeric(x) && isscalar(x));
addParameter(params,'RotationCostThreshold', 0.005, ...
    @(x) isnumeric(x) && isscalar(x));
addParameter(params,'InlierSelectionAlgorithm', 0, ...
    @(x) isnumeric(x) && isscalar(x));
addParameter(params,'KCoreHeuThreshold', 0.5, ...
    @(x) isnumeric(x) && isscalar(x));
parse(params, varargin{:});

[s, R, t, time_taken] = teaser_solve_mex(src, dst, params.Results.Cbar2, ...
        params.Results.NoiseBound, params.Results.EstimateScaling, ...
        params.Results.RotationEstimationAlgorithm, params.Results.RotationGNCFactor, ...
        params.Results.RotationMaxIterations, params.Results.RotationCostThreshold, ...
        params.Results.InlierSelectionAlgorithm, params.Results.KCoreHeuThreshold);
time_taken = time_taken / 1000;
end