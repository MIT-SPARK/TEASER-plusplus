src = rand(3,50);
dst = src + [1;0;1];

cbar2 = 1;
noise_bound = 0.01;
estimate_scaling = false;
rot_alg = 0;
rot_gnc_factor = 1.4;
rot_max_iters = 100;
rot_cost_threshold = 1e-12;
inlier_arg = 0;
kcore_thr = 0.5;

% Test the MEX function
[s, R, t, time_taken] = teaser_solve_mex(src, dst, cbar2, ...
        noise_bound, estimate_scaling, rot_alg, rot_gnc_factor, ...
        rot_max_iters, rot_cost_threshold, inlier_arg, kcore_thr);
assert(s==1);
assert(norm(R-eye(3)) < 1e-5);
assert(norm(t-[1;0;1]) < 1e-5);

% Test the MEX wrapper function
[s, R, t, time_taken] = teaser_solve(src, dst, 'Cbar2', cbar2, 'NoiseBound', noise_bound, ...
                                     'EstimateScaling', false, 'RotationCostThreshold', rot_cost_threshold);
assert(s==1);
assert(norm(R-eye(3)) < 1e-5);
assert(norm(t-[1;0;1]) < 1e-5);