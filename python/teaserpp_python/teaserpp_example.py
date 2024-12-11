import numpy as np
import teaserpp_python

if __name__ == "__main__":
    print("===========================================")
    print("TEASER++ robust registration solver example")
    print("===========================================")

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


    params = teaserpp_python.RobustRegistrationSolverParams(
        cbar2=1,
        noise_bound=1,
        estimate_scaling=True,
        rotation_estimation_algorithm=teaserpp_python.RotationEstimationAlgorithm.GNC_TLS,
        rotation_gnc_factor=1.4,
        rotation_max_iterations=100,
        rotation_cost_threshold=1e-12
    )

    solver = teaserpp_python.RobustRegistrationSolver(*params)
    solver.solve(src, dst)


    print("Solution is:", solver.solution)
    print("=======================================")
    print("Scale inliers (TIM pairs) are:")
    print("Note: they should not include the outlier points.")
    for i in range(len(solver.scale_inliers)):
        print(solver.scale_inliers[i], end=',')
    print("\n=======================================")

    print("Translation inliers are:", solver.translation_inliers)
    print("Translation inliers map is:", solver.translation_inliers_map)
