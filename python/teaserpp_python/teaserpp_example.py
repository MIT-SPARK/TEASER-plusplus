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

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = 0.01
    solver_params.estimate_scaling = True
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    print("Parameters are:", solver_params)

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    solver.solve(src, dst)

    solution = solver.getSolution()

    # Print the solution
    print("Solution is:", solution)

    # Print the inliers
    scale_inliers = solver.getScaleInliers()
    scale_inliers_map = solver.getScaleInliersMap()
    translation_inliers = solver.getTranslationInliers()
    translation_inliers_map = solver.getTranslationInliersMap()

    print("=======================================")
    print("Scale inliers (TIM pairs) are:")
    print("Note: they should not include the outlier points.")
    for i in range(len(scale_inliers)):
        print(scale_inliers[i], end=',')
    print("\n=======================================")

    print("Translation inliers are:", translation_inliers)
    print("Translation inliers map is:", translation_inliers_map)
