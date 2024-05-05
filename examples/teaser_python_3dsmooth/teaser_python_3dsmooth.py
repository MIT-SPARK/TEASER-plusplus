import os
import copy
import time
import line_mesh
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from timeit import default_timer as timer

import bench_utils
import teaserpp_python

VISUALIZE = True
NOISE_BOUND = 0.05
FRAG1_COLOR = [1, 0.3, 0.05]
FRAG2_COLOR = [0, 0.629, 0.9]
GT_COLOR = [1,1,0]
SPHERE_COLOR = [0,1,0.1]
SPHERE_COLOR_2 = [0.5,1,0.1]

def custom_draw_geometry_load_option(pcds, width=640, height=480):

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    for pcd in pcds:
        vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("./render_option.json")
    vis.run()
    vis.destroy_window()

def create_spheres(data, color=[0.1,0.6,0.2], radius=0.05):
    """
    Create a list of spheres from a 2D numpy array

    Numpy array needs to be N-by-3
    """
    vis_list = []
    for row in range(data.shape[0]):
        c_pt = data[row, :]
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color)
        mesh_sphere.translate(c_pt)
        vis_list.append(mesh_sphere)
    return vis_list

def visualize_correspondences(
    target_corrs_points, source_corrs_points, fragment1, fragment2, gt_inliers, translate=[-1.3,-1.5,0]
):
    """
    Helper function for visualizing the correspondences

    target is fragment1
    source is fragment2
    """
    TARGET_COLOR = [0.02, 0.551, 0.61]
    SOURCE_COLOR = [0.5, 0.5, 0.2]

    SCENE_COLOR = [0.02, 0.551, 0.61]

    INLIER_COLOR = [0, 0.9, 0.1]
    OUTLIER_COLOR = [1, 0.1, 0.1]

    temp_trans_dist = np.array([2, 2, 2])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_corrs_points)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_corrs_points)
    source.translate(translate)


    # create lineset
    outlier_set = []
    inlier_set = []
    # get inliers
    target_inlier_points = np.zeros([len(gt_inliers), 3])
    source_inlier_points = np.zeros([len(gt_inliers), 3])
    inlier_count = 0
    for i in range(target_corrs_points.shape[0]):
        if i in gt_inliers:
            inlier_set.append((i,i))
            target_inlier_points[inlier_count,:] = target_corrs_points[i,:]
            source_inlier_points[inlier_count,:] = np.asarray(source.points)[i,:]
            inlier_count+=1
        else:
            outlier_set.append((i,i))

    temp_target_points = np.asarray(target.points)
    temp_source_points = np.asarray(source.points)

    #target_spheres = create_spheres(target_inlier_points, color=[0.1, 0.1, 0.1], radius=0.01)
    #source_spheres = create_spheres(source_inlier_points, color=[0.1, 0.1, 0.1], radius=0.01)

    inlier_line_mesh = line_mesh.LineMesh(temp_target_points, temp_source_points, inlier_set, INLIER_COLOR, radius=0.012)
    inlier_line_mesh_geoms = inlier_line_mesh.cylinder_segments

    outlier_line_mesh = line_mesh.LineMesh(temp_target_points, temp_source_points, outlier_set, OUTLIER_COLOR, radius=0.001)
    outlier_line_mesh_geoms = outlier_line_mesh.cylinder_segments
    
    target.paint_uniform_color(TARGET_COLOR)
    source.paint_uniform_color(SOURCE_COLOR)

    frag1_temp = copy.deepcopy(fragment1)
    frag2_temp = copy.deepcopy(fragment2)

    frag1_temp.paint_uniform_color(FRAG1_COLOR)
    frag2_temp.paint_uniform_color(FRAG2_COLOR)
    frag2_temp.translate(translate)

    # estimate normals
    vis_list = [target, source, frag1_temp, frag2_temp]
    for ii in vis_list:
        ii.estimate_normals()
    vis_list.extend([*inlier_line_mesh_geoms, *outlier_line_mesh_geoms])
    #vis_list.extend(target_spheres)
    #vis_list.extend(source_spheres)

    custom_draw_geometry_load_option(vis_list)

def draw_registration_result(target_corrs_points, source_corrs_points, frag1, frag2, transformation, max_clique, gt=None, gt_inliers=None):

    frag1_temp = copy.deepcopy(frag1)
    frag2_temp = copy.deepcopy(frag2)

    frag1_temp.paint_uniform_color(FRAG1_COLOR)
    frag2_temp.paint_uniform_color(FRAG2_COLOR)

    frag1_temp.estimate_normals()
    frag2_temp.estimate_normals()

    frag2_temp.transform(transformation)

    inlier_spheres = []
    if max_clique is None:
        inlier_spheres = []
        target_inlier_points = np.zeros([0, 3])
    else:
        target_inlier_points = np.zeros([len(max_clique), 3])
        inlier_count = 0
        for i in range(target_corrs_points.shape[0]):
            if i in max_clique:
                target_inlier_points[inlier_count,:] = target_corrs_points[i,:]
                inlier_count+=1
        inlier_spheres = create_spheres(target_inlier_points, radius=0.3)

    vis_list = [frag1_temp, frag2_temp]
    if gt is not None:
        frag2_gt_temp = copy.deepcopy(frag2)
        frag2_gt_temp.paint_uniform_color(GT_COLOR)
        frag2_gt_temp.transform(gt)
        frag2_gt_temp.estimate_normals()

        gt_vis_list = [frag1_temp, frag2_gt_temp]
        # add gt inliers
        if gt_inliers is not None:
            gt_inlier_set = []
            gt_target_inlier_points = np.zeros([len(gt_inliers), 3])
            inlier_count = 0
            for i in range(target_corrs_points.shape[0]):
                if i in gt_inliers:
                    gt_inlier_set.append((i,i))
                    gt_target_inlier_points[inlier_count,:] = target_corrs_points[i,:]
                    inlier_count+=1

            gt_spheres = create_spheres(gt_target_inlier_points, radius=0.05)
            gt_vis_list.extend(gt_spheres)

        # ground truth alignment
        #print("Now showing ground truth alignment ...")
        #custom_draw_geometry_load_option(gt_vis_list)
        
        # TEASER++ alignment
        print("Now showing TEASER++ alignment ...")
        tpp_inlier_spheres = create_spheres(target_inlier_points, radius=0.04)
        vis_list.extend(tpp_inlier_spheres)
        custom_draw_geometry_load_option(vis_list, width=680, height=480)

        # together
        #print("Now showing GT & TEASER++ alignments ...")
        total_vis_list = vis_list
        total_vis_list.extend(gt_vis_list) 
        #custom_draw_geometry_load_option(total_vis_list)

def find_mutually_nn_keypoints(ref_key, test_key, ref, test):
    """
    Use kdtree to find mutually closest keypoints 

    ref_key: reference keypoints (source)
    test_key: test keypoints (target)
    ref: reference feature (source feature)
    test: test feature (target feature)
    """
    ref_features = ref.data.T
    test_features = test.data.T
    ref_keypoints = np.asarray(ref_key.points)
    test_keypoints = np.asarray(test_key.points)
    n_samples = test_features.shape[0]

    ref_tree = KDTree(ref_features)
    test_tree = KDTree(test.data.T)
    test_NN_idx = ref_tree.query(test_features, return_distance=False)
    ref_NN_idx = test_tree.query(ref_features, return_distance=False)

    # find mutually closest points
    ref_match_idx = np.nonzero(
        np.arange(n_samples) == np.squeeze(test_NN_idx[ref_NN_idx])
    )[0]
    ref_matched_keypoints = ref_keypoints[ref_match_idx]
    test_matched_keypoints = test_keypoints[ref_NN_idx[ref_match_idx]]

    return np.transpose(ref_matched_keypoints), np.transpose(test_matched_keypoints)


def execute_teaser_global_registration(source, target):
    """
    Use TEASER++ to perform global registration
    """
    # Prepare TEASER++ Solver
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    print("TEASER++ Parameters are:", solver_params)
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    # Solve with TEASER++
    start = timer()
    teaserpp_solver.solve(source, target)
    end = timer()
    est_solution = teaserpp_solver.getSolution()
    est_mat = bench_utils.compose_mat4_from_teaserpp_solution(est_solution)
    max_clique = teaserpp_solver.getTranslationInliersMap()
    print("Max clique size:", len(max_clique))
    final_inliers = teaserpp_solver.getTranslationInliers()
    return est_mat, max_clique, end - start

def pair_eval_helper(scene_path, desc_path, gt_mat, frag1_idx, frag2_idx):
    """
    Heper funtion for evaluating a pair in a scene
    Helper function for investigating matches between pairs
    """

    fragment1_name = "cloud_bin_" + str(frag1_idx)
    fragment2_name = "cloud_bin_" + str(frag2_idx)

    # load descriptors
    frag1_desc_file = os.path.join(
        desc_path, fragment1_name + ".ply_0.150000_16_1.750000_3DSmoothNet.npz"
    )
    frag1_desc = np.load(frag1_desc_file)
    frag1_desc = frag1_desc["data"]

    frag2_desc_file = os.path.join(
        desc_path, fragment2_name + ".ply_0.150000_16_1.750000_3DSmoothNet.npz"
    )
    frag2_desc = np.load(frag2_desc_file)
    frag2_desc = frag2_desc["data"]

    # save as o3d feature
    frag1 = o3d.pipelines.registration.Feature()
    frag1.data = frag1_desc.T

    frag2 = o3d.pipelines.registration.Feature()
    frag2.data = frag2_desc.T

    # load point clouds
    frag1_pc = o3d.io.read_point_cloud(
        os.path.join(scene_path, fragment1_name + ".ply")
    )
    frag2_pc = o3d.io.read_point_cloud(
        os.path.join(scene_path, fragment2_name + ".ply")
    )

    # load keypoints
    frag1_indices = np.genfromtxt(
        os.path.join(scene_path, "01_Keypoints/", fragment1_name + "Keypoints.txt")
    )
    frag2_indices = np.genfromtxt(
        os.path.join(scene_path, "01_Keypoints/", fragment2_name + "Keypoints.txt")
    )

    frag1_pc_keypoints = np.asarray(frag1_pc.points)[frag1_indices.astype(int), :]
    frag2_pc_keypoints = np.asarray(frag2_pc.points)[frag2_indices.astype(int), :]

    # Save as open3d point clouds
    frag1_key = o3d.geometry.PointCloud()
    frag1_key.points = o3d.utility.Vector3dVector(frag1_pc_keypoints)

    frag2_key = o3d.geometry.PointCloud()
    frag2_key.points = o3d.utility.Vector3dVector(frag2_pc_keypoints)

    ref_matched_key, test_matched_key = find_mutually_nn_keypoints(
        frag2_key, frag1_key, frag2, frag1
    )
    ref_matched_key = np.squeeze(ref_matched_key)
    test_matched_key = np.squeeze(test_matched_key)

    # TEASER++ registration
    # test: frag1
    # ref: frag2
    est_mat, max_clique, time = execute_teaser_global_registration(ref_matched_key, test_matched_key)

    # Plot point clouds after registration
    if VISUALIZE:
        gt_inliers = bench_utils.get_gt_inliers(test_matched_key.T, ref_matched_key.T, gt_mat)

        print("Now drawing correspondences ...")
        visualize_correspondences(test_matched_key.T, ref_matched_key.T, frag1_pc, frag2_pc, gt_inliers)

        print("Now drawing registration results ...")
        draw_registration_result(
            test_matched_key.T, ref_matched_key.T, frag1_pc, frag2_pc, est_mat, max_clique, gt=gt_mat, gt_inliers=gt_inliers
        )
        
        gt_clique_intersection = set(max_clique).intersection(gt_inliers)
        print("Max clique & gt inliers intersection:", gt_clique_intersection) 
        print("Max clique & gt inliers intersection len:", len(gt_clique_intersection))

    print("==================================================")
    print("Number of points: ", frag2_pc_keypoints.shape[0])
    print("Number of matched correspondences: ", ref_matched_key.shape[1])
    print("Time (s) for TEASER:", time)
    return est_mat

if __name__ == "__main__":
    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    scene_path = os.path.abspath("../example_data/3dmatch_sample/")
    desc_path = os.path.abspath("../example_data/3dmatch_sample/")
    gt_path = os.path.abspath("../example_data/3dmatch_sample/gt.log")
    pair = (2, 36)

    # load ground truth transformation
    gt_mat = bench_utils.load_gt_transformation(pair[0], pair[1], gt_path)
    est_mat = pair_eval_helper(
        scene_path, desc_path, gt_mat, pair[0], pair[1]
    )

    # compute errors
    rot_error, trans_error = bench_utils.compute_transformation_diff(est_mat, gt_mat)
    print("Rotation error (deg): ", rot_error)
    print("Translation error (m): ", trans_error)
