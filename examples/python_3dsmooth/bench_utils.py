import os
import open3d as o3d
import numpy as np
import copy
import networkx as nx
import math
import csv
import pickle

NOISE_BOUND = 0.05
FRAG1_COLOR =[0, 0.651, 0.929]
FRAG2_COLOR = [1, 0.706, 0]
GT_COLOR =[0, 1, 0]

class EvalRegResult:
    def __init__(self, gt_mat, est_mat, max_clique, final_inliers, rot_error, trs_error):
        self.gt_mat = gt_mat 
        self.est_mat = est_mat 
        self.max_clique = max_clique 
        self.final_inliers = final_inliers 
        self.rot_error = rot_error 
        self.trs_error = trs_error 

def load_all_gt_pairs(gt_log_path):
    """
    Load all possible pairs from GT
    """
    with open(gt_log_path) as f:
        content = f.readlines()

    gt_data = {}
    for i in range(len(content)):
        tokens = [k.strip() for k in content[i].split(" ")]
        if len(tokens) == 3:
            frag1 = int(tokens[0])
            frag2 = int(tokens[1])

            def line_to_list(line):
                return [float(k.strip()) for k in line.split("\t")[:4]]

            gt_mat = np.array([line_to_list(content[i+1]),
                               line_to_list(content[i+2]), 
                               line_to_list(content[i+3]), 
                               line_to_list(content[i+4])])

            gt_data[(frag1, frag2)] = gt_mat

    return gt_data

def load_gt_transformation(fragment1_idx, fragment2_idx, gt_log_path):
    """
    Load gt transformation
    """
    with open(gt_log_path) as f:
        content = f.readlines()

    for i in range(len(content)):
        tokens = [k.strip() for k in content[i].split(" ")]
        if tokens[0] == str(fragment1_idx) and tokens[1] == str(fragment2_idx):

            def line_to_list(line):
                return [float(k.strip()) for k in line.split("\t")[:4]]

            gt_mat = np.array([line_to_list(content[i+1]),
                               line_to_list(content[i+2]), 
                               line_to_list(content[i+3]), 
                               line_to_list(content[i+4])])
            return gt_mat
    return None

def visualize_max_clique_inliers(target_corrs_points, source_corrs_points, max_clique, fragment1, fragment2, est_mat):
    """
    Helper function for visualizing the 

    target is fragment1
    source is fragment2
    """
    import pdb; pdb.set_trace()
    TARGET_COLOR = [0.02, 0.551, 0.61]
    SCENE_COLOR = [0.02, 0.551, 0.61]
    SOURCE_COLOR = [0.8, 0.1, 0.2]
    INLIER_COLOR = [0, 0.9, 0.1]
    OUTLIER_COLOR = [1, 0.1, 0.1]

    temp_trans_dist = np.array([2,2,2])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_corrs_points)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_corrs_points)
    source.transform(est_mat)
    
    # create lineset
    outlier_set = []
    inlier_set = []
    # get inliers
    target_inlier_points = np.zeros([len(max_clique), 3])
    source_inlier_points = np.zeros([len(max_clique), 3])
    inlier_count = 0
    for i in range(target_corrs_points.shape[0]):
        if i in max_clique:
            inlier_set.append((i,i))
            target_inlier_points[inlier_count,:] = target_corrs_points[i,:]
            source_inlier_points[inlier_count,:] = np.asarray(source.points)[i,:]
            inlier_count+=1
        else:
            outlier_set.append((i,i))
    
    inlier_line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(target, source, inlier_set)
    outlier_line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(target, source, outlier_set)

    inlier_line_set.paint_uniform_color(INLIER_COLOR)
    outlier_line_set.paint_uniform_color(OUTLIER_COLOR)
    target.paint_uniform_color(TARGET_COLOR)
    source.paint_uniform_color(SOURCE_COLOR)

    frag1_temp = copy.deepcopy(fragment1)
    frag2_temp = copy.deepcopy(fragment2)

    frag1_temp.paint_uniform_color(FRAG1_COLOR)
    frag2_temp.paint_uniform_color(FRAG2_COLOR)

    frag2_temp.transform(est_mat)

    o3d.visualization.draw_geometries(
        [target, source, inlier_line_set, outlier_line_set, frag1_temp, frag2_temp]
    )

    target_inlier_spheres = create_spheres(target_inlier_points)
    source_inlier_spheres = create_spheres(source_inlier_points,color=[0.5, 0.5, 0.5])

    vis_list = [target, source, inlier_line_set, frag1_temp, frag2_temp]
    vis_list.extend(target_inlier_spheres)
    vis_list.extend(source_inlier_spheres)
    o3d.visualization.draw_geometries(
            vis_list
    )

    
def get_gt_inliers(fragment1_points, fragment2_points, gt_mat):
    """
    Get GT inliers
    """
    # gt transformed source
    gt_fragment1 = o3d.geometry.PointCloud()
    gt_fragment1.points = o3d.utility.Vector3dVector(fragment2_points)
    gt_fragment1.transform(gt_mat)

    # Count inliers and outliers
    inliers_set = set()
    num_inliers = 0
    total = 0
    for i in range(fragment1_points.shape[0]):
        total += 1

        p1 = np.asarray(gt_fragment1.points)[i, :]
        p2 = fragment1_points[i, :]

        dist = np.linalg.norm(p1 - p2)
        if dist <= NOISE_BOUND:
            inliers_set.add(i)
            num_inliers += 1

    # GT Inliers
    print("GT Inliers count:", num_inliers)
    print("GT Inliers:", inliers_set)
    return inliers_set

def eval_errors(fragment1_points, fragment2_points, fragment1, fragment2, gt_mat, est_mat, max_clique, inlier_adj_list=None, visualize=False):
    """
    Eval results
    """
    if visualize:
        visualize_max_clique_inliers(fragment1_points, fragment2_points, max_clique, fragment1, fragment2, est_mat)

    # gt transformed source
    gt_fragment1 = o3d.geometry.PointCloud()
    gt_fragment1.points = o3d.utility.Vector3dVector(fragment2_points)
    gt_fragment1.transform(gt_mat)

    # Count inliers and outliers
    inliers_set = set()
    num_inliers = 0
    total = 0
    for i in range(fragment1_points.shape[0]):
        total += 1

        p1 = np.asarray(gt_fragment1.points)[i, :]
        p2 = fragment1_points[i, :]

        dist = np.linalg.norm(p1 - p2)
        if dist <= NOISE_BOUND:
            inliers_set.add(i)
            num_inliers += 1

    # GT Inliers
    print("GT Inliers count:", num_inliers)
    print("GT Inliers:", inliers_set)

    # TEASER++ max clique
    print("MAX CLIQUE SIZE: ", len(max_clique))
    print("MAX CLIQUE : ", max_clique)


    # find intersection
    if not inlier_adj_list:
        temp_set_1 = inliers_set
        temp_set_2 = set(max_clique)
        print("Intersection size:", len(temp_set_1.intersection(temp_set_2)))
        print("Intersection:", temp_set_1.intersection(temp_set_2))
        if len(temp_set_1.intersection(temp_set_2)) != 0:
            print("Difference size:", len(temp_set_1 - temp_set_2))
            print("Difference:", temp_set_1 - temp_set_2)
        else:
            print("No intersection! \nOther maximal cliques (that have GT inliers as a subset).")
            # Find all maximal cliques in inlier graph
            temp_inlier_dict = {}
            for i in range(len(inlier_adj_list)):
                temp_inlier_dict[i] = inlier_adj_list[i]
            inlier_graph=nx.Graph(temp_inlier_dict)
            all_cliques = nx.algorithms.clique.find_cliques(inlier_graph)
            for clique in all_cliques:
                clique_set = set(clique)
                if inliers_set <= clique_set:
                    print("Clique:", clique)

            print("Other maximal cliques (that are subsets of GT inliers).")
            for clique in all_cliques:
                clique_set = set(clique)
                if clique_set <= inliers_set:
                    print("Clique:", clique)

    return

def compose_mat4_from_teaserpp_solution(solution):
    """
    Compose a 4-by-4 matrix from teaserpp solution
    """
    s = solution.scale
    rotR = solution.rotation
    t = solution.translation
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = rotR
    M = T.dot(R)

    if s == 1:
        M = T.dot(R)
    else:
        S = np.eye(4)
        S[0:3, 0:3] = np.diag([s, s, s])
        M = T.dot(R).dot(S)

    return M

def draw_registration_result(frag1, frag2, transformation, gt=None):

    frag1_temp = copy.deepcopy(frag1)
    frag2_temp = copy.deepcopy(frag2)

    frag1_temp.paint_uniform_color(FRAG1_COLOR)
    frag2_temp.paint_uniform_color(FRAG2_COLOR)

    frag2_temp.transform(transformation)
    if gt is not None:
        frag2_gt_temp = copy.deepcopy(frag2)
        frag2_gt_temp.paint_uniform_color(GT_COLOR)
        frag2_gt_temp.transform(gt)
        o3d.visualization.draw_geometries([frag1_temp, frag2_temp, frag2_gt_temp])
    else:
        o3d.visualization.draw_geometries([frag1_temp, frag2_temp])

def create_spheres(data, color=[0.7, 0.1, 0.1], radius=0.05):
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


def get_angular_error(R_gt, R_est):
    """
    Get angular error
    """
    try:
        A = (np.trace(np.dot(R_gt.T, R_est))-1) / 2.0
        if A < -1:
            A = -1
        if A > 1:
            A = 1
        rotError = math.fabs(math.acos(A));
        return math.degrees(rotError)
    except ValueError:
        import pdb; pdb.set_trace()
        return 99999

def compute_transformation_diff(est_mat, gt_mat):
    """
    Compute difference between two 4-by-4 SE3 transformation matrix
    """
    R_gt = gt_mat[:3,:3]
    R_est = est_mat[:3,:3]
    rot_error = get_angular_error(R_gt, R_est)

    t_gt = gt_mat[:,-1]
    t_est = est_mat[:,-1]
    trans_error = np.linalg.norm(t_gt - t_est)

    return rot_error, trans_error

def dump_scene_results(scene, result, file_path="./data/teaserpp_results"):
    """
    Dump scene result
    """
    fp = os.path.join(file_path, scene+"_tpp_results.csv")
    fp_b = os.path.join(file_path, scene+"_tpp_results.bin") 
    pickle.dump(result, open(fp_b, "wb"))

    with open(fp, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['frag1','frag2','v1_result_rot_error','v2_result_rot_error','v3_result_rot_error', 'v1_result_trs_error','v2_result_trs_error','v3_result_trs_error'])
        for key, values in result.items():
            # key is pair of scans, values are 3 different scan results
            writer.writerow([key[0], key[1], values[0].rot_error, values[1].rot_error, values[2].rot_error, values[0].trs_error, values[1].trs_error, values[2].trs_error])

    return
