import os
import open3d as o3d
import numpy as np
import copy
import math

NOISE_BOUND = 0.05
FRAG1_COLOR =[0, 0.651, 0.929]
FRAG2_COLOR = [1, 0.706, 0]
GT_COLOR =[0, 1, 0]

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
