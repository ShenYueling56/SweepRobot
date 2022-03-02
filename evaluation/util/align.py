#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys

sys.path.append('evaluation')
from util.associate import *
from util.metric import *
from liegroups.numpy import SE3
from util.plot_self import *

T_c0_robot = np.asarray([0.9998199476005879, 0.01896351998860963, 0.0006762319126025606, 0.04216179343179696,
                         0.0002432940438150989, 0.02282305763279319, -0.9997394904915473, 0.08578733170877284,
                         -0.01897401349125333, 0.9995596495206938, 0.02281433457504149, -0.06909713473004092,
                         0, 0, 0, 1]).reshape(4, 4)
T_robot_c0 = np.linalg.inv(T_c0_robot)

T_imu_c0 = np.asarray(
    [-0.01078808, -0.0053516, -0.99992749, -0.0190819, 0.99993588, 0.0033851, -0.01080629, -0.04077015, 0.00344269,
     -0.99997995, 0.00531474, 0.10690639, 0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
T_c0_imu = np.linalg.inv(T_imu_c0)

T_imu_robot = np.dot(T_imu_c0, T_c0_robot)
T_robot_imu = np.linalg.inv(T_imu_robot)

T_odo_imu = np.asarray([-1, 0, 0, 0.133, 0, -1, 0, 0.0, 0, 0, 1, 0.02564, 0., 0., 0., 1.]).reshape(4, 4)
T_imu_odo = np.linalg.inv(T_odo_imu)

T_odo_robot = np.dot(T_odo_imu, np.dot(T_imu_c0, T_c0_robot))
T_robot_odo = np.linalg.inv(T_odo_robot)

T_odo_c0 = np.dot(T_odo_imu, T_imu_c0)
T_c0_odo = np.linalg.inv(T_odo_c0)


def getTrajFromVICON(T_gv_lv, T_lv_lp):
    T_gv_lp = {}
    for key, value in T_gv_lv.items():
        T_gv_lp[key] = value.dot(T_lv_lp)
    return T_gv_lp


def getTrajFromSLAM(T_gp_lp, T_gv_gp):
    T_gv_lp = {}
    for key, value in T_gp_lp.items():
        T_gv_lp[key] = T_gv_gp.dot(value)
    return T_gv_lp


def trans_robot_slam(slamTraj):
    print("T_robot_c0", T_robot_c0)
    num, _ = slamTraj.shape
    for i in range(num):
        T = tum2kitti2(slamTraj[i, 1:])
        T = np.dot(T_robot_c0, T)
        T = np.dot(T, T_c0_robot)
        slamTraj[i, 1:] = np.asarray(kitti2tum(T))
    return slamTraj


def trans_robot_vio(slamTraj):
    num, _ = slamTraj.shape
    for i in range(num):
        T = tum2kitti2(slamTraj[i, 1:])
        T = np.dot(T_robot_imu, T)
        T = np.dot(T, T_imu_robot)
        slamTraj[i, 1:] = np.asarray(kitti2tum(T))

    T_origin = tum2kitti2(slamTraj[0, 1:])
    T_origin_inv = np.linalg.inv(T_origin)
    for i in range(num):
        T = tum2kitti2(slamTraj[i, 1:])
        T_ = np.dot(T_origin_inv, T)
        slamTraj[i, 1:] = np.asarray(kitti2tum(T_))

    return slamTraj


def trans_robot_vicon(viconTraj):
    T_vicon_robot = tum2kitti2(viconTraj[0, 1:])
    T_robot_vicon = np.linalg.inv(T_vicon_robot)
    T_vicon_c0 = np.dot(T_vicon_robot, T_robot_c0)
    print("T_vicon_c0", T_vicon_c0)
    # print(T_vicon_robot[:3,:3])
    # print(T_robot_vicon[:3,:3])
    # 返回欧拉角也是zyx顺序
    # print(Rotation.from_dcm(T_robot_vicon[:3,:3]).as_euler("zyx")/math.pi*180.0)
    # 返回欧拉角也是zyx顺序
    # print(Rotation.from_dcm(T_vicon_robot[:3,:3]).as_euler("zyx")/math.pi*180.0)
    num, _ = viconTraj.shape
    for i in range(num):
        try:
            T = tum2kitti2(viconTraj[i, 1:])
        except ValueError:
            print(viconTraj[i, 1:])
        T_ = np.dot(T_robot_vicon, T)
        # print(T[:3,3],T_[:3,3])
        # 返回欧拉角也是zyx顺序
        # print(Rotation.from_dcm(T[:3,:3]).as_euler("zyx")/math.pi*180.0)
        # T = np.dot(T, T_robot_vicon)
        viconTraj[i, 1:] = np.asarray(kitti2tum(T_))
    return viconTraj


def trans_robot_odometry(odometryTraj):
    num, _ = odometryTraj.shape
    for i in range(num):
        T = tum2kitti2(odometryTraj[i, 1:])
        T = np.dot(T_robot_odo, T)
        T = np.dot(T, T_odo_robot)
        odometryTraj[i, 1:] = np.asarray(kitti2tum(T))

    T_odo_origin = tum2kitti2(odometryTraj[0, 1:])
    T_origin_odo = np.linalg.inv(T_odo_origin)
    for i in range(num):
        try:
            T = tum2kitti2(odometryTraj[i, 1:])
        except ValueError:
            print(odometryTraj[i, 1:])
        T_ = np.dot(T_origin_odo, T)
        odometryTraj[i, 1:] = np.asarray(kitti2tum(T_))

    return odometryTraj


def alignTime(slamTraj, viconTraj):
    timeMax = slamTraj[-1, 0]
    timeMin = slamTraj[0, 0]
    num = viconTraj.shape[0]
    timeVicon = np.linspace(timeMin, timeMax, num)
    viconTraj[:, 0] = timeVicon.reshape(-1)
    return slamTraj, viconTraj


def getUsefulPart(traj, vicon=False):
    trans = traj[:, 1:4]
    num, _ = traj.shape;
    endIdx1 = 0
    if vicon:
        thresold = 0.00002
    else:
        thresold = 0.00055
    for i in range(num):
        egoMotion = np.sqrt(np.sum(np.square(trans[i + 1, :] - trans[i, :])))
        # print(egoMotion)
        if egoMotion < thresold:
            endIdx1 = i
        else:
            break
    traj = traj[endIdx1:]

    trans = traj[:, 1:4]
    num, _ = traj.shape;
    endIdx2 = num - 1
    for i in range(num):
        j = num - i - 1;
        egoMotion = np.sqrt(np.sum(np.square(trans[j - 1, :] - trans[j, :])))
        # print(egoMotion)
        if egoMotion < thresold:
            endIdx2 = j
        else:
            break
    traj = traj[:endIdx2]

    return traj, endIdx1, endIdx2


def align_and_interpolation_TUM(est_traj, gt_traj, offset=0, max_difference=0.02):
    est_traj, gt_traj_interp = align_and_interpolation_SE3(est_traj, gt_traj, offset=offset,
                                                           max_difference=max_difference)

    first_keys = list(est_traj.keys())

    for key in first_keys:
        if key in list(gt_traj_interp.keys()):
            est_traj[key] = kitti2tum(est_traj[key])
            gt_traj_interp[key] = kitti2tum(gt_traj_interp[key])
        else:
            del est_traj[key]

    return est_traj, gt_traj_interp


def align_and_interpolation_SE3(est_traj, gt_traj, max_difference=0.02):
    for key, value in est_traj.items():
        est_traj[key] = tum2kitti2(value)
    for key, value in gt_traj.items():
        gt_traj[key] = tum2kitti2(value)

    first_keys = list(est_traj.keys())
    second_keys = list(gt_traj.keys())

    potential_matches = []
    # 找到被估计轨迹每个点周围足够近的真实点
    for a in first_keys:
        potential_matches_single = []
        start = False
        for b in second_keys:
            if abs(a - b) < max_difference:
                start = True
                potential_matches_single.append([a, b, a - b])
            elif start == True:
                continue
        potential_matches.append(potential_matches_single)

    # 找到左右最近的
    matches = []
    for single in potential_matches:
        left_near = 0
        left_near_dis = 100
        right_near = 0
        right_near_dis = -100
        for a, b, difference in single:
            if difference > 0 and difference < left_near_dis:
                left_near = b
                left_near_dis = difference
            if difference < 0 and difference > right_near_dis:
                right_near = b
                right_near_dis = difference
        matches.append([a, left_near, left_near_dis, right_near, right_near_dis])

    # 插值
    gt_traj_interp = {}
    for a, left_near, left_near_dis, right_near, right_near_dis in matches:
        if left_near == 0 or right_near == 0:
            continue
        t = (a - left_near) / (right_near - left_near)
        delta = SE3.from_matrix(np.dot(np.linalg.inv(gt_traj[left_near]), gt_traj[right_near]))
        T = gt_traj[left_near].dot(SE3.exp(t * SE3.log(delta)).as_matrix())
        gt_traj_interp[a] = T

    for key in first_keys:
        if not key in list(gt_traj_interp.keys()):
            del est_traj[key]

    return est_traj, gt_traj_interp


def getEveryImgFromVicon(est_time, gt_traj):
    from sklearn.neighbors import NearestNeighbors
    dst = gt_traj[:, 0].reshape(-1, 1)
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(dst)

    matches = []
    for i, time in enumerate(est_time):
        # shape(1,2)
        indices = neigh.kneighbors(time.reshape(-1, 1), return_distance=False)
        matches.append([time, indices[0, 0], time - dst[indices[0, 0], 0], indices[0, 1], time - dst[indices[0, 1], 0]])

    gt_traj_dict = {}
    for i in range(gt_traj.shape[0]):
        gt_traj_dict[i] = tum2kitti2(gt_traj[i, 1:])

    # 插值
    gt_traj_interp = {}
    gt_traj_visual = []
    for num, match in enumerate(matches):
        a, left_near, left_near_dis, right_near, right_near_dis = match
        left = a - left_near_dis
        right = a - right_near_dis
        t = (a - left) / (right - left)
        delta = SE3.from_matrix(np.dot(np.linalg.inv(gt_traj_dict[left_near]), gt_traj_dict[right_near]))
        T = gt_traj_dict[left_near].dot(SE3.exp(t * SE3.log(delta)).as_matrix())
        gt_traj_interp[num] = T
        gt_traj_visual.append(T.ravel())

    gt_traj_visual = np.asarray(gt_traj_visual).reshape(-1, 16)

    # plot3d(gt_traj_visual[:,[3,7,11]])

    return gt_traj_visual


T_lv_lp_init = np.asarray([0.99967818, 0.01796308, 0.01791252, 0.03058822,
                           -0.01697347, 0.05113262, 0.99854762, 0.08129711,
                           0.0188529, -0.99853031, 0.05081127, 0.09255246,
                           0.0, 0.0, 0.0, 1.0]).reshape(4, 4)
T_gv_gp_init = np.asarray([0.9998446, 0.01666372, 0.00575364, 0.47558702,
                           0.00672279, 0.05869977, 0.99825304, 0.4705637,
                           0.01629687, 0.99813659, 0.05880268, 0.31821361,
                           0.0, 0.0, 0.0, 1.0]).reshape(4, 4)


def train_epochs_SE3(T_gp_lp, T_gv_lv, alignOdo=False):
    # init
    key_early = np.inf
    for key, value in T_gv_lv.items():
        if key < key_early:
            T_vicon_robot = value
    if alignOdo:
        T_init = T_odo_c0
    else:
        T_init = T_robot_c0
    T_vicon_c0 = np.dot(T_vicon_robot, T_init)
    T_gv_gp_opt = T_vicon_c0
    T_lv_lp_opt = T_init
    # T_gv_gp_opt = T_gv_gp_init
    # T_lv_lp_opt = T_lv_lp_init
    err_F = calErr(T_gv_gp_opt, T_gp_lp, T_gv_lv, T_lv_lp_opt)
    # T_gv_lp
    # plot_slam_eval_align(T_gv_gp_opt, T_gp_lp, T_gv_lv, T_lv_lp_opt)
    # print("best err: ",err_F)

    T_gv_gp, err_F = opt_T_gv_gp(T_gp_lp, T_gv_lv, T_lv_lp=T_lv_lp_opt)
    T_lv_lp, err_F = opt_T_lv_lp(T_gp_lp, T_gv_lv, T_gv_gp=T_gv_gp)

    # one epoch
    best_T_gv_gp = []
    best_T_lv_lp = []
    best_err_F = np.inf
    err_last = np.inf
    for i in range(100):
        T_gv_gp, err_stage1 = opt_T_gv_gp(T_gp_lp, T_gv_lv, T_lv_lp=T_lv_lp)
        T_lv_lp, err_F = opt_T_lv_lp(T_gp_lp, T_gv_lv, T_gv_gp=T_gv_gp)
        if err_F < best_err_F:
            best_err_F = err_F
            best_T_gv_gp = T_gv_gp
            best_T_lv_lp = T_lv_lp
        if (abs(err_F - err_last) < 0.1):
            break
        err_last = err_F
        # print("opt err_F: ",i," ",err_F)

    return best_T_gv_gp, best_T_lv_lp, best_err_F


def dictToNpTrans(traj):
    traj_np = [];
    traj_np = np.zeros(shape=(3, len(traj)))
    i = 0
    for key, value in traj.items():
        traj_np[0, i] = value[0, 3]
        traj_np[1, i] = value[1, 3]
        traj_np[2, i] = value[2, 3]
        i = i + 1
    return traj_np


def trans_use_umeyama(traj, r_a, t_a, s):
    T = np.eye(4)
    T[:3, :3] = r_a
    T[:3, 3] = t_a
    for key, value in traj.items():
        value[0, 3] = value[0, 3] * s
        value[1, 3] = value[1, 3] * s
        value[2, 3] = value[2, 3] * s
        value = np.dot(T, value)
        traj[key] = value
    return traj


def umeyama_alignment_orb(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise Exception("data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise Exception("Degenerate covariance rank, "
                        "Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def opt_T_gv_gp(T_gp_lp, T_gv_lv, T_lv_lp=T_robot_c0):
    num = len(T_gv_lv)
    T_gv_lp_arr = np.zeros(shape=(3, num))
    T_gp_lp_arr = np.zeros(shape=(3, num))
    i = 0
    for key, value in T_gv_lv.items():
        T_gv_lp_i = T_gv_lv[key].dot(T_lv_lp)
        T_gp_lp_i = T_gp_lp[key]
        T_gv_lp_arr[:, i] = T_gv_lp_i[:3, 3].reshape(3)
        T_gp_lp_arr[:, i] = T_gp_lp_i[:3, 3].reshape(3)
        i = i + 1

    T_gv_gp = optBySVD(T_gv_lp_arr, T_gp_lp_arr)
    # F范数误差
    err_F = calErr(T_gv_gp, T_gp_lp, T_gv_lv, T_lv_lp)

    return T_gv_gp, err_F


def opt_T_lv_lp(T_gp_lp, T_gv_lv, T_gv_gp=np.eye(4)):
    num = len(T_gv_lv)
    T_lv_gv_arr = np.zeros(shape=(3, num))
    T_lp_gv_arr = np.zeros(shape=(3, num))
    i = 0
    for key, value in T_gv_lv.items():
        T_lv_gv_i = np.linalg.inv(T_gv_lv[key])
        T_lv_gv_arr[:, i] = T_lv_gv_i[:3, 3].reshape(3)
        T_lp_gv_i = np.linalg.inv(T_gv_gp.dot(T_gp_lp[key]))
        T_lp_gv_arr[:, i] = T_lp_gv_i[:3, 3].reshape(3)
        i = i + 1

    T_lp_lv = optBySVD(T_lp_gv_arr, T_lv_gv_arr)
    T_lv_lp = np.linalg.inv(T_lp_lv)
    # F范数误差
    err_F = calErr(T_gv_gp, T_gp_lp, T_gv_lv, T_lv_lp)

    return T_lv_lp, err_F


# T_gv_gp*T_gp_lp_dict=T_gv_lv_dict*T_lv_lp
def calErr(T_gv_gp, T_gp_lp_dict, T_gv_lv_dict, T_lv_lp):
    # 获取keys
    T2_keys = list(T_gv_lv_dict.keys())
    err = 0
    for key, value in T_gp_lp_dict.items():
        if key in T2_keys:
            T_left = T_gv_gp.dot(T_gp_lp_dict[key])
            T_right = T_gv_lv_dict[key].dot(T_lv_lp)
            err = err + np.linalg.norm(T_left - T_right)
    return err


# T1_arr = T * T2_arr
def optBySVD(T1_arr, T2_arr):
    num = len(T1_arr)

    T1_avg = np.mean(T1_arr, axis=1, keepdims=True)
    T2_avg = np.mean(T2_arr, axis=1, keepdims=True)

    T1_arr = T1_arr - T1_avg
    T2_arr = T2_arr - T2_avg

    S = T2_arr.dot(T1_arr.T) / num

    u, s, vh = np.linalg.svd(S)
    v = vh.T
    w = v.dot(u.T)
    if (np.linalg.det(w) < 0):
        v[:, -1] = -v[:, -1]
        w = v.dot(u.T)
    opt_T = np.eye(4)
    opt_T[:3, :3] = w
    opt_T[:3, 3] = (T1_avg.reshape(3, 1) - opt_T[:3, :3].dot(T2_avg.reshape(3, 1))).reshape(3)

    return opt_T


def timeAlign(args):
    best_err = np.inf
    best_offset = 0
    interval = 0.5

    while (interval >= 0.0001):
        for offset in np.linspace(best_offset - interval, best_offset + interval, num=4):
            est_traj = npToDict(args.est_traj, offset=offset)
            gt_traj = npToDict(args.gt_traj)

            est_traj, gt_traj = align_and_interpolation_SE3(est_traj, gt_traj, args.max_difference)

            T_gv_gp, T_lv_lp, err = train_epochs_SE3(est_traj, gt_traj)

            if err < best_err:
                best_err = err
                best_offset = offset
        print("best_offset: ", best_offset, " best_err: ", best_err)

        interval = interval / 2.0

    return best_offset


if __name__ == "__main__":
    T = SE3.exp(np.zeros(shape=(6))).as_matrix()
    print(T)
    print(SE3.log(SE3.from_matrix(np.eye(4))))
    print(SE3.log(SE3.from_matrix(np.eye(4))) + 1)
