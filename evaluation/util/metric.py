#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import math
from scipy.spatial.transform import Rotation

rad2degree = 1 / math.pi * 180.0


def saveTum(file_name, traj):
    w = open(file_name, 'w')
    for i in range(traj.shape[0]):
        ss = str(traj[i, 0])
        for j in range(1, 8):
            ss = ss + " " + str(traj[i, j])
        w.write(ss + '\n')
    w.close()


def displacement(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def relaQuat(q1, q2):
    rotation = Rotation.from_quat(q1)
    r1 = rotation.as_matrix()
    rotation = Rotation.from_quat(q2)
    r2 = rotation.as_matrix()

    r = np.dot(np.linalg.inv(r1), r2)

    euler = Rotation.from_matrix(r).as_euler("zyx") / math.pi * 180.0

    return np.linalg.norm(euler)


def relaDcm(r1, r2):
    r = np.dot(np.linalg.inv(r1), r2)

    euler = Rotation.from_matrix(r).as_euler("zyx") / math.pi * 180.0

    return np.linalg.norm(euler)


def relaDcmOnYaw(r1, r2):
    yaw1 = math.atan2(r1[1, 0], r1[0, 0]) * rad2degree
    yaw2 = math.atan2(r2[1, 0], r2[0, 0]) * rad2degree

    # yaw -180-180, pitch , roll 因为是平面，几乎不变
    yaw = yaw1 - yaw2
    if yaw > 180:
        yaw = yaw - 360
    elif yaw < -180:
        yaw = yaw + 360

    return yaw


def tum2kitti(t, q):
    rotation = Rotation.from_quat(q)
    r = rotation.as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = r
    T[0:3, 3] = t
    return T


def tum2kitti2(tum):
    t = tum[:3]
    q = tum[3:]
    rotation = Rotation.from_quat(q)
    r = rotation.as_matrix()
    T = np.eye(4)
    T[0:3, 0:3] = r
    T[0:3, 3] = t
    return T


def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


def kitti2tum(kitti):
    dcm = kitti[:3, :3]
    t = kitti[:3, 3].tolist()
    q = Rotation.from_matrix(dcm).as_quat()
    tum = list(flat([t, q.tolist()]))
    return np.asarray(tum)


def tum_err(t1, q1, t2, q2):
    T1 = tum2kitti(t1, q1)
    T2 = tum2kitti(t2, q2)
    T = np.dot(np.linalg.inv(T1), T2)
    tum = kitti2tum(T)
    return tum[:3], tum[3:]


def se_err(pose1, pose2):
    T1 = tum2kitti(pose1)
    T2 = tum2kitti(pose2)

    deltaT = np.dot(np.linalg.inv(T1), T2)
    deltaTum = kitti2tum(deltaT)

    t = deltaTum[:3]
    q = deltaTum[3:]
    euler = Rotation.from_quat(q).as_euler("zyx") / math.pi * 180.0

    return t, euler


def quat2euler(quat):
    return Rotation.from_quat(quat).as_euler("zyx") / math.pi * 180.0


def dcm2euler(dcm, order="zyx"):
    return Rotation.from_matrix(dcm).as_euler(order) / math.pi * 180.0


def tum2simple(tum):
    return tum[:3], quat2euler(tum[3:])


def simpleSE3(T, order="zyx"):
    dcm = T[:3, :3]
    t = T[:3, 3]
    return dcm2euler(dcm, order=order).T, t.T


def evo_rpe(est_xyz, est_quat, gt_xyz, gt_quat):
    if est_xyz.shape[0] < est_xyz.shape[1]:
        est_xyz = est_xyz.transpose()
    if gt_xyz.shape[0] < gt_xyz.shape[1]:
        gt_xyz = gt_xyz.transpose()

    traj_len = est_xyz.shape[0]
    for i in range(1, traj_len):
        print(i)
        t_err, q_err = tum_err(est_xyz[i - 1], est_quat[i - 1], est_xyz[i], est_quat[i])
        print(t_err, quat2euler(q_err))
        t_err, q_err = tum_err(gt_xyz[i - 1], gt_quat[i - 1], gt_xyz[i], gt_quat[i])
        print(t_err, quat2euler(q_err))


def evo_ape_tum(est_xyz, est_quat, gt_xyz, gt_quat):
    if est_xyz.shape[0] < est_xyz.shape[1]:
        est_xyz = est_xyz.transpose()
    if gt_xyz.shape[0] < gt_xyz.shape[1]:
        gt_xyz = gt_xyz.transpose()

    traj_len = est_xyz.shape[0]
    trans_route = [0.0]
    rot_route = [0.0]
    trans_err_list = [0.0]
    rot_err_list = [0.0]
    for i in range(1, traj_len):
        trans_route.append(trans_route[i - 1] + displacement(est_xyz[i], est_xyz[i - 1]))
        rot_route.append(rot_route[i - 1] + relaQuat(est_quat[i], gt_quat[i - 1]))

        t_err, q_err = tum_err(est_xyz[i], est_quat[i], gt_xyz[i], gt_quat[i])
        # 走一段路程后再开始运算
        if trans_route[i] > 1.0:
            trans_err_list.append(np.linalg.norm(t_err) / trans_route[i] * 100)
        if rot_route[i] > 10.0:
            rot_err_list.append(np.linalg.norm(quat2euler(q_err)) / rot_route[i] * 100)

    return trans_err_list, rot_err_list


def evo_ape_se3(est_traj, gt_traj):
    traj_len = len(est_traj)
    trans_route = [0.0]
    rot_route = [0.0]
    trans_err_list = [0.0]
    rot_err_list = [0.0]

    i = 0
    last_key = 0
    est_xyz_last = np.zeros(3)
    est_dcm_last = np.eye(3)
    gt_xyz_last = np.zeros(3)
    gt_dcm_last = np.eye(3)
    for key in sorted(est_traj):
        est_xyz = est_traj[key][:3, 3]
        est_dcm = est_traj[key][:3, :3]
        gt_xyz = gt_traj[key][:3, 3]
        gt_dcm = gt_traj[key][:3, :3]

        trans_route.append(trans_route[i - 1] + displacement(est_xyz, est_xyz_last))
        rot_route.append(rot_route[i - 1] + relaDcm(est_dcm, est_dcm_last))

        t_err = est_xyz - gt_xyz
        euler_err = dcm2euler(np.dot(np.linalg.inv(est_dcm), gt_dcm))

        # 走一段路程后再开始运算
        if trans_route[i] > 2.0:
            # if trans_route[i] > 0.2:
            trans_err_list.append(np.linalg.norm(t_err) / trans_route[i] * 100)
        if rot_route[i] > 10.0:
            # if rot_route[i] > 5:
            rot_err_list.append(np.linalg.norm(euler_err) / rot_route[i] * 100)

        est_xyz_last = est_xyz
        est_dcm_last = np.eye(3)
        gt_xyz_last = np.zeros(3)
        gt_dcm_last = np.eye(3)
        i = i + 1

    return trans_err_list, rot_err_list


def evo_statics(trans_err_list, rot_err_list):
    trans_err_list.sort()
    rot_err_list.sort()

    trans_err_mean = np.mean(trans_err_list)
    trans_err_max = np.max(trans_err_list)
    trans_err_median = np.median(trans_err_list)

    rot_err_mean = np.mean(rot_err_list)
    rot_err_max = np.max(rot_err_list)
    rot_err_median = np.median(rot_err_list)

    return trans_err_mean, trans_err_max, trans_err_median, rot_err_mean, rot_err_max, rot_err_median
