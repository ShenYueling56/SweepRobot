#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def plotTraj(traj1, traj2):

    fig = plt.figure()

    ax = getAxis(fig)

    ax.plot(traj1[:,1], traj1[:,2], traj1[:,3], '.', color="blue", label="estimation", markersize=1)
    ax.plot(traj2[:,1], traj2[:,2], traj2[:,3], '.', color="black", label="ground truth", markersize=1)

    set_aspect_equal_3d(ax)

    ax.legend(frameon=True)

    plt.show()

def set_aspect_equal_3d(ax):
    """
    kudos to https://stackoverflow.com/a/35126679
    :param ax: matplotlib 3D axes object
    """
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)

    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def getAxis(fig):
    ax = fig.add_subplot("111", projection="3d")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    return ax


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib.

    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend

    """
    stamps.sort()
    interval = np.median([s - t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    z = []
    last = stamps[0]
    # 中间可能有断开的，所以分段画
    for i in range(len(stamps)):
        if stamps[i] - last < 2 * interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
            z.append(traj[i][2])
        elif len(x) > 0:
            ax.plot(x, y, z, style, color=color, label=label)
            label = ""
            x = []
            y = []
            z = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, z, style, color=color, label=label)


def plot_seq(data, labels='box'):
    plt.figure()
    data = np.asarray(data).flatten()
    plt.plot(data)
    plt.show()


def plot_slam_eval(second_stamps, est_xyz, first_stamps, gt_xyz):
    fig = plt.figure()

    ax = getAxis(fig)

    plot_traj(ax, first_stamps, gt_xyz, '-', "black", "ground truth")
    plot_traj(ax, second_stamps, est_xyz, '-', "blue", "estimated")

    set_aspect_equal_3d(ax)

    ax.legend(frameon=True)

    plt.show()


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


def plot_slam_eval_SE3(est_traj, gt_traj):
    num = min([len(est_traj), len(est_traj)])
    first_stamps = np.zeros(shape=(num))
    gt_xyz = np.zeros(shape=(num, 3))
    second_stamps = np.zeros(shape=(num))
    est_xyz = np.zeros(shape=(num, 3))
    i = 0
    # 获取keys
    T2_keys = list(gt_traj.keys())
    for key, value in est_traj.items():
        if key in T2_keys:
            first_stamps[i] = key
            second_stamps[i] = key
            gt_xyz[i, :] = gt_traj[key][:3, 3].reshape(3)
            est_xyz[i, :] = est_traj[key][:3, 3].reshape(3)
            i = i + 1

    plot_slam_eval(second_stamps, est_xyz, first_stamps, gt_xyz)


def plot_slam_eval_align(T_gv_gp, T_gp_lp_dict, T_gv_lv_dict, T_lv_lp):
    num = min([len(T_gp_lp_dict), len(T_gv_lv_dict)])
    first_stamps = np.zeros(shape=(num))
    gt_xyz = np.zeros(shape=(num, 3))
    second_stamps = np.zeros(shape=(num))
    est_xyz = np.zeros(shape=(num, 3))
    i = 0
    # 获取keys
    T2_keys = list(T_gv_lv_dict.keys())
    for key, value in T_gp_lp_dict.items():
        if key in T2_keys:
            first_stamps[i] = key
            second_stamps[i] = key
            gt_xyz[i, :] = T_gv_gp.dot(T_gp_lp_dict[key])[:3, 3].reshape(3)
            est_xyz[i, :] = T_gv_lv_dict[key].dot(T_lv_lp)[:3, 3].reshape(3)
            i = i + 1

    plot_slam_eval(second_stamps, est_xyz, first_stamps, gt_xyz)


def plot1d(x, label=None):
    time = np.arange(max(x.shape))
    plt.plot(time, x, '-', label=label, markersize=1)


def format(pcl):
    if len(pcl.shape) == 3:  pcl = pcl[0];
    if pcl.shape[0] > pcl.shape[1]: pcl = pcl.T;
    return pcl


def plot3d(pcl1, s1=1, color1='blue'):
    pcl1 = format(pcl1)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pcl1[0], pcl1[1], pcl1[2], color=color1, s=s1)

    set_aspect_equal_3d(ax)

    plt.show()
