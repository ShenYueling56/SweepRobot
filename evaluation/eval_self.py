# -*- coding: UTF-8 -*-
import sys
import time
import os
from util.plot_self import plotTraj
parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent)
# print(os.getcwd())
# print(sys.path)
from util.align import *

if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, default="01")
    parser.add_argument('--slam', type=str, default='orb3')
    parser.add_argument('--type_slam', type=str, default='slam')
    parser.add_argument('--gt_path', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default="")
    parser.add_argument('--end_time', default=0, type=float)
    parser.add_argument('--est_path', help='est_path trajectory (format: timestamp tx ty tz qx qy qz qw)', default="")
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument('--suffix', type=str, default="")
    parser.add_argument('--scaleAlign', default=False, action="store_true")
    parser.add_argument('--Align', type=bool, default=True)
    parser.add_argument('--offset', type=float,
                        help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.00)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.02 s)',
                        default=0.02)
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)',
                        default="re.pdf")
    parser.add_argument('--verbose',
                        help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)',
                        action='store_true')
    parser.add_argument('--odometry',
                        help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)',
                        action='store_true')
    args = parser.parse_args()

    if args.scaleAlign:
        args.suffix = args.suffix + " -s"
    if args.Align:
        args.suffix = args.suffix + " -a"

    path = '/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/{}'.format(args.seq)
    if args.gt_path == "":
        args.gt_path = os.path.join(path, "vicon_{}.txt".format(args.seq))
    if args.odometry:
        args.est_path = os.path.join(path, "odometry.txt")
    elif args.est_path == "":
        args.est_path = os.path.join(path, "robot{}_{}_stereo_{}.txt").format(args.seq, args.slam, args.type_slam)

    print("load " + str(args.est_path))
    args.est_traj = np.loadtxt(args.est_path)
    print("load " + str(args.gt_path))
    args.gt_traj = np.loadtxt(args.gt_path)

    # plotTraj(args.est_traj, args.gt_traj)

    # 1601012055.929288
    if (args.end_time > 0):
        est_traj = []
        gt_traj = []
        for i in range(args.gt_traj.shape[0]):
            if (args.gt_traj[i, 0] < args.end_time):
                gt_traj.append(args.gt_traj[i, :8])
        args.gt_traj = np.asarray(gt_traj).reshape(-1, 8)
        for i in range(args.est_traj.shape[0]):
            if (args.est_traj[i, 0] < args.end_time):
                est_traj.append(args.est_traj[i, :8])
        args.est_traj = np.asarray(est_traj).reshape(-1, 8)
    else:
        args.gt_traj = args.gt_traj[:, :8]
        args.est_traj = args.est_traj[:, :8]
    # 时间戳标定
    # args.offset = timeAlign(args)

    args.est_traj = npToDict(args.est_traj)
    args.gt_traj = npToDict(args.gt_traj, offset=args.offset)

    print('start align_and_interpolation_SE3')
    t1 = time.perf_counter()
    args.est_traj, args.gt_traj = align_and_interpolation_SE3(args.est_traj, args.gt_traj, args.max_difference)
    print('finish align_and_interpolation_SE3. Use ', time.perf_counter() - t1, " seconds")

    print('start train_epochs_SE3')
    t1 = time.perf_counter()
    if args.odometry:
        T_gv_gp, T_lv_lp, best_err_F = train_epochs_SE3(args.est_traj, args.gt_traj, alignOdo=True)
    else:
        T_gv_gp, T_lv_lp, best_err_F = train_epochs_SE3(args.est_traj, args.gt_traj)
    print('finish train_epochs_SE3. Use ', time.perf_counter() - t1, " seconds")
    # print("T_robot_c0", T_robot_c0)
    print("T_lv_lp", T_lv_lp)
    print("T_gv_gp", T_gv_gp)
    # print("T_robot_c0 euler,t: ", simpleSE3(T_robot_c0))
    # print("T_lv_lp euler,t:", simpleSE3(T_lv_lp))

    args.est_traj = getTrajFromSLAM(args.est_traj, T_gv_gp)
    args.gt_traj = getTrajFromVICON(args.gt_traj, T_lv_lp)

    r_a, t_a, s = umeyama_alignment_orb(dictToNpTrans(args.est_traj), dictToNpTrans(args.gt_traj))
    # print(r_a)
    # print(t_a)
    # print(s)
    s = 1.0
    args.est_traj = trans_use_umeyama(args.est_traj, r_a, t_a, s)

    evaluateSE3(args)

    if (False):
        args.est_traj = dictToNp(args.est_traj)
        eval_name = args.est_path[:-4] + "_BodyFrame.txt"
        saveTum(eval_name, args.est_traj)

        args.gt_traj = dictToNp(args.gt_traj)
        gt_file_new = args.gt_path[:-4] + "_BodyFrame.txt"
        saveTum(gt_file_new, args.gt_traj)

        print(eval_name)
        print(gt_file_new)

    if (False):
        # os.system(
        #     "evo_traj tum " + eval_name + " --save_plot result.pdf --ref " + gt_file_new )
        os.system(
            "evo_traj tum " + eval_name + " --ref " + gt_file_new)
        # angle_deg
        os.system("evo_ape tum " + gt_file_new + " " + eval_name + " -p -r trans_part" + args.suffix)
        os.system("evo_ape tum " + gt_file_new + " " + eval_name + " -r angle_deg" + args.suffix)
        # os.system("evo_rpe tum "+ gt_file_new+ " " + eval_name + suffix)
