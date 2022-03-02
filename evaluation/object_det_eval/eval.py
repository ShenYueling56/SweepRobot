# -*- coding:utf-8 -*-
# usage: gen_file_lists.py /path/to/data/

import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gen gt')
    parser.add_argument('root_dir', type=str, default="/home/shenyl/Documents/sweeper/data/", help='root_dir')
    parser.add_argument('det_file_name', type=str, default="object_det_matlab_sgbm.txt", help='root_dir')

    args = parser.parse_args()
    root_dir = args.root_dir
    gt_path = root_dir
    gt_file = gt_path + "gt.txt"
    det_file = gt_path + args.det_file_name
    error_z_file = gt_path + "error_z.txt"
    error_x_file = gt_path + "error_x.txt"
    error_z0_file = gt_path + "error_z0.txt"
    error_x0_file = gt_path + "error_x0.txt"
    error_z1_file = gt_path + "error_z1.txt"
    error_x1_file = gt_path + "error_x1.txt"
    error_z2_file = gt_path + "error_z2.txt"
    error_x2_file = gt_path + "error_x2.txt"
    error_z3_file = gt_path + "error_z3.txt"
    error_x3_file = gt_path + "error_x3.txt"

    e_x_0 = 0
    e_z_0 = 0
    e_x_1 = 0
    e_z_1 = 0
    e_x_2 = 0
    e_z_2 = 0
    e_x_3 = 0
    e_z_3 = 0
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0

    count_det_0 = 0
    count_det_1 = 0
    count_det_2 = 0
    count_det_3 = 0

    f_gt = open(gt_file)
    f_det = open(det_file)
    f_error_z = open(error_z_file, 'w')
    f_error_x = open(error_x_file, 'w')
    f_error_z0 = open(error_z0_file, 'w')
    f_error_x0 = open(error_x0_file, 'w')
    f_error_z1 = open(error_z1_file, 'w')
    f_error_x1 = open(error_x1_file, 'w')
    f_error_z2 = open(error_z2_file, 'w')
    f_error_x2 = open(error_x2_file, 'w')
    f_error_z3 = open(error_z3_file, 'w')
    f_error_x3 = open(error_x3_file, 'w')
    c_gt_list = []
    x_gt_list = []
    z_gt_list = []
    frame_det_list = []
    c_det_list = []
    x_det_list = []
    z_det_list = []

    count_gt_0 = 0
    count_gt_1 = 0
    count_gt_2 = 0
    count_gt_3 = 0

    for line_gt in f_gt.readlines():
        gt = line_gt.split(" ")

        c_gt, x_gt, z_gt = int(gt[0]), float(gt[1]), float(gt[3])
        # print("gt")
        # print(c_gt)
        # print(str(x_gt))
        # print(str(z_gt))
        c_gt_list.append(c_gt)
        x_gt_list.append(x_gt)
        z_gt_list.append(z_gt)
        if z_gt > 1.4:
            continue
        if c_gt == 0:
            count_gt_0 = count_gt_0 + 1
        if c_gt == 1:
            count_gt_1 = count_gt_1 + 1
        if c_gt == 2:
            # if z_gt > 1.4:
            #     continue
            count_gt_2 = count_gt_2 + 1
        if c_gt == 3:
            count_gt_3 = count_gt_3 + 1

    for line_det in f_det.readlines():
        det = line_det.split(" ")
        frame, c,  x_det, z_det = int(det[0]),  int(det[2]), float(det[3]), float(det[5])
        frame_det_list.append(frame)
        c_det_list.append(c)
        x_det_list.append(x_det)
        z_det_list.append(z_det)


    frame_list = []
    min_c_det_list = []
    min_x_det_list = []
    min_z_det_list = []
    last_frame = 0
    min_c_det = -1
    min_z_det = 3000
    min_x_det = 3000
    for i in range(len(frame_det_list)-1):
        frame = frame_det_list[i]
        if frame > 240:
            continue
        if (frame==last_frame):
            if min_z_det> z_det_list[i]:
                min_c_det = c_det_list[i]
                min_z_det = z_det_list[i]
                min_x_det = x_det_list[i]
        else:
            min_c_det_list.append(min_c_det)
            min_x_det_list.append(min_x_det)
            min_z_det_list.append(min_z_det)
            frame_list.append(last_frame)
            min_c_det = c_det_list[i]
            min_z_det = z_det_list[i]
            min_x_det = x_det_list[i]
        last_frame = frame
    min_c_det_list.append(min_c_det)
    min_x_det_list.append(min_x_det)
    min_z_det_list.append(min_z_det)
    frame_list.append(last_frame)
    print("finish")

    for i in range(len(frame_list)):
        frame = frame_list[i]
        x_gt = x_gt_list[frame]
        z_gt = z_gt_list[frame]
        c_gt = c_gt_list[frame]
        c_det = min_c_det_list[i]
        x_det = min_x_det_list[i]
        z_det = min_z_det_list[i]
        # print("det")
        # print(c_det)
        # print(str(x_det))
        # print(str(z_det))

        if z_gt > 1.4:
            continue

        if(c_gt == 0):
            count_0 = count_0 + 1
            e_x_0 = e_x_0 + abs(x_gt - x_det)
            e_z_0 = e_z_0 + abs(z_gt - z_det)
            if(c_det == 0):
                count_det_0 = count_det_0 + 1
        if (c_gt == 1):
            count_1 = count_1 + 1
            e_x_1 = e_x_1 + abs(x_gt - x_det)
            e_z_1 = e_z_1 + abs(z_gt - z_det)
            if (c_det == 1):
                count_det_1 = count_det_1 + 1
        if (c_gt == 2):
            # if z_gt > 1.4:
            #     continue
            count_2 = count_2 + 1
            e_x_2 = e_x_2 + abs(x_gt - x_det)
            e_z_2 = e_z_2 + abs(z_gt - z_det)
            if (c_det == 2):
                count_det_2 = count_det_2 + 1
        if (c_gt == 3):
            count_3 = count_3 + 1
            e_x_3 = e_x_3 + abs(x_gt - x_det)
            e_z_3 = e_z_3 + abs(z_gt - z_det)
            if (c_det == 3):
                count_det_3 = count_det_3 + 1

        f_error_z.write(str(frame)+' '+str(c_gt) + ' ' + str(z_gt)+" "+str(z_det)+" "+str(z_det-z_gt)+" "+str((z_det-z_gt)/z_gt*100)+'\n')
        f_error_x.write(str(frame) + ' '+str(c_gt) + ' '  + str(x_gt) + " " + str(x_det) + " " + str(x_det - x_gt) + '\n')

    count_gt_0 = count_gt_0 -1
    count_gt_3 = count_gt_3

    f_error_z.close()
    f_error_x.close()
    f_error_z0.close()
    f_error_x0.close()
    f_error_z1.close()
    f_error_x1.close()
    f_error_z2.close()
    f_error_x2.close()
    f_error_z3.close()
    f_error_x3.close()
    e_z_0, e_x_0 = e_z_0 / count_0, e_x_0 / count_0
    e_z_1, e_x_1 = e_z_1 / count_1, e_x_1 / count_1
    e_z_2, e_x_2 = e_z_2 / count_2, e_x_2 / count_2
    e_z_3, e_x_3 = e_z_3 / count_3, e_x_3 / count_3
    print("////////////////obsatcle result//////////////////")
    print("e_z_0: " + str(e_z_0))
    print("e_z_1: " + str(e_z_1))
    print("e_z_2: " + str(e_z_2))
    print("e_z_3: " + str(e_z_3))
    print("e_x_0: " + str(e_x_0))
    print("e_x_1: " + str(e_x_1))
    print("e_x_2: " + str(e_x_2))
    print("e_x_3: " + str(e_x_3))
    print("count0: " + str(count_0) + " "+str(count_0/count_gt_0))   #58
    print("count1: " + str(count_1) + " "+str(count_1/count_gt_1))   #60
    print("count2: " + str(count_2) + " "+str(count_2/count_gt_2))   #59
    print("count3: " + str(count_3) + " "+str(count_3/count_gt_3))   #61
    count_all = count_0 + count_1 + count_2 + count_3
    print(str(count_all / (count_gt_0 + count_gt_1 + count_gt_2 + count_gt_3))) #239
    print("////////////////classification result//////////////////")
    print("count0: " + str(count_det_0) + " " + str(count_gt_0) + " " + str(count_det_0 / count_gt_0))
    print("count1: " + str(count_det_1) + " " + str(count_gt_1) + " " + str(count_det_1 / count_gt_1))
    print("count2: " + str(count_det_2) + " " + str(count_gt_2) + " " + str(count_det_2 / count_gt_2))
    print("count3: " + str(count_det_3) + " " + str(count_gt_3) + " " + str(count_det_3 / count_gt_3))
    count_det_all = count_det_0 + count_det_1 + count_det_2 + count_det_3
    print(str(count_det_all / (count_gt_0 + count_gt_1 + count_gt_2 + count_gt_3)))


