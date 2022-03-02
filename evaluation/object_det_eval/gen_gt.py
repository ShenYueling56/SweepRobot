# -*- coding:utf-8 -*-
# usage: gen_file_lists.py /path/to/data/

import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gen gt')
    parser.add_argument('root_dir', type=str, default="/home/shenyl/Documents/sweeper/data/", help='root_dir')
    args = parser.parse_args()

    root_dir = args.root_dir
    gt_path = root_dir + "object_det_result/"
    gt_file = gt_path + "gt.txt"
    dis_x = [0, -0.3, 0.3, -0.15, 0.15]
    dis_z = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

    num_class = 4
    num_dis_z = 14
    num_dis_x = 5
    # frame = 0
    with open(gt_file, 'w') as f:
        for c in range(num_class):
            for x in range(num_dis_x):
                for z in range(num_dis_z):
                    # f.write(str(frame))
                    # f.write(" ")
                    f.write(str(c))
                    f.write(" ")
                    f.write(str(dis_x[x]))
                    f.write(" ")
                    f.write(str(0))
                    f.write(" ")
                    f.write(str(dis_z[z]))
                    f.write(" ")
                    f.write("\n")
                    # frame=frame+1

