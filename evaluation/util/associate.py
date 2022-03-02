#!/usr/bin/python
# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
The Kinect provides the color and depth images in an un-synchronized way. This means that the set of time stamps from the color images do not intersect with those of the depth images. Therefore, we need some way of associating color images to depth images.

For t script. It reads the time stamps from the rgb.txt file and the depth.txt file, and joins them by finding the best matches.
"""
# !/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import sys
import numpy
from util.metric import *
from util.plot_self import *


def npToDict(array, offset=0):
    dict_array = {}
    for i in range(array.shape[0]):
        dict_array[array[i, 0] + offset] = array[i, 1:]
    return dict_array


def dictToNp(dict):
    num = len(dict)
    array = numpy.zeros(shape=(num, 8))
    i = 0
    for key, value in dict.items():
        array[i, 0] = key
        array[i, 1:] = kitti2tum(value)
        i = i + 1
    return array


def read_file_list(filename, remove_bounds):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    if remove_bounds:
        lines = lines[100:-100]
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)


def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b - offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


def evaluateTUM(args):
    matches = associate(args.gt_traj, args.est_traj, 0, float(args.max_difference))
    if len(matches) < 2:
        sys.exit(
            "Couldn't find matching timestamp pairs between groundtruth and est_path trajectory! Did you choose the correct sequence?")
    gt_stamps = numpy.asarray([float(a) for a, b in matches]).transpose()
    gt_xyz = numpy.asarray([[float(value) for value in args.gt_traj[a][0:3]] for a, b in matches])
    gt_quat = numpy.asarray([[float(value) for value in args.gt_traj[a][3:]] for a, b in matches])

    est_stamps = numpy.asarray([float(b) for a, b in matches]).transpose()
    est_xyz = numpy.asarray([[float(value) * float(args.scale) for value in args.est_traj[b][0:3]] for a, b in matches])
    est_quat = numpy.asarray([[float(value) for value in args.est_traj[b][3:]] for a, b in matches])

    plot_slam_eval(est_stamps, est_xyz, gt_stamps, gt_xyz)

    trans_err_list, rot_err_list = evo_ape_tum(est_xyz, est_quat, gt_xyz, gt_quat)
    # plot_seq(trans_err_list)
    trans_err_mean, trans_err_max, trans_err_median, rot_err_mean, rot_err_max, rot_err_median = evo_statics(
        trans_err_list, rot_err_list)

    print("trans error max: ", trans_err_max)
    print("trans error mean: ", trans_err_mean)
    print("trans error median: ", trans_err_median)
    print("rot error max: ", rot_err_max)
    print("rot error mean: ", rot_err_mean)
    print("rot error median: ", rot_err_median)


def evaluateSE3(args):
    plot_slam_eval_SE3(args.est_traj, args.gt_traj)

    trans_err_list, rot_err_list = evo_ape_se3(args.est_traj, args.gt_traj)
    # plot_seq(trans_err_list)
    trans_err_mean, trans_err_max, trans_err_median, rot_err_mean, rot_err_max, rot_err_median = evo_statics(
        trans_err_list, rot_err_list)

    # print("trans error max: ", trans_err_max)
    print("trans error mean: ", trans_err_mean)
    print("trans error median: ", trans_err_median)
    # print("rot error max: ", rot_err_max)
    print("rot error mean: ", rot_err_mean)
    print("rot error median: ", rot_err_median)


if __name__ == '__main__':

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script takes two data files with timestamps and associates them   
    ''')
    parser.add_argument('first_file', help='first text file (format: timestamp data)')
    parser.add_argument('second_file', help='second text file (format: timestamp data)')
    parser.add_argument('--first_only', help='only output associated lines from first file', action='store_true')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    args = parser.parse_args()

    first_list = read_file_list(args.first_file)
    second_list = read_file_list(args.second_file)

    matches = associate(first_list, second_list, float(args.offset), float(args.max_difference))

    if args.first_only:
        for a, b in matches:
            print("%f %s" % (a, " ".join(first_list[a])))
    else:
        for a, b in matches:
            print("%f %s %f %s" % (a, " ".join(first_list[a]), b - float(args.offset), " ".join(second_list[b])))
