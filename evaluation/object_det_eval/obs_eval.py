import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gen gt')
    parser.add_argument('det_file', type=str, default="/media/shenyl/Elements/sweeper/dataset/exp1030/exp1030_0.7/result/", help='det_file')
    parser.add_argument('gt_file', type=str, default="/media/shenyl/Elements/sweeper/dataset/exp1030/exp1030_0.7/gt.txt", help='gt_file')


    args = parser.parse_args()
    det_file = args.det_file + "obs_boxes.txt"
    gt_file = args.gt_file
    result_file = args.det_file + "obs_det_precision.txt"

    f_gt = open(gt_file)
    f_det = open(det_file)


    x_gt_list = []
    z_gt_list = []
    w_gt_list = []
    h_gt_list = []

    frame_det_list = []
    x_det_list = []
    y_det_list = []
    z_det_list = []
    w_det_list = []
    h_det_list = []

    count_gt = 0
    for line_gt in f_gt.readlines():
        gt = line_gt.split(" ")

        x_gt, y_gt, z_gt, w_gt, h_gt = float(gt[0])/1000, float(gt[1])/1000, float(gt[2])/1000, float(gt[3])/1000, float(gt[4])/1000
        x_gt_list.append(x_gt)
        z_gt_list.append(z_gt)
        w_gt_list.append(w_gt)
        h_gt_list.append(h_gt)
        count_gt = count_gt + 1
        # print(h_gt)

    for line_det in f_det.readlines():
        det = line_det.split(" ")
        frame, x_det, y_det, z_det, w_det, h_det = int(det[0]),  float(det[1]), float(det[2]), float(det[3]), float(det[4]), float(det[5])
        if frame > 240:
            continue
        frame_det_list.append(frame)
        x_det_list.append(x_det)
        y_det_list.append(y_det)
        z_det_list.append(z_det)
        w_det_list.append(w_det)
        h_det_list.append(h_det)

    frame_list = []
    min_x_det_list = []
    min_y_det_list = []
    min_z_det_list = []
    min_w_det_list = []
    min_h_det_list = []
    last_frame = 0
    min_c_det = -1
    min_z_det = 3000
    min_x_det = 3000
    for i in range(len(frame_det_list)):
        frame = frame_det_list[i]
        if (frame==last_frame):
            if min_z_det> z_det_list[i]:
                min_x_det = x_det_list[i]
                min_y_det = y_det_list[i]
                min_z_det = z_det_list[i]
                min_w_det = w_det_list[i]
                min_h_det = h_det_list[i]

        else:
            min_x_det_list.append(min_x_det)
            min_y_det_list.append(min_y_det)
            min_z_det_list.append(min_z_det)
            min_w_det_list.append(min_w_det)
            min_h_det_list.append(min_h_det)
            frame_list.append(last_frame)
            min_x_det = x_det_list[i]
            min_y_det = y_det_list[i]
            min_z_det = z_det_list[i]
            min_w_det = w_det_list[i]
            min_h_det = h_det_list[i]
        last_frame = frame
    min_x_det_list.append(min_x_det)
    min_y_det_list.append(min_y_det)
    min_z_det_list.append(min_z_det)
    min_w_det_list.append(min_w_det)
    min_h_det_list.append(min_h_det)
    frame_list.append(last_frame)
    print("finish")

    e_x = 0
    e_z = 0
    e_w = 0
    e_h = 0

    for i in range(len(frame_list)):
        frame = frame_list[i]
        x_gt = x_gt_list[frame]
        z_gt = z_gt_list[frame]
        w_gt = w_gt_list[frame]
        h_gt = h_gt_list[frame]

        x_det = min_x_det_list[i]
        y_det = min_y_det_list[i]
        z_det = min_z_det_list[i]
        w_det = min_w_det_list[i]
        h_det = min_h_det_list[i]
        # print("det")
        # print(c_det)
        # print(str(x_det))
        # print(str(z_det))

        # e_x = e_x + abs(x_gt - x_det)/x_gt
        print(str(w_gt)+ " " + str(w_det) + " " + str(abs(w_gt - w_det)))
        e_z = e_z + abs(z_gt - z_det)/z_gt*100
        e_w = e_w + abs(w_gt - w_det)*100
        e_h = e_h + abs(h_gt - h_det)*100

        with open(result_file, 'a') as f:
            # f.write("e_x" + "\t" + "e_z" + "\t" + "e_w" + "\t" + "e_h")
            # f.write('\n')
            # f.write(str(e_x) + "\t" + str(e_z) + "\t" + str(e_w) + "\t" + str(e_h))
            f.write(str((abs(z_gt - z_det)/z_gt)*100) + "\t" + str(abs(w_gt - w_det)*100) + "\t" + str(abs(h_gt - h_det)*100)+"\n")

    # e_x = e_x / count_gt
    e_z = e_z / count_gt
    e_w = e_w / count_gt
    e_h = e_h / count_gt

    # print("e_x "+str(e_x))
    print("e_z "+str(e_z))
    print("e_w " + str(e_w))
    print("e_h " + str(e_h))

    # result = open(result_file, 'w')
    print(result_file)


