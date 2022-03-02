from util.align import *
from sklearn.neighbors import KDTree
from util.metric import *

offset = -0.0
path = "/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/"
train_idx = [3]
test_idx = [1, 2]
seq_interval = 100000

rot_pos = 15  # degree
trans_pos = 0.2  # m


def generate():
    test_database = []
    test_positive = []

    for num, i in enumerate(test_idx):
        vicon_path = path + str(i).zfill(2) + "/vicon_" + str(i).zfill(2) + ".txt"
        est_path = path + str(i).zfill(2) + "/cameraStamps" + ".txt"
        gt_traj = np.loadtxt(vicon_path)
        est_TimeStamps = np.loadtxt(est_path)
        gt_traj_interp = getEveryImgFromVicon(est_TimeStamps, gt_traj)

        index = (np.arange(est_TimeStamps.shape[0]) + seq_interval * i).reshape(-1, 1)
        gt_traj_interp = np.hstack([index, gt_traj_interp])

        test_database.append(gt_traj_interp)

    test_database = np.concatenate(test_database, axis=0)
    print("database: ", test_database.shape[0])

    test_database_idx = test_database[:, 0]
    test_positive_trans = []
    test_database_trans = test_database[:, [4, 8, 12]]

    tree_trans = KDTree(test_database_trans, leaf_size=2)
    for i, img_idx in enumerate(test_database[:, 0]):
        query = test_database_trans[i, :].reshape(1, -1)
        ind = tree_trans.query_radius(query, r=trans_pos)
        ind = ind[0].ravel().tolist()
        # 去掉本身
        ind.remove(i)
        test_positive_trans.append(ind)

    for i, ind in enumerate(test_positive_trans):
        rot_i = test_database[i, [1, 2, 3, 5, 6, 7, 9, 10, 11]].reshape(3, 3)
        ind_filter_rot = []
        for j in ind:
            rot_j = test_database[j, [1, 2, 3, 5, 6, 7, 9, 10, 11]].reshape(3, 3)
            yaw_ij = relaDcmOnYaw(rot_i, rot_j)
            if abs(yaw_ij) < rot_pos:
                ind_filter_rot.append(j)

        # 第一个元素是被查询的图像编号，后面是回环图像编号
        test_positive_i = [test_database_idx[i]]
        for j in ind_filter_rot:
            test_positive_i.append(test_database_idx[j])
        test_positive.append(test_positive_i)

    print("test_positive: ", len(test_positive))

    with open("./test_positive_" + str(rot_pos) + "_" + str(trans_pos) + ".txt", 'w') as f:
        for test_positive_i in test_positive:
            line = str(test_positive_i)
            line = line[1:-1]
            line = line.replace(".0,", "")
            line = line.replace(".0", "")
            f.write(line + '\n')


def inter(a, b):
    return list(set(a) & set(b))


def testRecall():
    f = open("./result_dbow_pl.txt", "r")
    ret_dbow = f.readlines()
    ret_dbow = [list(map(int, line.split())) for line in ret_dbow]
    f.close()

    f = open("./test_positive_15_0.2.txt", "r")
    ret_gt = f.readlines()
    ret_gt = [list(map(int, line.split())) for line in ret_gt]
    f.close()

    N = 50
    recall = []
    for n in range(N):
        recall_n = []
        for i in range(len(ret_dbow)):
            # 去除第一列编号
            loop_id = ret_dbow[i][1:n + 2]
            loop_id_gt = ret_gt[i][1:]
            intersection = inter(loop_id, loop_id_gt)
            recall_n.append(len(intersection) * 100.0 / len(loop_id))
        recall.append(np.mean(recall_n))
    for n, recall_n in enumerate(recall):
        print(n + 1, recall_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="../../../../dataset/")
    parser.add_argument('--test', default=False, action="store_true")
    args = parser.parse_args()
    path = args.path
    if args.test:
        testRecall()
    else:
        generate()
