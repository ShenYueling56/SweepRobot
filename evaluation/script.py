from util.plot_self import *

np.set_printoptions(suppress=True)


def check_IMU_CAM_By_baseline():
    T_c0_c1 = np.asarray([0.9999992350794548, 0.0012154557728128233, -2.2914573879299506E-4, 0.1197446053075405,
                          -0.001215825251510682, 0.9999979499237537, -0.0016192335371478666, -0.00012487470599842067,
                          2.2717716227647425E-4, 0.0016195108997383704, 0.9999986627865971, 0.0001870855595881091,
                          0., 0., 0., 1.]).reshape(4, 4)

    T_imu_c0 = np.asarray([-0.01078808, -0.0053516, -0.99992749, -0.0190819,
                           0.99993588, 0.0033851, -0.01080629, -0.04077015,
                           0.00344269, -0.99997995, 0.00531474, 0.10690639,
                           0., 0., 0., 1.]).reshape(4, 4)
    T_imu_c1 = np.asarray([-0.00633844, -0.00691795, -0.99995598, -0.02233173,
                           0.99997988, -0.00030443, -0.00633648, 0.07160075,
                           -0.00026058, -0.99997602, 0.00691974, 0.10088583,
                           0., 0., 0., 1.]).reshape(4, 4)

    T_c0_c1_est = np.dot(np.linalg.inv(T_imu_c0), T_imu_c1)
    T_imu_c1_new = np.dot(T_imu_c0, T_c0_c1)

    print(T_c0_c1)
    print(T_c0_c1_est)
    print(T_imu_c1)
    print(T_imu_c1_new)
    print(T_imu_c0)


if __name__ == "__main__":
    # check_IMU_CAM_By_baseline()
    acc = np.loadtxt('./acc.txt')


    def printMean(acc):
        print(np.mean(acc - acc[0]))


    acc_real = acc[:1500, 1]
    acc_odo = acc[:1500, 4] - 0.025
    printMean(acc_real)
    printMean(acc_odo)
    plot1d(acc_real, label='acc_real')
    plot1d(acc_odo, label='acc_odo')
    plt.legend()
    plt.show()
