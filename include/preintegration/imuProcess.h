//
// Created by qzj on 2020/9/18.
//

#ifndef SRC_IMUPROCESS_H
#define SRC_IMUPROCESS_H

#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "include/orb_slam3/KeyFrame.h"
#include "Util/parameters.h"

namespace ORB_SLAM3 {
    class IntegrationBase;

    class ImageFrame;

    class KeyFrame;

    class ZUPT;

    enum MarginalizationFlag {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };
    enum SolverFlag {
        INITIAL,// 还未成功初始化
        NON_LINEAR // 已成功初始化，正处于紧耦合优化状态
    };

    void IntegrateOdo(const double &deltaT, Eigen::Vector2d &delatEncoder, const double &alpha, Eigen::Vector3d &accOdo,
                      cv::Mat &T_pre_cur);

    class IMUProcess {
    public:
        IMUProcess();

        bool getIMUInterval(double t0, double t1, std::vector<std::pair<double, Eigen::Vector3d>> &accVector,
                            std::vector<std::pair<double, Eigen::Vector3d>> &gyrVector,
                            std::vector<std::pair<double, cv::Point2f>> &encVector);

        void inputIMU(double t, const Eigen::Vector3d &linearAcceleration, const Eigen::Vector3d &angularVelocity,
                      const cv::Point2f &encoder);

        void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);

        void preIntegrateIMU(double img_t);

        bool IMUAvailable(double t);

        void initFirstIMUPose(std::vector<std::pair<double, Eigen::Vector3d>> &accVector);

        void setParameter();

        void processImage(const double header);

        void updateLatestStates();

        void slideWindow();

        void updateIMUBias();

        void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs);

        void setNewPoseFromORB3(list<cv::Mat> &mlRelativeFramePoses, list<KeyFrame *> &mlpReferences,
                                list<double> &mlFrameTimes, list<bool> &mlbLost);

        void updatePoseFromORB3(Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

        void processIMU(double t, double dt, const Eigen::Vector3d &linear_acceleration,
                        const Eigen::Vector3d &angular_velocity,
                        const cv::Point2f &encoder);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        std::queue<std::pair<double, Eigen::Vector3d>> accBuf;
        std::queue<std::pair<double, Eigen::Vector3d>> gyrBuf;
        std::queue<std::pair<double, cv::Point2f>> encoderBuf;

        IntegrationBase *tmp_pre_integration;//这个是输入到图像中的预积分值
        Eigen::Vector3d acc_0, gyr_0;
        double yawLast;
        cv::Mat mPreOdo;
        Eigen::Vector3d mOdoReceive;
        cv::Point2f encoderLast;
        Eigen::Matrix3d mRwc_orb[(WINDOW_SIZE + 1)];
        Eigen::Vector3d mtwc_orb[(WINDOW_SIZE + 1)];

        /***********滑动窗口*************/
        std::map<double, ImageFrame> all_image_frame;
        //窗口内所有帧的时间
        double Headers[(WINDOW_SIZE + 1)];
        //窗口中的dt,a,v
        vector<double> dt_buf[(WINDOW_SIZE + 1)];
        vector<Eigen::Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
        vector<Eigen::Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];
        IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];//里边放的是imu的预积分
        Eigen::Vector3d Ps[(WINDOW_SIZE + 1)];//划窗内所有的p
        Eigen::Vector3d Vs[(WINDOW_SIZE + 1)];//划窗内所有的速度
        Eigen::Matrix3d Rs[(WINDOW_SIZE + 1)];//划窗内所有的R
        Eigen::Vector3d Bas[(WINDOW_SIZE + 1)];//划窗内所有的bias of a
        Eigen::Vector3d Bgs[(WINDOW_SIZE + 1)];//划窗内所有的bias of g

        Eigen::Matrix3d R_init;
        Eigen::Matrix3d Rwi_IMU_Pre;
        Eigen::Vector3d Pwi_IMU_Pre;

        Eigen::Matrix3d ric[2];
        Eigen::Vector3d tic[2];

        int frame_count; //窗口内的第几帧,最大值为WINDOW_SIZE + 1
        double curTime, prevTime;
        double latest_time;
        Eigen::Vector3d g;
        Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
        Eigen::Quaterniond latest_Q;

        bool estBiasFirst;
        bool first_imu;//该图像之后的第一个imu
        bool initFirstPoseFlag;//IMU初始位姿的flag
        SolverFlag solver_flag;
        MarginalizationFlag marginalization_flag;

        //检测静止估计IMU零漂
        //Zero-velocity Update,ZUPT
        ZUPT *mZUPT;

    public:
        bool getPreOdoUpdate();

        void setPreOdoUpdate(bool preOdoUpdate);

        bool getIsUpdateBias();

        void setIsUpdateBias(bool isUpdateBias);

        void setOdoTpcDelta(cv::Mat odoTpcDelta);

        void updateOdoTpcDelta(cv::Mat T_pre_cur);

        cv::Mat getOdoTpcDelta();

    private:
        std::mutex mOdoUpdate;
        bool mbPreOdoUpdate;

        std::mutex mIsUpdateBias;
        bool mbIsUpdateBias;

        std::mutex mutexOdoTpcDelta;
        cv::Mat mOdoTpcDelta;

        std::mutex mProcess;
        std::mutex mBuf;
        std::mutex mPropagate;
        std::mutex mBiasUpdate;
    };

    class ZUPT {
    public:
        ZUPT(double STATIC_THRESHOLD = 0.3, int STATIC_JUDGE_NUM = 20, int STATIC_BUFFER_NUM = 500) : STATIC_THRESHOLD(
                STATIC_THRESHOLD),
                                                                                                      STATIC_JUDGE_NUM(
                                                                                                              STATIC_JUDGE_NUM),
                                                                                                      STATIC_BUFFER_NUM(
                                                                                                              STATIC_BUFFER_NUM) {

            //for(int i=0;i<STATIC_BUFFER_NUM;i++)
            //    wRawSlide.push(Eigen::Vector3d::Zero());
            staticCount = 0;
            mbBiasUpdate = false;
            wBiasSum.setZero();
            wBias.setZero();
            staticCountLast = 0;
            staticNum = 0;
        };

        void estimateBias(const Eigen::Vector3d &wRaw);

        bool CheckStatic(double wz);

        void setBias(const Eigen::Vector3d &bias) {
            unique_lock<std::mutex> lock(mutexBias);
            wBias = bias;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Eigen::Vector3d getBias() {
            unique_lock<std::mutex> lock(mutexBias);
            return wBias;
        }

        bool mbBiasUpdate;
    private:
        const double RAD_TO_ANGLE = 57.295779515;
        int STATIC_BUFFER_NUM;
        int STATIC_JUDGE_NUM;
        double STATIC_THRESHOLD;
        queue<Eigen::Vector3d> wRawSlide;
        unsigned int staticCount;
        unsigned int staticNum = 0;
        Eigen::Vector3d wBiasSum;
        unsigned int staticCountLast;

        std::mutex mutexBias;
        Eigen::Vector3d wBias;
    };

    class IntegrationBase {
    public:
        IntegrationBase() = delete;

        IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                        const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg);

        void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

        void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg);

        void midPointIntegration(double _dt,
                                 const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                                 const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                                 const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
                                 const Eigen::Vector3d &delta_v,
                                 const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                                 Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q,
                                 Eigen::Vector3d &result_delta_v,
                                 Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg,
                                 bool update_jacobian);

        void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1);

        Eigen::Matrix<double, 15, 1>
        evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
                 const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                 const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
                 const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj);

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        double dt;
        Eigen::Vector3d acc_0, gyr_0;
        Eigen::Vector3d acc_1, gyr_1;

        const Eigen::Vector3d linearized_acc, linearized_gyr;
        Eigen::Vector3d linearized_ba, linearized_bg;

        Eigen::Matrix<double, 15, 15> jacobian, covariance;
        Eigen::Matrix<double, 15, 15> step_jacobian;
        Eigen::Matrix<double, 15, 18> step_V;
        Eigen::Matrix<double, 18, 18> noise;

        double sum_dt;
        Eigen::Vector3d delta_p;
        Eigen::Quaterniond delta_q;
        Eigen::Vector3d delta_v;

        std::vector<double> dt_buf;
        std::vector<Eigen::Vector3d> acc_buf; //这与estimator里的acc_buf不同
        std::vector<Eigen::Vector3d> gyr_buf;
    };

    class ImageFrame {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        ImageFrame() {};

        ImageFrame(double _t) : t{_t}, is_key_frame{false} {
        };
        double t;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        IntegrationBase *pre_integration;
        bool is_key_frame;
    };

    void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs);

    bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Eigen::Vector3d *Bgs, Eigen::Vector3d &g,
                            Eigen::VectorXd &x);
}

#endif //SRC_IMUPROCESS_H
