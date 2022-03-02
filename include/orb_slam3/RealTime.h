//
// Created by warren on 2021/6/19.
//

#ifndef SWEEPSLAM_REALTIME_H
#define SWEEPSLAM_REALTIME_H

#include <vector>
#include <queue>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include "include/preintegration/imuProcess.h"
#include "System.h"
#include "Util/parameters.h"
#include "include/Util/se3_util.hpp"
#include "KeyFrame.h"
#include <cmath>

using namespace std;

typedef struct{
    double time = -1.0;
    cv::Point2f encoder = cv::Point2f(0.f, 0.f);
    double yaw = 0.0;
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}RawData;

typedef struct{
    double startTime = 0.0;
    double endTime = 0.0;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}RelPose_t;

class RealTime {
public:
    RealTime(const string &strSettingFile);

    void SetIMUProcessor(std::shared_ptr<ORB_SLAM3::IMUProcess> IMUProcessor) {mpIMUProcessor = IMUProcessor;};

    void SetVSLAM(std::shared_ptr<ORB_SLAM3::System> VSLAM) {mpVSLAM = VSLAM;};

    bool getLatestIMUPose();

    void processIMU(double t, double dt, const Eigen::Vector3d &linear_acceleration,
                    const Eigen::Vector3d &angular_velocity,
                    const cv::Point2f &encoder);

    Eigen::Matrix4d IntegrateOdo(const double &deltaT, Eigen::Vector2d &delatEncoder, const double &alpha);

    bool getLatestCamPose();

    ORB_SLAM3::Pose_t saveLatestPose(const std::string &filename);

    void inputIMU(const double &t, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr, const cv::Point2f &encoder);

private:
    void updateRelPose();

    Eigen::Matrix4d mTc_odo;
    Eigen::Matrix4d mTodo_c;

    // pointers of other classes
    std::shared_ptr<ORB_SLAM3::IMUProcess> mpIMUProcessor;
    std::shared_ptr<ORB_SLAM3::System> mpVSLAM;

    RawData rawDataLast;
    RawData rawData;

    // flags
    bool overwrite = true;

    // imu, odo integration
    Eigen::Vector3d acc_last;
    Eigen::Vector3d gyr_last;
    double yaw_last;
    cv::Point2f encoder_last;
    Eigen::Vector3d Ba;
    Eigen::Vector3d Bg;

    // constants
    Eigen::Vector3d g = G; //ideal gravitational acc

    // latest pose and its time (either imu or cam)
    ORB_SLAM3::Pose_t mTwc;
    ORB_SLAM3::Pose_t mTwc_orb;
    RelPose_t relPose;
};


#endif //SWEEPSLAM_REALTIME_H
