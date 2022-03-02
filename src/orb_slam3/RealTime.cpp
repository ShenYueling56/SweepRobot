//
// Created by warren on 2021/6/19.
//

#include "include/orb_slam3/RealTime.h"
#include "include/orb_slam3/RealTime.h"
#include "include/orb_slam3/Converter.h"
//#define DEBUG_REAL_TIME

using namespace std;

RealTime::RealTime(const string &strSettingFile){

    cv::FileStorage fsSettings(strSettingFile, cv::FileStorage::READ);
    cv::Mat tmp;
    fsSettings["Tc_odo"] >> tmp;
    mTc_odo = ORB_SLAM3::Converter::toMatrix4d(tmp);
    mTodo_c = ORB_SLAM3::Converter::toEigen4dInverse(mTc_odo);
};

void RealTime::inputIMU(const double &t, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr, const cv::Point2f &encoder){

    rawData.time = t;
    rawData.acc = acc;
    rawData.gyr = gyr;
    rawData.encoder = encoder;

//  第一次， 重置参数
    if (rawDataLast.time < 0.0){
        rawDataLast = rawData;
        rawDataLast.yaw = 0.0;
        rawDataLast.R.setIdentity();
        relPose.pose.setIdentity();
        mTwc.pose.setIdentity();
        mTwc.time = relPose.endTime = relPose.startTime = rawData.time;
        return;
    }

    updateRelPose();
}

void RealTime::updateRelPose(){

    double dt = rawData.time - rawDataLast.time;

    Ba = mpIMUProcessor->Bas[mpIMUProcessor->frame_count];
    Bg = mpIMUProcessor->Bgs[mpIMUProcessor->frame_count];
    Eigen::Vector3d un_gyr = 0.5 * (rawDataLast.gyr + rawData.gyr) - Bg;//移除了偏执的gyro
    rawData.R = rawDataLast.R * Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    rawData.yaw = Utility::R2ypr(rawData.R).x(); // now the R_last is the newest R

    double alpha = (rawData.yaw - rawDataLast.yaw) / 180.0 * M_PI;
    Eigen::Vector2d delatEncoder(rawData.encoder.x - rawDataLast.encoder.x, rawData.encoder.y - rawDataLast.encoder.y);
//    cout << std::setiosflags(ios::fixed) << std::setprecision(4) <<"t: "<<rawData.time<<" last t: "<<rawDataLast.time<<" dt: "<<dt
//    <<" yaw: "<<rawData.yaw<<" last yaw: "<<rawDataLast.yaw <<" alpha: "<<alpha
//    <<" gyr: "<<rawData.gyr.transpose()<<" last gyr: "<<rawDataLast.gyr.transpose()
//    <<" delatEncoder: "<<delatEncoder.transpose()<<endl;
    Eigen::Matrix4d T_pre_cur = IntegrateOdo(dt, delatEncoder, alpha);
    relPose.pose = relPose.pose * T_pre_cur;
    relPose.endTime = rawData.time;
    rawDataLast = rawData;
}

bool RealTime::getLatestIMUPose() {

//    没有IMU数据时
    if(rawDataLast.time < 0.0){
        return false;
    }

    ORB_SLAM3::Pose_t curTwc_orb = mpVSLAM->getLatestCamPose();
    if(curTwc_orb.time < 0.0){
        cout<<"getLatestIMUPose 1"<<endl;
        mTwc.pose = relPose.pose;
        mTwc.time = relPose.endTime;
    }else if(curTwc_orb.time <= relPose.endTime){
//        cout<<"getLatestIMUPose 2"<<endl;
        double beta = (curTwc_orb.time - relPose.startTime) / (relPose.endTime - relPose.startTime);
//        cout << std::setiosflags(ios::fixed) << std::setprecision(4) <<" img time: "<<curTwc_orb.time <<" startTime: "<<relPose.startTime<<" endTime: "<<relPose.endTime<<endl;
        Eigen::Matrix4d M0 = Eigen::Matrix4d::Identity();
        relPose.pose = relPose.pose * ORB_SLAM3::Converter::toEigen4dInverse(interpolateSE3(M0, relPose.pose, beta));
        relPose.startTime = curTwc_orb.time;
//        cout << "beta: " <<beta << " relPose.pose: \n" << std::setiosflags(ios::fixed) << std::setprecision(6) << relPose.pose <<endl;
        mTwc.pose = curTwc_orb.pose * mTc_odo * relPose.pose * mTodo_c;
//        mTwc.pose = curTwc_orb.pose;
        mTwc.time = relPose.endTime;
    }else if(curTwc_orb.time > relPose.endTime){
        cout<<"getLatestIMUPose 3"<<endl;
//        relPose.startTime = relPose.endTime = curTwc_orb.time;
//        relPose.pose = Eigen::Matrix4d::Identity();

//        rawDataLast = rawData;
//        rawDataLast.time = curTwc_orb.time;
//        rawDataLast.R = curTwc_orb.pose.block<3,3>(0,0);
//        rawDataLast.yaw = Utility::R2ypr(rawDataLast.R).x();

        mTwc.pose = curTwc_orb.pose;
        mTwc.time = curTwc_orb.time;
    }else{
        cout<<"ERROR getLatestIMUPose"<<endl;
    }

    return true;
}

Eigen::Matrix4d RealTime::IntegrateOdo(const double &deltaT, Eigen::Vector2d &delatEncoder, const double &alpha) {
    static Eigen::Vector2d encoderLast = Eigen::Vector2d::Zero(), odoSelfLast = Eigen::Vector2d::Zero();
    static vector<pair<double, pair<double, double>>> delta_t_X;

    if (delatEncoder(0) >= (5.0 * 18.0 * 67.2))
        delatEncoder(0) = delatEncoder(0) - 10.0 * 18.0 * 67.2;
    else if (delatEncoder(0) <= -(5.0 * 18.0 * 67.2))
        delatEncoder(0) = delatEncoder(0) + 10.0 * 18.0 * 67.2;
    if (delatEncoder(1) >= (5.0 * 18.0 * 67.2))
        delatEncoder(1) = delatEncoder(1) - 10.0 * 18.0 * 67.2;
    else if (delatEncoder(1) <= -(5.0 * 18.0 * 67.2))
        delatEncoder(1) = delatEncoder(1) + 10.0 * 18.0 * 67.2;
    //delatEncoder = delatEncoder/(18.0*67.2)*M_PI*0.07;
    delatEncoder = delatEncoder * 0.0001818051304160764;

    double delatEncoderMid = (delatEncoder(0) + delatEncoder(1)) / 2.0;
    double curV = delatEncoderMid / deltaT;
    if (delta_t_X.size() < 6)
        delta_t_X.push_back(make_pair(deltaT, make_pair(delatEncoderMid, curV)));
    else {
        delta_t_X.erase(delta_t_X.begin());
        delta_t_X.push_back(make_pair(deltaT, make_pair(delatEncoderMid, curV)));
    }

    // 李群上积分
    Eigen::Matrix2d A;
    if (abs(alpha) > 0.0001)
        A << sin(alpha) / alpha, -(1 - cos(alpha)) / alpha, (1 - cos(alpha)) / alpha, sin(alpha) / alpha;
    else
        A << cos(alpha), -(0 + sin(alpha)) / 1, (0 + sin(alpha)) / 1, cos(alpha);
    Eigen::Vector2d v = Eigen::Vector2d::Zero();
    v(0) = delatEncoderMid;
    Eigen::Vector2d deltaOdo = A * v;

//    T_oi_oi+1
    Eigen::Matrix4d T_pre_cur = Eigen::Matrix4d::Identity();
    T_pre_cur.block<3,3>(0,0) = Eigen::AngleAxisd (alpha, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T_pre_cur(0, 3) = deltaOdo(0);
    T_pre_cur(1, 3) = deltaOdo(1);
    return T_pre_cur;
}

ORB_SLAM3::Pose_t RealTime::saveLatestPose(const std::string &filename) {
    ofstream f;
    if(overwrite){
        overwrite = false;
        f.open(filename.c_str(), ios::ate);
    }else
        f.open(filename.c_str(), ios::app);

    Eigen::Matrix3d Rwc = mTwc.pose.block<3,3>(0,0);
    Eigen::Quaterniond q(Rwc);
    Eigen::Vector3d twc = mTwc.pose.block<3,1>(0,3);

    f << fixed;
    f << setprecision(6) << mTwc.time << " " << setprecision(9) << twc(0) << " " << twc(1) << " "
      << twc(2) << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << endl;
    f.close();

    return mTwc;
}