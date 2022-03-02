/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<mutex>

#include<ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include<cv_bridge/cv_bridge.h>
#include<sensor_msgs/Imu.h>
#include"include/orb_slam3/Frame.h"

#include<opencv2/core/core.hpp>

#include"include/orb_slam3/System.h"
#include"include/preintegration/ImuTypes.h"
#include"include/preintegration/Odometer.h"

#include "Util/parameters.h"
#include "include/preintegration/imuProcess.h"
#include "include/orb_slam3/RealTime.h"

using namespace std;

class ImuGrabber {
public:
    ImuGrabber(std::shared_ptr<ORB_SLAM3::IMUProcess> imuProcessor) : mpImuProcessor(imuProcessor) {};

    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    void setRealTime(std::shared_ptr<RealTime> pRealTime) { mpRealTime = pRealTime; };

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::shared_ptr<ORB_SLAM3::IMUProcess> mpImuProcessor;
    std::shared_ptr<RealTime> mpRealTime;
    std::mutex mBufMutex;
};

class ImageGrabber {
public:
    ImageGrabber(std::shared_ptr<ORB_SLAM3::System> SLAM, std::shared_ptr<ORB_SLAM3::IMUProcess> imuProcessor,
                 ImuGrabber *pImuGb, const bool bRect, const bool bClahe, const string &strSettingsFile) :
    mpSLAM(SLAM), mpImuProcessor(imuProcessor), mpImuGb(pImuGb), do_rectify(bRect), mbClahe(bClahe) {
#ifdef ALTER_STEREO_MATCHING
        do_rectify = false;
#else
        do_rectify = true;
#endif
        if (do_rectify) {
            // Load settings related to stereo calibration
            cv::FileStorage fsSettings(strSettingsFile, cv::FileStorage::READ);
            if (!fsSettings.isOpened()) {
                cerr << "ERROR: Wrong path to settings" << endl;
                return;
            }

            cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
            fsSettings["LEFT.K"] >> K_l;
            fsSettings["RIGHT.K"] >> K_r;

            fsSettings["LEFT.P"] >> P_l;
            fsSettings["RIGHT.P"] >> P_r;

            fsSettings["LEFT.R"] >> R_l;
            fsSettings["RIGHT.R"] >> R_r;

            fsSettings["LEFT.D"] >> D_l;
            fsSettings["RIGHT.D"] >> D_r;

            int rows_l = fsSettings["LEFT.height"];
            int cols_l = fsSettings["LEFT.width"];
            int rows_r = fsSettings["RIGHT.height"];
            int cols_r = fsSettings["RIGHT.width"];

            if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() ||
                D_r.empty() ||
                rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
                cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
                return;
            }

            cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F,
                                        M1l, M2l);
            cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F,
                                        M1r, M2r);
        }
    };

    void GrabImageLeft(const sensor_msgs::ImageConstPtr &msg);

    void GrabImageRight(const sensor_msgs::ImageConstPtr &msg);

    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);

    bool SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft, mBufMutexRight;

    std::shared_ptr<ORB_SLAM3::IMUProcess> mpImuProcessor;
    std::shared_ptr<ORB_SLAM3::System> mpSLAM;
    ImuGrabber *mpImuGb;

    bool do_rectify;
    cv::Mat M1l, M2l, M1r, M2r;

    const bool mbClahe;
    //增强对比度
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};


static uint32_t imuCnt = 0;
static bool IMUReady = false;
double Td = -0.008;
std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
//ofstream imuBias;

int main(int argc, char **argv) {
    bool bEqual = false;
//    if (argc != 4) {
//        cerr << "wrong" << endl;
//        return 0;
//    }
    std::string sbRect("true");
    std::string sbEqual("false");
    if (sbEqual == "true")
        bEqual = true;
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::string strVocFile = "../Vocabulary/ORBvoc.txt";
//    std::string strSettingFile = argv[1];
    std::string strSettingFile = "../config/robot/robot_orb_stereo_new.yaml";

    // Location of the ROS bag we want to read in
//    std::string path_to_bag = "/media/warren/mobile_HDD/Ubuntu_Downloads_Synchronize/datasets/sweepSLAM/10/small_10.bag";
//    std::string path_to_bag = "/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/01/2020-07-26-19-47-34.bag";
    //std::string path_to_bag="/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/02/2020-07-26-19-49-21.bag";
    //std::string path_to_bag="/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/03/2020-07-26-19-50-56.bag";
    //std::string path_to_bag="/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/04/2020-07-29-18-40-03.bag";
    //std::string path_to_bag="/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/05/2020-07-29-18-41-52.bag";
    //std::string path_to_bag="/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/06/2020-07-29-18-43-57.bag";
    //std::string path_to_bag="/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/07/2020-08-12-16-41-28.bag";
    //std::string path_to_bag="/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/08/2020-08-12-16-47-23.bag";
//    std::string path_to_bag="/media/qzj/Document/grow/research/slamDataSet/sweepRobot/round3/09/2020-08-12-16-54-51.bag";
    std::string path_to_bag="/media/qzj/Dataset/slamDataSet/sweepRobot/round3/10/small_10.bag";
//    std::string path_to_bag= argv[3];

    cout << fixed << endl << "-------" << endl;
    ROS_INFO("ros bag path is: %s", path_to_bag.c_str());

//    imuBias.open("./imuBiasORB01.txt");
//    imuBias << fixed;
// 先读参数
    readParameters(strSettingFile);

    std::shared_ptr<ORB_SLAM3::IMUProcess> imuProcessor;    
    std::shared_ptr<ORB_SLAM3::System> SLAM;
    SLAM.reset( new ORB_SLAM3::System(strVocFile, strSettingFile, ORB_SLAM3::System::STEREO, true));
    imuProcessor.reset(new ORB_SLAM3::IMUProcess());
    imuProcessor->setParameter();
    SLAM->SetIMUProcessor(imuProcessor);

    std::shared_ptr<RealTime> pRealTime;
    pRealTime.reset(new RealTime(strSettingFile));
    pRealTime->SetIMUProcessor(imuProcessor);
    pRealTime->SetVSLAM(SLAM);

    // Our camera topics (left and right stereo)
    std::string topic_imu = "/imu0";
    std::string topic_camera0 = "/cam0/image_raw", topic_camera1 = "/cam1/image_raw";
    //n.param<std::string>("topic_imu", topic_imu, "/imu0");
    //n.param<std::string>("topic_camera0", topic_camera0, "/cam0/image_raw");
    //n.param<std::string>("topic_camera1", topic_camera1, "/cam1/image_raw");

    ImuGrabber imugb(imuProcessor);
    imugb.setRealTime(pRealTime);
    ImageGrabber igb(SLAM, imuProcessor, &imugb, sbRect == "true", bEqual, strSettingFile);

    cv::FileStorage fsSettings(strSettingFile, cv::FileStorage::READ);
    std::string save_path = fsSettings["save_path"];

    // Get our start location and how much of the bag we want to play
    // Make the bag duration < 0 to just process to the end of the bag
    double bag_start = 0, bag_durr = -1;
    //n.param<double>("bag_start", bag_start, 0);
    //n.param<double>("bag_durr", bag_durr, -1);
    ROS_INFO("bag start: %.1f", bag_start);
    ROS_INFO("bag duration: %.1f", bag_durr);

    // Load rosbag here, and find messages we can play
    rosbag::Bag bag;
    bag.open(path_to_bag, rosbag::bagmode::Read);

    // We should load the bag as a view
    // Here we go from beginning of the bag to the end of the bag
    rosbag::View view_full;
    rosbag::View view;

    // Start a few seconds in from the full view time
    // If we have a negative duration then use the full bag length
    view_full.addQuery(bag);
    ros::Time time_init = view_full.getBeginTime();
    time_init += ros::Duration(bag_start);
    ros::Time time_finish = (bag_durr < 0) ? view_full.getEndTime() : time_init + ros::Duration(bag_durr);
    ROS_INFO("time start = %.6f", time_init.toSec());
    ROS_INFO("time end   = %.6f", time_finish.toSec());
    view.addQuery(bag, time_init, time_finish);

    // Check to make sure we have data to play
    if (view.size() == 0) {
        ROS_ERROR("No messages to play on specified topics.  Exiting.");
        ros::shutdown();
        return EXIT_FAILURE;
    }

    vector<float> vTimesTrack;

    // Step through the rosbag
    double time = view_full.getBeginTime().toSec();
    for (const rosbag::MessageInstance &m : view) {
        // If ros is wants us to stop, break out
        //if (!ros::ok())
        //    break;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Handle IMU measurement
        sensor_msgs::Imu::ConstPtr s2 = m.instantiate<sensor_msgs::Imu>();
        if (s2 != nullptr && m.getTopic() == topic_imu) {
            imugb.GrabImu(s2);
        }

        // Handle LEFT camera
        sensor_msgs::Image::ConstPtr s0 = m.instantiate<sensor_msgs::Image>();
        if (s0 != nullptr && m.getTopic() == topic_camera0) {
            igb.GrabImageLeft(s0);
        }
        // Handle RIGHT camera
        sensor_msgs::Image::ConstPtr s1 = m.instantiate<sensor_msgs::Image>();
        if (s1 != nullptr && m.getTopic() == topic_camera1) {
            igb.GrabImageRight(s1);
        }

        if (igb.SyncWithImu()) {
            time = m.getTime().toSec();
            t2 = std::chrono::steady_clock::now();
            double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            vTimesTrack.push_back(ttrack);

            //if((time-timeLast)>ttrack)
            //    usleep(((time-timeLast)-ttrack)*1e6);
            //timeLast = time;
        }

//         pRealTime should run after imuProcessor has finished this round (need acc_0, gyr_0, Bg, Ba from imuPorcessor)
//         Each time, either camera or imu can be the latest info
        if((m.getTopic() == topic_imu || m.getTopic() == topic_camera0 || m.getTopic() == topic_camera1))
            if(pRealTime->getLatestIMUPose()){
                pRealTime->saveLatestPose(std::string("orb3_stereo_slam_realtime.txt"));
            }
    }

// Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < vTimesTrack.size(); ni++) {
        totaltime += vTimesTrack[ni];
    }
    cout << fixed << setprecision(4);

    while (ros::ok());

    // Stop all threads
    SLAM->Shutdown();

    SLAM->SaveTrajectoryTUM(std::string("orb3_stereo_slam.txt"));
    cout << "all finish " << endl;
    return 0;
}

int imgIdx = 0;
const int speedUp = 5;

void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg) {
    if (!IMUReady)
        return;
    mBufMutexLeft.lock();
    if (!imgLeftBuf.empty())
        imgLeftBuf.pop();
    //img_msg->header.stamp = ;
    imgIdx++;
    if (imgIdx % speedUp == 0)
        imgLeftBuf.push(img_msg);
    mBufMutexLeft.unlock();
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg) {
    if (!IMUReady)
        return;
    mBufMutexRight.lock();
    if (!imgRightBuf.empty())
        imgRightBuf.pop();
    if (imgIdx % speedUp == 0)
        imgRightBuf.push(img_msg);
    mBufMutexRight.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg) {
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    if (img_msg->encoding == sensor_msgs::image_encodings::BGR8) {
        try {
            cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
        if (cv_ptr->image.type() == 16) {
            return cv_ptr->image.clone();
        } else {
            std::cout << "Error type" << std::endl;
            return cv_ptr->image.clone();
        }
    } else if (img_msg->encoding == sensor_msgs::image_encodings::MONO8) {
        try {
            cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
        }
        catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
        if (cv_ptr->image.type() == 0) {
            return cv_ptr->image.clone();
        } else {
            std::cout << "Error type" << std::endl;
            return cv_ptr->image.clone();
        }
    }
}

bool ImageGrabber::SyncWithImu() {
    const double maxTimeDiff = 0.01;
    cv::Mat imLeft, imRight;
    double tImLeft = 0, tImRight = 0;
    if (!imgLeftBuf.empty() && !imgRightBuf.empty() && !mpImuGb->imuBuf.empty()) {
        tImLeft = imgLeftBuf.front()->header.stamp.toSec() + Td;
        tImRight = imgRightBuf.front()->header.stamp.toSec() + Td;
        //获得最近的右图像
        this->mBufMutexRight.lock();
        while ((tImLeft - tImRight) > maxTimeDiff && imgRightBuf.size() > 1) {
            imgRightBuf.pop();
            tImRight = imgRightBuf.front()->header.stamp.toSec() + Td;
        }
        this->mBufMutexRight.unlock();

        //获得最近的左图像
        this->mBufMutexLeft.lock();
        while ((tImRight - tImLeft) > maxTimeDiff && imgLeftBuf.size() > 1) {
            imgLeftBuf.pop();
            tImLeft = imgLeftBuf.front()->header.stamp.toSec() + Td;
        }
        this->mBufMutexLeft.unlock();

        if ((tImLeft - tImRight) > maxTimeDiff || (tImRight - tImLeft) > maxTimeDiff) {
            //std::cout << "big time difference" << std::endl;
            return false;
        }
        // IMU值太少了,需要在相机之后依然有
        if (tImLeft > mpImuGb->imuBuf.back()->header.stamp.toSec()) {
            //std::cout << "tImLeft > mpImuGb->imuBuf.back()->header.stamp.toSec()" << std::endl;
            return false;
        }

        this->mBufMutexLeft.lock();
        imLeft = GetImage(imgLeftBuf.front());
        imgLeftBuf.pop();
        this->mBufMutexLeft.unlock();

        this->mBufMutexRight.lock();
        imRight = GetImage(imgRightBuf.front());
        imgRightBuf.pop();
        this->mBufMutexRight.unlock();

        //载入IMU数据
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        vector<ORB_SLAM3::ODO::Point> vOdoMeas;
        mpImuGb->mBufMutex.lock();
        if (!mpImuGb->imuBuf.empty()) {
            // Load imu measurements from buffer
            vImuMeas.clear();
            vOdoMeas.clear();
            while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= tImLeft) {
                double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x,
                                mpImuGb->imuBuf.front()->linear_acceleration.y,
                                mpImuGb->imuBuf.front()->linear_acceleration.z);
                cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x,
                                mpImuGb->imuBuf.front()->angular_velocity.y,
                                mpImuGb->imuBuf.front()->angular_velocity.z);

                cv::Point3f odometer(mpImuGb->imuBuf.front()->angular_velocity_covariance[0],
                                     mpImuGb->imuBuf.front()->angular_velocity_covariance[1],
                                     mpImuGb->imuBuf.front()->angular_velocity_covariance[2]);
                cv::Point2f encoder(mpImuGb->imuBuf.front()->angular_velocity_covariance[4],
                                    mpImuGb->imuBuf.front()->angular_velocity_covariance[5]);
                cv::Point3f rpy(mpImuGb->imuBuf.front()->angular_velocity_covariance[6],
                                mpImuGb->imuBuf.front()->angular_velocity_covariance[7],
                                mpImuGb->imuBuf.front()->angular_velocity_covariance[8]);

                vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                vOdoMeas.push_back(ORB_SLAM3::ODO::Point(odometer, encoder, rpy, t));
//                cout<<t<<" "<<acc.x<<" "<<acc.y<<" "<<acc.z<<" "<<gyr.x<<" "<<gyr.y<<" "<<rpy.z<<" "<<encoder.x<<" "<<encoder.y<<endl;
                mpImuGb->imuBuf.pop();
            }
        }
        mpImuGb->mBufMutex.unlock();

        t1 = std::chrono::steady_clock::now();

        if (mbClahe) {
            mClahe->apply(imLeft, imLeft);
            mClahe->apply(imRight, imRight);
        }

        if (do_rectify) {
            cv::remap(imLeft, imLeft, M1l, M2l, cv::INTER_LINEAR);
            cv::remap(imRight, imRight, M1r, M2r, cv::INTER_LINEAR);
        }

        ORB_SLAM3::Verbose::PrintMess("TrackStereo 1", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        if (!vOdoMeas.empty())
            mpImuProcessor->mOdoReceive << vOdoMeas[vOdoMeas.size() - 1].odometer.x, vOdoMeas[vOdoMeas.size() -
                                                                                            1].odometer.y, vOdoMeas[
                    vOdoMeas.size() - 1].rpy.z;
        ORB_SLAM3::Verbose::PrintMess("tracking start", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        mpImuProcessor->preIntegrateIMU(tImLeft);
        ORB_SLAM3::Verbose::PrintMess("tracking preIntegrateIMU", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        mpSLAM->TrackStereo(imLeft, imRight, tImLeft, vImuMeas, vOdoMeas);
        ORB_SLAM3::Verbose::PrintMess("tracking TrackStereo", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        mpImuProcessor->updateIMUBias();
        ORB_SLAM3::Verbose::PrintMess("tracking over", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
//        imuBias<<fixed<<tImLeft<<" "<< imuProcessor->Bgs[0].x()<<" "<< imuProcessor->Bgs[0].y()<<" "<< imuProcessor->Bgs[0].z() <<endl;
        //cv::Mat Tcw_orb;
        //mpSLAM->GetCurPose(Tcw_orb);
        //estimator.setNewPoseFromORB3(Tcw_orb);
        //estimator.inputImage(tImLeft, imLeft, imRight);
        //cout<<"TrackStereo 2"<<endl;

        return true;
        //std::chrono::milliseconds tSleep(1);
        //std::this_thread::sleep_for(tSleep);
    }
    return false;
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg) {
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d acc(dx, dy, dz);
    Eigen::Vector3d gyr(rx, ry, rz);
//    cout << fixed <<"raw t: "<<t<<" gyr: "<<gyr.transpose()<<endl;
    cv::Point2f encoder(imu_msg->angular_velocity_covariance[4],
                        imu_msg->angular_velocity_covariance[5]);

    mpImuProcessor->inputIMU(t, acc, gyr, encoder);
    mpRealTime->inputIMU(t, acc, gyr, encoder);

    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    
    imuCnt++;
    if (imuCnt > 5)
        IMUReady = true;
}


