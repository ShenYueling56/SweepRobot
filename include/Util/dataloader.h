//
// Created by qzj on 2021/4/25.
//
#include"include/orb_slam3/System.h"
#include"include/preintegration/ImuTypes.h"
#include"include/preintegration/Odometer.h"
#include "include/preintegration/imuProcess.h"
#include "selfDefine.h"
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<vector>
#include<queue>
#include<thread>
#include<opencv2/videoio.hpp>
#include<mutex>
#include "include/orb_slam3/RealTime.h"
//#include <serial/serial.h>


#ifndef SWEEPSLAM_DATALOADER_H
#define SWEEPSLAM_DATALOADER_H

namespace ORB_SLAM3 {
    class System;

    typedef struct {
        IMU::Point imu;
        ODO::Point odo;
        double time;
    }IMUmsg;

    typedef struct {
        cv::Mat image;
        double time;
    }ImgMsg;

    class ImuGrabber {
    public:
        ImuGrabber(std::shared_ptr<ORB_SLAM3::IMUProcess> imuProcessor) {
            imuInit = std::chrono::steady_clock::now();
            mpImuProcessor = imuProcessor;
        };
        void setRealTime(std::shared_ptr<RealTime> pRealTime) { mpRealTime = pRealTime; };

        void GrabImu();

        queue<IMUmsg> imuBuf;

        std::mutex mBufMutex;

        std::chrono::steady_clock::time_point imuInit;

        std::shared_ptr<RealTime> mpRealTime;

        std::shared_ptr<ORB_SLAM3::IMUProcess> mpImuProcessor;

        void UpdateLatestPose();
//    public:
//        serial::Serial ros_ser;
    };

    class ImageGrabber {
    public:
        ImageGrabber(std::shared_ptr<ORB_SLAM3::System> pSLAM, ImuGrabber *pImuGb, std::shared_ptr<ORB_SLAM3::IMUProcess> imuProcessor, const bool bRect, const bool bClahe, const string &strSettingsFile){
            mpSLAM = pSLAM;
            mpImuGb = pImuGb;
#ifdef ALTER_STEREO_MATCHING
            do_rectify = false;
#else
            do_rectify = true;
#endif
            mbClahe = bClahe;
            imuInit = mpImuGb->imuInit;
            mpImuProcessor = imuProcessor;

//            id可能需要修改
            cap.open(24);                             //打开相机，电脑自带摄像头一般编号为0，外接摄像头编号为1，主要是在设备管理器中查看自己摄像头的编号。
            //--------------------------------------------------------------------------------------
            cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));//视频流格式
            cap.set(CV_CAP_PROP_FPS, 50);//帧率
            cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);  //设置捕获视频的宽度
            cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);  //设置捕获视频的高度

            if (!cap.isOpened())                         //判断是否成功打开相机
            {
                cout << "摄像头打开失败!" << endl;
                return;
            }

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

        }
        void setRealTime(std::shared_ptr<RealTime> pRealTime) { mpRealTime = pRealTime; };

        void GrabImage();

        void SyncWithImu();

        queue<std::pair<ImgMsg, ImgMsg>> imgBuf;

        std::mutex mBufMutex;

        std::shared_ptr<ORB_SLAM3::System> mpSLAM;
        ImuGrabber *mpImuGb;
        std::shared_ptr<RealTime> mpRealTime;

        bool do_rectify;
        bool mbClahe;

        cv::Mat M1l, M2l, M1r, M2r;

        vector<float> vTimesTrack;
        vector<float> vTimesNow;

        //增强对比度
        cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));

        std::chrono::steady_clock::time_point imuInit;

        std::shared_ptr<ORB_SLAM3::IMUProcess> mpImuProcessor;

        cv::VideoCapture cap;
    };
}

#endif //SWEEPSLAM_DATALOADER_H
