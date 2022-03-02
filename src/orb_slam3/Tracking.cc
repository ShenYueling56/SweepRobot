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


#include "include/orb_slam3/Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"include/orb_slam3/ORBmatcher.h"
#include"include/orb_slam3/FrameDrawer.h"
#include"include/orb_slam3/Converter.h"
#include"include/orb_slam3/Initializer.h"
#include"include/orb_slam3/G2oTypes.h"
#include"include/orb_slam3/Optimizer.h"
#include"include/orb_slam3/PnPsolver.h"
#include "Util/FileUtil.h"
#include<iostream>
#include "Util/parameters.h"

#include<mutex>
#include<chrono>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <include/orb_slam3/MLPnPsolver.h>
#include "Util/parameters.h"


using namespace std;

namespace ORB_SLAM3 {

    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer,
                       Atlas *pAtlas, KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor,
                       const string &_nameSeq) :
            mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
            mlastMapmLastFrameTcw(cv::Mat::eye(4, 4, CV_32F)), mInitPreOdo(false), mOdoTpc(cv::Mat()),
            mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
            mlastTcwUpdate(false), mpReferenceKF(static_cast<KeyFrame *>(NULL)),
            mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys), mpViewer(NULL), mbRestart(false),
            mbVisual(false), mRotAccumulate(0.f), mTransAccumulate(0.f),
            mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0),
            time_recently_lost(5.0), mbInertialFirst(
            false), usefulForIMU(0),
            mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr),
            mpIMUProcessor(static_cast<IMUProcess *>(NULL)) {
        // Load camera parameters from settings file
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        t_odo_sum = 0.0;
        t_cam_sum = 0.0;
        if (mbVisual) {
            rmByCpp("./images/");
            createDirectory("./images/");
            rmByCpp("./Reference/");
            createDirectory("./Reference/");
            rmByCpp("./Last/");
            createDirectory("./Last/");
        }
        bool b_parse_cam = ParseCamParamFile(fSettings);
        if (!b_parse_cam) {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if (!b_parse_orb) {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        initID = 0;
        lastID = 0;

        // Load IMU parameters
        bool b_parse_imu = true;
        if (sensor == System::IMU_MONOCULAR || sensor == System::IMU_STEREO) {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if (!b_parse_imu) {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        mbInitWith3KFs = false;

        mnNumDataset = 0;

        if (!b_parse_cam || !b_parse_orb || !b_parse_imu) {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try {
                throw -1;
            }
            catch (exception &e) {

            }
        }

        mpTrackingTime = new TrackingTime();
    }

    Tracking::~Tracking() {
    }

    void Tracking::SetIMUProcessor(std::shared_ptr<ORB_SLAM3::IMUProcess> IMUProcessor) {
        mpIMUProcessor = IMUProcessor;
    }

    bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings) {
        mDistCoef = cv::Mat::zeros(4, 1, CV_32F);
        cout << endl << "Camera Parameters: " << endl;
        bool b_miss_params = false;

        string sCameraName = fSettings["Camera.type"];
        fSettings["Tc_odo"] >> mTc_odo;
        mTodo_c = Converter::toCvMatInverse(mTc_odo);

        if (mSensor == System::STEREO || mSensor == System::IMU_STEREO) {
            cv::FileNode node = fSettings["Camera.bf"];
            if (!node.empty() && node.isReal()) {
                mbf = node.real();
            } else {
                std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
        }

        if (sCameraName == "PinHole") {
            float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;

            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera_fx"];
            if (!node.empty() && node.isReal()) {
                fx = node.real();
            } else {
                std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera_fy"];
            if (!node.empty() && node.isReal()) {
                fy = node.real();
            } else {
                std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera_cx"];
            if (!node.empty() && node.isReal()) {
                cx = node.real();
            } else {
                std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera_cy"];
            if (!node.empty() && node.isReal()) {
                cy = node.real();
            } else {
                std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            mDistCoef.at<float>(0) = fSettings["Camera_k1"];
//        node = fSettings["Camera_k1"];
//        if(!node.empty() && node.isReal())
//        {
//            mDistCoef.at<float>(0) = node.real();
//        }
//        else
//        {
//            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
//            b_miss_params = true;
//        }

            mDistCoef.at<float>(1) = fSettings["Camera_k2"];
//        if(!node.empty() && node.isReal())
//        {
//            mDistCoef.at<float>(1) = node.real();
//        }
//        else
//        {
//            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
//            b_miss_params = true;
//        }

            mDistCoef.at<float>(2) = fSettings["Camera_p1"];
//        if(!node.empty() && node.isReal())
//        {
//            mDistCoef.at<float>(2) = node.real();
//        }
//        else
//        {
//            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
//            b_miss_params = true;
//        }

            mDistCoef.at<float>(3) = fSettings["Camera_p2"];
//        if(!node.empty() && node.isReal())
//        {
//            mDistCoef.at<float>(3) = node.real();
//        }
//        else
//        {
//            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
//            b_miss_params = true;
//        }
            // todo: 会一直读不到Camera_k3
            node = fSettings["Camera_k3"];
            if (!node.empty() && node.isReal()) {
                mDistCoef.resize(5);
                mDistCoef.at<float>(4) = node.real();
            }

            if (b_miss_params) {
                return false;
            }

            vector<float> vCamCalib{fx, fy, cx, cy};

            mpCamera = new Pinhole(vCamCalib);

            mpAtlas->AddCamera(mpCamera);


            std::cout << "- Camera: Pinhole" << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
            std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


            std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
            std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

            if (mDistCoef.rows == 5)
                std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

            mK = cv::Mat::eye(3, 3, CV_32F);
            mK.at<float>(0, 0) = fx;
            mK.at<float>(1, 1) = fy;
            mK.at<float>(0, 2) = cx;
            mK.at<float>(1, 2) = cy;

            if (mSensor == System::STEREO) {
                // modified stereo with undistortion of both cameras
                // Load camera parameters from settings file
                cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
                fSettings["cameraMatrixL"] >> K_l;
                if (K_l.type() != CV_32F)
                    K_l.convertTo(K_l, CV_32F);
                fSettings["cameraMatrixR"] >> K_r;
                if (K_r.type() != CV_32F)
                    K_r.convertTo(K_r, CV_32F);

                fSettings["P1"] >> P_l;
                if (P_l.type() != CV_32F)
                    P_l.convertTo(P_l, CV_32F);
                fSettings["P2"] >> P_r;
                if (P_r.type() != CV_32F)
                    P_r.convertTo(P_r, CV_32F);

                fSettings["R1"] >> R_l;
                if (R_l.type() != CV_32F)
                    R_l.convertTo(R_l, CV_32F);
                fSettings["R2"] >> R_r;
                if (R_r.type() != CV_32F)
                    R_r.convertTo(R_r, CV_32F);

                fSettings["distCoeffsL"] >> D_l;
                if (D_l.type() != CV_32F)
                    D_l.convertTo(D_l, CV_32F);
                fSettings["distCoeffsR"] >> D_r;
                if (D_r.type() != CV_32F)
                    D_r.convertTo(D_r, CV_32F);

                cv::Size imageSize;
                fSettings["imageSize"] >> imageSize;

                int rows_l = imageSize.height;
                int cols_l = imageSize.width;
                int rows_r = imageSize.height;
                int cols_r = imageSize.width;

                if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() ||
                    D_l.empty() ||
                    D_r.empty() ||
                    rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0) {
                    cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
                    return false;
                }

                K_l.copyTo(mK_ori);
#ifdef ALTER_STEREO_MATCHING
                D_l.copyTo(mDistCoef);
#endif
                R_l.copyTo(mR);
                P_l.copyTo(mP);
                //
                K_r.copyTo(mK_right);
                D_r.copyTo(mDistCoef_right);
                R_r.copyTo(mR_right);
                P_r.copyTo(mP_right);
				
                mFrameAfterInital = 0;
                mbTrackLossAlert = 0;
                nFrameSinceLast = 0;
            }
        } else if (sCameraName == "KannalaBrandt8") {
            float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;
            float k1 = 0.f, k2 = 0.f, k3 = 0.f, k4 = 0.f;

            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera_fx"];
            if (!node.empty() && node.isReal()) {
                fx = node.real();
            } else {
                std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera_fy"];
            if (!node.empty() && node.isReal()) {
                fy = node.real();
            } else {
                std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera_cx"];
            if (!node.empty() && node.isReal()) {
                cx = node.real();
            } else {
                std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera_cy"];
            if (!node.empty() && node.isReal()) {
                cy = node.real();
            } else {
                std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera_k1"];
            if (!node.empty() && node.isReal()) {
                k1 = node.real();
            } else {
                std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera_k2"];
            if (!node.empty() && node.isReal()) {
                k2 = node.real();
            } else {
                std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera_k3"];
            if (!node.empty() && node.isReal()) {
                k3 = node.real();
            } else {
                std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera_k4"];
            if (!node.empty() && node.isReal()) {
                k4 = node.real();
            } else {
                std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            if (!b_miss_params) {
                vector<float> vCamCalib{fx, fy, cx, cy, k1, k2, k3, k4};
                mpCamera = new KannalaBrandt8(vCamCalib);

                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                mK = cv::Mat::eye(3, 3, CV_32F);
                mK.at<float>(0, 0) = fx;
                mK.at<float>(1, 1) = fy;
                mK.at<float>(0, 2) = cx;
                mK.at<float>(1, 2) = cy;
            }

            if (mSensor == System::STEREO || mSensor == System::IMU_STEREO) {
                // Right camera
                // Camera calibration parameters
                cv::FileNode node = fSettings["Camera2.fx"];
                if (!node.empty() && node.isReal()) {
                    fx = node.real();
                } else {
                    std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }
                node = fSettings["Camera2.fy"];
                if (!node.empty() && node.isReal()) {
                    fy = node.real();
                } else {
                    std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.cx"];
                if (!node.empty() && node.isReal()) {
                    cx = node.real();
                } else {
                    std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.cy"];
                if (!node.empty() && node.isReal()) {
                    cy = node.real();
                } else {
                    std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                // Distortion parameters
                node = fSettings["Camera2.k1"];
                if (!node.empty() && node.isReal()) {
                    k1 = node.real();
                } else {
                    std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }
                node = fSettings["Camera2.k2"];
                if (!node.empty() && node.isReal()) {
                    k2 = node.real();
                } else {
                    std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.k3"];
                if (!node.empty() && node.isReal()) {
                    k3 = node.real();
                } else {
                    std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }

                node = fSettings["Camera2.k4"];
                if (!node.empty() && node.isReal()) {
                    k4 = node.real();
                } else {
                    std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                    b_miss_params = true;
                }


                int leftLappingBegin = -1;
                int leftLappingEnd = -1;

                int rightLappingBegin = -1;
                int rightLappingEnd = -1;

                node = fSettings["Camera.lappingBegin"];
                if (!node.empty() && node.isInt()) {
                    leftLappingBegin = node.operator int();
                } else {
                    std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
                }
                node = fSettings["Camera.lappingEnd"];
                if (!node.empty() && node.isInt()) {
                    leftLappingEnd = node.operator int();
                } else {
                    std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
                }
                node = fSettings["Camera2.lappingBegin"];
                if (!node.empty() && node.isInt()) {
                    rightLappingBegin = node.operator int();
                } else {
                    std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
                }
                node = fSettings["Camera2.lappingEnd"];
                if (!node.empty() && node.isInt()) {
                    rightLappingEnd = node.operator int();
                } else {
                    std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
                }

                node = fSettings["Tlr"];
                if (!node.empty()) {
                    mTlr = node.mat();
                    if (mTlr.rows != 3 || mTlr.cols != 4) {
                        std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                        b_miss_params = true;
                    }
                } else {
                    std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                    b_miss_params = true;
                }

                if (!b_miss_params) {
                    static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                    static_cast<KannalaBrandt8 *>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                    mpFrameDrawer->both = true;

                    vector<float> vCamCalib2{fx, fy, cx, cy, k1, k2, k3, k4};
                    mpCamera2 = new KannalaBrandt8(vCamCalib2);

                    static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                    static_cast<KannalaBrandt8 *>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                    std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                    std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                    std::cout << "- Camera: Fisheye" << std::endl;
                    std::cout << "- fx: " << fx << std::endl;
                    std::cout << "- fy: " << fy << std::endl;
                    std::cout << "- cx: " << cx << std::endl;
                    std::cout << "- cy: " << cy << std::endl;
                    std::cout << "- k1: " << k1 << std::endl;
                    std::cout << "- k2: " << k2 << std::endl;
                    std::cout << "- k3: " << k3 << std::endl;
                    std::cout << "- k4: " << k4 << std::endl;

                    std::cout << "- mTlr: \n" << mTlr << std::endl;

                    std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
                }
            }

            if (b_miss_params) {
                return false;
            }

            mpAtlas->AddCamera(mpCamera);
            mpAtlas->AddCamera(mpCamera2);
        } else {
            std::cerr << "*Not Supported Camera Sensor*" << std::endl;
            std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
        }

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;
        camera_fps = fps;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << "- fps: " << fps << endl;
        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO) {
            float fx = mpCamera->getParameter(0);
            cv::FileNode node = fSettings["ThDepth"];
            if (!node.empty() && node.isReal()) {
                mThDepth = node.real();
                mThDepth = mbf * mThDepth / fx;
                cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
            } else {
                std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


        }

        if (mSensor == System::RGBD) {
            cv::FileNode node = fSettings["DepthMapFactor"];
            if (!node.empty() && node.isReal()) {
                mDepthMapFactor = node.real();
                if (fabs(mDepthMapFactor) < 1e-5)
                    mDepthMapFactor = 1;
                else
                    mDepthMapFactor = 1.0f / mDepthMapFactor;
            } else {
                std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

        }

        if (b_miss_params) {
            return false;
        }

        return true;
    }

    bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings) {
        bool b_miss_params = false;
        int nFeatures, nLevels, fIniThFAST, fMinThFAST;
        float fScaleFactor;

        cv::FileNode node = fSettings["ORBextractor.nFeatures"];
        if (!node.empty() && node.isInt()) {
            nFeatures = node.operator int();
        } else {
            std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.scaleFactor"];
        if (!node.empty() && node.isReal()) {
            fScaleFactor = node.real();
        } else {
            std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.nLevels"];
        if (!node.empty() && node.isInt()) {
            nLevels = node.operator int();
        } else {
            std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.iniThFAST"];
        if (!node.empty() && node.isInt()) {
            fIniThFAST = node.operator int();
        } else {
            std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["ORBextractor.minThFAST"];
        if (!node.empty() && node.isInt()) {
            fMinThFAST = node.operator int();
        } else {
            std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        if (b_miss_params) {
            return false;
        }

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::STEREO || mSensor == System::IMU_STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
            mpIniORBextractor = new ORBextractor(5 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        return true;
    }

    bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings) {
        bool b_miss_params = false;

        cv::Mat Tbc;
        cv::FileNode node = fSettings["Tbc"];
        if (!node.empty()) {
            Tbc = node.mat();
            if (Tbc.rows != 4 || Tbc.cols != 4) {
                std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
                b_miss_params = true;
            }
        } else {
            std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
            b_miss_params = true;
        }

        cout << endl;

        cout << "Left camera to Imu Transform (Tbc): " << endl << Tbc << endl;

        float freq = 0.f, Ng = 0.f, Na = 0.f, Ngw = 0.f, Naw = 0.f;

        node = fSettings["IMU.Frequency"];
        if (!node.empty() && node.isInt()) {
            freq = node.operator int();
        } else {
            std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseGyro"];
        if (!node.empty() && node.isReal()) {
            Ng = node.real();
        } else {
            std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.NoiseAcc"];
        if (!node.empty() && node.isReal()) {
            Na = node.real();
        } else {
            std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.GyroWalk"];
        if (!node.empty() && node.isReal()) {
            Ngw = node.real();
        } else {
            std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["IMU.AccWalk"];
        if (!node.empty() && node.isReal()) {
            Naw = node.real();
        } else {
            std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        if (b_miss_params) {
            return false;
        }

        const float sf = sqrt(freq);
        //cout << endl;
        //cout << "IMU frequency: " << freq << " Hz" << endl;
        //cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
        //cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
        //cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
        //cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

        mpImuCalib = new IMU::Calib(Tbc, Ng * sf, Na * sf, Ngw / sf, Naw / sf);

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);


        return true;
    }

    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer) {
        mpViewer = pViewer;
    }

    void Tracking::SetStepByStep(bool bSet) {
        bStepByStep = bSet;
    }

    cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp,
                                      string filename) {
        mImGray = imRectLeft;
        cv::Mat imGrayRight = imRectRight;
        mImRight = imRectRight;

        if (mImGray.channels() == 3) {
            if (mbRGB) {
//            cout<<"CV_RGB2GRAY"<<endl;
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
            } else {
//            cout<<"CV_BGR2GRAY"<<endl;
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
            }
        } else if (mImGray.channels() == 4) {
            if (mbRGB) {
//            cout<<"CV_BGRA2GRAY"<<endl;
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
            } else {
//            cout<<"CV_BGRA2GRAY"<<endl;
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
                cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
            }
        }

        if (mSensor == System::STEREO && !mpCamera2)
            mCurrentFrame = Frame(mImGray, imGrayRight, timestamp, mpORBextractorLeft, mpORBextractorRight,
                                  mpORBVocabulary, mK, mK_ori, mDistCoef, mR, mP,
                                  mK_right, mDistCoef_right, mR_right, mP_right,
                                  mbf, mThDepth, mpCamera);
//        else if(mSensor == System::STEREO && mpCamera2)
//            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
//        else if(mSensor == System::IMU_STEREO && !mpCamera2)
//            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
//        else if(mSensor == System::IMU_STEREO && mpCamera2)
//            mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);
        else
            cerr << "wrong frame create！" << endl;

        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;

        Track();

        return mCurrentFrame.mTcw.clone();
    }


    cv::Mat
    Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename) {
        mImGray = imRGB;
        cv::Mat imDepth = imD;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
            imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

        mCurrentFrame = Frame(mImGray, imDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf,
                              mThDepth, mpCamera);

        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;

        Track();

        return mCurrentFrame.mTcw.clone();
    }


    cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename) {
        mImGray = im;

        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if (mSensor == System::MONOCULAR) {
            if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET || (lastID - initID) < mMaxFrames)
                mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf,
                                      mThDepth);
            else
                mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf,
                                      mThDepth);
        } else if (mSensor == System::IMU_MONOCULAR) {
            if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) {
                mCurrentFrame = Frame(mImGray, timestamp, mpIniORBextractor, mpORBVocabulary, mpCamera, mDistCoef, mbf,
                                      mThDepth, &mLastFrame, *mpImuCalib);
            } else
                mCurrentFrame = Frame(mImGray, timestamp, mpORBextractorLeft, mpORBVocabulary, mpCamera, mDistCoef, mbf,
                                      mThDepth, &mLastFrame, *mpImuCalib);
        }

        if (mState == NO_IMAGES_YET)
            t0 = timestamp;

        mCurrentFrame.mNameFile = filename;
        mCurrentFrame.mnDataset = mnNumDataset;

        lastID = mCurrentFrame.mnId;
        Track();

        return mCurrentFrame.mTcw.clone();
    }


    void Tracking::GrabImuData(const IMU::Point &imuMeasurement) {
        unique_lock<mutex> lock(mMutexImuQueue);
        mlQueueImuData.push_back(imuMeasurement);
    }

    void Tracking::GrabOdoData(const ODO::Point &odoMeasurement) {
        unique_lock<mutex> lock(mMutexOdoQueue);
        mlQueueOdoData.push_back(odoMeasurement);
    }

//更新 mVelocityFromOdo
    void Tracking::PreIntFromIMUPro() {
        static cv::Mat velocityFromOdoLast = cv::Mat::eye(4, 4, CV_32F);
        // 如果未初始化，或者跟踪正常，则为OK
        if (mOdoTpc.empty() || mlastTcwUpdate) {
            mlastTcwUpdate = false;
            mOdoTpc = cv::Mat::eye(4, 4, CV_32F);
        }
        if (!mInitPreOdo) {
            mInitPreOdo = true;
            mVelocityFromOdo = cv::Mat::eye(4, 4, CV_32F);
            mlQueueOdoData.clear();
            Verbose::PrintMess("init PreOdo!!", Verbose::VERBOSITY_QUIET);
            return;
        }
        if (!mpIMUProcessor->getPreOdoUpdate()) {
            mVelocityFromOdo = velocityFromOdoLast;
            cout << "Use pre odo update." << endl;
            return;
        }
        mOdoTpc = mOdoTpc * mpIMUProcessor->getOdoTpcDelta();
        mVelocityFromOdo = mTc_odo * Converter::toCvMatInverse(mOdoTpc) * mTodo_c;
        velocityFromOdoLast = mVelocityFromOdo.clone();
        mpIMUProcessor->setPreOdoUpdate(false);
    }

// update mOdoTpc
    void Tracking::PreintegrateODO() {
        static cv::Mat velocityFromOdoLast = cv::Mat::eye(4, 4, CV_32F);
        //cout << "start PreintegrateODO "<< mlQueueOdoData.size() << endl;
        // 如果未初始化，或者跟踪正常，则为OK
        if (mOdoTpc.empty() || mlastTcwUpdate) {
            mlastTcwUpdate = false;
            mOdoTpc = cv::Mat::eye(4, 4, CV_32F);
        }
        if (mlQueueOdoData.size() == 0) {
            mVelocityFromOdo = velocityFromOdoLast;
            //Verbose::PrintMess("Not ODO data in mlQueueOdoData!!", Verbose::VERBOSITY_QUIET);
            return;
        }
        if (!mInitPreOdo) {
            mInitPreOdo = true;
            mVelocityFromOdo = cv::Mat::eye(4, 4, CV_32F);
            mlQueueOdoData.clear();
            Verbose::PrintMess("init PreOdo !!", Verbose::VERBOSITY_QUIET);
            return;
        }
        //bool debug= false;
        //if(mlQueueOdoData.size() <= 2)
        //{
        //    debug= true;
        //    cout<<"mlQueueOdoData.size() "<<mlQueueOdoData.size()<<endl;
        //}
        mvOdoFromLastFrame.clear();
        mvOdoFromLastFrame.reserve(mlQueueOdoData.size());
        while (true) {
            bool bSleep = false;
            {
                unique_lock<mutex> lock(mMutexOdoQueue);
                if (!mlQueueOdoData.empty()) {
                    ODO::Point *m = &mlQueueOdoData.front();
                    cout.precision(17);
                    if (m) {
                        // 取当前帧到上一个最早的一帧
                        if (m->t < mCurrentFrame.mTimeStamp - 0.001l) {
                            mvOdoFromLastFrame.push_back(*m);
                            mlQueueOdoData.pop_front();
                            //if(debug)
                            //    cout<<"put early"<<endl;
                        } else //只取超出 mCurrentFrame.mTimeStamp-0.001 后的一帧，并且不丢弃（留着下次用）
                        {
                            mvOdoFromLastFrame.push_back(*m);
                            //if(debug)
                            //    cout<<"put later"<<endl;
                            break;
                        }
                    }
                } else {
                    break;
                    bSleep = true;
                }
            }
            if (bSleep)
                usleep(500);
        }
        if (mvOdoFromLastFrame.size() < 2) {
            //cout<<"not enough odo datas! only "<<mvOdoFromLastFrame.size()<<" frames."<<endl;
            mVelocityFromOdo = velocityFromOdoLast;
            return;
        }
        const int n = mvOdoFromLastFrame.size() - 1;
        cv::Mat mOdoTpcDelta = cv::Mat::eye(4, 4, CV_32F);
        for (int i = 0; i < n; i++) {
            double deltaT = mvOdoFromLastFrame[i + 1].t - mvOdoFromLastFrame[i].t;
            double alpha = (mvOdoFromLastFrame[i + 1].rpy.z - mvOdoFromLastFrame[i].rpy.z) / 180.0 * M_PI;
            if (mvOdoFromLastFrame[i + 1].rpy.z * mvOdoFromLastFrame[i].rpy.z < 0
                && fabs(mvOdoFromLastFrame[i + 1].rpy.z) > 170) {
                double delta = mvOdoFromLastFrame[i + 1].rpy.z + mvOdoFromLastFrame[i].rpy.z;
                //更靠近负角，取平均应该是正的
                if (delta < 0)
                    alpha = 180 + delta / 2.0;
                else
                    alpha = -180 + delta / 2.0;
            }
            Eigen::Vector2d delatEncoder(mvOdoFromLastFrame[i + 1].encoder.x - mvOdoFromLastFrame[i].encoder.x,
                                         mvOdoFromLastFrame[i + 1].encoder.y - mvOdoFromLastFrame[i].encoder.y);
            Eigen::Vector3d accOdo = Eigen::Vector3d::Zero();
            cv::Mat T_pre_cur = cv::Mat();
            IntegrateOdo(deltaT, delatEncoder, alpha, accOdo, T_pre_cur);
            mOdoTpcDelta = mOdoTpcDelta * T_pre_cur;
        }
        //cout<<"r p y "<< mvOdoFromLastFrame[n].rpy.x<<" "<< mvOdoFromLastFrame[n].rpy.y
        //<<" "<< mvOdoFromLastFrame[n].rpy.z<<endl;
        //double yaw = Utility::R2ypr(mpIMUProcessor->Rwi_IMU_Pre).x();
        //double pitch = Utility::R2ypr(mpIMUProcessor->Rwi_IMU_Pre).y();
        //double roll = Utility::R2ypr(mpIMUProcessor->Rwi_IMU_Pre).z();
        //cout<<"my r p y "<< roll<<" "<< pitch
        //    <<" "<< yaw<<endl;
        //cout<<"delta "<< mvOdoFromLastFrame[n].rpy.z - yaw << mvOdoFromLastFrame[n].rpy.z<<" " << yaw <<endl;

        mOdoTpc = mOdoTpc * mOdoTpcDelta;
        mVelocityFromOdo = mTc_odo * Converter::toCvMatInverse(mOdoTpc) * mTodo_c;
        velocityFromOdoLast = mVelocityFromOdo.clone();
    }

    void Tracking::PreintegrateIMU() {
        //cout << "start preintegration" << endl;

        if (!mCurrentFrame.mpPrevFrame) {
            Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }

        // cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;

        mvImuFromLastFrame.clear();
        mvImuFromLastFrame.reserve(mlQueueImuData.size());
        if (mlQueueImuData.size() == 0) {
            Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
            mCurrentFrame.setIntegrated();
            return;
        }

        while (true) {
            bool bSleep = false;
            {
                unique_lock<mutex> lock(mMutexImuQueue);
                if (!mlQueueImuData.empty()) {
                    IMU::Point *m = &mlQueueImuData.front();
                    cout.precision(17);
                    if (m->t < mCurrentFrame.mpPrevFrame->mTimeStamp - 0.001l) {
                        mlQueueImuData.pop_front();
                    } else if (m->t < mCurrentFrame.mTimeStamp - 0.001l) {
                        mvImuFromLastFrame.push_back(*m);
                        mlQueueImuData.pop_front();
                    } else {
                        mvImuFromLastFrame.push_back(*m);
                        break;
                    }
                } else {
                    break;
                    bSleep = true;
                }
            }
            if (bSleep)
                usleep(500);
        }


        const int n = mvImuFromLastFrame.size() - 1;
        IMU::Preintegrated *pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,
                                                                                    mCurrentFrame.mImuCalib);

        for (int i = 0; i < n; i++) {
            float tstep;
            cv::Point3f acc, angVel;
            //得到平均的角速度，加速度
            if ((i == 0) && (i < (n - 1))) {
                float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
                float tini = mvImuFromLastFrame[i].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
                       (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tini / tab)) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                          (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tini / tab)) * 0.5f;
                tstep = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mpPrevFrame->mTimeStamp;
            } else if (i < (n - 1)) {
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w) * 0.5f;
                tstep = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
            } else if ((i > 0) && (i == (n - 1))) {
                float tab = mvImuFromLastFrame[i + 1].t - mvImuFromLastFrame[i].t;
                float tend = mvImuFromLastFrame[i + 1].t - mCurrentFrame.mTimeStamp;
                acc = (mvImuFromLastFrame[i].a + mvImuFromLastFrame[i + 1].a -
                       (mvImuFromLastFrame[i + 1].a - mvImuFromLastFrame[i].a) * (tend / tab)) * 0.5f;
                angVel = (mvImuFromLastFrame[i].w + mvImuFromLastFrame[i + 1].w -
                          (mvImuFromLastFrame[i + 1].w - mvImuFromLastFrame[i].w) * (tend / tab)) * 0.5f;
                tstep = mCurrentFrame.mTimeStamp - mvImuFromLastFrame[i].t;
            } else if ((i == 0) && (i == (n - 1))) {
                acc = mvImuFromLastFrame[i].a;
                angVel = mvImuFromLastFrame[i].w;
                tstep = mCurrentFrame.mTimeStamp - mCurrentFrame.mpPrevFrame->mTimeStamp;
            }

            if (!mpImuPreintegratedFromLastKF)
                cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
            mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc, angVel, tstep);
            pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc, angVel, tstep);
        }

        mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
        mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

        mCurrentFrame.setIntegrated();

        Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
    }


    bool Tracking::PredictStateIMU() {
        if (!mCurrentFrame.mpPrevFrame) {
            Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
            return false;
        }

        if (mbMapUpdated && mpLastKeyFrame) {
            const cv::Mat twb1 = mpLastKeyFrame->GetImuPosition();
            const cv::Mat Rwb1 = mpLastKeyFrame->GetImuRotation();
            const cv::Mat Vwb1 = mpLastKeyFrame->GetVelocity();

            const cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
            const float t12 = mpImuPreintegratedFromLastKF->dT;

            cv::Mat Rwb2 = IMU::NormalizeRotation(
                    Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
            cv::Mat twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz +
                           Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
            cv::Mat Vwb2 = Vwb1 + t12 * Gz +
                           Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
            mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);
            mCurrentFrame.mPredRwb = Rwb2.clone();
            mCurrentFrame.mPredtwb = twb2.clone();
            mCurrentFrame.mPredVwb = Vwb2.clone();
            mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        } else if (!mbMapUpdated) {
            const cv::Mat twb1 = mLastFrame.GetImuPosition();
            const cv::Mat Rwb1 = mLastFrame.GetImuRotation();
            const cv::Mat Vwb1 = mLastFrame.mVw;
            const cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);
            const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

            cv::Mat Rwb2 = IMU::NormalizeRotation(
                    Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
            cv::Mat twb2 = twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz +
                           Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
            cv::Mat Vwb2 = Vwb1 + t12 * Gz +
                           Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

            mCurrentFrame.SetImuPoseVelocity(Rwb2, twb2, Vwb2);
            mCurrentFrame.mPredRwb = Rwb2.clone();
            mCurrentFrame.mPredtwb = twb2.clone();
            mCurrentFrame.mPredVwb = Vwb2.clone();
            mCurrentFrame.mImuBias = mLastFrame.mImuBias;
            mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
            return true;
        } else
            cout << "not IMU prediction!!" << endl;

        return false;
    }


    void Tracking::ComputeGyroBias(const vector<Frame *> &vpFs, float &bwx, float &bwy, float &bwz) {
        const int N = vpFs.size();
        vector<float> vbx;
        vbx.reserve(N);
        vector<float> vby;
        vby.reserve(N);
        vector<float> vbz;
        vbz.reserve(N);

        cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
        cv::Mat grad = cv::Mat::zeros(3, 1, CV_32F);
        for (int i = 1; i < N; i++) {
            Frame *pF2 = vpFs[i];
            Frame *pF1 = vpFs[i - 1];
            cv::Mat VisionR = pF1->GetImuRotation().t() * pF2->GetImuRotation();
            cv::Mat JRg = pF2->mpImuPreintegratedFrame->JRg;
            cv::Mat E = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaRotation().t() * VisionR;
            cv::Mat e = IMU::LogSO3(E);
            assert(fabs(pF2->mTimeStamp - pF1->mTimeStamp - pF2->mpImuPreintegratedFrame->dT) < 0.01);

            cv::Mat J = -IMU::InverseRightJacobianSO3(e) * E.t() * JRg;
            grad += J.t() * e;
            H += J.t() * J;
        }

        cv::Mat bg = -H.inv(cv::DECOMP_SVD) * grad;
        bwx = bg.at<float>(0);
        bwy = bg.at<float>(1);
        bwz = bg.at<float>(2);

        for (int i = 1; i < N; i++) {
            Frame *pF = vpFs[i];
            pF->mImuBias.bwx = bwx;
            pF->mImuBias.bwy = bwy;
            pF->mImuBias.bwz = bwz;
            pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
            pF->mpImuPreintegratedFrame->Reintegrate();
        }
    }

    void Tracking::ComputeVelocitiesAccBias(const vector<Frame *> &vpFs, float &bax, float &bay, float &baz) {
        const int N = vpFs.size();
        const int nVar = 3 * N + 3; // 3 velocities/frame + acc bias
        const int nEqs = 6 * (N - 1);

        cv::Mat J(nEqs, nVar, CV_32F, cv::Scalar(0));
        cv::Mat e(nEqs, 1, CV_32F, cv::Scalar(0));
        cv::Mat g = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);

        for (int i = 0; i < N - 1; i++) {
            Frame *pF2 = vpFs[i + 1];
            Frame *pF1 = vpFs[i];
            cv::Mat twb1 = pF1->GetImuPosition();
            cv::Mat twb2 = pF2->GetImuPosition();
            cv::Mat Rwb1 = pF1->GetImuRotation();
            cv::Mat dP12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaPosition();
            cv::Mat dV12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaVelocity();
            cv::Mat JP12 = pF2->mpImuPreintegratedFrame->JPa;
            cv::Mat JV12 = pF2->mpImuPreintegratedFrame->JVa;
            float t12 = pF2->mpImuPreintegratedFrame->dT;
            // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
            J.rowRange(6 * i, 6 * i + 3).colRange(3 * i, 3 * i + 3) += cv::Mat::eye(3, 3, CV_32F) * t12;
            J.rowRange(6 * i, 6 * i + 3).colRange(3 * N, 3 * N + 3) += Rwb1 * JP12;
            e.rowRange(6 * i, 6 * i + 3) = twb2 - twb1 - 0.5f * g * t12 * t12 - Rwb1 * dP12;
            // Velocity v2=v1+g*t+R1*dV12
            J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * i, 3 * i + 3) += -cv::Mat::eye(3, 3, CV_32F);
            J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * (i + 1), 3 * (i + 1) + 3) += cv::Mat::eye(3, 3, CV_32F);
            J.rowRange(6 * i + 3, 6 * i + 6).colRange(3 * N, 3 * N + 3) -= Rwb1 * JV12;
            e.rowRange(6 * i + 3, 6 * i + 6) = g * t12 + Rwb1 * dV12;
        }

        cv::Mat H = J.t() * J;
        cv::Mat B = J.t() * e;
        cv::Mat x(nVar, 1, CV_32F);
        cv::solve(H, B, x);

        bax = x.at<float>(3 * N);
        bay = x.at<float>(3 * N + 1);
        baz = x.at<float>(3 * N + 2);

        for (int i = 0; i < N; i++) {
            Frame *pF = vpFs[i];
            x.rowRange(3 * i, 3 * i + 3).copyTo(pF->mVw);
            if (i > 0) {
                pF->mImuBias.bax = bax;
                pF->mImuBias.bay = bay;
                pF->mImuBias.baz = baz;
                pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
            }
        }
    }

    void Tracking::ResetFrameIMU() {
        // TODO To implement...
    }


// note LOST if(mState==LOST) → CreateMapInAtlas  →　mpAtlas->CreateNewMap() → StereoInitialization
    void Tracking::Track() {

        Map *pCurrentMap = mpAtlas->GetCurrentMap();

        if (mState != NO_IMAGES_YET) {
            if (mLastFrame.mTimeStamp > mCurrentFrame.mTimeStamp) {
                cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
                unique_lock<mutex> lock(mMutexImuQueue);
                mlQueueImuData.clear();
                CreateMapInAtlas();
                return;
            }
        }

        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }

        mLastProcessedState = mState;

        if (mSensor == System::STEREO) {
            PreIntFromIMUPro();
        }

        mbCreatedMap = false;

        // Get Map Mutex -> Map cannot be changed
        unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

        mbMapUpdated = false;

        int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
        int nMapChangeIndex = pCurrentMap->GetLastMapChange();
        if (nCurMapChangeIndex > nMapChangeIndex) {
            // cout << "Map update detected" << endl;
            pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
            mbMapUpdated = true;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now(), t2 = std::chrono::steady_clock::now(), t3 = std::chrono::steady_clock::now(), t4 = std::chrono::steady_clock::now();
        if (mState == NOT_INITIALIZED) {
            if (mSensor == System::STEREO || mSensor == System::RGBD || mSensor == System::IMU_STEREO) {
                if (mbRestart) {
                    ORB_SLAM3::Verbose::PrintMess("ReStereoInitialization", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    ReStereoInitialization();
                    ORB_SLAM3::Verbose::PrintMess("ReStereoInitialization over",
                                                  ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                } else {
                    ORB_SLAM3::Verbose::PrintMess("StereoInitialization", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    StereoInitialization();
                    ORB_SLAM3::Verbose::PrintMess("StereoInitialization over",
                                                  ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                }
            } else {
                MonocularInitialization();
            }

            ORB_SLAM3::Verbose::PrintMess("mpFrameDrawer->Update", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
            mpFrameDrawer->Update(this);
            ORB_SLAM3::Verbose::PrintMess("mpFrameDrawer->Update over", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);

            if (mState != OK) // If rightly initialized, mState=OK
            {
                ORB_SLAM3::Verbose::PrintMess("Initialization !=OK", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                // Update drawer
                mpFrameDrawer->Update(this);
                //这个判断应该是没用的
                if (!mCurrentFrame.mTcw.empty()) {
                    mLastFrame = Frame(mCurrentFrame);
                    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                    ORB_SLAM3::Verbose::PrintMess("start CreateNewKeyFrameWithoutMPs",
                                                  ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    CreateNewKeyFrameWithoutMPs();
                    if (mpReferenceKF) {
                        ORB_SLAM3::Verbose::PrintMess("update mpReferenceKF", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                        UpdateLastTcw();
                        mCurrentFrame.mpReferenceKF = mpReferenceKF;
                        //为了传达给imuprocess和轨迹，因为这个系统不丢失
                        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
                        mlRelativeFramePoses.push_back(Tcr);
                        mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                        mlbLost.push_back(mState == LOST);
                        ORB_SLAM3::Verbose::PrintMess("update mpReferenceKF over",
                                                      ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    } else {
                        cout << "init fall mpReferenceKF is null" << endl;
                    }
                }
                ORB_SLAM3::Verbose::PrintMess("Initialization !=OK over", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                return;
            }

            if (mpAtlas->GetAllMaps().size() == 1 && !mbRestart) {
                mnFirstFrameId = mCurrentFrame.mnId;
            }

            if (mbRestart) {
                mbRestart = false;
                Verbose::PrintMess("ReStereoInitialization finished", Verbose::VERBOSITY_QUIET);
            } else
                Verbose::PrintMess("StereoInitialization finished", Verbose::VERBOSITY_QUIET);
        } else {
            // System is initialized. Track Frame.
            bool bOK;

            t1 = std::chrono::steady_clock::now();
            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            if (!mbOnlyTracking) {
                // State OK
                // Local Mapping is activated. This is the normal behaviour, unless
                // you explicitly activate the "only tracking" mode.
                if (mState == OK) {
                    ORB_SLAM3::Verbose::PrintMess("Track ego", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                    // Local Mapping might have changed some MapPoints tracked in last frame
                    CheckReplacedInLastFrame();

                    if ((mVelocity.empty() && !pCurrentMap->isImuInitialized()) ||
                        mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
                        Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_QUIET);
                        bOK = TrackReferenceKeyFrame();
                        Verbose::PrintMess("TRACK: Track with respect to the reference KF finish",
                                           Verbose::VERBOSITY_QUIET);
                    } else {
                        //Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_QUIET);
                        bOK = TrackWithMotionModel();
                        if (!bOK) {
                            bOK = TrackReferenceKeyFrame();
                        }
                    }


                    if (!bOK) {
                        if (mCurrentFrame.mnId <= (mnLastRelocFrameId + mnFramesToResetIMU) &&
                            (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) {
                            mState = LOST;
                        } else if (pCurrentMap->KeyFramesInMap() > 10) {
                            cout << "RECENTLY_LOST KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                            mState = RECENTLY_LOST;
                            mTimeStampLost = mCurrentFrame.mTimeStamp;
                            //mCurrentFrame.SetPose(mLastFrame.mTcw);
                        } else {
                            mState = LOST;
                        }
                    }
                    ORB_SLAM3::Verbose::PrintMess("Track ego over", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
                } else {
                    if (mState == RECENTLY_LOST) {
                        Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);
                        bOK = true;

                        // TODO fix relocalization
                        bOK = Relocalization();
                        if (!bOK) {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK = false;
                        }
                    } else if (mState == LOST) {

                        Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                        if (pCurrentMap->KeyFramesInMap() < 10) {
                            mpSystem->ResetActiveMap();
                            cout << "Reseting current map..." << endl;
                        } else
                            CreateMapInAtlas();

                        if (mpLastKeyFrame)
                            mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

                        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                        return;
                    }
                }
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            ////cout<<"mLastFrame.mTcw.empty(): 0 "<<mLastFrame.mTcw.empty()<<endl;
            // If we have an initial estimation of the camera pose and matching. Track the local map.

            t2 = std::chrono::steady_clock::now();

            ORB_SLAM3::Verbose::PrintMess("Track TrackLocalMap", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
            if (!mbOnlyTracking) {
                if (bOK)
                    bOK = TrackLocalMap();
                if (!bOK)
                    cout << "Fail to track local map!" << endl;
            }
            ORB_SLAM3::Verbose::PrintMess("Track TrackLocalMap over", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);

            t3 = std::chrono::steady_clock::now();

            //判断跟踪局部地图是否成功
            if (bOK)
                mState = OK;
            else if (mState == OK) {

                mState = LOST; // visual to lost
                cout<<"mState = LOST;"<<endl;
                if (mCurrentFrame.mnId > mnLastRelocFrameId + mMaxFrames) {
                    mTimeStampLost = mCurrentFrame.mTimeStamp;
                }
            }

            ORB_SLAM3::Verbose::PrintMess("Track post", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);

            // Update drawer
            mpFrameDrawer->Update(this);
            if (!mCurrentFrame.mTcw.empty())
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            //cout<<"mLastFrame.mTcw.empty(): 3 "<<mLastFrame.mTcw.empty()<<endl;
            if (bOK || mState == RECENTLY_LOST) {
                //cout<<"mLastFrame.mTcw.empty(): 4 "<<mLastFrame.mTcw.empty()<<endl;
                // Update motion model
                if (!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty()) {
                    if(USE_ODO_PREDICT){
                        mVelocity = mVelocityFromOdo;
                    }else{
                        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                        mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                        mVelocity = mCurrentFrame.mTcw * LastTwc;
                    }
//                    std::vector<float> velocityDebug = Converter::PrintSE(mVelocity, "mVelocity ", true);
//                    std::vector<float> velocityFromOdoDebug = Converter::PrintSE(mVelocityFromOdo, "mVelocityFromOdo ", true);
//                CheckOdoVelocity();
                } else
                    mVelocity = cv::Mat();

                // Clean VO matches
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                     lit != lend; lit++) {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                mlpTemporalPoints.clear();

                bool bNeedKF = false;
                if(bOK)
                    bNeedKF = NeedNewKeyFrame();

                // Check if we need to insert a new keyframe
                if (bNeedKF) {
                    if (bOK ||
                        (mState == RECENTLY_LOST &&
                         (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))) {
                        CreateNewKeyFrame();
                    }
                }
                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame. Only has effect if lastframe is tracked
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }
            }

            ORB_SLAM3::Verbose::PrintMess("Track post over", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
            if (mState == RECENTLY_LOST)
                cout << "RECENTLY_LOST curFrameId: " << mCurrentFrame.mnId << endl;
            else if (mState == LOST)
                cout << "LOST curFrameId: " << mCurrentFrame.mnId << endl;
            // Reset if the camera get lost soon after initialization
//            mState = LOST;
            if (mState == LOST) {
                //fixme 保存每一个小地图
                if (pCurrentMap->KeyFramesInMap() <= 5) {
//                    mRotAccumulate = 0.f;
//                    mTransAccumulate = 0.f;
                    cout << "KeyFramesInMap < 5 mpSystem ResetActiveMap." << endl;
                    //mpSystem->ResetActiveMap();
                    //return;
                } else {
                    mRotAccumulate = 0.f;
                    mTransAccumulate = 0.f;
                    static int count = 0;
                    cout << "KeyFramesInMap > 5 mpSystem ResetActiveMap. "<<count++ << endl;
                }
//                CreateMapInAtlas();
                RestartTrack();
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
            if (mbVisual)
                mImGray.copyTo(mImKFGray);
        }

        t4 = std::chrono::steady_clock::now();
        if (mState == OK || mState == RECENTLY_LOST) {
            // Store frame pose information to retrieve the complete camera trajectory afterwards.
            if (!mCurrentFrame.mTcw.empty()) {
                usefulForIMU++;
                UpdateLastTcw();
                cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
                mlRelativeFramePoses.push_back(Tcr);
                mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
                mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
                mlbLost.push_back(mState == LOST);

                mpTrackingTime->ExtractFeature.update(mCurrentFrame.mTimeORB_Ext);
                mpTrackingTime->StereoMatch.update(mCurrentFrame.mTimeStereoMatch);
                mpTrackingTime->TrackFrame.update(
                        std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t2 - t1).count());
                mpTrackingTime->TrackMap.update(
                        std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t3 - t2).count());
                mpTrackingTime->PostTrack.update(
                        std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t4 - t3).count());
                mpTrackingTime->timesCur.push_back(GetTimeNow());
            } else {
                usefulForIMU = 0;
                //cout<<" tra6 ";
                // This can happen if tracking is lost
                mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
                mlpReferences.push_back(mlpReferences.back());
                mlFrameTimes.push_back(mlFrameTimes.back());
                mlbLost.push_back(mState == LOST);
            }
        } else
            usefulForIMU = 0;

        if (usefulForIMU > WINDOW_SIZE)
            mpIMUProcessor->setIsUpdateBias(true);
        else
            mpIMUProcessor->setIsUpdateBias(false);
        // push the time log of current frame into the vector
    }

    void Tracking::UpdateLastTcw() {
        mlastMapmLastFrameTcw.release();
        mlastMapmLastFrameTcw = mCurrentFrame.mTcw.clone();
        mlastTcwUpdate = true;
    }

    void Tracking::StereoInitialization() {
        if (mCurrentFrame.N > 500 && mCurrentFrame.N_R > 500) {
            if (mSensor == System::IMU_STEREO) {
                if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated) {
                    cout << "not IMU meas" << endl;
                    return;
                }

                if (cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA - mLastFrame.mpImuPreintegratedFrame->avgA) <
                    0.5) {
                    cout << "not enough acceleration" << endl;
                    return;
                }

                if (mpImuPreintegratedFromLastKF)
                    delete mpImuPreintegratedFromLastKF;

                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(-0.0, 0.0, -0.0), *mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            }

            // Set Frame pose to the origin (In case of inertial SLAM to imu)
            if (mSensor == System::IMU_STEREO) {
                cv::Mat Rwb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0, 3).colRange(0, 3).clone();
                cv::Mat twb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0, 3).col(3).clone();
                mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, cv::Mat::zeros(3, 1, CV_32F));
            } else {
                //mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
                if (mVelocityFromOdo.empty())
                    cerr << "mVelocityFromOdo empty()" << endl;
                if (mlastMapmLastFrameTcw.empty())
                    cerr << "mlastMapmLastFrameTcw empty()" << endl;
                if (!mbInertialFirst) {
                    mbInertialFirst = true;
                    Eigen::Matrix3d R0 = mpIMUProcessor->R_init;

                    cv::Mat Twi_init = cv::Mat::eye(4, 4, CV_32F);
                    Converter::toCvMat(R0).copyTo(Twi_init.rowRange(0, 3).colRange(0, 3));

                    cv::Mat Tic = cv::Mat::eye(4, 4, CV_32F);
                    Converter::toCvMat(mpIMUProcessor->ric[0]).copyTo(Tic.rowRange(0, 3).colRange(0, 3));
                    Converter::toCvMat(mpIMUProcessor->tic[0]).copyTo(Tic.rowRange(0, 3).col(3));
                    cv::Mat Tci = Converter::toCvMatInverse(Tic);

                    cv::Mat Twc_init = Tci * Twi_init * Tic;
                    mCurrentFrame.SetPose(Converter::toCvMatInverse(Twc_init));
                } else
                    mCurrentFrame.SetPose(mVelocityFromOdo * mlastMapmLastFrameTcw);
            }

            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

            // Insert KeyFrame in the map
            mpAtlas->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            if (!mpCamera2) {
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    float z = mCurrentFrame.mvDepth[i];
                    if (z > 0) {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                        pNewMP->AddObservation(pKFini, i);
                        pKFini->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    }
                }
            } else {
                for (int i = 0; i < mCurrentFrame.Nleft; i++) {
                    int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                    if (rightIndex != -1) {
                        cv::Mat x3D = mCurrentFrame.mvStereo3Dpoints[i];

                        MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                        pNewMP->AddObservation(pKFini, i);
                        pNewMP->AddObservation(pKFini, rightIndex + mCurrentFrame.Nleft);

                        pKFini->AddMapPoint(pNewMP, i);
                        pKFini->AddMapPoint(pNewMP, rightIndex + mCurrentFrame.Nleft);

                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft] = pNewMP;
                    }
                }
            }

            Verbose::PrintMess("New Map has " + to_string(mpAtlas->MapPointsInMap()) + " points",
                               Verbose::VERBOSITY_QUIET);

            mpLocalMapper->InsertKeyFrame(pKFini);

            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;
            mnLastRelocFrameId = mCurrentFrame.mnId;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;
            mLastFrame = Frame(mCurrentFrame);

            mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

            mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;
            if (mbVisual)
                mImGray.copyTo(mImKFGray);
        }
    }

    void Tracking::ReStereoInitialization() {
        if (mVelocityFromOdo.empty())
            cerr << "mVelocityFromOdo empty()" << endl;
        if (mlastMapmLastFrameTcw.empty())
            cerr << "mlastMapmLastFrameTcw empty()" << endl;
        mCurrentFrame.SetPose(mVelocityFromOdo * mlastMapmLastFrameTcw);
        mCurrentFrame.UpdatePoseMatrices();
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        if (!mpLocalMapper->SetNotStop(true)) {
            cout << "!mpLocalMapper->SetNotStop(true)" << endl;
            return;
        }
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
            cout << "mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()" << endl;
            return;
        }
        if (!mpLocalMapper->AcceptKeyFrames()) {
            cout << "!mpLocalMapper->AcceptKeyFrames()" << endl;
            mpLocalMapper->InterruptBA();
            return;
        }

        //如果不能保持局部建图器开启的状态,就无法顺利插入关键帧
        if (mCurrentFrame.N > 500 && mCurrentFrame.N_R > 500) {
            mvpLocalMapPoints = vector<MapPoint *>(NULL);

            //mCurrentFrame.SetPose(mVelocityFromOdo * mlastMapmLastFrameTcw);
            // Converter::PrintSE(mCurrentFrame.mTcw);

            // Create KeyFrame
            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

            // Create MapPoints and asscoiate to KeyFrame
            if (!mpCamera2) {
                // 根据Tcw计算mRcw、mtcw和mRwc、mOw
                //mCurrentFrame.UpdatePoseMatrices();
                vector<pair<float, int> > vDepthIdx;
                vDepthIdx.reserve(mCurrentFrame.N);
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    float z = mCurrentFrame.mvDepth[i];
                    if (z > 0) {
                        vDepthIdx.push_back(make_pair(z, i));
                    }
                }
                if (!vDepthIdx.empty()) {
                    sort(vDepthIdx.begin(), vDepthIdx.end());
                    int nPoints = 0;
                    for (size_t j = 0; j < vDepthIdx.size(); j++) {
                        int i = vDepthIdx[j].second;
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                        // 这些添加属性的操作是每次创建MapPoint后都要做的
                        pNewMP->AddObservation(pKFini, i);
                        pKFini->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        mvpLocalMapPoints.push_back(pNewMP);

                        nPoints++;

                        if(vDepthIdx[j].first>mThDepth || nPoints>200)
                            break;
                    }
                }

            }

//            if (mvpLocalMapPoints.size() < 100) {
//                delete pKFini;
//                pKFini = NULL;
//
//                for (int i = 0; i < mvpLocalMapPoints.size(); i++) {
//                    mpAtlas->EraseMapPoint(mvpLocalMapPoints[i]);
//                    delete mvpLocalMapPoints[i];
//                    mvpLocalMapPoints[i] = NULL;
//                }
//                mvpLocalMapPoints.clear();
//                return;
//            }

            // Insert KeyFrame in the map
            mpAtlas->AddKeyFrame(pKFini);
            mpAtlas->AddRecentInitId(pKFini);

            Verbose::PrintMess(
                    "\nframe ID: " + to_string(mCurrentFrame.mnId) + ". Keyframe ID: " + to_string(pKFini->mnId) +
                    ". New Map created with " + to_string(mvpLocalMapPoints.size()) + " points. ",
                    Verbose::VERBOSITY_QUIET,
                    true);
            //Converter::PrintSE(mVelocityFromOdo);
            //Verbose::PrintMess("New Map has " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

            mpLocalMapper->InsertKeyFrame(pKFini);
            mpLocalMapper->SetNotStop(false);

            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;
            //mnLastRelocFrameId = mCurrentFrame.mnId;

            mvpLocalKeyFrames.clear();
            mvpLocalKeyFrames.push_back(pKFini);

            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

            //mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.clear();
            //mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

            //mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mLastFrame = Frame(mCurrentFrame);
            mState = OK;
            if (mbVisual)
                mImGray.copyTo(mImKFGray);
        }
    }

    void Tracking::MonocularInitialization() {

        if (!mpInitializer) {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > 100) {

                mInitialFrame = Frame(mCurrentFrame);
                mLastFrame = Frame(mCurrentFrame);
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                if (mpInitializer)
                    delete mpInitializer;

                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                if (mSensor == System::IMU_MONOCULAR) {
                    if (mpImuPreintegratedFromLastKF) {
                        delete mpImuPreintegratedFromLastKF;
                    }
                    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
                    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

                }
                return;
            }
        } else {
            if (((int) mCurrentFrame.mvKeys.size() <= 100) ||
                ((mSensor == System::IMU_MONOCULAR) && (mLastFrame.mTimeStamp - mInitialFrame.mTimeStamp > 1.0))) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                return;
            }

            // Find correspondences
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches,
                                                           100);

            // Check if there are enough correspondences
            if (nmatches < 100) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }

            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

            if (mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn, mCurrentFrame.mvKeysUn, mvIniMatches, Rcw,
                                                  tcw, mvIniP3D, vbTriangulated)) {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

                CreateInitialMapMonocular();

                // Just for video
                // bStepByStep = true;
            }
        }
    }


    void Tracking::CreateInitialMapMonocular() {
        // Create KeyFrames
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        if (mSensor == System::IMU_MONOCULAR)
            pKFini->mpImuPreintegrated = (IMU::Preintegrated *) (NULL);


        if (mbVisual)
            mImGray.copyTo(mImKFGray);
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpAtlas->AddKeyFrame(pKFini);
        mpAtlas->AddKeyFrame(pKFcur);

        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);
            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpAtlas->GetCurrentMap());

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            pMP->ComputeDistinctiveDescriptors();
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            //Add to Map
            mpAtlas->AddMapPoint(pMP);
        }


        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        std::set<MapPoint *> sMPs;
        sMPs = pKFini->GetMapPoints();

        // Bundle Adjustment
        Verbose::PrintMess("\nNew Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points",
                           Verbose::VERBOSITY_QUIET);
        Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(), 20);

        pKFcur->PrintPointDistribution();

        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth;
        if (mSensor == System::IMU_MONOCULAR)
            invMedianDepth = 4.0f / medianDepth; // 4.0f
        else
            invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 50) // TODO Check, originally 100 tracks
        {
            Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_NORMAL);
            mpSystem->ResetActiveMap();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
                pMP->UpdateNormalAndDepth();
            }
        }

        if (mSensor == System::IMU_MONOCULAR) {
            pKFcur->mPrevKF = pKFini;
            pKFini->mNextKF = pKFcur;
            pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),
                                                                  pKFcur->mImuCalib);
        }


        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);
        mpLocalMapper->mFirstTs = pKFcur->mTimeStamp;

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;
        mnLastRelocFrameId = mInitialFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        // Compute here initial velocity
        vector<KeyFrame *> vKFs = mpAtlas->GetAllKeyFrames();

        cv::Mat deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();
        mVelocity = cv::Mat();
        Eigen::Vector3d phi = LogSO3(Converter::toMatrix3d(deltaT.rowRange(0, 3).colRange(0, 3)));


        double aux = (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp) /
                     (mCurrentFrame.mTimeStamp - mInitialFrame.mTimeStamp);
        phi *= aux;

        mLastFrame = Frame(mCurrentFrame);

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;

        initID = pKFcur->mnId;
    }

    void Tracking::RestartTrack() {
        cout << "RestartTrack" << endl;
        //cout<<"mLastFrame.mTcw.clone():1 "<<mLastFrame.mTcw.empty()<<endl;

        //mnLastInitFrameId = mCurrentFrame.mnId;
        //mpAtlas->CreateNewMap();
        mbSetInit = false;

        //mnInitialFrameId = mCurrentFrame.mnId+1;
        mState = NO_IMAGES_YET;

        // Restart the variable with information about the last KF
        mVelocity = cv::Mat();
        //mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
        Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId + 1), Verbose::VERBOSITY_NORMAL);
        mbVO = false; // Init value for know if there are enough MapPoints in the last KF

        ORB_SLAM3::Verbose::PrintMess("RestartTrack 2", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        if (mpLastKeyFrame)
            mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

        ORB_SLAM3::Verbose::PrintMess("RestartTrack 3", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        //这里设计到浅拷贝和深拷贝了，应该用clone 初始化里会新建mLastFrame，所以一定不为空
        //cout<<"mLastFrame.mTcw.clone():3 "<<mLastFrame.mTcw.empty()<<endl;
        //mlastMapmLastFrameTcw = mLastFrame.mTcw.clone();
        //if(mlastMapmLastFrameTcw.empty())
        //{
        //    cout<<"mlastMapmLastFrameTcw empty!!!!!!!!!"<<endl;
        //}
        mLastFrame = Frame();
        mCurrentFrame = Frame();
        mvIniMatches.clear();
        ORB_SLAM3::Verbose::PrintMess("RestartTrack 4", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);

        mbRestart = true;

        mCurrentFrame.SetPose(mVelocityFromOdo * mlastMapmLastFrameTcw);

        ORB_SLAM3::Verbose::PrintMess("RestartTrack 5", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        if (mpReferenceKF) {
            ORB_SLAM3::Verbose::PrintMess("RestartTrack 6", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
            UpdateLastTcw();
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
            //为了传达给imuprocess和轨迹，因为这个系统不丢失
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
            ORB_SLAM3::Verbose::PrintMess("RestartTrack 7", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        } else
            cout << "restart mpReferenceKF is null" << endl;

        //if(mpReferenceKF)
        //    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    }

    void Tracking::CreateMapInAtlas() {
        cout << "CreateMapInAtlas" << endl;

        mnLastInitFrameId = mCurrentFrame.mnId;
        mpAtlas->CreateNewMap();
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
            mpAtlas->SetInertialSensor();
        mbSetInit = false;

        mnInitialFrameId = mCurrentFrame.mnId + 1;
        mState = NO_IMAGES_YET;

        // Restart the variable with information about the last KF
        mVelocity = cv::Mat();
        mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
        Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId + 1), Verbose::VERBOSITY_NORMAL);
        mbVO = false; // Init value for know if there are enough MapPoints in the last KF

        if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR) {
            if (mpInitializer)
                delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }

        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mpImuPreintegratedFromLastKF) {
            delete mpImuPreintegratedFromLastKF;
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(), *mpImuCalib);
        }

        if (mpLastKeyFrame)
            mpLastKeyFrame = static_cast<KeyFrame *>(NULL);

        if (mpReferenceKF)
            mpReferenceKF = static_cast<KeyFrame *>(NULL);

        //这里设计到浅拷贝和深拷贝了，应该用clone 初始化里会新建mLastFrame，所以一定不为空
        //mlastMapmLastFrameTcw = mLastFrame.mTcw.clone();
        //cout<<"xyz: "<<mlastMapmLastFrameTcw.at<float>(0,3)<<" "<<mlastMapmLastFrameTcw.at<float>(1,3)<<" "<<mlastMapmLastFrameTcw.at<float>(2,3)<<endl;
        mLastFrame = Frame();
        mCurrentFrame = Frame();
        mvIniMatches.clear();

        mbCreatedMap = true;
    }

    void Tracking::CheckReplacedInLastFrame() {
        for (int i = 0; i < mLastFrame.N; i++) {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (pMP) {
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }
    }

    void Tracking::VisualPointMatch(string s) {
        if (!mbVisual)
            return;
        // plot the matches
        cv::Mat pic_Temp(mImGray.rows, mImGray.cols * 2, mImGray.type());
        (mImGray.rowRange(0, mImGray.rows).colRange(0, mImGray.cols)).copyTo(pic_Temp.colRange(0, mImGray.cols));
        (mImKFGray.rowRange(0, mImKFGray.rows).colRange(0, mImKFGray.cols)).copyTo(
                pic_Temp.colRange(mImGray.cols, mImGray.cols * 2));
        cv::cvtColor(pic_Temp, pic_Temp, 8);
        vector<cv::KeyPoint> keypoints1;
        vector<cv::KeyPoint> keypoints2;

        cv::putText(pic_Temp, "cur", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 0, 0), 1, 8);

        cout << "mCurrentFrame.mvpMapPoints: " << mCurrentFrame.mvpMapPoints.size() << endl;

        ORBmatcher matcher(0.9, true);

        for (int i = 0; i < mCurrentFrame.mvpMapPoints.size(); i++) {
            if (!mCurrentFrame.mvpMapPoints[i])
                continue;
            cv::Point Point_1, Point_2;
            cv::KeyPoint point_1, point_2;
            point_1 = mCurrentFrame.mvKeysUn[i];
            uint16_t pkfPointIdx = 65535;
            if (s == "Reference") {
                pkfPointIdx = get<0>(mCurrentFrame.mvpMapPoints[i]->GetIndexInKeyFrame(mpReferenceKF));
                point_2 = mpReferenceKF->mvKeys[pkfPointIdx];
            } else if (s == "Last") {
                // if (vLastIdx[i].second == -1)
                //     continue;
                // pkfPointIdx = vLastIdx[i].second;
                cerr << "暂时不支持" <<endl;
//                for (int j = 0; j < matcher.vLastIdx.size(); j++) {
//                    if (i == matcher.vLastIdx[j].first) {
//                        pkfPointIdx = matcher.vLastIdx[j].second;
//                        point_2 = mLastFrame.mvKeysUn[pkfPointIdx];
//                    }
//                }
            }
            if (pkfPointIdx == 65535)
                continue;
            Point_1.x = point_1.pt.x;
            Point_1.y = point_1.pt.y;
            Point_2.x = point_2.pt.x + mImGray.cols;
            Point_2.y = point_2.pt.y;

            cv::line(pic_Temp, Point_1, Point_2, cv::Scalar(0, 0, 255), 2, CV_AA);
            cv::putText(pic_Temp, std::to_string(i), Point_1, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 0, 0), 1,
                        8);
            cv::putText(pic_Temp, std::to_string(pkfPointIdx), Point_2, cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(0, 200, 0), 1, 8);
        }

        static int i = 0;
        i++;
        s = "./" + s + '/';
        //createDirectory(s);
        cv::imwrite(s + "/" + to_string(i) + ".png", pic_Temp);
    }

    bool Tracking::TrackReferenceKeyFrame() {
//    arma::wall_clock timer;
        // Compute Bag of Words vector
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.7, true);
        vector<MapPoint *> vpMapPointMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

        if (nmatches < 15) {
            cout << "TRACK_REF_KF: Less than 15 matches!!\n";
            return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.SetPose(mLastFrame.mTcw);

        //mCurrentFrame.PrintPointDistribution();
        VisualPointMatch("Reference");

        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            //if(i >= mCurrentFrame.Nleft) break;
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    if (i < mCurrentFrame.Nleft) {
                        pMP->mbTrackInView = false;
                    } else {
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        // TODO check these conditions
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
            return true;
        else
            return nmatchesMap >= 10;
    }

    void Tracking::UpdateLastFrame() {
        // Update pose according to reference keyframe
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        //cout<<"UpdateLastFrame 1"<<endl;
        cv::Mat Tlr = mlRelativeFramePoses.back();
        mLastFrame.SetPose(Tlr * pRef->GetPose());

        //cout<<"UpdateLastFrame 2"<<endl;
        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR ||
            !mbOnlyTracking)
            return;

        //cout<<"UpdateLastFrame 3"<<endl;
        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        //cout<<"UpdateLastFrame 4"<<endl;
        for (int i = 0; i < mLastFrame.N; i++) {
            float z = mLastFrame.mvDepth[i];
            if (z > 0) {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }
        //cout<<"UpdateLastFrame 5"<<endl;

        if (vDepthIdx.empty())
            return;

        //cout<<"UpdateLastFrame 6"<<endl;
        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        int nPoints = 0;
        //cout<<"UpdateLastFrame 7"<<endl;
        for (size_t j = 0; j < vDepthIdx.size(); j++) {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            //cout<<"UpdateLastFrame 8"<<endl;
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1) {
                bCreateNew = true;
            }

            //cout<<"UpdateLastFrame 9"<<endl;
            if (bCreateNew) {
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpAtlas->GetCurrentMap(), &mLastFrame, i);

                mLastFrame.mvpMapPoints[i] = pNewMP;

                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else {
                nPoints++;
            }

            //cout<<"UpdateLastFrame 10"<<endl;
            if (vDepthIdx[j].first > mThDepth && nPoints > 100) {
                break;
            }
        }
        //cout<<"UpdateLastFrame 11"<<endl;
    }

    bool Tracking::TrackWithMotionModel() {
//    arma::wall_clock timer;
        ORBmatcher matcher(0.9, true);
        //cout<<"TrackWithMotionModel 1"<<endl;
        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        UpdateLastFrame();

        //cout<<"TrackWithMotionModel 2"<<endl;
        if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId > mnLastRelocFrameId + mnFramesToResetIMU)) {
            // Predict ste with IMU if it is initialized and it doesnt need reset
            PredictStateIMU();
            return true;
        } else {
            //mCurrentFrame.SetPose(mVelocityFromOdo*mLastFrame.mTcw);
            mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
        }

        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        // Project points seen in previous frame
        int th;

        if (mSensor == System::STEREO)
            th = 7;
        else
            th = 15;

        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th,
                                                  mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);

        // If few matches, uses a wider window search
        if (nmatches < 20) {
            Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * th,
                                                  mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR);
            Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);
        }
        VisualPointMatch("Last");

        if (nmatches < 20) {
            Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
                return true;
            else
                return false;
        }

        // Optimize frame pose with all matches
        Optimizer::PoseOptimization(&mCurrentFrame);

        // Discard outliers
        int nmatchesMap = 0;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    if (i < mCurrentFrame.Nleft) {
                        pMP->mbTrackInView = false;
                    } else {
                        pMP->mbTrackInViewR = false;
                    }
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
            return true;
        else
            return nmatchesMap >= 10;
    }

    bool Tracking::TrackLocalMap() {
        // We have an estimation of the camera pose and some map points tracked in the frame.
        // We retrieve the local map and try to find matches to points in the local map.
        mTrackedFr++;

        UpdateLocalMap();
        SearchLocalPoints();

        // TOO check outliers before PO
        int aux1 = 0, aux2 = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i]) {
                aux1++;
                if (mCurrentFrame.mvbOutlier[i])
                    aux2++;
            }

        int inliers;
        if (!mpAtlas->isImuInitialized())
            Optimizer::PoseOptimization(&mCurrentFrame);
        else {
            if (mCurrentFrame.mnId <= mnLastRelocFrameId + mnFramesToResetIMU) {
                Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
                Optimizer::PoseOptimization(&mCurrentFrame);
            } else {
                // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
                if (!mbMapUpdated) //  && (mnMatchesInliers>30))
                {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastFrame(
                            &mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                } else {
                    Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                    inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(
                            &mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
                }
            }
        }

        aux1 = 0, aux2 = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
            if (mCurrentFrame.mvpMapPoints[i]) {
                aux1++;
                if (mCurrentFrame.mvbOutlier[i])
                    aux2++;
            }

        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (!mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                } else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        mpLocalMapper->mnMatchesInliers = mnMatchesInliers;
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
            return false;

        if ((mnMatchesInliers > 10) && (mState == RECENTLY_LOST))
            return true;


        if (mSensor == System::IMU_MONOCULAR) {
            if (mnMatchesInliers < 15) {
                return false;
            } else
                return true;
        } else if (mSensor == System::IMU_STEREO) {
            if (mnMatchesInliers < 15) {
                return false;
            } else
                return true;
        } else {
            if (mnMatchesInliers < 30)
                return false;
            else
                return true;
        }
    }

    bool Tracking::NeedNewKeyFrame() {
        if (((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) &&
            !mpAtlas->GetCurrentMap()->isImuInitialized()) {
            if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
                return true;
            else if (mSensor == System::IMU_STEREO && (mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.25)
                return true;
            else
                return false;
        }

        if (mbOnlyTracking)
            return false;

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
            return false;
        }

        // Return false if IMU is initialazing
        if (mpLocalMapper->IsInitializing())
            return false;
        const int nKFs = mpAtlas->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames) {
            return false;
        }

#ifdef LIMIT_NEW_KEYFRAME
//    自定义的关键帧限制 accelerate
        float increTheta = 0.f, increT = 0.f;
        //计算增量大小
        if(mpLastKeyFrame->GetPoseInverse().empty())
            cout<<"mpLastKeyFrame->GetPoseInverse().empty()"<<endl;
        if(mCurrentFrame.mTcw.empty())
            cout<<"mCurrentFrame.mTcw.empty()"<<endl;

        cv::Mat delatT = mCurrentFrame.mTcw * mpLastKeyFrame->GetPoseInverse();
        Converter::increT(delatT, increTheta, increT);
        increTheta = fabs(increTheta) / M_PI * 180.f;
        increT = fabs(increT);
        //统计得，正常情况，特征丰富下，纯平移0.15建一帧,角度1-3度建一帧(最小到0.06度)
        float score = increTheta / 10.f + increT / 0.15f;
//    cout<<" score: "<<score<<" increTheta: "<<increTheta<<" increT: "<<increT<<endl;
        if (score < 1.f)
            return false;
#endif

        // Tracked MapPoints in the reference keyframe
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

        // Check how many "close" points are being tracked and how many could be potentially created.
        int nNonTrackedClose = 0;
        int nTrackedClose = 0;

        if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) {
            int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
            for (int i = 0; i < N; i++) {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nTrackedClose++;
                    else
                        nNonTrackedClose++;

                }
            }
        }

        bool bNeedToInsertClose;
        bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

        // Thresholds
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        if (mpCamera2) thRefRatio = 0.75f;

        if (mSensor == System::IMU_MONOCULAR) {
            if (mnMatchesInliers > 350) // Points tracked from the local map
                thRefRatio = 0.75f;
            else
                thRefRatio = 0.90f;
        }

        // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        //cout<< "mCurrentFrame.mnId: "<<mCurrentFrame.mnId<< " mnLastKeyFrameId: "<<mnLastKeyFrameId<<endl;
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
        // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        //cout<< "bLocalMappingIdle: "<<bLocalMappingIdle<<endl;
        const bool c1b = ((mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames) && bLocalMappingIdle);
        //Condition 1c: tracking is weak
        const bool c1c =
                mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR && mSensor != System::IMU_STEREO &&
                (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        const bool c2 = (((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose)) &&
                         mnMatchesInliers > 15);

        // Temporal condition for Inertial cases
        bool c3 = false;
        if (mpLastKeyFrame) {
            if (mSensor == System::IMU_MONOCULAR) {
                if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
                    c3 = true;
            } else if (mSensor == System::IMU_STEREO) {
                if ((mCurrentFrame.mTimeStamp - mpLastKeyFrame->mTimeStamp) >= 0.5)
                    c3 = true;
            }
        }

        bool c4 = false;
        if ((((mnMatchesInliers < 75) && (mnMatchesInliers > 15)) || mState == RECENTLY_LOST) && ((mSensor ==
                                                                                                   System::IMU_MONOCULAR))) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
            c4 = true;
        else
            c4 = false;
        //cout<< "c1a: "<<c1a<< " c1b: "<<c1b<< " c1c: "<<c1c<< " c2: "<<c2<< " c3: "<<c3<< " c4: "<<c4<<endl;
        if (((c1a || c1b || c1c) && c2) || c3 || c4) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            if (bLocalMappingIdle) {
                return true;
            } else {
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                        return true;
                    else
                        return false;
                } else
                    return false;
            }
        } else
            return false;
    }

    bool Tracking::CreateNewKeyFrameWithoutMPs() {
        float increTheta = 0.f, increT = 0.f;
        //计算增量大小
        Converter::increT(mpIMUProcessor->getOdoTpcDelta(), increTheta, increT);
        mRotAccumulate = mRotAccumulate + fabs(increTheta) / M_PI * 180.f;
        mTransAccumulate = mTransAccumulate + fabs(increT);
        //cout<<"mRotAccumulate: "<<mRotAccumulate<<". mTransAccumulate: "<<mTransAccumulate<<endl;
        float score = mRotAccumulate / 10.f + mTransAccumulate / 0.1f;
        if (score < 1.f)
            return false;
        else {
            mRotAccumulate = 0.f;
            mTransAccumulate = 0.f;
        }

        if (mpLocalMapper->IsInitializing())
            return false;

        ORB_SLAM3::Verbose::PrintMess("create KeyFrame", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        mpAtlas->AddRecentInitId(pKF);
        mpAtlas->AddKeyFrame(pKF);

        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        ORB_SLAM3::Verbose::PrintMess("create KeyFrame over", ORB_SLAM3::Verbose::VERBOSITY_SELF_DEBUG);
        pKF->UpdateConnections();
        //mpLocalMapper->InsertKeyFrame(pKF);

        cout << "create kf without MPs. frame ID: " << mCurrentFrame.mnId << ". KF ID: " << pKF->mnId << endl;
        //Converter::PrintSE(mVelocityFromOdo);

        return true;
    }

    void Tracking::CreateNewKeyFrame() {
        if (mpLocalMapper->IsInitializing())
            return;

        if (!mpLocalMapper->SetNotStop(true))
            return;

        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        if (mbVisual)
            mImGray.copyTo(mImKFGray);

        if (mpAtlas->isImuInitialized())
            pKF->bImu = true;

        pKF->SetNewBias(mCurrentFrame.mImuBias);
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        if (mpLastKeyFrame) {
            pKF->mPrevKF = mpLastKeyFrame;
            mpLastKeyFrame->mNextKF = pKF;
        } else
            Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

        // Reset preintegration from last KF (Create new object)
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) {
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(), pKF->mImuCalib);
        }

        if (mSensor != System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
        {
            mCurrentFrame.UpdatePoseMatrices();
            // cout << "create new MPs" << endl;
            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.
            int maxPoint = 100;
            if (mSensor == System::IMU_STEREO)
                maxPoint = 100;

            vector<pair<float, int> > vDepthIdx;
            int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
            vDepthIdx.reserve(mCurrentFrame.N);
            for (int i = 0; i < N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty()) {
                sort(vDepthIdx.begin(), vDepthIdx.end());

                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size(); j++) {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }

                    if (bCreateNew) {
                        cv::Mat x3D;

                        if (mCurrentFrame.Nleft == -1) {
                            x3D = mCurrentFrame.UnprojectStereo(i);
                        } else {
                            x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                        }

                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpAtlas->GetCurrentMap());
                        pNewMP->AddObservation(pKF, i);

                        //Check if it is a stereo observation in order to not
                        //duplicate mappoints
                        if (mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0) {
                            mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft +
                                                       mCurrentFrame.mvLeftToRightMatch[i]] = pNewMP;
                            pNewMP->AddObservation(pKF, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                            pKF->AddMapPoint(pNewMP, mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        }

                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpAtlas->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    } else {
                        nPoints++; // TODO check ???
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > maxPoint) {
                        break;
                    }
                }
                Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
        //cout  << "end creating new KF" << endl;
    }

    void Tracking::SearchLocalPoints() {
        // Do not search map points already matched
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPoint *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                    pMP->mbTrackInViewR = false;
                }
            }
        }

        int nToMatch = 0;

        // Project points in frame and check its visibility
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;

            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
                pMP->IncreaseVisible();
                nToMatch++;
            }
            if (pMP->mbTrackInView) {
                mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
            }
        }

        if (nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = 1;
            if (mSensor == System::RGBD)
                th = 3;
            if (mpAtlas->isImuInitialized()) {
                if (mpAtlas->GetCurrentMap()->GetIniertialBA2())
                    th = 2;
                else
                    th = 3;
            } else if (!mpAtlas->isImuInitialized() &&
                       (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) {
                th = 10;
            }

            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            if (mState == LOST || mState == RECENTLY_LOST) // Lost for less than 1 second
                th = 15; // 15

            int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints,
                                                     mpLocalMapper->mThFarPoints);
        }
    }

    void Tracking::UpdateLocalMap() {
        // This is for visualization
        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        // Update
        UpdateLocalKeyFrames();
        UpdateLocalPoints();
    }

    void Tracking::UpdateLocalPoints() {
        mvpLocalMapPoints.clear();

        int count_pts = 0;

        for (vector<KeyFrame *>::const_reverse_iterator itKF = mvpLocalKeyFrames.rbegin(), itEndKF = mvpLocalKeyFrames.rend();
             itKF != itEndKF; ++itKF) {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
                 itMP != itEndMP; itMP++) {

                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad()) {
                    count_pts++;
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }


    void Tracking::UpdateLocalKeyFrames() {
        // Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;
        if (!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId < mnLastRelocFrameId + 2)) {
            for (int i = 0; i < mCurrentFrame.N; i++) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP) {
                    if (!pMP->isBad()) {
                        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                        for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end();
                             it != itend; it++)
                            keyframeCounter[it->first]++;
                    } else {
                        mCurrentFrame.mvpMapPoints[i] = NULL;
                    }
                }
            }
        } else {
            for (int i = 0; i < mLastFrame.N; i++) {
                // Using lastframe since current frame has not matches yet
                if (mLastFrame.mvpMapPoints[i]) {
                    MapPoint *pMP = mLastFrame.mvpMapPoints[i];
                    if (!pMP)
                        continue;
                    if (!pMP->isBad()) {
                        const map<KeyFrame *, tuple<int, int>> observations = pMP->GetObservations();
                        for (map<KeyFrame *, tuple<int, int>>::const_iterator it = observations.begin(), itend = observations.end();
                             it != itend; it++)
                            keyframeCounter[it->first]++;
                    } else {
                        // MODIFICATION
                        mLastFrame.mvpMapPoints[i] = NULL;
                    }
                }
            }
        }


        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

        // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++) {
            KeyFrame *pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(pKF);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }

        // Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80) // 80
                break;

            KeyFrame *pKF = *itKF;

            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);


            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++) {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad()) {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad()) {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame *pParent = pKF->GetParent();
            if (pParent) {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // Add 10 last temporal KFs (mainly for IMU)
        if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mvpLocalKeyFrames.size() < 80) {
            //cout << "CurrentKF: " << mCurrentFrame.mnId << endl;
            KeyFrame *tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

            const int Nd = 20;
            for (int i = 0; i < Nd; i++) {
                if (!tempKeyFrame)
                    break;
                //cout << "tempKF: " << tempKeyFrame << endl;
                if (tempKeyFrame->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(tempKeyFrame);
                    tempKeyFrame->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    tempKeyFrame = tempKeyFrame->mPrevKF;
                }
            }
        }

        if (pKFmax) {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization() {
        Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame,
                                                                                         mpAtlas->GetCurrentMap());

        if (vpCandidateKFs.empty()) {
            Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
            return false;
        }

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);

        vector<MLPnPsolver *> vpMLPnPsolvers;
        vpMLPnPsolvers.resize(nKFs);

        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;

        for (int i = 0; i < nKFs; i++) {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15) {
                    vbDiscarded[i] = true;
                    continue;
                } else {
                    MLPnPsolver *pSolver = new MLPnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 6, 0.5, 5.991);  //This solver needs at least 6 points
                    vpMLPnPsolvers[i] = pSolver;
                }
            }
        }

        // Alternatively perform some iterations of P4P RANSAC
        // Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch) {
            for (int i = 0; i < nKFs; i++) {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                MLPnPsolver *pSolver = vpMLPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (!Tcw.empty()) {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++) {
                        if (vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50) {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10,
                                                                      100);

                        if (nadditional + nGood >= 50) {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50) {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                                                                          64);

                                // Final optimization
                                if (nGood + nadditional >= 50) {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50) {
                        bMatch = true;
                        break;
                    }
                }
            }
        }

        if (!bMatch) {
            return false;
        } else {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            cout << "Relocalized!!" << endl;
            return true;
        }

    }

    void Tracking::Reset(bool bLocMap) {
        Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

        if (mpViewer) {
            mpViewer->RequestStop();
            while (!mpViewer->isStopped())
                usleep(3000);
        }

        // Reset Local Mapping
        if (!bLocMap) {
            Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
            mpLocalMapper->RequestReset();
            Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        }


        // Reset Loop Closing
        Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
        mpLoopClosing->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear BoW Database
        Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
        mpKeyFrameDB->clear();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear Map (this erase MapPoints and KeyFrames)
        mpAtlas->clearAtlas();
        mpAtlas->CreateNewMap();
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
            mpAtlas->SetInertialSensor();
        mnInitialFrameId = 0;

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        if (mpInitializer) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }
        mbSetInit = false;

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();
        mCurrentFrame = Frame();
        mnLastRelocFrameId = 0;
        mLastFrame = Frame();
        mpReferenceKF = static_cast<KeyFrame *>(NULL);
        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
        mvIniMatches.clear();

        if (mpViewer)
            mpViewer->Release();

        Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
    }

    void Tracking::ResetActiveMap(bool bLocMap) {
        Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
        if (mpViewer) {
            mpViewer->RequestStop();
            while (!mpViewer->isStopped())
                usleep(3000);
        }

        Map *pMap = mpAtlas->GetCurrentMap();

        if (!bLocMap) {
            Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
            mpLocalMapper->RequestResetActiveMap(pMap);
            Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
        }

        // Reset Loop Closing
        Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
        mpLoopClosing->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear BoW Database
        Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
        mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

        // Clear Map (this erase MapPoints and KeyFrames)
        mpAtlas->clearMap();


        //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
        //Frame::nNextId = mnLastInitFrameId;
        mnLastInitFrameId = Frame::nNextId;
        mnLastRelocFrameId = mnLastInitFrameId;
        mState = NO_IMAGES_YET; //NOT_INITIALIZED;

        if (mpInitializer) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }

        list<bool> lbLost;
        // lbLost.reserve(mlbLost.size());
        unsigned int index = mnFirstFrameId;
        cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
        for (Map *pMap : mpAtlas->GetAllMaps()) {
            if (pMap->GetAllKeyFrames().size() > 0) {
                if (index > pMap->GetLowerKFID())
                    index = pMap->GetLowerKFID();
            }
        }

        //cout << "First Frame id: " << index << endl;
        int num_lost = 0;
        cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

        for (list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++) {
            if (index < mnInitialFrameId)
                lbLost.push_back(*ilbL);
            else {
                lbLost.push_back(true);
                num_lost += 1;
            }

            index++;
        }
        cout << num_lost << " Frames set to lost" << endl;

        mlbLost = lbLost;

        mnInitialFrameId = mCurrentFrame.mnId;
        mnLastRelocFrameId = mCurrentFrame.mnId;

        mCurrentFrame = Frame();
        mLastFrame = Frame();
        mpReferenceKF = static_cast<KeyFrame *>(NULL);
        mpLastKeyFrame = static_cast<KeyFrame *>(NULL);
        mvIniMatches.clear();

        if (mpViewer)
            mpViewer->Release();

        Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
    }

    vector<MapPoint *> Tracking::GetLocalMapMPS() {
        return mvpLocalMapPoints;
    }

    void Tracking::ChangeCalibration(const string &strSettingPath) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera_fx"];
        float fy = fSettings["Camera_fy"];
        float cx = fSettings["Camera_cx"];
        float cy = fSettings["Camera_cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera_k1"];
        DistCoef.at<float>(1) = fSettings["Camera_k2"];
        DistCoef.at<float>(2) = fSettings["Camera_p1"];
        DistCoef.at<float>(3) = fSettings["Camera_p2"];
        const float k3 = fSettings["Camera_k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool &flag) {
        mbOnlyTracking = flag;
    }

    void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame) {
        Map *pMap = pCurrentKeyFrame->GetMap();
        unsigned int index = mnFirstFrameId;
        list<ORB_SLAM3::KeyFrame *>::iterator lRit = mlpReferences.begin();
        list<bool>::iterator lbL = mlbLost.begin();
        for (list<cv::Mat>::iterator lit = mlRelativeFramePoses.begin(), lend = mlRelativeFramePoses.end();
             lit != lend; lit++, lRit++, lbL++) {
            if (*lbL)
                continue;

            KeyFrame *pKF = *lRit;

            while (pKF->isBad()) {
                pKF = pKF->GetParent();
            }

            if (pKF->GetMap() == pMap) {
                (*lit).rowRange(0, 3).col(3) = (*lit).rowRange(0, 3).col(3) * s;
            }
        }

        mLastBias = b;

        mpLastKeyFrame = pCurrentKeyFrame;

        mLastFrame.SetNewBias(mLastBias);
        mCurrentFrame.SetNewBias(mLastBias);

        cv::Mat Gz = (cv::Mat_<float>(3, 1) << 0, 0, -IMU::GRAVITY_VALUE);

        cv::Mat twb1;
        cv::Mat Rwb1;
        cv::Mat Vwb1;
        float t12;

        while (!mCurrentFrame.imuIsPreintegrated()) {
            usleep(500);
        }


        if (mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId) {
            mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                          mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                          mLastFrame.mpLastKeyFrame->GetVelocity());
        } else {
            twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
            Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
            Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
            t12 = mLastFrame.mpImuPreintegrated->dT;

            mLastFrame.SetImuPoseVelocity(Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                          twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz +
                                          Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                          Vwb1 + Gz * t12 +
                                          Rwb1 * mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        if (mCurrentFrame.mpImuPreintegrated) {
            twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
            Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
            Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
            t12 = mCurrentFrame.mpImuPreintegrated->dT;

            mCurrentFrame.SetImuPoseVelocity(Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                             twb1 + Vwb1 * t12 + 0.5f * t12 * t12 * Gz +
                                             Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                             Vwb1 + Gz * t12 +
                                             Rwb1 * mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
        }

        mnFirstImuFrameId = mCurrentFrame.mnId;
    }


    cv::Mat Tracking::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2) {
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        cv::Mat R12 = R1w * R2w.t();
        cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

        cv::Mat t12x = Converter::tocvSkewMatrix(t12);

        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;


        return K1.t().inv() * t12x * R12 * K2.inv();
    }


    void Tracking::CreateNewMapPoints() {
        // Retrieve neighbor keyframes in covisibility graph
        const vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();

        ORBmatcher matcher(0.6, false);

        cv::Mat Rcw1 = mpLastKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpLastKeyFrame->GetTranslation();
        cv::Mat Tcw1(3, 4, CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0, 3));
        tcw1.copyTo(Tcw1.col(3));
        cv::Mat Ow1 = mpLastKeyFrame->GetCameraCenter();

        const float &fx1 = mpLastKeyFrame->fx;
        const float &fy1 = mpLastKeyFrame->fy;
        const float &cx1 = mpLastKeyFrame->cx;
        const float &cy1 = mpLastKeyFrame->cy;
        const float &invfx1 = mpLastKeyFrame->invfx;
        const float &invfy1 = mpLastKeyFrame->invfy;

        const float ratioFactor = 1.5f * mpLastKeyFrame->mfScaleFactor;

        int nnew = 0;

        // Search matches with epipolar restriction and triangulate
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF2 = vpKFs[i];
            if (pKF2 == mpLastKeyFrame)
                continue;

            // Check first that baseline is not too short
            cv::Mat Ow2 = pKF2->GetCameraCenter();
            cv::Mat vBaseline = Ow2 - Ow1;
            const float baseline = cv::norm(vBaseline);

            if ((mSensor != System::MONOCULAR) || (mSensor != System::IMU_MONOCULAR)) {
                if (baseline < pKF2->mb)
                    continue;
            } else {
                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                const float ratioBaselineDepth = baseline / medianDepthKF2;

                if (ratioBaselineDepth < 0.01)
                    continue;
            }

            // Compute Fundamental Matrix
            cv::Mat F12 = ComputeF12(mpLastKeyFrame, pKF2);

            // Search matches that fullfil epipolar constraint
            vector<pair<size_t, size_t> > vMatchedIndices;
            matcher.SearchForTriangulation(mpLastKeyFrame, pKF2, F12, vMatchedIndices, false);

            cv::Mat Rcw2 = pKF2->GetRotation();
            cv::Mat Rwc2 = Rcw2.t();
            cv::Mat tcw2 = pKF2->GetTranslation();
            cv::Mat Tcw2(3, 4, CV_32F);
            Rcw2.copyTo(Tcw2.colRange(0, 3));
            tcw2.copyTo(Tcw2.col(3));

            const float &fx2 = pKF2->fx;
            const float &fy2 = pKF2->fy;
            const float &cx2 = pKF2->cx;
            const float &cy2 = pKF2->cy;
            const float &invfx2 = pKF2->invfx;
            const float &invfy2 = pKF2->invfy;

            // Triangulate each match
            const int nmatches = vMatchedIndices.size();
            for (int ikp = 0; ikp < nmatches; ikp++) {
                const int &idx1 = vMatchedIndices[ikp].first;
                const int &idx2 = vMatchedIndices[ikp].second;

                const cv::KeyPoint &kp1 = mpLastKeyFrame->mvKeysUn[idx1];
                const float kp1_ur = mpLastKeyFrame->mvuRight[idx1];
                bool bStereo1 = kp1_ur >= 0;

                const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
                const float kp2_ur = pKF2->mvuRight[idx2];
                bool bStereo2 = kp2_ur >= 0;

                // Check parallax between rays
                cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
                cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

                cv::Mat ray1 = Rwc1 * xn1;
                cv::Mat ray2 = Rwc2 * xn2;
                const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

                float cosParallaxStereo = cosParallaxRays + 1;
                float cosParallaxStereo1 = cosParallaxStereo;
                float cosParallaxStereo2 = cosParallaxStereo;

                if (bStereo1)
                    cosParallaxStereo1 = cos(2 * atan2(mpLastKeyFrame->mb / 2, mpLastKeyFrame->mvDepth[idx1]));
                else if (bStereo2)
                    cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

                cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

                cv::Mat x3D;
                if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
                    (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
                    // Linear Triangulation Method
                    cv::Mat A(4, 4, CV_32F);
                    A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                    A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                    A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                    A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                    cv::Mat w, u, vt;
                    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                    x3D = vt.row(3).t();

                    if (x3D.at<float>(3) == 0)
                        continue;

                    // Euclidean coordinates
                    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

                } else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
                    x3D = mpLastKeyFrame->UnprojectStereo(idx1);
                } else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
                    x3D = pKF2->UnprojectStereo(idx2);
                } else
                    continue; //No stereo and very low parallax

                cv::Mat x3Dt = x3D.t();

                //Check triangulation in front of cameras
                float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
                if (z1 <= 0)
                    continue;

                float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
                if (z2 <= 0)
                    continue;

                //Check reprojection error in first keyframe
                const float &sigmaSquare1 = mpLastKeyFrame->mvLevelSigma2[kp1.octave];
                const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
                const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
                const float invz1 = 1.0 / z1;

                if (!bStereo1) {
                    float u1 = fx1 * x1 * invz1 + cx1;
                    float v1 = fy1 * y1 * invz1 + cy1;
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;
                    if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                        continue;
                } else {
                    float u1 = fx1 * x1 * invz1 + cx1;
                    float u1_r = u1 - mpLastKeyFrame->mbf * invz1;
                    float v1 = fy1 * y1 * invz1 + cy1;
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;
                    float errX1_r = u1_r - kp1_ur;
                    if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
                        continue;
                }

                //Check reprojection error in second keyframe
                const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
                const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
                const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
                const float invz2 = 1.0 / z2;
                if (!bStereo2) {
                    float u2 = fx2 * x2 * invz2 + cx2;
                    float v2 = fy2 * y2 * invz2 + cy2;
                    float errX2 = u2 - kp2.pt.x;
                    float errY2 = v2 - kp2.pt.y;
                    if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                        continue;
                } else {
                    float u2 = fx2 * x2 * invz2 + cx2;
                    float u2_r = u2 - mpLastKeyFrame->mbf * invz2;
                    float v2 = fy2 * y2 * invz2 + cy2;
                    float errX2 = u2 - kp2.pt.x;
                    float errY2 = v2 - kp2.pt.y;
                    float errX2_r = u2_r - kp2_ur;
                    if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)
                        continue;
                }

                //Check scale consistency
                cv::Mat normal1 = x3D - Ow1;
                float dist1 = cv::norm(normal1);

                cv::Mat normal2 = x3D - Ow2;
                float dist2 = cv::norm(normal2);

                if (dist1 == 0 || dist2 == 0)
                    continue;

                const float ratioDist = dist2 / dist1;
                const float ratioOctave = mpLastKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

                if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                    continue;

                // Triangulation is succesfull
                MapPoint *pMP = new MapPoint(x3D, mpLastKeyFrame, mpAtlas->GetCurrentMap());

                pMP->AddObservation(mpLastKeyFrame, idx1);
                pMP->AddObservation(pKF2, idx2);

                mpLastKeyFrame->AddMapPoint(pMP, idx1);
                pKF2->AddMapPoint(pMP, idx2);

                pMP->ComputeDistinctiveDescriptors();

                pMP->UpdateNormalAndDepth();

                mpAtlas->AddMapPoint(pMP);
                nnew++;
            }
        }
        TrackReferenceKeyFrame();
    }

    void Tracking::NewDataset() {
        mnNumDataset++;
    }

    int Tracking::GetNumberDataset() {
        return mnNumDataset;
    }

    int Tracking::GetMatchesInliers() {
        return mnMatchesInliers;
    }

    void Tracking::CheckOdoVelocity() {
        static int forward = 0;
        std::vector<float> velocityDebug = Converter::PrintSE(mVelocity, "mVelocity ", false);
        std::vector<float> velocityFromOdoDebug = Converter::PrintSE(mVelocityFromOdo, "mVelocityFromOdo ", false);
        double t_cam = pow(velocityDebug[0], 2) + pow(velocityDebug[1], 2) + pow(velocityDebug[2], 2);
        t_cam = sqrt(t_cam);
        double t_odo =
                pow(velocityFromOdoDebug[0], 2) + pow(velocityFromOdoDebug[1], 2) + pow(velocityFromOdoDebug[2], 2);
        t_odo = sqrt(t_odo);
        if (t_odo > 0.004)
            forward++;
        else
            forward = 0;
        if (forward > 3) {
            t_odo_sum = t_odo_sum + t_odo;
            t_cam_sum = t_cam_sum + t_cam;
            cout << "scale: " << t_cam_sum / t_odo_sum << " t_cam: " << t_cam << " t_odo: " << t_odo << endl;
        }

//    mpSystem->mfDebugForOdo <<  setprecision(9);
//    for(int i=0;i<6;i++)
//        mpSystem->mfDebugForOdo<<velocityDebug[i]<<" ";
//    for(int i=0;i<6;i++)
//    {
//        if(i!=5)
//            mpSystem->mfDebugForOdo<<velocityFromOdoDebug[i]<<" ";
//        else
//            mpSystem->mfDebugForOdo<<velocityFromOdoDebug[i]<<endl;
//    }
    }

} //namespace ORB_SLAM
