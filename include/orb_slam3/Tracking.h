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


#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Atlas.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "include/preintegration/ImuTypes.h"
#include "include/preintegration/Odometer.h"

#include "CameraModels/GeometricCamera.h"
#include <mutex>
#include <unordered_set>
#include "include/preintegration/imuProcess.h"
#include "include/Util/selfDefine.h"
#include "Util/record.h"

#ifdef ENABLE_GOOD_GRAPH
#include <include/goodGraph/Observability.h>
#include "include/goodGraph/gfUtil.hpp"
#endif


class ProbabilityMapping;
namespace ORB_SLAM3 {

    class Viewer;

    class FrameDrawer;

    class Atlas;

    class LocalMapping;

    class LoopClosing;

    class System;

    class Tracking {

    public:
        Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas,
                 KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor,
                 const string &_nameSeq = std::string());

        ~Tracking();

        // Parse the config file
        bool ParseCamParamFile(cv::FileStorage &fSettings);

        bool ParseORBParamFile(cv::FileStorage &fSettings);

        bool ParseIMUParamFile(cv::FileStorage &fSettings);

        // Preprocess the input and call Track(). Extract features and performs stereo matching.
        cv::Mat GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp,
                                string filename);

        cv::Mat GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename);

        cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);
        // cv::Mat GrabImageImuMonocular(const cv::Mat &im, const double &timestamp);

        void GrabImuData(const IMU::Point &imuMeasurement);

        void GrabOdoData(const ODO::Point &odoMeasurement);


        void SetLocalMapper(LocalMapping *pLocalMapper);

        void SetLoopClosing(LoopClosing *pLoopClosing);

        void SetViewer(Viewer *pViewer);

        void SetStepByStep(bool bSet);

#ifdef USE_SEMI_DENSE_MAP
        void SetSemiDenseMapping(ProbabilityMapping* pSemiDenseMapping);
#endif

        // Load new settings
        // The focal lenght should be similar or scale prediction will fail when projecting points
        void ChangeCalibration(const string &strSettingPath);

        // Use this function if you have deactivated local mapping and you only want to localize the camera.
        void InformOnlyTracking(const bool &flag);

        void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame *pCurrentKeyFrame);

        KeyFrame *GetLastKeyFrame() {
            return mpLastKeyFrame;
        }

        void CreateMapInAtlas();

        void RestartTrack();

        std::mutex mMutexTracks;

        //--
        void NewDataset();

        int GetNumberDataset();

        int GetMatchesInliers();

        void CheckOdoVelocity();

        void SetIMUProcessor(std::shared_ptr<ORB_SLAM3::IMUProcess> IMUProcessor);

    public:

        // Tracking states
        enum eTrackingState {
            SYSTEM_NOT_READY = -1,
            NO_IMAGES_YET = 0,
            NOT_INITIALIZED = 1,
            OK = 2,
            RECENTLY_LOST = 3,
            LOST = 4,
            OK_KLT = 5
        };

        eTrackingState mState;
        eTrackingState mLastProcessedState;

        // Input sensor
        int mSensor;

        // Current Frame
        Frame mCurrentFrame;
        Frame mLastFrame;

        cv::Mat mImGray;
        cv::Mat mImKFGray;

#ifdef ENABLE_GOOD_GRAPH
        std::vector<TrackingLog> mFrameTimeLog;
        TrackingLog logCurrentFrame;
#endif

        // Initialization Variables (Monocular)
        std::vector<int> mvIniLastMatches;
        std::vector<int> mvIniMatches;
        std::vector<cv::Point2f> mvbPrevMatched;
        std::vector<cv::Point3f> mvIniP3D;
        Frame mInitialFrame;

        // Lists used to recover the full camera trajectory at the end of the execution.
        // Basically we store the reference keyframe for each frame and its relative transformation
        list <cv::Mat> mlRelativeFramePoses;
        list<KeyFrame *> mlpReferences;
        list<double> mlFrameTimes;
        list<bool> mlbLost;

        // frames with estimated pose
        int mTrackedFr;
        bool mbStep;

        // True if local mapping is deactivated and we are performing only localization
        bool mbOnlyTracking;

        void Reset(bool bLocMap = false);

        void ResetActiveMap(bool bLocMap = false);

        float mMeanTrack;
        bool mbInitWith3KFs;
        double t0; // time-stamp of first read frame
        double t0vis; // time-stamp of first inserted keyframe
        double t0IMU; // time-stamp of IMU initialization


        vector<MapPoint *> GetLocalMapMPS();

        bool mbInertialFirst;

        //TEST--
        bool mbNeedRectify;
        //cv::Mat M1l, M2l;
        //cv::Mat M1r, M2r;

        bool mbWriteStats;

        uint32_t usefulForIMU;

#ifdef ENABLE_GOOD_GRAPH
        Observability * mObsHandler;
#endif

        int mnInitStereo;

        double t_cam_sum;
        double t_odo_sum;

    protected:

        // Main tracking function. It is independent of the input sensor.
        void Track();

        // Map initialization for stereo and RGB-D
        void StereoInitialization();

        void ReStereoInitialization();

        // Map initialization for monocular
        void MonocularInitialization();

        void CreateNewMapPoints();

        cv::Mat ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2);

        void CreateInitialMapMonocular();

        void CheckReplacedInLastFrame();

        void VisualPointMatch(string s);

        bool TrackReferenceKeyFrame();

        void UpdateLastFrame();

        bool TrackWithMotionModel();

        bool PredictStateIMU();

        bool Relocalization();

        void UpdateLocalMap();

        void UpdateLocalPoints();

        void UpdateLocalKeyFrames();

        bool TrackLocalMap();

        bool TrackLocalMap_old();

        bool SearchAdditionalMatchesInFrame(const double time_for_search, Frame &F);

        void SearchLocalPoints();

        void PredictJacobianNextFrame(const double time_for_predict, const size_t pred_horizon);

        bool NeedNewKeyFrame();

        void CreateNewKeyFrame();

        bool CreateNewKeyFrameWithoutMPs();

        void UpdateLastTcw();

        // Perform preintegration from last frame
        void PreintegrateIMU();

        void PreintegrateODO();

        void PreIntFromIMUPro();


        // Reset IMU biases and compute frame velocity
        void ResetFrameIMU();

        void ComputeGyroBias(const vector<Frame *> &vpFs, float &bwx, float &bwy, float &bwz);

        void ComputeVelocitiesAccBias(const vector<Frame *> &vpFs, float &bax, float &bay, float &baz);


        bool mbMapUpdated;

        // Imu preintegration from last frame
        IMU::Preintegrated *mpImuPreintegratedFromLastKF;

        // Queue of IMU measurements between frames
        std::list<IMU::Point> mlQueueImuData;
        std::list<ODO::Point> mlQueueOdoData;

        // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
        std::vector<IMU::Point> mvImuFromLastFrame;
        std::mutex mMutexImuQueue;

        std::vector<ODO::Point> mvOdoFromLastFrame;
        std::mutex mMutexOdoQueue;

        // Imu calibration parameters
        IMU::Calib *mpImuCalib;

        // Last Bias Estimation (at keyframe creation)
        IMU::Bias mLastBias;

        // In case of performing only localization, this flag is true when there are no matches to
        // points in the map. Still tracking will continue if there are enough matches with temporal points.
        // In that case we are doing visual odometry. The system will try to do relocalization to recover
        // "zero-drift" localization to the map.
        bool mbVO;
        bool mInitPreOdo;

        //Other Thread Pointers
        LocalMapping *mpLocalMapper;
        LoopClosing *mpLoopClosing;

        /**semi-dense**/
#ifdef USE_SEMI_DENSE_MAP
        ProbabilityMapping* mpSemiDenseMapping;
#endif

        //ORB
        ORBextractor *mpORBextractorLeft, *mpORBextractorRight;
        ORBextractor *mpIniORBextractor;

        //BoW
        ORBVocabulary *mpORBVocabulary;
        KeyFrameDatabase *mpKeyFrameDB;

        // Initalization (only for monocular)
        Initializer *mpInitializer;
        bool mbSetInit;

        //Local Map
        KeyFrame *mpReferenceKF;
        std::vector<KeyFrame *> mvpLocalKeyFrames;
        std::vector<MapPoint *> mvpLocalMapPoints;

        // System
        System *mpSystem;

        std::shared_ptr<ORB_SLAM3::IMUProcess> mpIMUProcessor;

        //Drawers
        Viewer *mpViewer;
        FrameDrawer *mpFrameDrawer;
        MapDrawer *mpMapDrawer;

        bool bStepByStep;

        //Atlas
        Atlas *mpAtlas;

        //Calibration matrix
        cv::Mat mK;
        cv::Mat mK_ori, mDistCoef, mR, mP;
        cv::Mat mK_right, mDistCoef_right, mR_right, mP_right;
        float mbf;

        //New KeyFrame rules (according to fps)
        int mMinFrames;
        int mMaxFrames;
        int camera_fps;

        int mnFirstImuFrameId;
        int mnFramesToResetIMU;

        // Threshold close/far points
        // Points seen as close by the stereo/RGBD sensor are considered reliable
        // and inserted from just one frame. Far points requiere a match in two keyframes.
        float mThDepth;

        // frame counter after initialization
        size_t mFrameAfterInital;
        size_t nFrameSinceLast;
        size_t mbTrackLossAlert;

        // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
        float mDepthMapFactor;

        //Current matches in frame
        int mnMatchesInliers;

        // Number of visible map points at current KF
        double mNumVisibleMpt;

        //Last Frame, KeyFrame and Relocalisation Info
        KeyFrame *mpLastKeyFrame;
        unsigned int mnLastKeyFrameId;
        unsigned int mnLastRelocFrameId;
        double mTimeStampLost;
        double time_recently_lost;

        unsigned int mnFirstFrameId;
        unsigned int mnInitialFrameId;
        unsigned int mnLastInitFrameId; //上一个子地图的最后一帧

        bool mbCreatedMap;
        bool mbRestart;
        bool mbVisual;
        bool mlastTcwUpdate;

        cv::Mat mlastMapmLastFrameTcw;

        //Motion Model
        cv::Mat mVelocity;
        cv::Mat mVelocityFromOdo;

        // 当前帧相对于上一帧（最近一个地图的“mlastframe”）的位姿变换, 因为相机有些丢失，所以可能会累乘起来
        cv::Mat mOdoTpc;
        float mRotAccumulate;
        float mTransAccumulate;

        cv::Mat mTc_odo;
        cv::Mat mTodo_c;

        //Color order (true RGB, false BGR, ignored if grayscale)
        bool mbRGB;

        list<MapPoint *> mlpTemporalPoints;

        //int nMapChangeIndex;

        int mnNumDataset;

        GeometricCamera *mpCamera, *mpCamera2;

        int initID, lastID;

        cv::Mat mTlr;

    public:
        cv::Mat mImRight;
        TrackingTime *mpTrackingTime;
    };

} //namespace ORB_SLAM

#endif // TRACKING_H
