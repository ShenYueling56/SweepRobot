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



#include "include/orb_slam3/System.h"
#include "Util/FileUtil.h"
#include "include/orb_slam3/Converter.h"
#include <thread>
#include <iomanip>
#include <openssl/md5.h>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
// #include "Thirdparty/fbow/src/vocabulary_creator.h"
#include "include/preintegration/imuProcess.h"
#include "include/Util/selfDefine.h"
#include "Util/record.h"

#ifdef SAVE_MAP_PCD
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#endif

namespace ORB_SLAM3 {

    Verbose::eLevel Verbose::th = Verbose::VERBOSITY_NORMAL;

    System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
                   const bool bUseViewer, const int initFr, const string &strSequence, const string &strLoadingFile,
                   const bool &bCreateVoc) :
            mSensor(sensor), mpViewer(static_cast<Viewer *>(NULL)), mbReset(false), mbResetActiveMap(false),
            mbCreateVoc(bCreateVoc),
            mbActivateLocalizationMode(false), mbDeactivateLocalizationMode(false),
            mpIMUProcessor(static_cast<IMUProcess *>(NULL)) {
        cout << "Input sensor was set to: ";
        //rmByCpp("./test_LBA/");
        //createDirectory("./test_LBA/");

        //mfDebugForOdo.open("debug.txt");
        //mfDebugForOdo << fixed;
#ifdef CEATE_VOC
        mbCreateVoc = true;
#endif

        if (mSensor == MONOCULAR)
            cout << "Monocular" << endl;
        else if (mSensor == STEREO)
            cout << "Stereo" << endl;
        else if (mSensor == RGBD)
            cout << "RGB-D" << endl;
        else if (mSensor == IMU_MONOCULAR)
            cout << "Monocular-Inertial" << endl;
        else if (mSensor == IMU_STEREO)
            cout << "Stereo-Inertial" << endl;

        //Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if (!fsSettings.isOpened()) {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        bool loadedAtlas = false;

        //----
        //Load ORB Vocabulary
        LoadVoc(strVocFile);

        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

        // 初始地图建立
        mpAtlas = new Atlas(0);

        if (mSensor == IMU_STEREO || mSensor == IMU_MONOCULAR)
            mpAtlas->SetInertialSensor();

        //Create Drawers. These are used by the Viewer
        mpFrameDrawer = new FrameDrawer(mpAtlas);
        mpMapDrawer = new MapDrawer(mpAtlas, strSettingsFile);

        //Initialize the Tracking thread
        //(it will live in the main thread of execution, the one that called this constructor)
        cout << "Seq. Name: " << strSequence << endl;
        mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                                 mpAtlas, mpKeyFrameDatabase, strSettingsFile, mSensor, strSequence);

        //Initialize the Local Mapping thread and launch
        mpLocalMapper = new LocalMapping(this, mpAtlas, mSensor == MONOCULAR || mSensor == IMU_MONOCULAR,
                                         mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO, strSequence);
        mptLocalMapping = new thread(&ORB_SLAM3::LocalMapping::Run, mpLocalMapper);
        mpLocalMapper->mInitFr = initFr;
        mpLocalMapper->mThFarPoints = fsSettings["thFarPoints"];
        if (mpLocalMapper->mThFarPoints != 0) {
            cout << "Discard points further than " << mpLocalMapper->mThFarPoints << " m from current camera" << endl;
            mpLocalMapper->mbFarPoints = true;
        } else
            mpLocalMapper->mbFarPoints = false;

        //Initialize the Loop Closing thread and launch
        // mSensor!=MONOCULAR && mSensor!=IMU_MONOCULAR
        mpLoopCloser = new LoopClosing(mpAtlas, mpKeyFrameDatabase, mpVocabulary,
                                       mSensor != MONOCULAR); // mSensor!=MONOCULAR);
        mptLoopClosing = new thread(&ORB_SLAM3::LoopClosing::Run, mpLoopCloser);

        //Initialize the Viewer thread and launch
#ifdef USE_PANGOLIN
        if (bUseViewer) {
            mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile);
            mptViewer = new thread(&Viewer::Run, mpViewer);
            mpTracker->SetViewer(mpViewer);
            mpLoopCloser->mpViewer = mpViewer;
            mpViewer->both = mpFrameDrawer->both;
        }
#endif

        //Set pointers between threads
        mpTracker->SetLocalMapper(mpLocalMapper);
        mpTracker->SetLoopClosing(mpLoopCloser);

        mpLocalMapper->SetTracker(mpTracker);
        mpLocalMapper->SetLoopCloser(mpLoopCloser);

        mpLoopCloser->SetTracker(mpTracker);
        mpLoopCloser->SetLocalMapper(mpLocalMapper);

        // Fix verbosity
        Verbose::SetTh(Verbose::VERBOSITY_QUIET);
//    Verbose::SetTh(Verbose::VERBOSITY_SELF_DEBUG);

    }

    void System::LoadVoc(const std::string &strVocFile) {
#if defined USE_FBOW
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
        cout << "open at: " << strVocFile << endl;
        mpVocabulary = new ORBVocabulary();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        mpVocabulary->readFromFile(strVocFile);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t_loadVoc = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        cout << "Vocabulary loaded! Spend "<<t_loadVoc<<" seconds. K:"<<mpVocabulary->getK() << " L:"<<mpVocabulary->getL() << endl;
        cout<<"Vocabulary. leafSize: "<<mpVocabulary->leafSize()<<" blockSize: "<<mpVocabulary->blockSize()
            <<" DescSize: "<<mpVocabulary->getDescSize()<<endl << endl;
#elif defined USE_DBOW3
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        mpVocabulary = new ORBVocabulary();
        mpVocabulary->load(strVocFile);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t_loadVoc = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        if (mpVocabulary->empty()) {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Failed to open at: " << strVocFile << endl;
            exit(-1);
        }
        printf("Vocabulary loaded in %.2fs\n", t_loadVoc);
#elif defined USE_DBOW2
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        mpVocabulary = new ORBVocabulary();
        // bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        bool bVocLoad = false; // chose loading method based on file extension
        if (has_suffix(strVocFile, ".txt"))
            bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        else
            bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t_loadVoc = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Failed to open at: " << strVocFile << endl;
            exit(-1);
        }
        printf("Vocabulary loaded in %.2fs\n", t_loadVoc);
#endif
    }

    void System::SetIMUProcessor(std::shared_ptr<ORB_SLAM3::IMUProcess> IMUProcessor) {
        mpIMUProcessor = IMUProcessor;
        mpTracker->SetIMUProcessor(mpIMUProcessor);
    }

    cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp,
                                const vector<IMU::Point> &vImuMeas, const vector<ODO::Point> &vOdoMeas,
                                string filename) {
        if (mSensor != STEREO && mSensor != IMU_STEREO) {
            cerr << "ERROR: you called TrackStereo but input sensor was not set to Stereo nor Stereo-Inertial." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode) {
                cout << "mbActivateLocalizationMode..." << endl;
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped()) {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode) {
                cout << "mbDeactivateLocalizationMode..." << endl;
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset) {
                mpTracker->Reset();
                cout << "Reset stereo..." << endl;
                mbReset = false;
                mbResetActiveMap = false;
            } else if (mbResetActiveMap) {
                cout << "mpTracker->ResetActiveMap()..." << endl;
                mpTracker->ResetActiveMap();
                mbResetActiveMap = false;
            }
        }

        if (mSensor == System::IMU_STEREO)
            for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
                mpTracker->GrabImuData(vImuMeas[i_imu]);

        if (mSensor == System::STEREO)
            for (size_t i_odo = 0; i_odo < vOdoMeas.size(); i_odo++)
                mpTracker->GrabOdoData(vOdoMeas[i_odo]);

        // std::cout << "start GrabImageStereo" << std::endl;
//    cout<<"GrabImageStereo 1"<<endl;
        cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft, imRight, timestamp, filename);
        //更新IMU 偏置估计的内容
        mpIMUProcessor->setNewPoseFromORB3(mpTracker->mlRelativeFramePoses, mpTracker->mlpReferences,
                                           mpTracker->mlFrameTimes, mpTracker->mlbLost);
//    cout<<"GrabImageStereo 2"<<endl;

        // std::cout << "out grabber" << std::endl;

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

        return Tcw;
    }

    cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp, string filename) {
        if (mSensor != RGBD) {
            cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode) {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped()) {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode) {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset) {
                mpTracker->Reset();
                mbReset = false;
                mbResetActiveMap = false;
            } else if (mbResetActiveMap) {
                mpTracker->ResetActiveMap();
                mbResetActiveMap = false;
            }
        }


        cv::Mat Tcw = mpTracker->GrabImageRGBD(im, depthmap, timestamp, filename);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        return Tcw;
    }

    cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp, const vector<IMU::Point> &vImuMeas,
                                   string filename) {
        if (mSensor != MONOCULAR && mSensor != IMU_MONOCULAR) {
            cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular nor Monocular-Inertial."
                 << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode) {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped()) {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode) {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset) {
                mpTracker->Reset();
                mbReset = false;
                mbResetActiveMap = false;
            } else if (mbResetActiveMap) {
                cout << "SYSTEM-> Reseting active map in monocular case" << endl;
                mpTracker->ResetActiveMap();
                mbResetActiveMap = false;
            }
        }

        if (mSensor == System::IMU_MONOCULAR)
            for (size_t i_imu = 0; i_imu < vImuMeas.size(); i_imu++)
                mpTracker->GrabImuData(vImuMeas[i_imu]);

        cv::Mat Tcw = mpTracker->GrabImageMonocular(im, timestamp, filename);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

        return Tcw;
    }


    void System::ActivateLocalizationMode() {
        unique_lock<mutex> lock(mMutexMode);
        mbActivateLocalizationMode = true;
    }

    void System::DeactivateLocalizationMode() {
        unique_lock<mutex> lock(mMutexMode);
        mbDeactivateLocalizationMode = true;
    }

    bool System::MapChanged() {
        static int n = 0;
        int curn = mpAtlas->GetLastBigChangeIdx();
        if (n < curn) {
            n = curn;
            return true;
        } else
            return false;
    }

    void System::Reset() {
        unique_lock<mutex> lock(mMutexReset);
        mbReset = true;
    }

    void System::ResetActiveMap() {
        unique_lock<mutex> lock(mMutexReset);
        mbResetActiveMap = true;
    }

    void System::Shutdown() {
        //mfDebugForOdo.close();
        mpLocalMapper->RequestFinish();
        mpLoopCloser->RequestFinish();

#ifdef USE_PANGOLIN
        if (mpViewer) {
            mpViewer->RequestFinish();
            while (!mpViewer->isFinished())
                usleep(5000);
        }
#endif
#ifdef USE_SEMI_DENSE_MAP
        mpSemiDenseMapping->RequestFinish();
#endif

        // Wait until all thread have effectively stopped
        while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA()) {
//            if(!mpLocalMapper->isFinished())
//                cout << "mpLocalMapper is not finished" << endl;
//            if(!mpLoopCloser->isFinished())
//                cout << "mpLoopCloser is not finished" << endl;
            if(mpLoopCloser->isRunningGBA()){
                cout << "mpLoopCloser is running GBA" << endl;
//                cout << "break anyway..." << endl;
//                break;
            }
            usleep(5000);
        }

#ifdef USE_PANGOLIN
        if (mpViewer) {
//            pangolin::BindToContext("ORB-SLAM3: Map Viewer");
            delete mpViewer;
            mpViewer = NULL;
        }
#endif
#ifdef USE_SEMI_DENSE_MAP
        while(!mpSemiDenseMapping->isFinished())
            usleep(5000);
#endif
    }

#ifdef SAVE_MAP_PCD
    void System::SaveMapPCD(const string &filename)
    {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        const vector<MapPoint *> &vpMPs = mpAtlas->GetCurrentMap()->GetAllMapPoints();

        if (vpMPs.empty())
            return;

        cloud.width = vpMPs.size();
        cloud.height = 1;
        cloud.is_dense = false;
        cloud.points.resize(cloud.width * cloud.height);

        for (size_t i = 0, iend = vpMPs.size(); i < iend; i++) {
            if (vpMPs[i]->isBad())
                continue;
            cv::Mat pos = vpMPs[i]->GetWorldPos();
            cloud.points[i].x = pos.at<float>(0);
            cloud.points[i].y = pos.at<float>(1);
            cloud.points[i].z = pos.at<float>(2);
        }

        pcl::io::savePCDFileASCII(filename, cloud);
    }
#endif

    void System::SaveTrajectoryTUM(const string &filename) {
        cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
        if (mSensor == MONOCULAR) {
            cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
            return;
        }

        vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<ORB_SLAM3::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        list<bool>::iterator lbL = mpTracker->mlbLost.begin();
        for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
                     lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++) {
            if (*lbL)
                continue;

            KeyFrame *pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            while (pKF->isBad()) {
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw * pKF->GetPose() * Two;

            cv::Mat Tcw = (*lit) * Trw;
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            vector<float> q = Converter::toQuaternion(Rwc);

            f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " "
              << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        }
        f.close();
        // cout << endl << "trajectory saved!" << endl;
    }

    Pose_t System::getLatestCamPose() {
        Pose_t Twc_orb;
        std::vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);
        if(vpKFs.empty()) { // no key frame exist
            Twc_orb.pose = Eigen::Matrix4d::Identity();
            Twc_orb.time = -1;
            return Twc_orb;
        }

        cv::Mat Two = vpKFs[0]->GetPoseInverse();
        list<ORB_SLAM3::KeyFrame *>::const_reverse_iterator lRit = mpTracker->mlpReferences.rbegin();
        list<double>::const_reverse_iterator lT = mpTracker->mlFrameTimes.rbegin();
        list<bool>::const_reverse_iterator lbL = mpTracker->mlbLost.rbegin();
        for (list<cv::Mat>::const_reverse_iterator lit = mpTracker->mlRelativeFramePoses.rbegin(),
                     lend = mpTracker->mlRelativeFramePoses.rend(); lit != lend; lit++, lRit++, lT++, lbL++) {
//            std::cerr << "4" << endl;

            KeyFrame *pKF = *lRit;
            if (!pKF)
                continue;
            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);
//            std::cerr << "pKF point: " << pKF << endl;
//            std::cerr << "pKF id: " << pKF->mnId << endl;

            // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            while (pKF->isBad()) {
//                std::cerr << "6" << endl;
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }
//            std::cerr << "7" << endl;

            Trw = Trw * pKF->GetPose() * Two;

            cv::Mat Tcw = (*lit) * Trw;
            Twc_orb.pose = Converter::toEigen4dInverse(Converter::toMatrix4d(Tcw));
            Twc_orb.time = *lT;
            return Twc_orb;
        }
        Twc_orb.pose = Eigen::Matrix4d::Identity();
        Twc_orb.time = -1;
        return Twc_orb;
    }


    void System::SaveKeyFrameTrajectoryTUM(const string &filename) {
        cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

        vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];

            // pKF->SetPose(pKF->GetPose()*Two);

            if (pKF->isBad())
                continue;

            cv::Mat R = pKF->GetRotation().t();
            vector<float> q = Converter::toQuaternion(R);
            cv::Mat t = pKF->GetCameraCenter();
            f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1)
              << " " << t.at<float>(2)
              << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        }

        f.close();
    }

    void System::SaveTrajectoryEuRoC(const string &filename) {

        cout << endl << "Saving trajectory to " << filename << " ..." << endl;
        /*if(mSensor==MONOCULAR)
        {
            cerr << "ERROR: SaveTrajectoryEuRoC cannot be used for monocular." << endl;
            return;
        }*/

        vector<Map *> vpMaps = mpAtlas->GetAllMaps();
        Map *pBiggerMap;
        int numMaxKFs = 0;
        for (Map *pMap :vpMaps) {
            if (pMap->GetAllKeyFrames().size() > numMaxKFs) {
                numMaxKFs = pMap->GetAllKeyFrames().size();
                pBiggerMap = pMap;
            }
        }

        vector<KeyFrame *> vpKFs = pBiggerMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Twb; // Can be word to cam0 or world to b dependingo on IMU or not.
        if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO)
            Twb = vpKFs[0]->GetImuPose();
        else
            Twb = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        // cout << "file open" << endl;
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<ORB_SLAM3::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        list<bool>::iterator lbL = mpTracker->mlbLost.begin();

        //cout << "size mlpReferences: " << mpTracker->mlpReferences.size() << endl;
        //cout << "size mlRelativeFramePoses: " << mpTracker->mlRelativeFramePoses.size() << endl;
        //cout << "size mpTracker->mlFrameTimes: " << mpTracker->mlFrameTimes.size() << endl;
        //cout << "size mpTracker->mlbLost: " << mpTracker->mlbLost.size() << endl;


        for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
                     lend = mpTracker->mlRelativeFramePoses.end(); lit != lend; lit++, lRit++, lT++, lbL++) {
            //cout << "1" << endl;
            if (*lbL)
                continue;


            KeyFrame *pKF = *lRit;
            //cout << "KF: " << pKF->mnId << endl;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            /*cout << "2" << endl;
            cout << "KF id: " << pKF->mnId << endl;*/

            // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            if (!pKF)
                continue;

            //cout << "2.5" << endl;

            while (pKF->isBad()) {
                //cout << " 2.bad" << endl;
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
                //cout << "--Parent KF: " << pKF->mnId << endl;
            }

            if (!pKF || pKF->GetMap() != pBiggerMap) {
                //cout << "--Parent KF is from another map" << endl;
                /*if(pKF)
                    cout << "--Parent KF " << pKF->mnId << " is from another map " << pKF->GetMap()->GetId() << endl;*/
                continue;
            }

            //cout << "3" << endl;

            Trw = Trw * pKF->GetPose() * Twb; // Tcp*Tpw*Twb0=Tcb0 where b0 is the new world reference

            // cout << "4" << endl;

            if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO) {
                cv::Mat Tbw = pKF->mImuCalib.Tbc * (*lit) * Trw;
                cv::Mat Rwb = Tbw.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twb = -Rwb * Tbw.rowRange(0, 3).col(3);
                vector<float> q = Converter::toQuaternion(Rwb);
                f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twb.at<float>(0) << " "
                  << twb.at<float>(1) << " " << twb.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " "
                  << q[3] << endl;
            } else {
                cv::Mat Tcw = (*lit) * Trw;
                cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
                cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
                vector<float> q = Converter::toQuaternion(Rwc);
                f << setprecision(6) << 1e9 * (*lT) << " " << setprecision(9) << twc.at<float>(0) << " "
                  << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " "
                  << q[3] << endl;
            }

            // cout << "5" << endl;
        }
        //cout << "end saving trajectory" << endl;
        f.close();
        cout << endl << "End of saving trajectory to " << filename << " ..." << endl;
    }


    void System::SaveKeyFrameTrajectoryEuRoC(const string &filename) {
        cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

        vector<Map *> vpMaps = mpAtlas->GetAllMaps();
        Map *pBiggerMap;
        int numMaxKFs = 0;
        for (Map *pMap :vpMaps) {
            if (pMap->GetAllKeyFrames().size() > numMaxKFs) {
                numMaxKFs = pMap->GetAllKeyFrames().size();
                pBiggerMap = pMap;
            }
        }

        vector<KeyFrame *> vpKFs = pBiggerMap->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];

            // pKF->SetPose(pKF->GetPose()*Two);

            if (pKF->isBad())
                continue;
            if (mSensor == IMU_MONOCULAR || mSensor == IMU_STEREO) {
                cv::Mat R = pKF->GetImuRotation().t();
                vector<float> q = Converter::toQuaternion(R);
                cv::Mat twb = pKF->GetImuPosition();
                f << setprecision(6) << 1e9 * pKF->mTimeStamp << " " << setprecision(9) << twb.at<float>(0) << " "
                  << twb.at<float>(1) << " " << twb.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " "
                  << q[3] << endl;

            } else {
                cv::Mat R = pKF->GetRotation();
                vector<float> q = Converter::toQuaternion(R);
                cv::Mat t = pKF->GetCameraCenter();
                f << setprecision(6) << 1e9 * pKF->mTimeStamp << " " << setprecision(9) << t.at<float>(0) << " "
                  << t.at<float>(1) << " " << t.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3]
                  << endl;
            }
        }
        f.close();
    }

    void System::SaveTrajectoryKITTI(const string &filename) {
        cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
        if (mSensor == MONOCULAR) {
            cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
            return;
        }

        vector<KeyFrame *> vpKFs = mpAtlas->GetAllKeyFrames();
        sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // We need to get first the keyframe pose and then concatenate the relative transformation.
        // Frames not localized (tracking failure) are not saved.

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<ORB_SLAM3::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(), lend = mpTracker->mlRelativeFramePoses.end();
             lit != lend; lit++, lRit++, lT++) {
            ORB_SLAM3::KeyFrame *pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            while (pKF->isBad()) {
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw * pKF->GetPose() * Two;

            cv::Mat Tcw = (*lit) * Trw;
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

            f << setprecision(9) << Rwc.at<float>(0, 0) << " " << Rwc.at<float>(0, 1) << " " << Rwc.at<float>(0, 2)
              << " " << twc.at<float>(0) << " " <<
              Rwc.at<float>(1, 0) << " " << Rwc.at<float>(1, 1) << " " << Rwc.at<float>(1, 2) << " " << twc.at<float>(1)
              << " " <<
              Rwc.at<float>(2, 0) << " " << Rwc.at<float>(2, 1) << " " << Rwc.at<float>(2, 2) << " " << twc.at<float>(2)
              << endl;
        }
        f.close();
    }


    void System::SaveDebugData(const int &initIdx) {
        // 0. Save initialization trajectory
        SaveTrajectoryEuRoC(
                "init_FrameTrajectoy_" + to_string(mpLocalMapper->mInitSect) + "_" + to_string(initIdx) + ".txt");

        // 1. Save scale
        ofstream f;
        f.open("init_Scale_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
        f << fixed;
        f << mpLocalMapper->mScale << endl;
        f.close();

        // 2. Save gravity direction
        f.open("init_GDir_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
        f << fixed;
        f << mpLocalMapper->mRwg(0, 0) << "," << mpLocalMapper->mRwg(0, 1) << "," << mpLocalMapper->mRwg(0, 2) << endl;
        f << mpLocalMapper->mRwg(1, 0) << "," << mpLocalMapper->mRwg(1, 1) << "," << mpLocalMapper->mRwg(1, 2) << endl;
        f << mpLocalMapper->mRwg(2, 0) << "," << mpLocalMapper->mRwg(2, 1) << "," << mpLocalMapper->mRwg(2, 2) << endl;
        f.close();

        // 3. Save computational cost
        f.open("init_CompCost_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
        f << fixed;
        f << mpLocalMapper->mCostTime << endl;
        f.close();

        // 4. Save biases
        f.open("init_Biases_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
        f << fixed;
        f << mpLocalMapper->mbg(0) << "," << mpLocalMapper->mbg(1) << "," << mpLocalMapper->mbg(2) << endl;
        f << mpLocalMapper->mba(0) << "," << mpLocalMapper->mba(1) << "," << mpLocalMapper->mba(2) << endl;
        f.close();

        // 5. Save covariance matrix
        f.open("init_CovMatrix_" + to_string(mpLocalMapper->mInitSect) + "_" + to_string(initIdx) + ".txt",
               ios_base::app);
        f << fixed;
        for (int i = 0; i < mpLocalMapper->mcovInertial.rows(); i++) {
            for (int j = 0; j < mpLocalMapper->mcovInertial.cols(); j++) {
                if (j != 0)
                    f << ",";
                f << setprecision(15) << mpLocalMapper->mcovInertial(i, j);
            }
            f << endl;
        }
        f.close();

        // 6. Save initialization time
        f.open("init_Time_" + to_string(mpLocalMapper->mInitSect) + ".txt", ios_base::app);
        f << fixed;
        f << mpLocalMapper->mInitTime << endl;
        f.close();
    }


    int System::GetTrackingState() {
        unique_lock<mutex> lock(mMutexState);
        return mTrackingState;
    }

    vector<MapPoint *> System::GetTrackedMapPoints() {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedMapPoints;
    }

    vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
        unique_lock<mutex> lock(mMutexState);
        return mTrackedKeyPointsUn;
    }

    double System::GetTimeFromIMUInit() {
        double aux = mpLocalMapper->GetCurrKFTime() - mpLocalMapper->mFirstTs;
        if ((aux > 0.) && mpAtlas->isImuInitialized())
            return mpLocalMapper->GetCurrKFTime() - mpLocalMapper->mFirstTs;
        else
            return 0.f;
    }

    bool System::isLost() {
        if (!mpAtlas->isImuInitialized())
            return false;
        else {
            if ((mpTracker->mState == Tracking::LOST)) //||(mpTracker->mState==Tracking::RECENTLY_LOST))
                return true;
            else
                return false;
        }
    }

    void System::GetCurPose(cv::Mat &Tcw) {
        mpTracker->mCurrentFrame.GetPose(Tcw);
    }

#if defined USE_FBOW
    void System::CreateVoc()
    {
        if(!mbCreateVoc)
            return;
        string desc_name = "orb";
        cout << "DescName=" << desc_name << endl;

        vector<cv::Mat> features;
        for(Map* pMap : mpAtlas->GetAllMaps())
        {
            for(KeyFrame* pKF : pMap->GetAllKeyFrames())
            {
                cv::Mat desc = pKF->mDescriptors;
                if(!desc.empty())
                    features.push_back(desc);
            }
        }
        if(features.empty())
        {
            std::cout << "features.empty()"<< std::endl;
            return;
        }

        fbow::VocabularyCreator::Params params;
        params.k = 10;
        params.L = 6;
        params.nthreads = 1;
        params.maxIters = 11;
        srand(0);
        fbow::VocabularyCreator voc_creator;
        fbow::Vocabulary voc;
        cout << "Creating a " << params.k << "^" << params.L << " vocabulary..." << endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        voc_creator.create(voc, features, desc_name, params);
        auto t_end = std::chrono::high_resolution_clock::now();
        //12 s
        cout << "time=" << double(std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count())
        << " msecs" << endl;
        cout << "nblocks=" << voc.blockSize() << endl;
        cout << "nleaf=" << voc.leafSize() << endl;

        string filePath = "/media/qzj/Software/code/catkin_ws/src/ORB-SLAM3/self.fbow";
        voc.saveToFile(filePath);
        std::cout << "saving to " << filePath << std::endl;
    }
#elif defined USE_DBOW3

    void System::CreateVoc() {
        if (!mbCreateVoc)
            return;
        string desc_name = "orb";
        cout << "DescName=" << desc_name << endl;

        vector<cv::Mat> pdescs;
        for (Map *pMap : mpAtlas->GetAllMaps()) {
            for (KeyFrame *pKF : pMap->GetAllKeyFrames()) {
                cv::Mat desc = pKF->mDescriptors;
                if (!desc.empty()) {
                    pdescs.push_back(desc);
                }
            }
        }
        if (pdescs.empty()) {
            std::cout << "features.empty()" << std::endl;
            return;
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        //Create Voc text
        const int K = 10;
        const int L = 6;
        ORBVocabulary voc(K, L, DBoW3::TF_IDF, DBoW3::L1_NORM);
        std::cout << "Creating a " << K << "^" << L << " vocabulary…" << endl;
        voc.create(pdescs);
        std::cout << "…done!" << endl;
        //save the vocabulary to disk
        std::cout << endl << "Saving vocabulary… " << endl;
        //voc.save("voc.yml.gz");
        voc.save("MyVoc_dbow3.yml.gz");
        auto t_end = std::chrono::high_resolution_clock::now();
        cout << "Done time=" << double(std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count())
             << " msecs" << endl;
    }

#elif defined USE_DBOW2
    void System::CreateVoc()
    {
        vector<vector<cv::Mat > > features;
        if(!mbCreateVoc)
            return;
        string desc_name = "orb";
        cout << "DescName=" << desc_name << endl;

        for(Map* pMap : mpAtlas->GetAllMaps())
        {
            for(KeyFrame* pKF : pMap->GetAllKeyFrames())
            {
                cv::Mat desc = pKF->mDescriptors;
                if(!desc.empty())
                {
                    features.push_back(vector<cv::Mat>());
                    VOCProsessor::changeStructure(desc, features.back());
                }
            }
        }
        if(features.empty())
        {
            std::cout << "features.empty()"<< std::endl;
            return;
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        //Create Voc text
        const int K=10;
        const int L=6;
        ORBVocabulary voc(K,L,DBoW2::TF_IDF,DBoW2::L1_NORM);
        std::cout<<"Creating a "<< K <<"^" << L <<" vocabulary…"<<endl;
        voc.create(features);
        std::cout<<"…done!"<<endl;
        //save the vocabulary to disk
        std::cout<<endl<<"Saving vocabulary… "<<endl;
        //voc.save("voc.yml.gz");
        voc.saveToBinaryFile("MyVoc_dbow2.bin");
        auto t_end = std::chrono::high_resolution_clock::now();
        cout << "Done time=" << double(std::chrono::duration_cast<std::chrono::seconds>(t_end - t_start).count())
             << " msecs" << endl;
    }
#endif

    bool System::isFinished() {
        return (GetTimeFromIMUInit() > 0.1);
    }

    void System::ChangeDataset() {
        if (mpAtlas->GetCurrentMap()->KeyFramesInMap() < 12) {
            mpTracker->ResetActiveMap();
        } else {
            mpTracker->CreateMapInAtlas();
        }

        mpTracker->NewDataset();
    }

    void System::SaveTime(string save_path) {
        ofstream f;

        f.open(save_path + "Track_time.txt", std::ios_base::out);
        f << fixed;
        f << "#(timeCur) ExtractFeature StereoMatch TrackFrame TrackMap PostTrack" << endl;
        f << "# mean: " << mpTracker->mpTrackingTime->ExtractFeature.get() << " "
          << mpTracker->mpTrackingTime->StereoMatch.get() << " "
          << mpTracker->mpTrackingTime->TrackFrame.get() << " " << mpTracker->mpTrackingTime->TrackMap.get() << " "
          << mpTracker->mpTrackingTime->PostTrack.get() << endl;
        f << "# max: " << mpTracker->mpTrackingTime->ExtractFeature.getMax() << " "
          << mpTracker->mpTrackingTime->StereoMatch.getMax() << " "
          << mpTracker->mpTrackingTime->TrackFrame.getMax() << " " << mpTracker->mpTrackingTime->TrackMap.getMax()
          << " "
          << mpTracker->mpTrackingTime->PostTrack.getMax() << endl;
        int n = mpTracker->mpTrackingTime->ExtractFeature.times.size();
        for (int i = 0; i < n; i++) {
            f << mpTracker->mpTrackingTime->timesCur[i] << " " << mpTracker->mpTrackingTime->StereoMatch.times[i] << " "
              << mpTracker->mpTrackingTime->TrackFrame.times[i] << " " << mpTracker->mpTrackingTime->TrackMap.times[i]
              << " "
              << mpTracker->mpTrackingTime->PostTrack.times[i] << endl;
        }
        f.close();

        f.open(save_path + "LocalMapper_time.txt", std::ios_base::out);
        f << fixed;
        f << "#(timeCur) kf_num procKF MPcull CheckMP searchNeigh Opt KF_cull Insert" << endl;
        f << "# mean: " << GetMean(mpLocalMapper->mpLocalMapTime->kf_cnt) << " "
          << mpLocalMapper->mpLocalMapTime->procKF.get() << " "
          << mpLocalMapper->mpLocalMapTime->MPcull.get() << " " << mpLocalMapper->mpLocalMapTime->CheckMP.get() << " "
          << mpLocalMapper->mpLocalMapTime->searchNeigh.get() << " " << mpLocalMapper->mpLocalMapTime->Opt.get() << " "
          << mpLocalMapper->mpLocalMapTime->KF_cull.get() << " " << mpLocalMapper->mpLocalMapTime->Insert.get() << endl;
        f << "# max: " << GetMax(mpLocalMapper->mpLocalMapTime->kf_cnt) << " "
          << mpLocalMapper->mpLocalMapTime->procKF.getMax() << " " << mpLocalMapper->mpLocalMapTime->MPcull.getMax()
          << " "
          << mpLocalMapper->mpLocalMapTime->CheckMP.getMax() << " "
          << mpLocalMapper->mpLocalMapTime->searchNeigh.getMax() << " " << mpLocalMapper->mpLocalMapTime->Opt.getMax()
          << " "
          << mpLocalMapper->mpLocalMapTime->KF_cull.getMax() << " " << mpLocalMapper->mpLocalMapTime->Insert.getMax()
          << endl;
        n = mpLocalMapper->mpLocalMapTime->procKF.times.size();
        for (int i = 0; i < n; i++) {
            f << mpLocalMapper->mpLocalMapTime->timesCur[i] << " " << mpLocalMapper->mpLocalMapTime->kf_cnt[i] << " "
              << mpLocalMapper->mpLocalMapTime->procKF.times[i] << " "
              << mpLocalMapper->mpLocalMapTime->MPcull.times[i] << " "
              << mpLocalMapper->mpLocalMapTime->CheckMP.times[i] << " "
              << mpLocalMapper->mpLocalMapTime->searchNeigh.times[i] << " "
              << mpLocalMapper->mpLocalMapTime->Opt.times[i] << " " << mpLocalMapper->mpLocalMapTime->KF_cull.times[i]
              << " "
              << mpLocalMapper->mpLocalMapTime->Insert.times[i] << endl;
        }
        f.close();

        f.open(save_path + "LoopClose_time.txt", std::ios_base::out);
        f << fixed;
        f << "#(timeCur) Detect Loop Merge" << endl;
        f << "# mean: " << " " << mpLoopCloser->mpLoopCloseTime->Detect.get() << " "
          << mpLoopCloser->mpLoopCloseTime->Loop.get() << " " << mpLoopCloser->mpLoopCloseTime->Merge.get() << endl;
        f << "# max: " << " " << mpLoopCloser->mpLoopCloseTime->Detect.getMax() << " "
          << mpLoopCloser->mpLoopCloseTime->Loop.getMax() << " " << mpLoopCloser->mpLoopCloseTime->Merge.getMax()
          << endl;
        n = mpLoopCloser->mpLoopCloseTime->Detect.times.size();
        for (int i = 0; i < n; i++) {
            f << mpLoopCloser->mpLoopCloseTime->timesCur[i] << " " << mpLoopCloser->mpLoopCloseTime->Detect.times[i]
              << " "
              << mpLoopCloser->mpLoopCloseTime->Loop.times[i] << " " << mpLoopCloser->mpLoopCloseTime->Merge.times[i]
              << endl;
        }
        f.close();

//        cout << "finish save time" << endl;
    }

/*void System::SaveAtlas(int type){
    cout << endl << "Enter the name of the file if you want to save the current Atlas session. To exit press ENTER: ";
    string saveFileName;
    getline(cin,saveFileName);
    if(!saveFileName.empty())
    {
        //clock_t start = clock();

        // Save the current session
        mpAtlas->PreSave();
        mpKeyFrameDatabase->PreSave();

        string pathSaveFileName = "./";
        pathSaveFileName = pathSaveFileName.append(saveFileName);
        pathSaveFileName = pathSaveFileName.append(".osa");

        string strVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);
        std::size_t found = mStrVocabularyFilePath.find_last_of("/\\");
        string strVocabularyName = mStrVocabularyFilePath.substr(found+1);

        if(type == TEXT_FILE) // File text
        {
            cout << "Starting to write the save text file " << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::text_oarchive oa(ofs);

            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            oa << mpKeyFrameDatabase;
            cout << "End to write the save text file" << endl;
        }
        else if(type == BINARY_FILE) // File binary
        {
            cout << "Starting to write the save binary file" << endl;
            std::remove(pathSaveFileName.c_str());
            std::ofstream ofs(pathSaveFileName, std::ios::binary);
            boost::archive::binary_oarchive oa(ofs);
            oa << strVocabularyName;
            oa << strVocabularyChecksum;
            oa << mpAtlas;
            oa << mpKeyFrameDatabase;
            cout << "End to write save binary file" << endl;
        }

        //clock_t timeElapsed = clock() - start;
        //unsigned msElapsed = timeElapsed / (CLOCKS_PER_SEC / 1000);
        //cout << "Binary file saved in " << msElapsed << " ms" << endl;
    }
}

bool System::LoadAtlas(string filename, int type)
{
    string strFileVoc, strVocChecksum;
    bool isRead = false;

    if(type == TEXT_FILE) // File text
    {
        cout << "Starting to read the save text file " << endl;
        std::ifstream ifs(filename, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::text_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        //ia >> mpKeyFrameDatabase;
        cout << "End to load the save text file " << endl;
        isRead = true;
    }
    else if(type == BINARY_FILE) // File binary
    {
        cout << "Starting to read the save binary file"  << endl;
        std::ifstream ifs(filename, std::ios::binary);
        if(!ifs.good())
        {
            cout << "Load file not found" << endl;
            return false;
        }
        boost::archive::binary_iarchive ia(ifs);
        ia >> strFileVoc;
        ia >> strVocChecksum;
        ia >> mpAtlas;
        //ia >> mpKeyFrameDatabase;
        cout << "End to load the save binary file" << endl;
        isRead = true;
    }

    if(isRead)
    {
        //Check if the vocabulary is the same
        string strInputVocabularyChecksum = CalculateCheckSum(mStrVocabularyFilePath,TEXT_FILE);

        if(strInputVocabularyChecksum.compare(strVocChecksum) != 0)
        {
            cout << "The vocabulary load isn't the same which the load session was created " << endl;
            cout << "-Vocabulary name: " << strFileVoc << endl;
            return false; // Both are differents
        }

        return true;
    }
    return false;
}

string System::CalculateCheckSum(string filename, int type)
{
    string checksum = "";

    unsigned char c[MD5_DIGEST_LENGTH];

    std::ios_base::openmode flags = std::ios::in;
    if(type == BINARY_FILE) // Binary file
        flags = std::ios::in | std::ios::binary;

    ifstream f(filename.c_str(), flags);
    if ( !f.is_open() )
    {
        cout << "[E] Unable to open the in file " << filename << " for Md5 hash." << endl;
        return checksum;
    }

    MD5_CTX md5Context;
    char buffer[1024];

    MD5_Init (&md5Context);
    while ( int count = f.readsome(buffer, sizeof(buffer)))
    {
        MD5_Update(&md5Context, buffer, count);
    }

    f.close();

    MD5_Final(c, &md5Context );

    for(int i = 0; i < MD5_DIGEST_LENGTH; i++)
    {
        char aux[10];
        sprintf(aux,"%02x", c[i]);
        checksum = checksum + aux;
    }

    return checksum;
}*/

} //namespace ORB_SLAM


