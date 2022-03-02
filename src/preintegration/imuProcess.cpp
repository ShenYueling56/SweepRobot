//
// Created by qzj on 2020/9/18.
//
#include "include/preintegration/imuProcess.h"
#include "include/orb_slam3/Converter.h"
#include "Util/RotUtil.h"
#include <list>
#include<iostream>

using namespace std;
using namespace Eigen;

//#define DEBUG_THIS

namespace ORB_SLAM3 {

    IMUProcess::IMUProcess() : initFirstPoseFlag(false), first_imu(false),
                               prevTime(-1), curTime(0), mbPreOdoUpdate(false), estBiasFirst(false),
                               mbIsUpdateBias(false),
                               frame_count(0), solver_flag(INITIAL) {
        mProcess.lock();
        while (!accBuf.empty())
            accBuf.pop();
        while (!gyrBuf.empty())
            gyrBuf.pop();

        mPreOdo = cv::Mat::eye(4, 4, CV_32F);
        mOdoReceive.setZero();
        for (int i = 0; i < WINDOW_SIZE + 1; i++) {
            Rs[i].setIdentity();
            mRwc_orb[i].setIdentity();
            mtwc_orb[i].setZero();
            Ps[i].setZero();
            Vs[i].setZero();
            Bas[i].setZero();
            Bgs[i].setZero();
            dt_buf[i].clear();
            linear_acceleration_buf[i].clear();
            angular_velocity_buf[i].clear();
            pre_integrations[i] = nullptr;
        }

        mZUPT = new ZUPT(0.3, 20, 500);

        R_init.setIdentity();
        Rwi_IMU_Pre.setIdentity();
        Pwi_IMU_Pre.setIdentity();

        for (int i = 0; i < 2; i++) {
            tic[i] = Vector3d::Zero();
            ric[i] = Matrix3d::Identity();
        }

        all_image_frame.clear();

        tmp_pre_integration = nullptr;

        mProcess.unlock();
        cout << "IMUProcess Create." << endl;
    }

    // 设置参数，并开启processMeasurements线程
    void IMUProcess::setParameter() {
        mProcess.lock();//涉及到多线程，暂时不了解
        // 讲相机参数传入
        for (int i = 0; i < NUM_OF_CAM; i++) {
            tic[i] = TIC[i];
            ric[i] = RIC[i];
            //cout << " exitrinsic cam " << i << endl << ric[i] << endl << tic[i].transpose() << endl;
        }
        g = G;//理想的中立加速度
        cout << "set g " << g.transpose() << endl;
        mProcess.unlock();
    }

    //预测当前相机的PVQ
    void IMUProcess::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity) {
        double dt = t - latest_time;
        latest_time = t;
        Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g; //上一个世界坐标系下的加速度
        Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg; //机体坐标系下的角速度
        latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt); //更新
        Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
        latest_V = latest_V + dt * un_acc;
        latest_acc_0 = linear_acceleration;
        latest_gyr_0 = angular_velocity;
    }

    // 输入一个imu的量测
// 填充了accBuf和gyrBuf
    void
    IMUProcess::inputIMU(double t, const Eigen::Vector3d &linearAcceleration, const Eigen::Vector3d &angularVelocity,
                         const cv::Point2f &encoder) {
        mBuf.lock();
        accBuf.push(make_pair(t, linearAcceleration));
        gyrBuf.push(make_pair(t, angularVelocity));
        encoderBuf.push(make_pair(t, encoder));
        //printf("input imu with time %f \n", t);
        mBuf.unlock();
        mZUPT->estimateBias(angularVelocity);

        //if (fillWindows == true) {
        //    mPropagate.lock();
        //    //计算当前时刻的pvq 先计算q的
        //    fastPredictIMU(t, linearAcceleration, angularVelocity);
        //    mPropagate.unlock();
        //}
    }


    bool ZUPT::CheckStatic(double wz) {
        wz = fabs(wz * RAD_TO_ANGLE);
        //ROS_INFO("%f",wz);
        if (wz < STATIC_THRESHOLD) {
            staticNum++;
            //cout<<"ZUPT staticNum "<<staticNum<<endl;
        } else
            staticNum = 0;
        if (staticNum > STATIC_JUDGE_NUM) {
            staticNum = STATIC_JUDGE_NUM + 1;
            return true;
        } else
            return false;
    }

// 输入原始角速度wRaw，输出去零飘的角速度wNew
    void ZUPT::estimateBias(const Eigen::Vector3d &wRaw) {
        Eigen::Vector3d wNew;
        if (staticCount >= STATIC_BUFFER_NUM) {
            mbBiasUpdate = true;
            setBias(wBiasSum / double(STATIC_BUFFER_NUM));
            wNew = wRaw - getBias();
        } else
            wNew = wRaw;

        if (CheckStatic(wNew[2])) {
            staticCount++;
            wRawSlide.push(wRaw);
            wBiasSum = wBiasSum + wRawSlide.back();
            if (staticCount > STATIC_BUFFER_NUM) {
                wBiasSum = wBiasSum - wRawSlide.front();
                wRawSlide.pop();
                staticCount = STATIC_BUFFER_NUM;
            }
        }
        //else
        //    staticCount = 0;

        if (staticCount > 0 && staticCountLast == 0)
            cout << "ZUPT robot is static" << endl;
        else if (staticCount == 0 && staticCountLast != 0)
            cout << "ZUPT robot start to move" << endl;
        if (staticCount == STATIC_BUFFER_NUM && staticCountLast != STATIC_BUFFER_NUM)
            cout << "ZUPT imu bais update" << endl;
        staticCountLast = staticCount;

        //for (unsigned char axis = 0; axis < 3; axis++)
        //    if (fabs(wNew[axis] * RAD_TO_ANGLE) < STATIC_THRESHOLD)
        //        wNew[axis] = 0.0;
    }

    bool IMUProcess::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                                    vector<pair<double, Eigen::Vector3d>> &gyrVector,
                                    std::vector<std::pair<double, cv::Point2f>> &encVector) {
        if (accBuf.empty()) {
            printf("not receive imu\n");
            return false;
        }
        //printf("get imu from %f %f\n", t0, t1);
        //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);

        // 有足够的imu值
        if (t1 <= accBuf.back().first) {
            //如果队列里第一个数据的时间小于起始时间，则删除第一个元素
            while (accBuf.front().first <= t0) {
                accBuf.pop();//.pop删除栈顶元素
                gyrBuf.pop();
                encoderBuf.pop();
            }
            // 讲队列里所有的acc和gyr输入到accvector个gyrvector中
            while (accBuf.front().first < t1) {
                accVector.push_back(accBuf.front());
                accBuf.pop();
                gyrVector.push_back(gyrBuf.front());
                gyrBuf.pop();
                encVector.push_back(encoderBuf.front());
                encoderBuf.pop();
            }
            //fixme 这里不需要判断accBuf是否还有值吗
            //再多加一个超过时间t1的
            accVector.push_back(accBuf.front());
            gyrVector.push_back(gyrBuf.front());
            encVector.push_back(encoderBuf.front());
        } else {
            printf("wait for imu\n");
            return false;
        }
        return true;
    }

    // 判断输入的时间t时候的imu是否可用
    bool IMUProcess::IMUAvailable(double t) {
//        cout<<"size: "<<accBuf.size()<<" time "<<accBuf.back().first<<endl;
        if (!accBuf.empty() && t <= accBuf.back().first)
            return true;
        else
            return false;
    }

    //初始第一个imu位姿
    void IMUProcess::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector) {
        printf("init first imu pose\n");
        initFirstPoseFlag = true;
        //return;
        //计算加速度的均值
        Eigen::Vector3d averAcc(0, 0, 0);
        int n = (int) accVector.size();
        for (size_t i = 0; i < accVector.size(); i++) {
            averAcc = averAcc + accVector[i].second;
        }
        averAcc = averAcc / n;
//        printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());

        Eigen::Matrix3d R0 = Utility::g2R(averAcc);
        double yaw = Utility::R2ypr(R0).x();
        R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;//另初始的航向为0
        Rs[0] = R0;
        R_init = R0;
        cout << "init R0 " << endl << Rs[0] << endl;
        //Vs[0] = Vector3d(5, 0, 0);
    }

    void IMUProcess::preIntegrateIMU(double img_t) {
        //printf("process measurments\n");
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        vector<pair<double, cv::Point2f>> encVector;
        curTime = img_t;
        //判断是否有可用的IMU
        while (1) {
            if ((IMUAvailable(curTime)))
                break;
            else {
                printf("wait for imu ... \n");
                std::chrono::milliseconds dura(5);//定义5ms的延迟
                std::this_thread::sleep_for(dura);//这个线程延迟5ms
            }
        }
        mBuf.lock();
        // 对imu的时间进行判断，讲队列里的imu数据放入到accVector和gyrVector中
        getIMUInterval(prevTime, curTime, accVector, gyrVector, encVector);
        mBuf.unlock();

        //初始化旋转
        if (!initFirstPoseFlag)
            initFirstIMUPose(accVector);
        setOdoTpcDelta(cv::Mat::eye(4, 4, CV_32F));
        for (size_t i = 0; i < accVector.size(); i++) {
            // 对于前n-1个，accVector[i].first 大于 prevTime 小于 curTime， 第n个 大于 curTime
            double dt;//计算每次imu量测之间的dt
            if (i == 0)
                dt = accVector[i].first - prevTime;
            else if (i == accVector.size() - 1) {
                // 因为accVector[i].first大于curTime，这个时候用curTime - accVector[i - 1].first
                dt = curTime - accVector[i - 1].first;
                //printf("time: %f.\n",dt);
            } else
                dt = accVector[i].first - accVector[i - 1].first;
            //进行了预积分，改变了Rs[frame_count]，Ps[frame_count]，Vs[frame_count]
            processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second, encVector[i].second);
        }
        setPreOdoUpdate(true);
    }

    void IMUProcess::updateIMUBias() {
        mProcess.lock();
        processImage(curTime);
        prevTime = curTime;
        mProcess.unlock();
    }

    /**
 * @brief   处理IMU数据
 * @Description IMU预积分，中值积分得到当前PQV作为优化初值
 * @param[in]   dt 时间间隔
 * @param[in]   linear_acceleration 线加速度
 * @param[in]   angular_velocity 角速度
 * @return  void
*/
    void IMUProcess::processIMU(double t, double dt, const Eigen::Vector3d &linear_acceleration,
                                const Eigen::Vector3d &angular_velocity,
                                const cv::Point2f &encoder) {
        double yaw = 0.0;
        // 第一个imu处理
        if (!first_imu) {
            first_imu = true;
            acc_0 = linear_acceleration;
            gyr_0 = angular_velocity;
            yawLast = yaw;
            encoderLast = encoder;
        }

        // 如果是新的一帧,则新建一个预积分项目
        if (!pre_integrations[frame_count]) {
            //初始化协方差矩阵
            pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
        }

        //f rame_count是窗内图像帧的计数
        // 一个窗内有是个相机帧，每个相机帧之间又有多个IMU数据
        if (frame_count != 0) {
            //更新两帧间pvq变换量，雅克比，协方差，偏置
            pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);

            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);//跟上面一样的输入，一样的操作

            //dt,linear_acceleration,angular_velocity经过上面两步并未改变
            dt_buf[frame_count].push_back(dt);
            linear_acceleration_buf[frame_count].push_back(linear_acceleration);
            angular_velocity_buf[frame_count].push_back(angular_velocity);

            // 得到当前帧的PVQ，与fastPredictIMU中的操作类似
            // Rs Ps Vs是frame_count这一个图像帧开始的预积分值,是在世界坐标系下的.
            int j = frame_count;
            Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;//移除了偏执的加速度
            Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];//移除了偏执的gyro
            Rs[j] = Rs[j] * Utility::deltaQ(un_gyr * dt).toRotationMatrix();
            Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
            Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            Ps[j] = Ps[j] + dt * Vs[j] + 0.5 * dt * dt * un_acc;
            Vs[j] = Vs[j] + dt * un_acc;

            Rwi_IMU_Pre = Rs[j];
            Pwi_IMU_Pre = Ps[j];

            yaw = Utility::R2ypr(Rwi_IMU_Pre).x();
            double alpha = (yaw - yawLast) / 180.0 * M_PI;
            Eigen::Vector2d delatEncoder(encoder.x - encoderLast.x, encoder.y - encoderLast.y);
            Eigen::Vector3d accOdo = Eigen::Vector3d::Zero();
            cv::Mat T_pre_cur = cv::Mat();
            IntegrateOdo(dt, delatEncoder, alpha, accOdo, T_pre_cur);
            updateOdoTpcDelta(T_pre_cur);
#ifdef DEBUG_THIS
            mPreOdo = mPreOdo * T_pre_cur;
#endif
        }
        // 让此时刻的值等于上一时刻的值，为下一次计算做准备
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
        yawLast = yaw;
        encoderLast = encoder;
    }

// image 里边放的就是该图像的特征点 header 时间
    void IMUProcess::processImage(const double header) {
        //cout<<std::fixed<<"frame_count "<<frame_count<<" header: "<<header<<endl;
        //他俩应该相等的
        Headers[frame_count] = header;
        //将图像数据、时间、临时预积分值存到图像帧类中
        ImageFrame imageframe(header);
        imageframe.pre_integration = tmp_pre_integration;
        // 检测关键帧
        if (true) {
            imageframe.is_key_frame = true;
            marginalization_flag = MARGIN_OLD;//新一阵将被作为关键帧!
        } else {
            marginalization_flag = MARGIN_SECOND_NEW;
        }
        all_image_frame.insert(make_pair(header, imageframe));
        //更新临时预积分初始值
        tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

        //frameCnt 为第一帧时不进行计算
        // 将结果放入到队列当中
        if (frame_count == WINDOW_SIZE) {
            updatePoseFromORB3(tic, ric);

            if (getIsUpdateBias())
                solveGyroscopeBias(all_image_frame, Bgs);

            // 对之前预积分得到的结果进行更新。
            // 预积分的好处查看就在于你得到新的Bgs，不需要又重新再积分一遍，可以通过Bgs对位姿，速度的一阶导数，进行线性近似，得到新的Bgs求解出MU的最终结果。
            for (int i = 0; i <= WINDOW_SIZE; i++) {
                pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
            }
            //updateLatestStates();
            //solver_flag = NON_LINEAR;
            slideWindow();
#ifdef DEBUG_THIS
            printf("yaw: %f %f x: %f %f y: %f %f Bgs: %f %f %f ZUPT: %f %f %f\n",mOdoReceive[2],
                   Utility::R2ypr(Converter::toMatrix3d(mPreOdo.rowRange(0,3).colRange(0,3))).x(),
                   mOdoReceive[0],mPreOdo.at<float>(0,3),
                   mOdoReceive[1],mPreOdo.at<float>(1,3),
                   Bgs[WINDOW_SIZE].x(), Bgs[WINDOW_SIZE].y(), Bgs[WINDOW_SIZE].z(),
                   mZUPT->getBias()[0],mZUPT->getBias()[1],mZUPT->getBias()[2]);
#endif
        }

        // 如果划窗内的没有放满,进行状态更新
        if (frame_count < WINDOW_SIZE) {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }
    }

    void IntegrateOdo(const double &deltaT, Eigen::Vector2d &delatEncoder, const double &alpha, Eigen::Vector3d &accOdo,
                      cv::Mat &T_pre_cur) {
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

        accOdo = Eigen::Vector3d::Zero(); //不够六个时用初始值
        if (delta_t_X.size() == 6) {
            vector<double> acc(5);
            for (int i = 0; i < 5; i++)
                acc.at(i) = (delta_t_X.at(i + 1).second.second - delta_t_X.at(i).second.second) /
                            (delta_t_X.at(i + 1).first + delta_t_X.at(i).first) / 2.0;
            Eigen::Vector3d acc3;
            for (int i = 0; i < 3; i++)
                acc3(i) = (acc.at(i) + acc.at(i + 1) + acc.at(i + 2)) / 3.0;
            accOdo(0) = (acc3(0) + acc3(1) + acc3(2)) / 3.0;
            //accOdo(0) = acc.at(4);
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
        T_pre_cur = cv::Mat::eye(4, 4, CV_32F);
        T_pre_cur.at<float>(0, 0) = cos(alpha);
        T_pre_cur.at<float>(0, 1) = -sin(alpha);
        T_pre_cur.at<float>(1, 0) = sin(alpha);
        T_pre_cur.at<float>(1, 1) = cos(alpha);
        T_pre_cur.at<float>(0, 3) = deltaOdo(0);
        T_pre_cur.at<float>(1, 3) = deltaOdo(1);
        //.copyTo(LastTwc.rowRange(0,3).col(3))
    }

    // 滑动窗口法
    void IMUProcess::slideWindow() {
        if (marginalization_flag == MARGIN_OLD) {
            double t_0 = Headers[0];
            //滑动窗口左移一个单位
            //cout<<"all_image size 1 "<<all_image_frame.size()<<endl;
            if (frame_count == WINDOW_SIZE) {
                for (int i = 0; i < WINDOW_SIZE; i++) {
                    Headers[i] = Headers[i + 1];
                    Rs[i].swap(Rs[i + 1]);//交换
                    Ps[i].swap(Ps[i + 1]);
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);//交换预积分值

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }

                // 滑动窗口最后一个与倒数第二个一样
                Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
                Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
                Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];
                delete pre_integrations[WINDOW_SIZE];//讲预积分的最后一个值删除
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE],
                                                                    Bgs[WINDOW_SIZE]};//在构造一个新的
                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();


                map<double, ImageFrame>::iterator it_0;
                //cout<<"t_0 "<<t_0<<endl;
                //it_0 = all_image_frame.find(t_0);//找到第一个
                it_0 = all_image_frame.begin();//找到第一个
                delete it_0->second.pre_integration;
                all_image_frame.erase(it_0);
                //all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            //cout<<"all_image size 2 "<<all_image_frame.size()<<endl;
        }
            // 把滑动窗口第二新的去掉（第一新的因为是最新的所以就挪到第二新的位置上去）
        else {
            if (frame_count == WINDOW_SIZE) {
                Headers[frame_count - 1] = Headers[frame_count];
                Ps[frame_count - 1] = Ps[frame_count];
                Rs[frame_count - 1] = Rs[frame_count];

                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration,
                                                                 tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE],
                                                                    Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
        }
    }

    void IMUProcess::setNewPoseFromORB3(list<cv::Mat> &mlRelativeFramePoses, list<KeyFrame *> &mlpReferences,
                                        list<double> &mlFrameTimes, list<bool> &mlbLost) {
        int cnt = 0;
        if (mlpReferences.size() <= WINDOW_SIZE || !getIsUpdateBias()) {
            setIsUpdateBias(false);
            return;
        }

        list<ORB_SLAM3::KeyFrame *>::reverse_iterator lRit = mlpReferences.rbegin();
        list<double>::reverse_iterator lT = mlFrameTimes.rbegin();
        list<bool>::reverse_iterator lbL = mlbLost.rbegin();
        for (list<cv::Mat>::reverse_iterator lit = mlRelativeFramePoses.rbegin(),
                     lend = mlRelativeFramePoses.rend(); lit != lend; lit++, lRit++, lT++, lbL++) {
            if (*lbL) {
                //丢失 在这个系统里不会出现
                continue;
            }

            KeyFrame *pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

            // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            while (pKF->isBad()) {
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw * pKF->GetPose();

            cv::Mat Tcw = (*lit) * Trw;
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
            mRwc_orb[WINDOW_SIZE - cnt] = Converter::toMatrix3d(Rwc);
            mtwc_orb[WINDOW_SIZE - cnt] = Converter::toVector3d(twc);
            //cout<<"Rwc "<<cnt<<" "<<Converter::toMatrix3d(Rwc)<<endl;
            //Headers[WINDOW_SIZE - cnt] = *lT;
            if (cnt == WINDOW_SIZE)
                break;
            cnt++;
        }
    }

    void IMUProcess::updatePoseFromORB3(Vector3d tic[], Matrix3d ric[]) {
        //cout<<"ric[0] "<<ric[0]<<endl;
        //cout<<"tic[0] "<<tic[0]<<endl;

        map<double, ImageFrame>::iterator frame_it;
        int i = 0;
        Eigen::Matrix3d Rs;
        Eigen::Vector3d Ps;
        for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++) {
            Rs = mRwc_orb[i] * ric[0].transpose();
            Ps = -mRwc_orb[i] * ric[0].transpose() * tic[0] + mtwc_orb[i];
            Rs = ric[0] * Rs;
            Ps = ric[0] * Ps + tic[0];
            //cout<<"Rs "<<i<<" "<<Rs<<endl;
            frame_it->second.R = Rs;
            frame_it->second.T = Ps;
            i++;
        }

        //cout<<"mtwc_orb "<<mtwc_orb.transpose()<<endl;
        //cout<<"Ps[frameCnt] "<<Ps[frameCnt].transpose()<<endl;
    }

    void IMUProcess::solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs) {
        Matrix3d A;
        Vector3d b;
        Vector3d delta_bg;
        A.setZero();
        b.setZero();
        map<double, ImageFrame>::iterator frame_i;
        map<double, ImageFrame>::iterator frame_j;
        //cout<<"all_image size "<<all_image_frame.size()<<endl;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
            frame_j = next(frame_i);
            if (frame_i->first >= frame_j->first) {
                cout << "map order error" << endl;
            }
            MatrixXd tmp_A(3, 3);
            tmp_A.setZero();
            VectorXd tmp_b(3);
            tmp_b.setZero();
            Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
            //cout<<"cam: "<<Utility::R2ypr(q_ij.toRotationMatrix()).transpose();
            //cout<<" imu: "<<Utility::R2ypr(frame_j->second.pre_integration->delta_q.toRotationMatrix()).transpose()<<endl;
            tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
            tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
            A += tmp_A.transpose() * tmp_A;
            b += tmp_A.transpose() * tmp_b;
        }
        delta_bg = A.ldlt().solve(b);
        //ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

        //    静止初始化，始终相信
        {
            unique_lock<mutex> lock(mBiasUpdate);

            if (!estBiasFirst) {
                estBiasFirst = true;
                for (int i = 0; i <= WINDOW_SIZE; i++)
                    Bgs[i] = Bgs[i] + delta_bg;
            } else {
                //cout<<"delta_bg.norm(): "<< delta_bg.norm()<<endl;
                //控制偏置估计不能浮动太大
                if (delta_bg.norm() < 0.01) {
                    for (int i = 0; i <= WINDOW_SIZE; i++) {
                        float alpha = 0.5;
                        Vector3d BgsNew = Bgs[i] + delta_bg;
                        Bgs[i] = alpha * Bgs[i] + (1 - alpha) * BgsNew;
                    }
                }
            }
            float alphaForZUPT = 0.2;
            Vector3d bias = mZUPT->getBias();
            if (mZUPT->mbBiasUpdate)
                for (int i = 0; i <= WINDOW_SIZE; i++)
                    Bgs[i] = alphaForZUPT * bias + (1 - alphaForZUPT) * Bgs[i];
        }

        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
            frame_j = next(frame_i);
            frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
        }
    }

    void IMUProcess::updateLatestStates() {
        mPropagate.lock();
        latest_time = Headers[frame_count];
        latest_P = Ps[frame_count];
        latest_Q = Rs[frame_count];
        latest_V = Vs[frame_count];
        latest_Ba = Bas[frame_count];
        latest_Bg = Bgs[frame_count];
        latest_acc_0 = acc_0;
        latest_gyr_0 = gyr_0;
        mBuf.lock();
        queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
        queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
        mBuf.unlock();
        while (!tmp_accBuf.empty()) {
            double t = tmp_accBuf.front().first;
            Eigen::Vector3d acc = tmp_accBuf.front().second;
            Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
            fastPredictIMU(t, acc, gyr);
            tmp_accBuf.pop();
            tmp_gyrBuf.pop();
        }
        mPropagate.unlock();
    }

    bool IMUProcess::getPreOdoUpdate() {
        unique_lock<mutex> lock(mOdoUpdate);
        return mbPreOdoUpdate;
    }

    void IMUProcess::setPreOdoUpdate(bool preOdoUpdate) {
        unique_lock<mutex> lock(mOdoUpdate);
        mbPreOdoUpdate = preOdoUpdate;
    }

    bool IMUProcess::getIsUpdateBias() {
        unique_lock<mutex> lock(mIsUpdateBias);
        return mbIsUpdateBias;
    }

    void IMUProcess::setIsUpdateBias(bool isUpdateBias) {
        unique_lock<mutex> lock(mIsUpdateBias);
        mbIsUpdateBias = isUpdateBias;
    }

    void IMUProcess::setOdoTpcDelta(cv::Mat odoTpcDelta) {
        unique_lock<mutex> lock(mutexOdoTpcDelta);
        mOdoTpcDelta = odoTpcDelta.clone();
    }

    void IMUProcess::updateOdoTpcDelta(cv::Mat T_pre_cur) {
        unique_lock<mutex> lock(mutexOdoTpcDelta);
        mOdoTpcDelta = mOdoTpcDelta * T_pre_cur;
    }

    cv::Mat IMUProcess::getOdoTpcDelta() {
        unique_lock<mutex> lock(mutexOdoTpcDelta);
        return mOdoTpcDelta.clone();
    }

    IntegrationBase::IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                                     const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
            : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
              linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
              jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
              sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()},
              delta_v{Eigen::Vector3d::Zero()} {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    void IntegrationBase::push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void IntegrationBase::repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg) {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    void IntegrationBase::midPointIntegration(double _dt,
                                              const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                                              const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                                              const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
                                              const Eigen::Vector3d &delta_v,
                                              const Eigen::Vector3d &linearized_ba,
                                              const Eigen::Vector3d &linearized_bg,
                                              Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q,
                                              Eigen::Vector3d &result_delta_v,
                                              Eigen::Vector3d &result_linearized_ba,
                                              Eigen::Vector3d &result_linearized_bg, bool update_jacobian) {
        //cout<<"midpoint integration");
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        if (update_jacobian) {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x << 0, -w_x(2), w_x(1),
                    w_x(2), 0, -w_x(0),
                    -w_x(1), w_x(0), 0;
            R_a_0_x << 0, -a_0_x(2), a_0_x(1),
                    a_0_x(2), 0, -a_0_x(0),
                    -a_0_x(1), a_0_x(0), 0;
            R_a_1_x << 0, -a_1_x(2), a_1_x(1),
                    a_1_x(2), 0, -a_1_x(0),
                    -a_1_x(1), a_1_x(0), 0;

            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x *
                                  (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x *
                                  (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            MatrixXd V = MatrixXd::Zero(15, 18);
            V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
            V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
            V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

            //step_jacobian = F;
            //step_V = V;
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }

    }

    /**
     * @brief   IMU预积分传播方程
     * @Description  积分计算两个关键帧之间IMU测量的变化量：
     *               旋转delta_q 速度delta_v 位移delta_p
     *               加速度的biaslinearized_ba 陀螺仪的Bias linearized_bg
     *               同时维护更新预积分的Jacobian和Covariance,计算优化时必要的参数
     * @param[in]   _dt 时间间隔
     * @param[in]   _acc_1 线加速度
     * @param[in]   _gyr_1 角速度
     * @return  void
     */
    void IntegrationBase::propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1) {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;
        /**
        * @brief   IMU预积分中采用中值积分递推Jacobian和Covariance, PVQ, ba, bg
        *          构造误差的线性化递推方程，得到Jacobian和Covariance递推公式-> Paper 式9、10、11
        * @return  void
        */
        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
        //                    linearized_ba, linearized_bg);
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;

    }

    Eigen::Matrix<double, 15, 1>
    IntegrationBase::evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
                              const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                              const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
                              const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) =
                Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    MatrixXd TangentBasis(Vector3d &g0) {
        Vector3d b, c;
        Vector3d a = g0.normalized();
        Vector3d tmp(0, 0, 1);
        if (a == tmp)
            tmp << 1, 0, 0;
        b = (tmp - a * (a.transpose() * tmp)).normalized();
        c = a.cross(b);
        MatrixXd bc(3, 2);
        bc.block<3, 1>(0, 0) = b;
        bc.block<3, 1>(0, 1) = c;
        return bc;
    }

    void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x) {
        Vector3d g0 = g.normalized() * G.norm();
        Vector3d lx, ly;
        //VectorXd x;
        int all_frame_count = all_image_frame.size();
        int n_state = all_frame_count * 3 + 2 + 1;

        MatrixXd A{n_state, n_state};
        A.setZero();
        VectorXd b{n_state};
        b.setZero();

        map<double, ImageFrame>::iterator frame_i;
        map<double, ImageFrame>::iterator frame_j;
        for (int k = 0; k < 4; k++) {
            MatrixXd lxly(3, 2);
            lxly = TangentBasis(g0);
            int i = 0;
            for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
                frame_j = next(frame_i);

                MatrixXd tmp_A(6, 9);
                tmp_A.setZero();
                VectorXd tmp_b(6);
                tmp_b.setZero();

                double dt = frame_j->second.pre_integration->sum_dt;


                tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
                tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
                tmp_A.block<3, 1>(0, 8) =
                        frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
                tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p +
                                          frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] -
                                          frame_i->second.R.transpose() * dt * dt / 2 * g0;

                tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
                tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
                tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
                tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v -
                                          frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


                Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
                //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
                //MatrixXd cov_inv = cov.inverse();
                cov_inv.setIdentity();

                MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
                VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

                A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
                b.segment<6>(i * 3) += r_b.head<6>();

                A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
                b.tail<3>() += r_b.tail<3>();

                A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
                A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
            }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
        }
        g = g0;
    }

    bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x) {
        int all_frame_count = all_image_frame.size();
        int n_state = all_frame_count * 3 + 3 + 1;

        MatrixXd A{n_state, n_state};
        A.setZero();
        VectorXd b{n_state};
        b.setZero();

        map<double, ImageFrame>::iterator frame_i;
        map<double, ImageFrame>::iterator frame_j;
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 10);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
            tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p +
                                      frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
            //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
            //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
            b.tail<4>() += r_b.tail<4>();

            A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
            A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        double s = x(n_state - 1) / 100.0;
//        ROS_DEBUG("estimated scale: %f", s);
        g = x.segment<3>(n_state - 4);
//        ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
        if (fabs(g.norm() - G.norm()) > 0.5 || s < 0) {
            return false;
        }

        RefineGravity(all_image_frame, g, x);
        s = (x.tail<1>())(0) / 100.0;
        (x.tail<1>())(0) = s;
//        ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
        if (s < 0.0)
            return false;
        else
            return true;
    }

    //bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x) {
    //    solveGyroscopeBias(all_image_frame, Bgs);
    //
    //    if (LinearAlignment(all_image_frame, g, x))
    //        return true;
    //    else
    //        return false;
    //}

}
