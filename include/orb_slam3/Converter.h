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


#ifndef CONVERTER_H
#define CONVERTER_H

#include<opencv2/core/core.hpp>

#include<Eigen/Dense>
#include"Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include"Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include <cmath>
#include <cassert>
#include <cstring>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

namespace ORB_SLAM3 {

    class Converter {
    public:
        static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

        static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);

        static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

        static cv::Mat toCvMat(const g2o::SE3Quat &SE3);

        static cv::Mat toCvMat(const g2o::Sim3 &Sim3);

        static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);

        static cv::Mat toCvMat(const Eigen::Matrix3d &m);

        static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);

        static cv::Mat toCvMat(const Eigen::MatrixXd &m);

        static cv::Mat toCvMat(const Eigen::Matrix3f &m);

        static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);

        static cv::Mat tocvSkewMatrix(const cv::Mat &v);

        static cv::Mat toCvMatInverse(const cv::Mat &Tcw);

        static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);

        static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);

        static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);

        static Eigen::Matrix<double, 4, 4> toMatrix4d(const cv::Mat &cvMat4);

        static std::vector<float> toQuaternion(const cv::Mat &M);

        static bool isRotationMatrix(const cv::Mat &R);

        static std::vector<float> toEuler(const cv::Mat &R);

        static void increT(const cv::Mat &T, float &theta, float &t);

        static std::vector<float> PrintSE(const cv::Mat &R, std::string s = "", bool verbose = true);

        static void Orthogonalize(cv::Mat &T);

        template<typename Derived>
        static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta) {
            typedef typename Derived::Scalar Scalar_t;

            Eigen::Quaternion<Scalar_t> dq;
            Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
            half_theta /= static_cast<Scalar_t>(2.0);
            dq.w() = static_cast<Scalar_t>(1.0);
            dq.x() = half_theta.x();
            dq.y() = half_theta.y();
            dq.z() = half_theta.z();
            return dq;
        }

        static Eigen::Matrix4d toEigen4dInverse(const Eigen::Matrix4d &Tcw) {
            Eigen::Matrix3d Rcw = Tcw.block(0, 0, 3, 3);
            Eigen::Vector3d tcw = Tcw.block(0, 3, 3, 1);
            Eigen::Matrix3d Rwc = Rcw.transpose();
            Eigen::Vector3d twc = -Rwc * tcw;

            Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();

            Twc.block(0, 0, 3, 3) = Rwc;
            Twc.block(0, 3, 3, 1) = twc;

            return Twc;
        }

    };

}// namespace ORB_SLAM

#endif // CONVERTER_H