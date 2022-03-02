#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

template<typename T>
inline Eigen::Matrix<T,4,4> interpolateSE3(const Eigen::Matrix<T,4,4> & source, const Eigen::Matrix<T,4,4> & target, const T alpha){

//    if(alpha<0 || alpha>1)
//    {
//        std::cerr << "warning: alpha < 0 or alpha > 1" <<std::endl;
//    }
    Eigen::Matrix<T,4,4> source_ = source;
    Eigen::Matrix<T,4,4> target_ = target;
    if(abs(source(0,0))>0.999 && abs(source(1,1))>0.999 && abs(source(0,1))<0.001 && abs(source(1,0))<0.001)
        source_ = Eigen::Matrix<T,4,4>::Identity();
    if(abs(target(0,0))>0.999 && abs(target(1,1))>0.999 && abs(target(0,1))<0.001 && abs(target(1,0))<0.001)
        target_ = Eigen::Matrix<T,4,4>::Identity();

    Sophus::SE3<T> SE1(source_);
    Sophus::SE3<T> SE2(target_);

    Eigen::Matrix<T, 6, 1> se1 = SE1.log();
    Eigen::Matrix<T, 6, 1> se2 = SE2.log();

    Sophus::SE3<T> SE3t = SE1 * Sophus::SE3<T>::exp(alpha * (se2-se1));

    return SE3t.matrix();
}
