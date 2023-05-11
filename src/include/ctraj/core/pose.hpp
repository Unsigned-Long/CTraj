//
// Created by csl on 10/3/22.
//

#ifndef CTRAJ_ODOMETER_POSE_H
#define CTRAJ_ODOMETER_POSE_H

#include "Eigen/Dense"
#include "sophus/se3.hpp"
#include "ctraj/utils/macros.hpp"

namespace ns_ctraj {
    template<class ScalarType>
    inline Sophus::Matrix3<ScalarType> AdjustRotationMatrix(const Sophus::Matrix3<ScalarType> &rotMat) {
        // adjust
        Eigen::JacobiSVD<Sophus::Matrix3<ScalarType>> svd(rotMat, Eigen::ComputeFullV | Eigen::ComputeFullU);
        const Sophus::Matrix3<ScalarType> &vMatrix = svd.matrixV();
        const Sophus::Matrix3<ScalarType> &uMatrix = svd.matrixU();
        Sophus::Matrix3<ScalarType> adjustedRotMat = uMatrix * vMatrix.transpose();
        return adjustedRotMat;
    }

    template<class ScalarType>
    struct OdomPose {
        double timeStamp;
        Eigen::Matrix<ScalarType, 4, 4> pose;

        explicit OdomPose(double timeStamp = INVALID_TIME_STAMP,
                          const Eigen::Matrix<ScalarType, 4, 4> &pose = Eigen::Matrix<ScalarType, 4, 4>::Identity())
                : timeStamp(timeStamp), pose(pose) {
        }

    public:
        template<class Archive>
        void serialize(Archive &ar) {
            ar(CEREAL_NVP(timeStamp), CEREAL_NVP(pose));
        }
    };

    using OdomPosed = OdomPose<double>;
    using OdomPosef = OdomPose<float>;

    template<class ScalarType>
    struct Pose {
    public:

        using Scale = ScalarType;
        using Rotation = Sophus::SO3<Scale>;
        using Translation = Sophus::Vector3<Scale>;

        Rotation so3;
        Translation t;
        double timeStamp;

        Pose(const Rotation &so3, const Translation &t, double timeStamp = INVALID_TIME_STAMP)
                : so3(so3), t(t), timeStamp(timeStamp) {}

        explicit Pose(double timeStamp = INVALID_TIME_STAMP)
                : so3(), t(Translation::Zero()), timeStamp(timeStamp) {}

        Eigen::Quaternion<ScalarType> q() const {
            return so3.unit_quaternion();
        }

        Sophus::Matrix3<ScalarType> R() const {
            return q().toRotationMatrix();
        }

        Sophus::SE3<ScalarType> se3() const {
            return Sophus::SE3<ScalarType>(so3, t);
        }

        Sophus::Matrix4<ScalarType> T() const {
            Sophus::Matrix4<ScalarType> T = Sophus::Matrix4<ScalarType>::Identity();
            T.template block<3, 3>(0, 0) = R();
            T.template block<3, 1>(0, 3) = t;
            return T;
        }

        static Pose
        FromT(const Sophus::Matrix4<ScalarType> &T, double timeStamp = INVALID_TIME_STAMP) {
            Sophus::Matrix3<ScalarType> rotMat = T.template block<3, 3>(0, 0);
            rotMat = AdjustRotationMatrix(rotMat);

            Pose pose(timeStamp);
            pose.so3 = Rotation(rotMat);
            pose.t = T.template block<3, 1>(0, 3);
            return pose;
        }

        static Pose
        FromRt(const Sophus::Matrix3<ScalarType> &R,
               const Sophus::Vector3<ScalarType> &t,
               double timeStamp = INVALID_TIME_STAMP) {
            Sophus::Matrix3<ScalarType> rotMat = AdjustRotationMatrix(R);

            Pose pose(timeStamp);
            pose.so3 = Rotation(rotMat);
            pose.t = t;
            return pose;
        }

        static Pose
        FromSE3(const Sophus::SE3<ScalarType> &se3, double timeStamp = INVALID_TIME_STAMP) {
            Pose pose(timeStamp);
            pose.so3 = se3.so3();
            pose.t = se3.translation();
            return pose;
        }


    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        template<class Archive>
        void serialize(Archive &ar) {
            ar(CEREAL_NVP(timeStamp), CEREAL_NVP(so3), CEREAL_NVP(t));
        }
    };

    using Posed = Pose<double>;
    using Posef = Pose<float>;
}

#endif //CTRAJ_ODOMETER_POSE_H
