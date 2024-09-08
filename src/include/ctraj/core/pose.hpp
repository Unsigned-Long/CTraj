// CTraj: Continuous-Time Trajectory (Time-Varying State) Representation and Estimation Library
// Copyright 2024, the School of Geodesy and Geomatics (SGG), Wuhan University, China
// https://github.com/Unsigned-Long/CTraj.git
//
// Author: Shuolong Chen (shlchen@whu.edu.cn)
// GitHub: https://github.com/Unsigned-Long
//  ORCID: 0000-0002-5283-9057
//
// Purpose: See .h/.hpp file.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * The names of its contributors can not be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef CTRAJ_ODOMETER_POSE_H
#define CTRAJ_ODOMETER_POSE_H

#include "Eigen/Dense"
#include "sophus/se3.hpp"
#include "ctraj/utils/macros.hpp"
#include "tiny-viewer/entity/utils.h"
#include "tiny-viewer/core/pose.hpp"

namespace ns_ctraj {
template <class ScalarType>
inline Sophus::Matrix3<ScalarType> AdjustRotationMatrix(const Sophus::Matrix3<ScalarType> &rotMat) {
    // our implement
    // Eigen::JacobiSVD<Sophus::Matrix3<ScalarType>> svd(rotMat,
    //                                                   Eigen::ComputeFullV | Eigen::ComputeFullU);
    // const Sophus::Matrix3<ScalarType> &vMatrix = svd.matrixV();
    // const Sophus::Matrix3<ScalarType> &uMatrix = svd.matrixU();
    // Sophus::Matrix3<ScalarType> adjustedRotMat = uMatrix * vMatrix.transpose();
    // implement of sophus
    Sophus::Matrix3<ScalarType> adjustedRotMat = Sophus::makeRotationMatrix(rotMat);
    return adjustedRotMat;
}

template <class ScalarType>
struct OdomPose {
    double timeStamp;
    Eigen::Matrix<ScalarType, 4, 4> pose;

    explicit OdomPose(
        double timeStamp = INVALID_TIME_STAMP,
        const Eigen::Matrix<ScalarType, 4, 4> &pose = Eigen::Matrix<ScalarType, 4, 4>::Identity())
        : timeStamp(timeStamp),
          pose(pose) {}

public:
    template <class Archive>
    void serialize(Archive &ar) {
        ar(CEREAL_NVP(timeStamp), CEREAL_NVP(pose));
    }
};

using OdomPosed = OdomPose<double>;
using OdomPosef = OdomPose<float>;

template <class ScalarType>
struct Pose {
public:
    using Scale = ScalarType;
    using Rotation = Sophus::SO3<Scale>;
    using Translation = Sophus::Vector3<Scale>;

    Rotation so3;
    Translation t;
    double timeStamp;

    Pose(const Rotation &so3, const Translation &t, double timeStamp = INVALID_TIME_STAMP)
        : so3(so3),
          t(t),
          timeStamp(timeStamp) {}

    explicit Pose(double timeStamp = INVALID_TIME_STAMP)
        : so3(),
          t(Translation::Zero()),
          timeStamp(timeStamp) {}

    [[nodiscard]] Eigen::Quaternion<ScalarType> q() const { return so3.unit_quaternion(); }

    [[nodiscard]] Sophus::Matrix3<ScalarType> R() const { return q().toRotationMatrix(); }

    [[nodiscard]] Sophus::SE3<ScalarType> se3() const { return Sophus::SE3<ScalarType>(so3, t); }

    [[nodiscard]] Sophus::Matrix4<ScalarType> T() const {
        Sophus::Matrix4<ScalarType> T = Sophus::Matrix4<ScalarType>::Identity();
        T.template block<3, 3>(0, 0) = R();
        T.template block<3, 1>(0, 3) = t;
        return T;
    }

    static Pose FromT(const Sophus::Matrix4<ScalarType> &T, double timeStamp = INVALID_TIME_STAMP) {
        Sophus::Matrix3<ScalarType> rotMat = T.template block<3, 3>(0, 0);
        rotMat = AdjustRotationMatrix(rotMat);

        Pose pose(timeStamp);
        pose.so3 = Rotation(rotMat);
        pose.t = T.template block<3, 1>(0, 3);
        return pose;
    }

    static Pose FromRt(const Sophus::Matrix3<ScalarType> &R,
                       const Sophus::Vector3<ScalarType> &t,
                       double timeStamp = INVALID_TIME_STAMP) {
        Sophus::Matrix3<ScalarType> rotMat = AdjustRotationMatrix(R);

        Pose pose(timeStamp);
        pose.so3 = Rotation(rotMat);
        pose.t = t;
        return pose;
    }

    static Pose FromSE3(const Sophus::SE3<ScalarType> &se3, double timeStamp = INVALID_TIME_STAMP) {
        Pose pose(timeStamp);
        pose.so3 = se3.so3();
        pose.t = se3.translation();
        return pose;
    }

    static Pose FromTinyViewerPose(const ns_viewer::Pose<ScalarType> &tvPose,
                                   double timeStamp = INVALID_TIME_STAMP) {
        Pose pose(timeStamp);
        pose.so3 = Sophus::SO3<ScalarType>(AdjustRotationMatrix(tvPose.rotation));
        pose.t = tvPose.translation;
        return pose;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    template <class Archive>
    void serialize(Archive &ar) {
        ar(CEREAL_NVP(timeStamp), CEREAL_NVP(so3), CEREAL_NVP(t));
    }
};

using Posed = Pose<double>;
using Posef = Pose<float>;

extern template struct Pose<double>;
extern template struct Pose<float>;
}  // namespace ns_ctraj

#endif  // CTRAJ_ODOMETER_POSE_H
