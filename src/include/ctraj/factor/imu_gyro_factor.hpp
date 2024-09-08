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

#ifndef CTRAJ_IMU_GYRO_FACTOR_HPP
#define CTRAJ_IMU_GYRO_FACTOR_HPP

#include "ctraj/utils/eigen_utils.hpp"
#include "ctraj/utils/sophus_utils.hpp"
#include "ctraj/core/imu.h"

namespace ns_ctraj {
template <int Order>
struct IMUGyroFactor {
private:
    ns_ctraj::SplineMeta<Order> _splineMeta;
    IMUFrame::Ptr _imuFrame{};

    double _dtInv;
    double _gyroWeight;

public:
    explicit IMUGyroFactor(ns_ctraj::SplineMeta<Order> splineMeta,
                           IMUFrame::Ptr imuFrame,
                           double gyroWeight)
        : _splineMeta(std::move(splineMeta)),
          _imuFrame(std::move(imuFrame)),
          _dtInv(1.0 / _splineMeta.segments.front().dt),
          _gyroWeight(gyroWeight) {}

    static auto Create(const ns_ctraj::SplineMeta<Order> &splineMeta,
                       const IMUFrame::Ptr &imuFrame,
                       double gyroWeight) {
        return new ceres::DynamicAutoDiffCostFunction<IMUGyroFactor>(
            new IMUGyroFactor(splineMeta, imuFrame, gyroWeight));
    }

    static std::size_t TypeHashCode() { return typeid(IMUGyroFactor).hash_code(); }

public:
    /**
     * param blocks:
     * [ SO3 | ... | SO3 | GYRO_BIAS | GYRO_MAP_COEFF | SO3_AtoG ]
     */
    template <class T>
    bool operator()(T const *const *sKnots, T *sResiduals) const {
        // array offset
        std::size_t SO3_OFFSET;
        double u;
        _splineMeta.template ComputeSplineIndex(_imuFrame->GetTimestamp(), SO3_OFFSET, u);

        std::size_t GYRO_BIAS_OFFSET = _splineMeta.NumParameters();
        std::size_t GYRO_MAP_COEFF_OFFSET = GYRO_BIAS_OFFSET + 1;
        std::size_t SO3_AtoG_OFFSET = GYRO_MAP_COEFF_OFFSET + 1;

        Sophus::SO3Tangent<T> gyroVel;
        ns_ctraj::CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(
            sKnots + SO3_OFFSET, u, _dtInv, nullptr, &gyroVel);

        Eigen::Map<const Eigen::Vector3<T>> gyroBias(sKnots[GYRO_BIAS_OFFSET]);
        auto gyroCoeff = sKnots[GYRO_MAP_COEFF_OFFSET];
        Eigen::Matrix33<T> gyroMapMat = Eigen::Matrix33<T>::Zero();
        gyroMapMat.diagonal() = Eigen::Map<const Eigen::Vector3<T>>(gyroCoeff, 3);
        gyroMapMat(0, 1) = *(gyroCoeff + 3);
        gyroMapMat(0, 2) = *(gyroCoeff + 4);
        gyroMapMat(1, 2) = *(gyroCoeff + 5);

        Eigen::Map<Sophus::SO3<T> const> const SO3_AtoG(sKnots[SO3_AtoG_OFFSET]);

        Eigen::Map<Eigen::Vector3<T>> residuals(sResiduals);
        residuals = (gyroMapMat * (SO3_AtoG * gyroVel)).eval() + gyroBias -
                    _imuFrame->GetGyro().template cast<T>();
        residuals = T(_gyroWeight) * residuals;

        return true;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ns_ctraj

#endif  // CTRAJ_IMU_GYRO_FACTOR_HPP
