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

#ifndef CTRAJ_IMU_ACCE_FACTOR_HPP
#define CTRAJ_IMU_ACCE_FACTOR_HPP

#include "ctraj/utils/eigen_utils.hpp"
#include "ctraj/core/imu.h"

namespace ns_ctraj {
template <int Order>
struct IMUAcceFactor {
private:
    ns_ctraj::SplineMeta<Order> _splineMeta;
    IMUFrame::Ptr _imuFrame{};

    double _dtInv;
    double _acceWeight;

public:
    explicit IMUAcceFactor(ns_ctraj::SplineMeta<Order> splineMeta,
                           IMUFrame::Ptr imuFrame,
                           double acceWeight)
        : _splineMeta(std::move(splineMeta)),
          _imuFrame(std::move(imuFrame)),
          _dtInv(1.0 / _splineMeta.segments.front().dt),
          _acceWeight(acceWeight) {}

    static auto Create(const ns_ctraj::SplineMeta<Order> &splineMeta,
                       const IMUFrame::Ptr &imuFrame,
                       double acceWeight) {
        return new ceres::DynamicAutoDiffCostFunction<IMUAcceFactor>(
            new IMUAcceFactor(splineMeta, imuFrame, acceWeight));
    }

    static std::size_t TypeHashCode() { return typeid(IMUAcceFactor).hash_code(); }

public:
    /**
     * param blocks:
     * [ SO3 | ... | SO3 | VEL | ... | VEL | ACCE_BIAS | ACCE_MAP_COEFF | GRAVITY ]
     */
    template <class T>
    bool operator()(T const *const *sKnots, T *sResiduals) const {
        std::size_t SO3_OFFSET;
        std::size_t POS_OFFSET;
        double u;
        _splineMeta.ComputeSplineIndex(_imuFrame->GetTimestamp(), SO3_OFFSET, u);
        POS_OFFSET = SO3_OFFSET + _splineMeta.NumParameters();

        std::size_t ACCE_BIAS_OFFSET = 2 * _splineMeta.NumParameters();
        std::size_t ACCE_MAP_COEFF_OFFSET = ACCE_BIAS_OFFSET + 1;
        std::size_t GRAVITY_OFFSET = ACCE_MAP_COEFF_OFFSET + 1;

        Sophus::SO3<T> so3;
        Sophus::SO3Tangent<T> so3Vel;
        ns_ctraj::CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(
            sKnots + SO3_OFFSET, u, _dtInv, &so3, &so3Vel);

        /**
         * @attention: current R^3 trajectory is the velocity b-spline, whose
         * first order derivative is the linear acceleration, not the second order derivative!!!
         */
        Eigen::Vector3<T> posAccel;
        ns_ctraj::CeresSplineHelper<Order>::template Evaluate<T, 3, 1>(sKnots + POS_OFFSET, u,
                                                                       _dtInv, &posAccel);

        Eigen::Map<const Eigen::Vector3<T>> acceBias(sKnots[ACCE_BIAS_OFFSET]);
        Eigen::Map<const Eigen::Vector3<T>> gravity(sKnots[GRAVITY_OFFSET]);

        auto acceCoeff = sKnots[ACCE_MAP_COEFF_OFFSET];

        Eigen::Matrix33<T> acceMapMat = Eigen::Matrix33<T>::Zero();

        acceMapMat.diagonal() = Eigen::Map<const Eigen::Vector3<T>>(acceCoeff, 3);
        acceMapMat(0, 1) = *(acceCoeff + 3);
        acceMapMat(0, 2) = *(acceCoeff + 4);
        acceMapMat(1, 2) = *(acceCoeff + 5);

        Eigen::Vector3<T> accePred =
            (acceMapMat * (so3.inverse() * (posAccel - gravity))).eval() + acceBias;

        Eigen::Vector3<T> acceResiduals = accePred - _imuFrame->GetAcce().template cast<T>();

        Eigen::Map<Eigen::Vector3<T>> residuals(sResiduals);
        residuals = T(_acceWeight) * acceResiduals;

        return true;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ns_ctraj

#endif  // CTRAJ_IMU_ACCE_FACTOR_HPP
