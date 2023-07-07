// Copyright (c) 2023. Created on 7/7/23 1:20 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_IMU_ACCE_FACTOR_HPP
#define CTRAJ_IMU_ACCE_FACTOR_HPP

#include "ctraj/utils/eigen_utils.hpp"
#include "ctraj/core/imu.h"

namespace ns_ctraj {
    template<int Order>
    struct IMUAcceFactor {
    private:
        ns_ctraj::SplineMeta<Order> _splineMeta;
        IMUFrame::Ptr _imuFrame{};

        double _dtInv;
        double _acceWeight;
    public:
        explicit IMUAcceFactor(ns_ctraj::SplineMeta<Order> splineMeta, IMUFrame::Ptr imuFrame, double acceWeight)
                : _splineMeta(std::move(splineMeta)), _imuFrame(std::move(imuFrame)),
                  _dtInv(1.0 / _splineMeta.segments.front().dt), _acceWeight(acceWeight) {}

        static auto
        Create(const ns_ctraj::SplineMeta<Order> &splineMeta, const IMUFrame::Ptr &imuFrame, double acceWeight) {
            return new ceres::DynamicAutoDiffCostFunction<IMUAcceFactor>(
                    new IMUAcceFactor(splineMeta, imuFrame, acceWeight)
            );
        }

        static std::size_t TypeHashCode() {
            return typeid(IMUAcceFactor).hash_code();
        }

    public:
        /**
         * param blocks:
         * [ SO3 | ... | SO3 | VEL | ... | VEL | ACCE_BIAS | ACCE_MAP_COEFF | GRAVITY ]
         */
        template<class T>
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
                    sKnots + SO3_OFFSET, u, _dtInv, &so3, &so3Vel
            );

            /**
             * @attention: current R^3 trajectory is the velocity b-spline, whose
             * first order derivative is the linear acceleration, not the second order derivative!!!
             */
            Eigen::Vector3<T> posAccel;
            ns_ctraj::CeresSplineHelper<Order>::template Evaluate<T, 3, 1>(
                    sKnots + POS_OFFSET, u, _dtInv, &posAccel
            );

            Eigen::Map<const Eigen::Vector3<T>> acceBias(sKnots[ACCE_BIAS_OFFSET]);
            Eigen::Map<const Eigen::Vector3<T>> gravity(sKnots[GRAVITY_OFFSET]);

            auto acceCoeff = sKnots[ACCE_MAP_COEFF_OFFSET];

            Eigen::Matrix33<T> acceMapMat = Eigen::Matrix33<T>::Zero();

            acceMapMat.diagonal() = Eigen::Map<const Eigen::Vector3<T>>(acceCoeff, 3);
            acceMapMat(0, 1) = *(acceCoeff + 3);
            acceMapMat(0, 2) = *(acceCoeff + 4);
            acceMapMat(1, 2) = *(acceCoeff + 5);

            Eigen::Vector3<T> accePred = (acceMapMat * (so3.inverse() * (posAccel - gravity))).eval() + acceBias;

            Eigen::Vector3<T> acceResiduals = accePred - _imuFrame->GetAcce().template cast<T>();

            Eigen::Map<Eigen::Vector3<T>> residuals(sResiduals);
            residuals = T(_acceWeight) * acceResiduals;

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

#endif //CTRAJ_IMU_ACCE_FACTOR_HPP
