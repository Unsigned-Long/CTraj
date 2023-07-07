//
// Created by csl on 10/2/22.
//

#ifndef CTRAJ_MULTI_IMU_GYRO_FACTOR_HPP
#define CTRAJ_MULTI_IMU_GYRO_FACTOR_HPP

#include "ctraj/factors/functor_typedef.hpp"
#include "ctraj/core/imu.h"

namespace ns_ctraj {
    template<int Order>
    struct MultiIMUGyroFactor {
    private:
        ns_ctraj::SplineMeta<Order> _splineMeta;
        IMUFrame::Ptr _imuFrame{};

        double _dtInv;
        double _gyroWeight;

    public:
        explicit MultiIMUGyroFactor(ns_ctraj::SplineMeta<Order> splineMeta, IMUFrame::Ptr imuFrame, double gyroWeight)
                : _splineMeta(std::move(splineMeta)), _imuFrame(std::move(imuFrame)),
                  _dtInv(1.0 / _splineMeta.segments.front().dt), _gyroWeight(gyroWeight) {}

        static auto
        Create(const ns_ctraj::SplineMeta<Order> &splineMeta, const IMUFrame::Ptr &imuFrame, double gyroWeight) {
            return new ceres::DynamicAutoDiffCostFunction<MultiIMUGyroFactor>(
                    new MultiIMUGyroFactor(splineMeta, imuFrame, gyroWeight)
            );
        }

        static std::size_t TypeHashCode() {
            return typeid(MultiIMUGyroFactor).hash_code();
        }

    public:
        /**
         * param blocks:
         * [ SO3 | ... | SO3 | GYRO_BIAS | GYRO_MAP_COEFF | SO3_AtoG | SO3_BiToBc | TIME_OFFSET_BiToBc ]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {
            // array offset
            std::size_t SO3_OFFSET;
            std::size_t GYRO_BIAS_OFFSET = _splineMeta.NumParameters();
            std::size_t GYRO_MAP_COEFF_OFFSET = GYRO_BIAS_OFFSET + 1;
            std::size_t SO3_AtoG_OFFSET = GYRO_MAP_COEFF_OFFSET + 1;
            std::size_t SO3_BiToBc_OFFSET = SO3_AtoG_OFFSET + 1;
            std::size_t TIME_OFFSET_BiToBc_OFFSET = SO3_BiToBc_OFFSET + 1;

            T TIME_OFFSET_BiToBc = sKnots[TIME_OFFSET_BiToBc_OFFSET][0];


            auto timeByBc = _imuFrame->GetTimestamp() + TIME_OFFSET_BiToBc;

            // calculate the so3 and pos offset
            std::pair<std::size_t, T> pointIU;
            _splineMeta.template ComputeSplineIndex(timeByBc, pointIU.first, pointIU.second);
            SO3_OFFSET = pointIU.first;

            Sophus::SO3<T> SO3_BcToBc0;
            ns_ctraj::SO3Tangent<T> SO3_VEL_BcToBc0InBc;
            ns_ctraj::CeresSplineHelperJet<T, Order>::template EvaluateLie(
                    sKnots + SO3_OFFSET, pointIU.second, _dtInv, &SO3_BcToBc0, &SO3_VEL_BcToBc0InBc
            );

            Eigen::Map<const ns_ctraj::Vector3<T>> gyroBias(sKnots[GYRO_BIAS_OFFSET]);
            auto gyroCoeff = sKnots[GYRO_MAP_COEFF_OFFSET];
            ns_ctraj::Matrix3<T> gyroMapMat = ns_ctraj::Matrix3<T>::Zero();
            gyroMapMat.diagonal() = Eigen::Map<const ns_ctraj::Vector3<T>>(gyroCoeff, 3);
            gyroMapMat(0, 1) = *(gyroCoeff + 3);
            gyroMapMat(0, 2) = *(gyroCoeff + 4);
            gyroMapMat(1, 2) = *(gyroCoeff + 5);

            Eigen::Map<Sophus::SO3<T> const> const SO3_AtoG(sKnots[SO3_AtoG_OFFSET]);
            Eigen::Map<Sophus::SO3<T> const> const SO3_BiToBc(sKnots[SO3_BiToBc_OFFSET]);
            Sophus::SO3<T> SO3_BiToBc0 = SO3_BcToBc0 * SO3_BiToBc;
            ns_ctraj::SO3Tangent<T> SO3_VEL_BcToBc0InBc0 = SO3_BcToBc0 * SO3_VEL_BcToBc0InBc;

            ns_ctraj::Vector3<T> pred =
                    (gyroMapMat * (SO3_AtoG * SO3_BiToBc0.inverse() * SO3_VEL_BcToBc0InBc0)).eval() + gyroBias;

            Eigen::Map<ns_ctraj::Vector3<T>> residuals(sResiduals);
            residuals = pred - _imuFrame->GetGyro().template cast<T>();
            residuals = T(_gyroWeight) * residuals;

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

#endif //CTRAJ_MULTI_IMU_GYRO_FACTOR_HPP
