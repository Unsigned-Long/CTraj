// Copyright (c) 2023. Created on 7/7/23 1:20 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_SE3_FACTOR_HPP
#define CTRAJ_SE3_FACTOR_HPP

#include "ctraj/utils/eigen_utils.hpp"

namespace ns_ctraj {
    template<int Order>
    struct SE3Functor {
    private:
        SplineMeta<Order> _splineMeta;
        Posed ItoG{};

        double _dtInv;
        double _so3Weight;
        double _posWeight;

    public:
        explicit SE3Functor(SplineMeta<Order> splineMeta, Posed ItoG, double so3Weight, double posWeight)
                : _splineMeta(std::move(splineMeta)), ItoG(std::move(ItoG)),
                  _dtInv(1.0 / _splineMeta.segments.front().dt), _so3Weight(so3Weight), _posWeight(posWeight) {}

        static auto Create(const SplineMeta<Order> &splineMeta, const Posed &ItoG, double so3Weight, double posWeight) {
            return new ceres::DynamicAutoDiffCostFunction<SE3Functor>(
                    new SE3Functor(splineMeta, ItoG, so3Weight, posWeight)
            );
        }

        static std::size_t TypeHashCode() {
            return typeid(SE3Functor).hash_code();
        }

    public:
        /**
         * param blocks:
         * [ SO3 | ... | SO3 | POS | ... | POS ]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {
            // array offset
            std::size_t SO3_OFFSET;
            double u;
            // generate the param: 'SO3_OFFSET', 'u'
            _splineMeta.template ComputeSplineIndex(ItoG.timeStamp, SO3_OFFSET, u);
            std::size_t POS_OFFSET = SO3_OFFSET + _splineMeta.NumParameters();

            Sophus::SO3<T> predSO3ItoG;
            CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(
                    sKnots + SO3_OFFSET, u, _dtInv, &predSO3ItoG
            );
            Eigen::Vector3<T> predPOSTtoG;
            // if 'DERIV = 0' returns value of the spline, otherwise corresponding derivative.
            CeresSplineHelper<Order>::template Evaluate<T, 3, 0>(
                    sKnots + POS_OFFSET, u, _dtInv, &predPOSTtoG
            );

            Eigen::Map<Eigen::Vector3<T>> so3Residuals(sResiduals);
            Eigen::Map<Eigen::Vector3<T>> poseResiduals(sResiduals + 3);

            so3Residuals = (ItoG.so3.template cast<T>() * predSO3ItoG.inverse()).log();
            poseResiduals = predPOSTtoG - ItoG.t.template cast<T>();

            so3Residuals = T(_so3Weight) * so3Residuals;
            poseResiduals = T(_posWeight) * poseResiduals;

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}

#endif //CTRAJ_SE3_FACTOR_HPP
