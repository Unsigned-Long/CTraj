//
// Created by csl on 10/4/22.
//

#ifndef CTRAJ_SO3_FACTOR_HPP
#define CTRAJ_SO3_FACTOR_HPP

#include "ctraj/utils/eigen_utils.hpp"

namespace ns_ctraj {
    template<int Order>
    struct SO3Functor {
    private:
        SplineMeta<Order> _splineMeta;
        Posed ItoG{};

        double _dtInv;
        double _so3Weight;

    public:
        explicit SO3Functor(SplineMeta<Order> splineMeta, Posed ItoG, double so3Weight)
                : _splineMeta(std::move(splineMeta)), ItoG(std::move(ItoG)),
                  _dtInv(1.0 / _splineMeta.segments.front().dt), _so3Weight(so3Weight) {}

        static auto Create(const SplineMeta<Order> &splineMeta, const Posed &ItoG, double so3Weight) {
            return new ceres::DynamicAutoDiffCostFunction<SO3Functor>(
                    new SO3Functor(splineMeta, ItoG, so3Weight)
            );
        }

        static std::size_t TypeHashCode() {
            return typeid(SO3Functor).hash_code();
        }

    public:
        /**
         * param blocks:
         * [ SO3 | ... | SO3]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {
            // array offset
            std::size_t SO3_OFFSET;
            double u;
            // generate the param: 'SO3_OFFSET', 'u'
            _splineMeta.template ComputeSplineIndex(ItoG.timeStamp, SO3_OFFSET, u);

            Sophus::SO3<T> predSO3ItoG;
            CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(
                    sKnots + SO3_OFFSET, u, _dtInv, &predSO3ItoG
            );

            Eigen::Map<Eigen::Vector3<T>> so3Residuals(sResiduals);

            so3Residuals = (predSO3ItoG * ItoG.so3.inverse().template cast<T>()).log();

            so3Residuals = T(_so3Weight) * so3Residuals;

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

}

#endif //CTRAJ_SO3_FACTOR_HPP
