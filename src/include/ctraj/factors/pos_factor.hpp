//
// Created by csl on 10/4/22.
//

#ifndef CTRAJ_POS_FACTOR_HPP
#define CTRAJ_POS_FACTOR_HPP

#include "ctraj/utils/eigen_utils.hpp"

namespace ns_ctraj {
    template<int Order>
    struct PO3Functor {
    private:
        SplineMeta<Order> _splineMeta;
        Posed ItoG{};

        double _dtInv;
        double _posWeight;

    public:
        explicit PO3Functor(SplineMeta<Order> splineMeta, Posed ItoG, double posWeight)
                : _splineMeta(std::move(splineMeta)), ItoG(std::move(ItoG)),
                  _dtInv(1.0 / _splineMeta.segments.front().dt), _posWeight(posWeight) {}

        static auto Create(const SplineMeta<Order> &splineMeta, const Posed &ItoG, double posWeight) {
            return new ceres::DynamicAutoDiffCostFunction<PO3Functor>(
                    new PO3Functor(splineMeta, ItoG, posWeight)
            );
        }


        static std::size_t TypeHashCode() {
            return typeid(PO3Functor).hash_code();
        }

    public:
        /**
         * param blocks:
         * [ POS | ... | POS ]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {
            // array offset
            std::size_t POS_OFFSET;
            double u;
            _splineMeta.template ComputeSplineIndex(ItoG.timeStamp, POS_OFFSET, u);

            Eigen::Vector3<T> predPOSTtoG;
            // if 'DERIV = 0' returns value of the spline, otherwise corresponding derivative.
            CeresSplineHelper<Order>::template Evaluate<T, 3, 0>(sKnots + POS_OFFSET, u, _dtInv, &predPOSTtoG);

            Eigen::Map<Eigen::Vector3<T>> poseResiduals(sResiduals + 0);

            poseResiduals = predPOSTtoG - ItoG.t.template cast<T>();

            poseResiduals = T(_posWeight) * poseResiduals;

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

#endif //CTRAJ_POS_FACTOR_HPP
