//
// Created by csl on 2/3/23.
//

#ifndef CTRAJ_CENTRALIZATION_FACTOR_HPP
#define CTRAJ_CENTRALIZATION_FACTOR_HPP

#include "ctraj/factors/functor_typedef.hpp"

namespace ns_ctraj {
    struct SO3CentralizationFunctor {
    private:
        std::size_t _blockSize;
        double _weight;

    public:
        explicit SO3CentralizationFunctor(std::size_t blockSize, double weight)
                : _blockSize(blockSize), _weight(weight) {}

        static auto Create(std::size_t blockSize, double weight) {
            return new ceres::DynamicAutoDiffCostFunction<SO3CentralizationFunctor>(
                    new SO3CentralizationFunctor(blockSize, weight)
            );
        }

        static std::size_t TypeHashCode() {
            return typeid(SO3CentralizationFunctor).hash_code();
        }

    public:
        /**
         * param blocks:
         * [ SO3_BiToBc | ... | SO3_BiToBc | ... ]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {

            Eigen::Map<Eigen::Vector3<T>> residuals(sResiduals);

            Eigen::Vector3<T> normSO3 = Eigen::Vector3<T>::Zero();
            for (int SO3_BiToBc_OFFSET = 0; SO3_BiToBc_OFFSET < _blockSize; ++SO3_BiToBc_OFFSET) {
                Eigen::Map<Sophus::SO3<T> const> const SO3_BiToBc(sKnots[SO3_BiToBc_OFFSET]);
                normSO3 += SO3_BiToBc.log();
            }

            residuals.template block<3, 1>(0, 0) = T(_weight) * normSO3;

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct POSCentralizationFunctor {
    private:
        std::size_t _blockSize;
        double _weight;

    public:
        explicit POSCentralizationFunctor(std::size_t blockSize, double weight)
                : _blockSize(blockSize), _weight(weight) {}

        static auto Create(std::size_t blockSize, double weight) {
            return new ceres::DynamicAutoDiffCostFunction<POSCentralizationFunctor>(
                    new POSCentralizationFunctor(blockSize, weight)
            );
        }

        static std::size_t TypeHashCode() {
            return typeid(POSCentralizationFunctor).hash_code();
        }

    public:
        /**
         * param blocks:
         * [ POS_BiInBc | ... | POS_BiInBc | ... ]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {

            Eigen::Map<Eigen::Vector3<T>> residuals(sResiduals);

            Eigen::Vector3<T> normPOS = Eigen::Vector3<T>::Zero();
            for (int POS_BiInBc_OFFSET = 0; POS_BiInBc_OFFSET < _blockSize; ++POS_BiInBc_OFFSET) {
                Eigen::Map<Eigen::Vector3<T> const> const POS_BiInBc(sKnots[POS_BiInBc_OFFSET]);
                normPOS += POS_BiInBc;
            }

            residuals.template block<3, 1>(0, 0) = T(_weight) * normPOS;

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct TimeOffsetCentralizationFunctor {
    private:
        std::size_t _blockSize;
        double _weight;

    public:
        explicit TimeOffsetCentralizationFunctor(std::size_t blockSize, double weight)
                : _blockSize(blockSize), _weight(weight) {}

        static auto Create(std::size_t blockSize, double weight) {
            return new ceres::DynamicAutoDiffCostFunction<TimeOffsetCentralizationFunctor>(
                    new TimeOffsetCentralizationFunctor(blockSize, weight)
            );
        }

        static std::size_t TypeHashCode() {
            return typeid(TimeOffsetCentralizationFunctor).hash_code();
        }

    public:
        /**
         * param blocks:
         * [ TIME_OFFSET_BiToBc | ... | TIME_OFFSET_BiToBc | ... ]
         */
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {

            Eigen::Map<Eigen::Vector1<T>>
                    residuals(sResiduals);

            Eigen::Vector1<T> normTimeOffset = Eigen::Vector1<T>::Zero();
            for (int TIME_OFFSET_BiToBc_OFFSET = 0;
                 TIME_OFFSET_BiToBc_OFFSET < _blockSize; ++TIME_OFFSET_BiToBc_OFFSET) {
                Eigen::Map<Eigen::Vector1<T> const> const TIME_OFFSET_BiToBc(sKnots[TIME_OFFSET_BiToBc_OFFSET]);
                normTimeOffset += TIME_OFFSET_BiToBc;
            }

            residuals.template block<1, 1>(0, 0) = T(_weight) * normTimeOffset;

            return true;
        }

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}
#endif //CTRAJ_CENTRALIZATION_FACTOR_HPP
