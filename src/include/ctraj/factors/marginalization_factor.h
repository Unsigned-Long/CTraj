// Copyright (c) 2023. Created on 10/24/23 4:52 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_MARGINALIZATION_FACTOR_H
#define CTRAJ_MARGINALIZATION_FACTOR_H

#include <utility>

#include "ctraj/utils/eigen_utils.hpp"
#include "ceres/ceres.h"

namespace ns_ctraj {
    class MarginalizationInfo {
    public:
        using Ptr = std::shared_ptr<MarginalizationInfo>;
        using ProblemPtr = std::shared_ptr<ceres::Problem>;

    private:
        ProblemPtr prob;

        // address, global size, local size
        std::vector<std::tuple<double *, int, int, const ceres::Manifold *>> margParBlocks;
        std::vector<std::tuple<double *, int, int, const ceres::Manifold *>> keepParBlocks;

        int margParDime;
        int keepParDime;

        Eigen::MatrixXd JMat;
        Eigen::VectorXd rVec;

        Eigen::MatrixXd HMat;
        Eigen::VectorXd bVec;

        constexpr static double EPS = 1E-8;

        Eigen::MatrixXd linJMat;
        Eigen::VectorXd linRVec;

    public:
        explicit MarginalizationInfo(ProblemPtr prob, const std::set<double *> &margParBlockAddVec);

        static Ptr Create(const ProblemPtr &prob, const std::set<double *> &margParBlockAddVec);

        [[nodiscard]] const std::vector<std::tuple<double *, int, int, const ceres::Manifold *>> &
        GetKeepParBlocks() const;

        [[nodiscard]] const std::vector<std::tuple<double *, int, int, const ceres::Manifold *>> &
        GetMargParBlocks() const;

        [[nodiscard]] int GetMargParDime() const;

        [[nodiscard]] int GetKeepParDime() const;

    protected:
        static Eigen::MatrixXd CRSMatrix2EigenMatrix(ceres::CRSMatrix *jacobianCRSMat);

        void PreMarginalization(const std::set<double *> &margParBlockAddVec);

        void SchurComplement();
    };

    class MarginalizationFactor {
    private:
        MarginalizationInfo::Ptr margInfo;
        double weight;

    protected:
        explicit MarginalizationFactor(MarginalizationInfo::Ptr margInfo, double weight)
                : margInfo(std::move(margInfo)), weight(weight) {}

    public:
        static auto AddMarginalizationFactorToProblem(ceres::Problem *prob, const MarginalizationInfo::Ptr &margInfo,
                                                      double weight) {
            auto func = new ceres::DynamicAutoDiffCostFunction<MarginalizationFactor>(
                    new MarginalizationFactor(margInfo, weight)
            );

            std::vector<double *> keepParBlocksAdd;
            keepParBlocksAdd.reserve(margInfo->GetKeepParBlocks().size());

            // assign param blocks for cost function
            for (const auto &[address, globalSize, localSize, manifold]: margInfo->GetKeepParBlocks()) {
                func->AddParameterBlock(globalSize);
                keepParBlocksAdd.push_back(address);
            }

            // set residuals dime
            func->SetNumResiduals(margInfo->GetKeepParDime());

            // add to problem
            prob->AddResidualBlock(func, nullptr, keepParBlocksAdd);
            // set manifold
            for (const auto &[address, globalSize, localSize, manifold]: margInfo->GetKeepParBlocks()) {
                // attention: problem should not own the manifolds!!!
                // i.e., Problem::Options::manifold_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP
                if (manifold != nullptr) { prob->SetManifold(address, const_cast<ceres::Manifold *>(manifold)); }
            }
            return func;
        }

        static std::size_t TypeHashCode() {
            return typeid(MarginalizationFactor).hash_code();
        }

    public:
        template<class T>
        bool operator()(T const *const *sKnots, T *sResiduals) const {
            // todo: ...
            return true;
        };

    };
}

#endif //CTRAJ_MARGINALIZATION_FACTOR_H
