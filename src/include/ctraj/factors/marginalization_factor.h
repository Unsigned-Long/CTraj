// Copyright (c) 2023. Created on 10/24/23 4:52 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_MARGINALIZATION_FACTOR_H
#define CTRAJ_MARGINALIZATION_FACTOR_H

#include <utility>
#include <ostream>
#include "ctraj/utils/eigen_utils.hpp"
#include "ceres/ceres.h"

namespace ns_ctraj {
    class MarginalizationInfo {
    public:
        using Ptr = std::shared_ptr<MarginalizationInfo>;

        struct ParBlockInfo {
        public:

            double *address;
            int globalSize;
            int localSize;
            const ceres::Manifold *manifold;
            // this member is owned by this BlockInfo
            Eigen::VectorXd oldState;
            int index;

        public:
            explicit ParBlockInfo(double *address = nullptr, int globalSize = 0, int localSize = 0,
                                  const ceres::Manifold *manifold = nullptr, int index = 0);

            friend std::ostream &operator<<(std::ostream &os, const ParBlockInfo &info);
        };

    private:


        // address, global size, local size
        std::vector<ParBlockInfo> margParBlocks;
        std::vector<ParBlockInfo> keepParBlocks;

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
        explicit MarginalizationInfo(ceres::Problem *prob, const std::set<double *> &margParBlockAddVec);

        static Ptr Create(ceres::Problem *prob, const std::set<double *> &margParBlockAddVec);

        [[nodiscard]] inline const std::vector<MarginalizationInfo::ParBlockInfo> &GetKeepParBlocks() const {
            return keepParBlocks;
        }

        [[nodiscard]] inline int GetMargParDime() const { return margParDime; }

        [[nodiscard]] inline int GetKeepParDime() const { return keepParDime; }

        [[nodiscard]] inline const std::vector<MarginalizationInfo::ParBlockInfo> &GetMargParBlocks() const {
            return margParBlocks;
        }

        [[nodiscard]] inline const Eigen::MatrixXd &GetLinJMat() const { return linJMat; }

        [[nodiscard]] inline const Eigen::VectorXd &GetLinRVec() const { return linRVec; }

    protected:
        static Eigen::MatrixXd CRSMatrix2EigenMatrix(ceres::CRSMatrix *jacobianCRSMat);

        void PreMarginalization(ceres::Problem *prob, const std::set<double *> &margParBlockAddVec);

        void SchurComplement();
    };

    class MarginalizationFactor : public ceres::CostFunction {
    private:
        MarginalizationInfo::Ptr margInfo;
        double weight;

    protected:
        explicit MarginalizationFactor(MarginalizationInfo::Ptr margInfo, double weight);

    public:
        static MarginalizationFactor *
        AddToProblem(ceres::Problem *prob, const MarginalizationInfo::Ptr &margInfo, double weight);

        static std::size_t TypeHashCode();

    public:
        bool Evaluate(double const *const *parBlocks, double *residuals, double **jacobians) const override;
    };
}

#endif //CTRAJ_MARGINALIZATION_FACTOR_H
