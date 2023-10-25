// Copyright (c) 2023. Created on 10/24/23 4:52 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_MARGINALIZATION_FACTOR_H
#define CTRAJ_MARGINALIZATION_FACTOR_H

#include "ctraj/utils/eigen_utils.hpp"
#include "ceres/ceres.h"

namespace ns_ctraj {
    class MarginalizationInfo {
    public:
        using Ptr = std::shared_ptr<MarginalizationInfo>;
        using ProblemPtr = std::shared_ptr<ceres::Problem>;

    private:
        ProblemPtr prob;

        std::vector<std::pair<double *, int>> margParBlocks;
        std::vector<std::pair<double *, int>> keepParBlocks;

        int margParDime;
        int keepParDime;

        Eigen::MatrixXd JMat;
        Eigen::VectorXd rVec;

        Eigen::MatrixXd HMat;
        Eigen::VectorXd bVec;

        const double EPS = 1E-8;

        Eigen::MatrixXd linJMat;
        Eigen::VectorXd linRVec;

    public:

        explicit MarginalizationInfo(ProblemPtr prob, const std::set<double *> &margParBlockAddVec);

    protected:
        static Eigen::MatrixXd CRSMatrix2EigenMatrix(ceres::CRSMatrix *jacobianCRSMat);

        void PreMarginalization(const std::set<double *> &margParBlockAddVec);

        void SchurComplement();
    };
}

#endif //CTRAJ_MARGINALIZATION_FACTOR_H
