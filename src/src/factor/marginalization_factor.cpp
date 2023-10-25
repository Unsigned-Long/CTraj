// Copyright (c) 2023. Created on 10/24/23 4:52 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#include <utility>
#include "ctraj/factors/marginalization_factor.h"

#define FORMAT_VECTOR
#define FORMAT_SET

#include "artwork/logger/logger.h"

namespace ns_ctraj {

    MarginalizationInfo::MarginalizationInfo(ceres::Problem *prob,
                                             const std::set<double *> &margParBlockAddVec)
            : margParDime(), keepParDime() {
        this->PreMarginalization(prob, margParBlockAddVec);
        this->SchurComplement();
    }

    MarginalizationInfo::Ptr MarginalizationInfo::Create(ceres::Problem *prob,
                                                         const std::set<double *> &margParBlockAddVec) {
        return std::make_shared<MarginalizationInfo>(prob, margParBlockAddVec);
    }

    Eigen::MatrixXd MarginalizationInfo::CRSMatrix2EigenMatrix(ceres::CRSMatrix *jacobianCRSMat) {
        Eigen::MatrixXd JMat(jacobianCRSMat->num_rows, jacobianCRSMat->num_cols);
        JMat.setZero();

        std::vector<int> jacobianCRSMatRows, jacobianCRSMatCols;
        std::vector<double> jacobianCRSMatVal;
        jacobianCRSMatRows = jacobianCRSMat->rows;
        jacobianCRSMatCols = jacobianCRSMat->cols;
        jacobianCRSMatVal = jacobianCRSMat->values;

        int curIdxInColsAndVals = 0;
        // rows is a num_rows + 1 sized array
        int rowSize = static_cast<int>(jacobianCRSMatRows.size()) - 1;
        // outer loop traverse rows, inner loop traverse cols and values
        for (int rowIdx = 0; rowIdx < rowSize; ++rowIdx) {
            while (curIdxInColsAndVals < jacobianCRSMatRows[rowIdx + 1]) {
                JMat(rowIdx, jacobianCRSMatCols[curIdxInColsAndVals]) = jacobianCRSMatVal[curIdxInColsAndVals];
                curIdxInColsAndVals++;
            }
        }
        return JMat;
    }

    void
    MarginalizationInfo::PreMarginalization(ceres::Problem *prob, const std::set<double *> &margParBlockAddVec) {
        // obtain the total parameter blocks
        std::vector<double *> totalParBlocksAdd;
        prob->GetParameterBlocks(&totalParBlocksAdd);

        // size <= margParBlockAddVec.size()
        this->margParBlocks.reserve(margParBlockAddVec.size());
        // size <= totalParBlocksAdd.size()
        this->keepParBlocks.reserve(totalParBlocksAdd.size());
        margParDime = 0, keepParDime = 0;
        // reorganize parameter blocks: [ marg | keep ]
        for (const auto &address: totalParBlocksAdd) {
            auto globalSize = prob->ParameterBlockSize(address);
            auto localSize = prob->ParameterBlockTangentSize(address);
            if (margParBlockAddVec.find(address) != margParBlockAddVec.cend()) {
                // this param block needs to be marg
                this->margParBlocks.emplace_back(address, globalSize, localSize, prob->GetManifold(address));
                margParDime += localSize;
            } else {
                // this param block needs to be kept
                this->keepParBlocks.emplace_back(address, globalSize, localSize, prob->GetManifold(address));
                keepParDime += localSize;
            }
        }
        // LOG_VAR(totalParBlocksAdd)
        // LOG_VAR(margParBlockAddVec)
        // LOG_VAR(margParBlocks)
        // LOG_VAR(keepParBlocks)
        // LOG_VAR(margParDime, keepParDime)

        // obtain JMat (jacobian matrix) and rVec (residuals vector)
        ceres::Problem::EvaluateOptions evalOpt;
        evalOpt.parameter_blocks.resize(totalParBlocksAdd.size());
        for (int i = 0; i < static_cast<int>(totalParBlocksAdd.size()); ++i) {
            if (i < static_cast<int>(margParBlocks.size())) {
                evalOpt.parameter_blocks.at(i) = std::get<0>(margParBlocks.at(i));
            } else {
                evalOpt.parameter_blocks.at(i) = std::get<0>(keepParBlocks.at(i - margParBlocks.size()));
            }
        }

        // LOG_VAR(evalOpt.parameter_blocks)

        ceres::CRSMatrix jacobianCRSMatrix;
        std::vector<double> residuals;
        prob->Evaluate(evalOpt, nullptr, &residuals, nullptr, &jacobianCRSMatrix);
        JMat = CRSMatrix2EigenMatrix(&jacobianCRSMatrix);
        rVec = Eigen::VectorXd(residuals.size());
        for (int i = 0; i < static_cast<int>(residuals.size()); ++i) { rVec(i) = residuals.at(i); }

        // LOG_VAR(JMat)
        // LOG_VAR(JMat.transpose() * JMat)
        // LOG_VAR(rVec)
        // LOG_VAR(-JMat.transpose() * rVec)

        HMat = JMat.transpose() * JMat;
        bVec = -JMat.transpose() * rVec;
    }

    void MarginalizationInfo::SchurComplement() {
        int m = margParDime, n = keepParDime;

        Eigen::MatrixXd Hmn = HMat.block(0, m, m, n);
        Eigen::MatrixXd Hnm = HMat.block(m, 0, n, m);
        Eigen::MatrixXd Hnn = HMat.block(m, m, n, n);

        Eigen::VectorXd bm = bVec.segment(0, m);
        Eigen::VectorXd bn = bVec.segment(m, n);

        Eigen::MatrixXd Hmm = 0.5 * (HMat.block(0, 0, m, m) + HMat.block(0, 0, m, m).transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Hmm);

        // LOG_VAR(Hmm)
        // LOG_VAR(Hmn)
        // LOG_VAR(Hnm)
        // LOG_VAR(Hnn)
        // LOG_VAR(bm)
        // LOG_VAR(bn)

        Eigen::MatrixXd HmmInv = saes.eigenvectors() * Eigen::VectorXd(
                (saes.eigenvalues().array() > EPS).select(saes.eigenvalues().array().inverse(), 0)
        ).asDiagonal() * saes.eigenvectors().transpose();

        // LOG_VAR(HmmInv)

        Eigen::MatrixXd AMatSchur = Hnn - Hnm * HmmInv * Hmn;
        Eigen::VectorXd bVecSchur = bn - Hnm * HmmInv * bm;

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(AMatSchur);
        Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > EPS).select(saes2.eigenvalues().array(), 0));
        Eigen::VectorXd SInv = Eigen::VectorXd(
                (saes2.eigenvalues().array() > EPS).select(saes2.eigenvalues().array().inverse(), 0)
        );

        Eigen::VectorXd SSqrt = S.cwiseSqrt();
        Eigen::VectorXd SInvSqrt = SInv.cwiseSqrt();

        // JMat
        linJMat = SSqrt.asDiagonal() * saes2.eigenvectors().transpose();
        // JMat.t().inv() * bVec
        linRVec = SInvSqrt.asDiagonal() * saes2.eigenvectors().transpose() * bVecSchur;

        // Eigen::MatrixXd temp = SInvSqrt.asDiagonal() * saes2.eigenvectors().transpose();
        // LOG_VAR(temp.inverse().transpose())
        // LOG_VAR(linJMat)
    }

    const std::vector<std::tuple<double *, int, int, const ceres::Manifold *>> &
    MarginalizationInfo::GetKeepParBlocks() const {
        return keepParBlocks;
    }

    int MarginalizationInfo::GetMargParDime() const {
        return margParDime;
    }

    int MarginalizationInfo::GetKeepParDime() const {
        return keepParDime;
    }

    const std::vector<std::tuple<double *, int, int, const ceres::Manifold *>> &
    MarginalizationInfo::GetMargParBlocks() const {
        return margParBlocks;
    }
}