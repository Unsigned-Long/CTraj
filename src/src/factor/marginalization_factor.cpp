// Copyright (c) 2023. Created on 10/24/23 4:52 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#include <utility>
#include "ctraj/factors/marginalization_factor.h"

namespace ns_ctraj {

    // -------------------
    // MarginalizationInfo
    // -------------------

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
                this->margParBlocks.emplace_back(
                        address, globalSize, localSize, prob->GetManifold(address), margParDime
                );
                margParDime += localSize;
            } else {
                // this param block needs to be kept
                this->keepParBlocks.emplace_back(
                        address, globalSize, localSize, prob->GetManifold(address), keepParDime
                );
                keepParDime += localSize;
            }
        }

        // obtain JMat (jacobian matrix) and rVec (residuals vector)
        ceres::Problem::EvaluateOptions evalOpt;
        evalOpt.parameter_blocks.resize(totalParBlocksAdd.size());
        for (int i = 0; i < static_cast<int>(totalParBlocksAdd.size()); ++i) {
            if (i < static_cast<int>(margParBlocks.size())) {
                evalOpt.parameter_blocks.at(i) = margParBlocks.at(i).address;
            } else {
                evalOpt.parameter_blocks.at(i) = keepParBlocks.at(i - margParBlocks.size()).address;
            }
        }

        ceres::CRSMatrix jacobianCRSMatrix;
        std::vector<double> residuals;
        prob->Evaluate(evalOpt, nullptr, &residuals, nullptr, &jacobianCRSMatrix);
        JMat = CRSMatrix2EigenMatrix(&jacobianCRSMatrix);
        rVec = Eigen::VectorXd(residuals.size());
        for (int i = 0; i < static_cast<int>(residuals.size()); ++i) { rVec(i) = residuals.at(i); }

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
    }

    // ---------------------
    // MarginalizationFactor
    // ---------------------

    MarginalizationFactor::MarginalizationFactor(MarginalizationInfo::Ptr margInfo, double weight)
            : margInfo(std::move(margInfo)), weight(weight) {}

    ceres::DynamicNumericDiffCostFunction<MarginalizationFactor> *
    MarginalizationFactor::AddToProblem(ceres::Problem *prob, const MarginalizationInfo::Ptr &margInfo, double weight) {
        auto func = new ceres::DynamicNumericDiffCostFunction<MarginalizationFactor>(
                new MarginalizationFactor(margInfo, weight)
        );

        std::vector<double *> keepParBlocksAdd;
        keepParBlocksAdd.reserve(margInfo->GetKeepParBlocks().size());

        // assign param blocks for cost function
        for (const auto &block: margInfo->GetKeepParBlocks()) {
            func->AddParameterBlock(block.globalSize);
            keepParBlocksAdd.push_back(block.address);
        }
        func->SetNumResiduals(margInfo->GetKeepParDime());

        // add to problem
        prob->AddResidualBlock(func, nullptr, keepParBlocksAdd);

        // set manifold
        for (const auto &block: margInfo->GetKeepParBlocks()) {
            // attention: problem should not own the manifolds!!!
            // i.e., Problem::Options::manifold_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP
            if (block.manifold != nullptr) {
                prob->SetManifold(block.address, const_cast<ceres::Manifold *>(block.manifold));
            }
        }

        return func;
    }

    bool MarginalizationFactor::operator()(const double *const *parBlocks, double *residuals) const {
        // delta x
        Eigen::VectorXd dx(this->margInfo->GetKeepParDime());

        // parameter blocks to keep
        const auto &keepParBlocks = margInfo->GetKeepParBlocks();

        // for each parameter block to optimized
        for (int i = 0; i < static_cast<int>(keepParBlocks.size()); ++i) {
            const auto &block = keepParBlocks.at(i);

            // obtain current state
            Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parBlocks[i], block.globalSize);

            if (block.manifold == nullptr) {
                // if no manifold employed, perform minus in vector space
                dx.segment(block.index, block.localSize) = x - block.oldState;
            } else {
                // otherwise, perform minus in manifold space

                // obtain delta x on manifold
                Eigen::VectorXd delta(block.localSize);
                block.manifold->Minus(x.data(), block.oldState.data(), delta.data());

                dx.segment(block.index, block.localSize) = delta;
            }
        }

        Eigen::Map<Eigen::VectorXd> residual(residuals, margInfo->GetKeepParDime());
        residual = weight * (margInfo->GetLinJMat() * dx - margInfo->GetLinRVec());

        return true;
    }

    std::size_t MarginalizationFactor::TypeHashCode() {
        return typeid(MarginalizationFactor).hash_code();
    }

    // -----------------------------------
    // MarginalizationFactor::ParBlockInfo
    // -----------------------------------

    MarginalizationInfo::ParBlockInfo::ParBlockInfo(double *address, int globalSize, int localSize,
                                                    const ceres::Manifold *manifold, int index)
            : address(address), globalSize(globalSize), localSize(localSize),
              manifold(manifold), oldState(globalSize), index(index) {
        for (int i = 0; i < globalSize; ++i) { oldState(i) = address[i]; }
    }

    std::ostream &operator<<(std::ostream &os, const MarginalizationInfo::ParBlockInfo &info) {
        os << "address: " << info.address << " globalSize: " << info.globalSize << " localSize: " << info.localSize
           << " oldState: " << info.oldState << " index: " << info.index;
        return os;
    }
}