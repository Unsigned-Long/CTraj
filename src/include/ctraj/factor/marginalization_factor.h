// CTraj: Continuous-Time Trajectory (Time-Varying State) Representation and Estimation Library
// Copyright 2024, the School of Geodesy and Geomatics (SGG), Wuhan University, China
// https://github.com/Unsigned-Long/CTraj.git
//
// Author: Shuolong Chen (shlchen@whu.edu.cn)
// GitHub: https://github.com/Unsigned-Long
//  ORCID: 0000-0002-5283-9057
//
// Purpose: See .h/.hpp file.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * The names of its contributors can not be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef CTRAJ_MARGINALIZATION_FACTOR_H
#define CTRAJ_MARGINALIZATION_FACTOR_H

#include <utility>
#include <ostream>
#include "ctraj/utils/eigen_utils.hpp"
#include "ceres/ceres.h"
#include "cereal/types/vector.hpp"
#include "tiny-viewer/entity/utils.h"

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
        explicit ParBlockInfo(double *address = nullptr,
                              int globalSize = 0,
                              int localSize = 0,
                              const ceres::Manifold *manifold = nullptr,
                              int index = 0);

        friend std::ostream &operator<<(std::ostream &os, const ParBlockInfo &info);

    public:
        template <class Archive>
        void save(Archive &ar) const {
            ar(cereal::make_nvp("address", reinterpret_cast<long>(address)), CEREAL_NVP(globalSize),
               CEREAL_NVP(localSize),
               cereal::make_nvp("manifold", reinterpret_cast<long>(manifold)), CEREAL_NVP(oldState),
               CEREAL_NVP(index));
        }
    };

private:
    constexpr static double EPS = 1E-8;

    // address, global size, local size
    std::vector<ParBlockInfo> margParBlocks;
    std::vector<ParBlockInfo> keepParBlocks;

    int margParDime;
    int keepParDime;

    Eigen::MatrixXd JMat;
    Eigen::VectorXd rVec;

    Eigen::MatrixXd HMat;
    Eigen::VectorXd bVec;

    Eigen::MatrixXd HMatSchur;
    Eigen::VectorXd bVecSchur;

    Eigen::MatrixXd linJMat;
    Eigen::VectorXd linRVec;

public:
    explicit MarginalizationInfo(ceres::Problem *prob,
                                 const std::set<double *> &margParBlockAddVec,
                                 const std::vector<double *> &consideredParBlocks = {},
                                 int numThreads = 1);

    static Ptr Create(ceres::Problem *prob,
                      const std::set<double *> &margParBlockAddVec,
                      const std::vector<double *> &consideredParBlocks = {},
                      int numThreads = 1);

    [[nodiscard]] inline const std::vector<MarginalizationInfo::ParBlockInfo> &GetKeepParBlocks()
        const {
        return keepParBlocks;
    }

    [[nodiscard]] inline int GetMargParDime() const { return margParDime; }

    [[nodiscard]] inline int GetKeepParDime() const { return keepParDime; }

    [[nodiscard]] inline const std::vector<MarginalizationInfo::ParBlockInfo> &GetMargParBlocks()
        const {
        return margParBlocks;
    }

    [[nodiscard]] inline const Eigen::MatrixXd &GetLinJMat() const { return linJMat; }

    [[nodiscard]] inline const Eigen::VectorXd &GetLinRVec() const { return linRVec; }

    void Save(const std::string &filename) const;

    void ShiftKeepParBlockAddress(const std::map<double *, double *> &oldToNew);

    void ShiftMargParBlockAddress(const std::map<double *, double *> &oldToNew);

protected:
    static Eigen::MatrixXd CRSMatrix2EigenMatrix(ceres::CRSMatrix *jacobianCRSMat);

    void PreMarginalization(ceres::Problem *prob,
                            const std::set<double *> &margParBlockAddVec,
                            const std::vector<double *> &consideredParBlocks,
                            int numThreads);

    void SchurComplement();

public:
    template <class Archive>
    void save(Archive &ar) const {
        ar(CEREAL_NVP(margParBlocks), CEREAL_NVP(keepParBlocks), CEREAL_NVP(margParDime),
           CEREAL_NVP(keepParDime), CEREAL_NVP(JMat), CEREAL_NVP(rVec), CEREAL_NVP(HMat),
           CEREAL_NVP(bVec), CEREAL_NVP(HMatSchur), CEREAL_NVP(bVecSchur), CEREAL_NVP(linJMat),
           CEREAL_NVP(linRVec));
    }
};

struct MarginalizationFactor {
private:
    MarginalizationInfo::Ptr margInfo;
    double weight;

public:
    explicit MarginalizationFactor(MarginalizationInfo::Ptr margInfo, double weight);

    static ceres::DynamicNumericDiffCostFunction<MarginalizationFactor> *AddToProblem(
        ceres::Problem *prob, const MarginalizationInfo::Ptr &margInfo, double weight);

    static std::size_t TypeHashCode();

public:
    bool operator()(double const *const *parBlocks, double *residuals) const;
};
}  // namespace ns_ctraj

#endif  // CTRAJ_MARGINALIZATION_FACTOR_H
