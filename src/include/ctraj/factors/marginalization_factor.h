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
            explicit ParBlockInfo(double *address = nullptr, int globalSize = 0, int localSize = 0,
                                  const ceres::Manifold *manifold = nullptr, int index = 0);

            friend std::ostream &operator<<(std::ostream &os, const ParBlockInfo &info);

        public:
            template<class Archive>
            void save(Archive &ar) const {
                ar(
                        cereal::make_nvp("address", reinterpret_cast<long>(address)),
                        CEREAL_NVP(globalSize), CEREAL_NVP(localSize),
                        cereal::make_nvp("manifold", reinterpret_cast<long>(manifold)),
                        CEREAL_NVP(oldState), CEREAL_NVP(index)
                );
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
        explicit MarginalizationInfo(ceres::Problem *prob, const std::set<double *> &margParBlockAddVec,
                                     const std::vector<double *> &consideredParBlocks = {}, int numThreads = 1);

        static Ptr Create(ceres::Problem *prob, const std::set<double *> &margParBlockAddVec,
                          const std::vector<double *> &consideredParBlocks = {}, int numThreads = 1);

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

        void Save(const std::string &filename) const;

        void ShiftKeepParBlockAddress(const std::map<double *, double *> &oldToNew);

        void ShiftMargParBlockAddress(const std::map<double *, double *> &oldToNew);

    protected:
        static Eigen::MatrixXd CRSMatrix2EigenMatrix(ceres::CRSMatrix *jacobianCRSMat);

        void PreMarginalization(ceres::Problem *prob, const std::set<double *> &margParBlockAddVec,
                                const std::vector<double *> &consideredParBlocks, int numThreads);

        void SchurComplement();

    public:
        template<class Archive>
        void save(Archive &ar) const {
            ar(
                    CEREAL_NVP(margParBlocks), CEREAL_NVP(keepParBlocks), CEREAL_NVP(margParDime),
                    CEREAL_NVP(keepParDime), CEREAL_NVP(JMat), CEREAL_NVP(rVec), CEREAL_NVP(HMat), CEREAL_NVP(bVec),
                    CEREAL_NVP(HMatSchur), CEREAL_NVP(bVecSchur), CEREAL_NVP(linJMat), CEREAL_NVP(linRVec)
            );
        }
    };

    struct MarginalizationFactor {
    private:
        MarginalizationInfo::Ptr margInfo;
        double weight;

    public:
        explicit MarginalizationFactor(MarginalizationInfo::Ptr margInfo, double weight);

        static ceres::DynamicNumericDiffCostFunction<MarginalizationFactor> *
        AddToProblem(ceres::Problem *prob, const MarginalizationInfo::Ptr &margInfo, double weight);

        static std::size_t TypeHashCode();

    public:

        bool operator()(double const *const *parBlocks, double *residuals) const;
    };
}

#endif //CTRAJ_MARGINALIZATION_FACTOR_H
