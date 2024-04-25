// Copyright (c) 2023. Created on 7/7/23 1:20 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_TRAJECTORY_ESTIMATOR_H
#define CTRAJ_TRAJECTORY_ESTIMATOR_H

#include <utility>
#include "thread"

#include "ctraj/core/imu.h"
#include "ctraj/core/pose.hpp"
#include "ctraj/core/trajectory.h"

#include "ceres/manifold.h"
#include "ceres/ceres.h"

#include "ctraj/factor/se3_factor.hpp"
#include "ctraj/factor/so3_factor.hpp"
#include "ctraj/factor/pos_factor.hpp"
#include "ctraj/factor/multi_imu_acce_factor.hpp"
#include "ctraj/factor/multi_imu_gyro_factor.hpp"
#include "ctraj/factor/imu_acce_factor.hpp"
#include "ctraj/factor/imu_gyro_factor.hpp"
#include "ctraj/factor/centralization_factor.hpp"

namespace ns_ctraj {

    struct OptimizationOption {
    public:
        enum Option : std::uint32_t {
            /**
             * @brief options
             */
            NONE = 1 << 0,
            OPT_SO3 = 1 << 1,
            OPT_POS = 1 << 2,
            ALL = OPT_SO3 | OPT_POS
        };

        static bool IsOptionWith(std::uint32_t desired, std::uint32_t curOption) {
            return (desired == (desired & curOption));
        }

        /**
         * @brief override operator '<<' for type 'Option'
         */
        friend std::ostream &operator<<(std::ostream &os, const Option &curOption) {
            std::stringstream stream;
            int count = 0;
            if (IsOptionWith(OPT_SO3, curOption)) {
                stream << "OPT_SO3";
                ++count;
            }
            if (IsOptionWith(OPT_POS, curOption)) {
                stream << " | OPT_POS";
                ++count;
            }
            if (count == 0) {
                os << "NONE";
            } else if (count == 2) {
                os << "ALL";
            } else {
                std::string str = stream.str();
                if (str.at(1) == '|') {
                    str = str.substr(3, str.size() - 3);
                }
                os << str;
            }
            return os;
        };
    };

    template<int Order>
    class TrajectoryEstimator : public ceres::Problem {
    public:
        using Ptr = std::shared_ptr<TrajectoryEstimator>;
        using Self = TrajectoryEstimator<Order>;
        using SelfPtr = Self::Ptr;

        using Traj = Trajectory<Order>;
        using TrajPtr = typename Traj::Ptr;

    protected:
        TrajPtr _trajectory;
        static std::shared_ptr<ceres::EigenQuaternionManifold> QUATER_MANIFOLD;
        static std::shared_ptr<ceres::SphereManifold<3>> S2_MANIFOLD;

    public:
        // using default problem options to create a 'TrajectoryEstimator'
        explicit TrajectoryEstimator(
                TrajPtr trajectory,
                const ceres::Problem::Options &options = TrajectoryEstimator::DefaultProblemOptions())
                : ceres::Problem(options), _trajectory(std::move(trajectory)) {}

        static ceres::Problem::Options DefaultProblemOptions() {

            // organize the default problem options
            ceres::Problem::Options defaultProblemOptions;
            defaultProblemOptions.loss_function_ownership = ceres::TAKE_OWNERSHIP;
            // we want to own the manifold ourselves, not the ceres::Problem itself
            defaultProblemOptions.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;

            return defaultProblemOptions;
        }

        static ceres::Solver::Options
        DefaultSolverOptions(int threadNum = -1, bool toStdout = true, bool useCUDA = false) {

            // organize the default solver option
            ceres::Solver::Options defaultSolverOptions;
            defaultSolverOptions.minimizer_type = ceres::TRUST_REGION;
            if (useCUDA) {
                defaultSolverOptions.linear_solver_type = ceres::DENSE_SCHUR;
                defaultSolverOptions.dense_linear_algebra_library_type = ceres::CUDA;
            } else {
                defaultSolverOptions.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            }
            defaultSolverOptions.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            defaultSolverOptions.minimizer_progress_to_stdout = toStdout;

            if (threadNum < 1) {
                defaultSolverOptions.num_threads = static_cast<int>(std::thread::hardware_concurrency());
            } else {
                defaultSolverOptions.num_threads = threadNum;
            }

            return defaultSolverOptions;
        }

        static TrajectoryEstimator::Ptr Create(
                const TrajPtr &trajectory,
                const ceres::Problem::Options &options = TrajectoryEstimator::DefaultProblemOptions()) {
            return std::make_shared<TrajectoryEstimator>(trajectory, options);
        }

        ceres::Solver::Summary
        Solve(const ceres::Solver::Options &options = TrajectoryEstimator::DefaultSolverOptions()) {
            ceres::Solver::Summary summary;
            ceres::Solve(options, this, &summary);
            return summary;
        }

        bool TimeInRange(double time) {
            // left closed right open interval
            return _trajectory->TimeStampInRange(time);
        }

    public:
        /**
         * different type optimize targets
         */

        void AddSE3Measurement(const Posed &ItoG, int option, double so3Weight, double posWeight) {
            // check time stamp
            if (!_trajectory->TimeStampInRange(ItoG.timeStamp)) {
                return;
            }

            // find the affected control points
            SplineMeta<Order> splineMeta;
            _trajectory->CalculateSplineMeta({{ItoG.timeStamp, ItoG.timeStamp}}, splineMeta);

            // create a cost function
            auto costFunc = SE3Functor<Order>::Create(splineMeta, ItoG, so3Weight, posWeight);

            // so3 knots param block [each has four sub params]
            for (int i = 0; i < static_cast<int>(splineMeta.NumParameters()); ++i) {
                costFunc->AddParameterBlock(4);
            }
            // pos knots param block [each has three sub params]
            for (int i = 0; i < static_cast<int>(splineMeta.NumParameters()); ++i) {
                costFunc->AddParameterBlock(3);
            }
            // set Residuals
            costFunc->SetNumResiduals(6);

            // organize the param block vector
            std::vector<double *> paramBlockVec;

            // so3 knots param block
            AddCtrlPointsData(
                    paramBlockVec, AddCtrlPointsDataFlag::SO3_KNOTS,
                    splineMeta, !OptimizationOption::IsOptionWith(OptimizationOption::OPT_SO3, option)
            );

            AddCtrlPointsData(
                    paramBlockVec, AddCtrlPointsDataFlag::POS_KNOTS,
                    splineMeta, !OptimizationOption::IsOptionWith(OptimizationOption::OPT_POS, option)
            );

            // pass to problem
            this->AddResidualBlockToProblem(costFunc, nullptr, paramBlockVec);
        }

        void AddSO3Measurement(const Posed &ItoG, int option, double so3Weight) {
            // check time stamp
            if (!_trajectory->TimeStampInRange(ItoG.timeStamp)) {
                return;
            }

            // find the affected control points
            SplineMeta<Order> splineMeta;
            _trajectory->CalculateSplineMeta({{ItoG.timeStamp, ItoG.timeStamp}}, splineMeta);

            // create a cost function
            auto costFunc = SO3Functor<Order>::Create(splineMeta, ItoG, so3Weight);

            // so3 knots param block [each has four sub params]
            for (int i = 0; i < static_cast<int>(splineMeta.NumParameters()); ++i) {
                costFunc->AddParameterBlock(4);
            }
            // set Residuals
            costFunc->SetNumResiduals(3);

            // organize the param block vector
            std::vector<double *> paramBlockVec;

            // so3 knots param block
            AddCtrlPointsData(
                    paramBlockVec, AddCtrlPointsDataFlag::SO3_KNOTS, splineMeta,
                    !OptimizationOption::IsOptionWith(OptimizationOption::OPT_SO3, option)
            );

            // pass to problem
            this->AddResidualBlockToProblem(costFunc, nullptr, paramBlockVec);
        }

        void AddPOSMeasurement(const Posed &ItoG, int option, double posWeight) {
            // check time stamp
            if (!_trajectory->TimeStampInRange(ItoG.timeStamp)) {
                return;
            }

            // find the affected control points
            SplineMeta<Order> splineMeta;
            _trajectory->CalculateSplineMeta({{ItoG.timeStamp, ItoG.timeStamp}}, splineMeta);

            // create a cost function
            auto costFunc = PO3Functor<Order>::Create(splineMeta, ItoG, posWeight);

            // pos knots param block [each has three sub params]
            for (int i = 0; i < static_cast<int>(splineMeta.NumParameters()); ++i) {
                costFunc->AddParameterBlock(3);
            }
            // set Residuals
            costFunc->SetNumResiduals(3);

            // organize the param block vector
            std::vector<double *> paramBlockVec;

            AddCtrlPointsData(
                    paramBlockVec, AddCtrlPointsDataFlag::POS_KNOTS,
                    splineMeta, !OptimizationOption::IsOptionWith(OptimizationOption::OPT_POS, option)
            );

            // pass to problem
            this->AddResidualBlockToProblem(costFunc, nullptr, paramBlockVec);
        }

    protected:
        enum class AddCtrlPointsDataFlag {
            SO3_KNOTS, POS_KNOTS
        };

        void AddCtrlPointsData(std::vector<double *> &paramBlockVec, AddCtrlPointsDataFlag flag,
                               const SplineMeta<Order> &splineMeta, bool setToConst
        ) {

            // for each segment
            for (const auto &seg: splineMeta.segments) {
                // the factor '+ seg.dt * 0.5' is the treatment of numerical accuracy
                auto idxMaster = _trajectory->ComputeTIndex(seg.t0 + seg.dt * 0.5).second;

                // from the first control point to the last control point
                for (std::size_t i = idxMaster; i < idxMaster + seg.NumParameters(); ++i) {
                    // switch
                    double *data;
                    switch (flag) {
                        case AddCtrlPointsDataFlag::SO3_KNOTS:
                            data = _trajectory->GetKnotSO3(i).data();
                            // the local parameterization is very, very important!!!
                            this->AddParameterBlock(data, 4, QUATER_MANIFOLD.get());
                            break;
                        case AddCtrlPointsDataFlag::POS_KNOTS:
                            data = _trajectory->GetKnotPos(i).data();
                            this->AddParameterBlock(data, 3);
                            break;
                        default:
                            throw std::runtime_error(
                                    "add unknown control points in the function 'TrajectoryEstimator::AddCtrlPointsData'"
                            );
                    }
                    paramBlockVec.push_back(data);
                    // set this param block to be constant
                    if (setToConst) {
                        this->SetParameterBlockConstant(data);
                    }
                }
            }
        }

        template<typename CostFunctor, int Stride = 4>
        ceres::ResidualBlockId
        AddResidualBlockToProblem(ceres::DynamicAutoDiffCostFunction<CostFunctor, Stride> *costFunc,
                                  ceres::LossFunction *lossFunc, const std::vector<double *> &paramBlocks) {
            return Problem::AddResidualBlock(costFunc, lossFunc, paramBlocks);
        }
    };

    template<int Order>
    std::shared_ptr<ceres::EigenQuaternionManifold> TrajectoryEstimator<Order>::QUATER_MANIFOLD
            = std::make_shared<ceres::EigenQuaternionManifold>();

    template<int Order>
    std::shared_ptr<ceres::SphereManifold<3>> TrajectoryEstimator<Order>::S2_MANIFOLD
            = std::make_shared<ceres::SphereManifold<3>>();
}

#endif //CTRAJ_TRAJECTORY_ESTIMATOR_H
