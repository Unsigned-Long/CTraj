// Copyright (c) 2023. Created on 10/24/23 5:16 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_MARG_TEST_HPP
#define CTRAJ_MARG_TEST_HPP

#include "ctraj/factor/marginalization_factor.h"

namespace ns_ctraj {
    struct F1 {
        template<typename T>
        bool operator()(const T *const x1, const T *const x2, T *residual) const {
            // f1 = x1 + 10 * x2;
            residual[0] = x1[0] + 10.0 * x2[0];
            return true;
        }
    };

    struct F2 {
        template<typename T>
        bool operator()(const T *const x3, const T *const x4, T *residual) const {
            // f2 = sqrt(5) (x3 - x4)
            residual[0] = sqrt(5.0) * (x3[0] - x4[0]);
            return true;
        }
    };

    struct F3 {
        template<typename T>
        bool operator()(const T *const x2, const T *const x3, T *residual) const {
            // f3 = (x2 - 2 x3)^2
            residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
            return true;
        }
    };

    struct F4 {
        template<typename T>
        bool operator()(const T *const x1, const T *const x4, T *residual) const {
            // f4 = sqrt(10) (x1 - x4)^2
            residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
            return true;
        }
    };

    struct MargTest {

        static void OrganizePowellProblem() {
            double x1 = 3.0;
            double x2 = -1.0;
            double x3 = 0.0;
            double x4 = 1.0;

            MarginalizationInfo::Ptr marg;

            ceres::Problem::Options opt;
            opt.manifold_ownership = ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
            {

                auto problem = std::make_shared<ceres::Problem>(opt);
                // Add residual terms to the problem using the autodiff
                // wrapper to get the derivatives automatically. The parameters, x1 through
                // x4, are modified in place.
                problem->AddResidualBlock(new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1), nullptr, &x1, &x2);
                problem->AddResidualBlock(new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2), nullptr, &x3, &x4);
                problem->AddResidualBlock(new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3), nullptr, &x2, &x3);
                problem->AddResidualBlock(new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4), nullptr, &x1, &x4);

                // perform solving, for example.
                marg = MarginalizationInfo::Create(problem.get(), {&x1, &x4});
                // marg = MarginalizationInfo::Create(problem.get(), {});
            }
            // testing
            auto problem = std::make_shared<ceres::Problem>(opt);
            MarginalizationFactor::AddToProblem(problem.get(), marg, 1.0);

            ceres::Solver::Options options;
            options.max_num_iterations = 100;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;
            // clang-format off
            std::cout << "Initial x1 = " << x1
                      << ", x2 = " << x2
                      << ", x3 = " << x3
                      << ", x4 = " << x4
                      << "\n";
            // clang-format on
            // Run the solver!
            ceres::Solver::Summary summary;
            ceres::Solve(options, problem.get(), &summary);
            // clang-format off
            std::cout << "Final x1 = " << x1
                      << ", x2 = " << x2
                      << ", x3 = " << x3
                      << ", x4 = " << x4
                      << "\n";
        }

        static void IncrementalSplineFitting() {
            constexpr double sTime = 0.0;
            constexpr double eTime = 15.0;

            // generate data sequence
            Eigen::aligned_vector<ns_ctraj::Posed> poseSeq;
            for (double t = sTime; t < eTime;) {
                ns_ctraj::Posed pose;
                if (t > 6.0 && t < 9.0) {
                    pose = ns_ctraj::Posed(Sophus::SO3d(), Eigen::Vector3d{t, -1.5 * t * t + 23.5 * t - 81, 0}, t);
                } else {
                    pose = ns_ctraj::Posed(Sophus::SO3d(), Eigen::Vector3d{t, t, 0}, t);
                }
                poseSeq.push_back(pose);
                t += 0.2;
            }
            // save to disk
            ns_ctraj::SavePoseSequence(poseSeq, "/home/csl/CppWorks/artwork/ctraj/output/marg/samples.json");

            {
                // --------------------------
                // Case 1: batch optimization
                // --------------------------
                auto traj = ns_ctraj::Trajectory<3>::Create(1.0, sTime, eTime);
                ns_ctraj::TrajectoryEstimator<3> estimator(traj);

                for (const auto &pose: poseSeq) {
                    estimator.AddPOSMeasurement(pose, ns_ctraj::OptimizationOption::OPT_POS, 1.0);
                    estimator.AddSO3Measurement(pose, ns_ctraj::OptimizationOption::OPT_SO3, 1.0);
                }

                estimator.Solve();
                traj->SamplingWithSaving("/home/csl/CppWorks/artwork/ctraj/output/marg/data1/curve.json", 0.01);
                traj->Save("/home/csl/CppWorks/artwork/ctraj/output/marg/data1/knots.json");
            }
            {
                // ----------------------------
                // Case 1.1: batch optimization
                // ----------------------------
                auto traj = ns_ctraj::Trajectory<3>::Create(1.0, sTime, 10.0);
                ns_ctraj::TrajectoryEstimator<3> estimator(traj);

                for (const auto &pose: poseSeq) {
                    estimator.AddPOSMeasurement(pose, ns_ctraj::OptimizationOption::OPT_POS, 1.0);
                    estimator.AddSO3Measurement(pose, ns_ctraj::OptimizationOption::OPT_SO3, 1.0);
                }

                std::set<double *> margParBlockAddVec;
                for (int i = 0; i < 6; ++i) {
                    margParBlockAddVec.insert(traj->GetKnotPos(i).data());
                    margParBlockAddVec.insert(traj->GetKnotSO3(i).data());
                }

                MarginalizationInfo::Create(&estimator, margParBlockAddVec)->Save(
                        "/home/csl/CppWorks/artwork/ctraj/output/marg/marg.json"
                );
            }
            {
                // -------------------------------------------
                // Case 2: batch optimization (sub data piece)
                // -------------------------------------------
                auto traj = ns_ctraj::Trajectory<3>::Create(1.0, sTime, 6.0);
                ns_ctraj::TrajectoryEstimator<3> estimator(traj);

                for (const auto &pose: poseSeq) {
                    estimator.AddPOSMeasurement(pose, ns_ctraj::OptimizationOption::OPT_POS, 1.0);
                    estimator.AddSO3Measurement(pose, ns_ctraj::OptimizationOption::OPT_SO3, 1.0);
                }

                estimator.Solve();
                traj->SamplingWithSaving("/home/csl/CppWorks/artwork/ctraj/output/marg/data2/curve.json", 0.01);
                traj->Save("/home/csl/CppWorks/artwork/ctraj/output/marg/data2/knots.json");
            }
            {
                // -------------------------------------------------
                // Case 3: incremental optimization (two data piece)
                // -------------------------------------------------
                auto traj = ns_ctraj::Trajectory<3>::Create(1.0, sTime, 6.0);
                {
                    ns_ctraj::TrajectoryEstimator<3> estimator(traj);
                    for (const auto &pose: poseSeq) {
                        estimator.AddPOSMeasurement(pose, ns_ctraj::OptimizationOption::OPT_POS, 1.0);
                        estimator.AddSO3Measurement(pose, ns_ctraj::OptimizationOption::OPT_SO3, 1.0);
                    }
                    estimator.Solve();
                }
                traj->ExtendKnotsTo(eTime, {});
                {
                    ns_ctraj::TrajectoryEstimator<3> estimator(traj);
                    for (const auto &pose: poseSeq) {
                        if (pose.timeStamp < 6.0) { continue; }
                        estimator.AddPOSMeasurement(pose, ns_ctraj::OptimizationOption::OPT_POS, 1.0);
                        estimator.AddSO3Measurement(pose, ns_ctraj::OptimizationOption::OPT_SO3, 1.0);
                    }
                    estimator.Solve();
                }

                traj->SamplingWithSaving("/home/csl/CppWorks/artwork/ctraj/output/marg/data3/curve.json", 0.01);
                traj->Save("/home/csl/CppWorks/artwork/ctraj/output/marg/data3/knots.json");
            }
            {
                // ----------------------------------------------------------------------
                // Case 3: incremental optimization with marginalization (two data piece)
                // ----------------------------------------------------------------------
                auto traj = ns_ctraj::Trajectory<3>::Create(1.0, sTime, 6.0);
                MarginalizationInfo::Ptr marg;
                {
                    ns_ctraj::TrajectoryEstimator<3> estimator(traj);
                    for (const auto &pose: poseSeq) {
                        estimator.AddPOSMeasurement(pose, ns_ctraj::OptimizationOption::OPT_POS, 1.0);
                        estimator.AddSO3Measurement(pose, ns_ctraj::OptimizationOption::OPT_SO3, 1.0);
                    }
                    estimator.Solve();
                    std::set<double *> margParBlockAddVec;
                    for (int i = 0; i < 6; ++i) {
                        margParBlockAddVec.insert(traj->GetKnotPos(i).data());
                        margParBlockAddVec.insert(traj->GetKnotSO3(i).data());
                    }
                    marg = MarginalizationInfo::Create(&estimator, margParBlockAddVec);
                    // marg = MarginalizationInfo::Create(&estimator, {});
                }
                traj->ExtendKnotsTo(eTime, {});
                {
                    ns_ctraj::TrajectoryEstimator<3> estimator(traj);
                    for (const auto &pose: poseSeq) {
                        if (pose.timeStamp < 6.0) { continue; }
                        estimator.AddPOSMeasurement(pose, ns_ctraj::OptimizationOption::OPT_POS, 1.0);
                        estimator.AddSO3Measurement(pose, ns_ctraj::OptimizationOption::OPT_SO3, 1.0);
                    }
                    MarginalizationFactor::AddToProblem(&estimator, marg, 1.0);
                    estimator.Solve();
                }

                traj->SamplingWithSaving("/home/csl/CppWorks/artwork/ctraj/output/marg/data4/curve.json", 0.01);
                traj->Save("/home/csl/CppWorks/artwork/ctraj/output/marg/data4/knots.json");
            }

            // ns_ctraj::Viewer viewer;
            // traj->Visualization(viewer);
            // viewer.RunInSingleThread();
        }
    };
}


#endif //CTRAJ_MARG_TEST_HPP
