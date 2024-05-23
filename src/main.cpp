//
// Created by csl on 4/7/23.
//
#include "ctraj/core/trajectory.h"
#include "ctraj/core/simu_trajectory.h"
#include "ctraj/core/spline_bundle.h"
#include "ctraj/nofree/marg_test.hpp"
#include "ctraj/core/preintegration.h"

void TEST_SPLINE_BUNDLE() {
    using namespace ns_ctraj;
    auto bundle = SplineBundle<4>::Create({
                                                  SplineInfo("so3", SplineType::So3Spline, 0.0, 10.0, 0.55),
                                                  SplineInfo("vel", SplineType::RdSpline, 0.0, 10.0, 0.12),
                                                  SplineInfo("ba", SplineType::RdSpline, 0.0, 10.0, 1.2),
                                                  SplineInfo("bg", SplineType::RdSpline, 0.0, 10.0, 1.5),
                                          });
    std::cout << *bundle << std::endl;
    bundle->Save("/home/csl/CppWorks/artwork/ctraj/output/bundle.json");
    {
        std::cout << *SplineBundle<4>::Load("/home/csl/CppWorks/artwork/ctraj/output/bundle.json") << std::endl;
    }
}

void TEST_TRAJECTORY() {
    using namespace ns_ctraj;
    // double sTime = 0.0, eTime = 2.0 * M_PI;
    // auto trajItoW = ns_ctraj::SimuCircularMotion<4>(2.0, sTime, eTime);
    // auto trajItoW = ns_ctraj::SimuSpiralMotion<4>(2.0, 2.0);
    // auto trajItoW = ns_ctraj::SimuEightShapeMotion<4>(5.0, 4.0, 0.5);
    auto trajItoW = ns_ctraj::SimuWaveMotion2<4>(2.0, 0.5, 0.0, 2 * M_PI, 1000.0);
    // auto trajItoW = ns_ctraj::SimuUniformLinearMotion<4>({0.0, 0.0, 0.0}, {5.0, 5.0, 5.0});
    // auto trajItoW = ns_ctraj::SimuUniformAcceleratedMotion<4>({0.0, 0.0, 0.0}, {5.0, 5.0, 5.0});
    // auto trajItoW = ns_ctraj::SimuDrunkardMotion<4>({0.0, 0.0, 0.0}, 0.5, 60.0);
    // auto traj2 = trajItoW * Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1, 1, 1));
    // auto traj3 = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1, 1, 1)) * trajItoW;

    trajItoW.Visualization("/home/csl/CppWorks/artwork/ctraj/img", true);
    trajItoW.VisualizationDynamic("/home/csl/CppWorks/artwork/ctraj/img");

    trajItoW.GetTrajectory()->Save("/home/csl/CppWorks/artwork/ctraj/output/simu_wave_motion.json");
    trajItoW.GetTrajectory()->SamplingWithSaving("/home/csl/CppWorks/artwork/ctraj/output/pose_seq.json");

    trajItoW.GetTrajectory()->ComputeIMUMeasurement({0.0, 0.0, -9.8}, 1.0 / 400.0);
    IMUFrame::SaveFramesToDisk(
            "/home/csl/CppWorks/artwork/ctraj/output/measurements.json",
            trajItoW.GetTrajectory()->ComputeIMUMeasurement({0.0, 0.0, -9.8}, 1.0 / 400.0), 5
    );
}

int main() {
    TEST_TRAJECTORY();
    // TEST_SPLINE_BUNDLE();
    // ns_ctraj::MargTest::OrganizePowellProblem();
    // ns_ctraj::MargTest::IncrementalSplineFitting();
    return 0;
}