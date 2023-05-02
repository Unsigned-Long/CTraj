//
// Created by csl on 4/7/23.
//
#include "ctraj/core/trajectory.h"
#include "ctraj/core/simu_trajectory.h"

int main() {
    using namespace ns_ctraj;
    double sTime = 0.0, eTime = 2.0 * M_PI;
    // auto trajItoW = ns_ctraj::SimuCircularMotion<4>(2.0, sTime, eTime);
    // auto trajItoW = ns_ctraj::SimuSpiralMotion<4>(2.0, 2.0);
    // auto trajItoW = ns_ctraj::SimuEightShapeMotion<4>(5.0, 4.0, 0.5);
    auto trajItoW = ns_ctraj::SimuWaveMotion2<4>(2.0, 0.5, 0.0, 2 * M_PI, 1000.0);
    // auto trajItoW = ns_ctraj::SimuUniformLinearMotion<4>({0.0, 0.0, 0.0}, {5.0, 5.0, 5.0});
    // auto trajItoW = ns_ctraj::SimuUniformAcceleratedMotion<4>({0.0, 0.0, 0.0}, {5.0, 5.0, 5.0});
    // auto trajItoW = ns_ctraj::SimuDrunkardMotion<4>({0.0, 0.0, 0.0}, 0.5, 60.0);
    // auto traj2 = trajItoW * Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1, 1, 1));
    // auto traj3 = Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(1, 1, 1)) * trajItoW;

    trajItoW.VisualizationDynamic();

    trajItoW.GetTrajectory()->Save("/home/csl/CppWorks/artwork/ctraj/output/simu_wave_motion.json");
    return 0;
}