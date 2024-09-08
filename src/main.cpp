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
        std::cout << *SplineBundle<4>::Load("/home/csl/CppWorks/artwork/ctraj/output/bundle.json")
                  << std::endl;
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
    trajItoW.GetTrajectory()->SamplingWithSaving(
        "/home/csl/CppWorks/artwork/ctraj/output/pose_seq.json");

    trajItoW.GetTrajectory()->ComputeIMUMeasurement({0.0, 0.0, -9.8}, 1.0 / 400.0);
    IMUFrame::SaveFramesToDisk(
        "/home/csl/CppWorks/artwork/ctraj/output/measurements.json",
        trajItoW.GetTrajectory()->ComputeIMUMeasurement({0.0, 0.0, -9.8}, 1.0 / 400.0), 5);
}

int main() {
    TEST_TRAJECTORY();
    // TEST_SPLINE_BUNDLE();
    // ns_ctraj::MargTest::OrganizePowellProblem();
    // ns_ctraj::MargTest::IncrementalSplineFitting();
    return 0;
}