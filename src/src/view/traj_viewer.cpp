//
// Created by csl on 10/7/22.
//

#include "ctraj/view/traj_viewer.h"

#ifdef USE_SLAM_SCENE_VIEWER

namespace ns_ctraj {

    void Viewer::ShowPoseSequence(const Eigen::aligned_vector <PoseSeqDisplay> &seq, float size) {

        for (int j = 0; j < seq.size(); ++j) {
            const auto &s = seq.at(j);
            const auto &poseSeq = s.poseSeq;

            const auto &c = s.colour;

            if (s.mode == PoseSeqDisplay::Mode::ARROW) {
                for (int i = 0; i < poseSeq.size(); ++i) {
                    const auto &pose = poseSeq.at(i);

                    Eigen::Vector3d dirVec = Eigen::AngleAxisd(pose.so3.unit_quaternion()).axis();
                    auto trans = pose.t;

                    // add arrow
                    pcl::PointXYZ from(trans(0), trans(1), trans(2));
                    pcl::PointXYZ to(trans(0) + dirVec(0), trans(1) + dirVec(1), trans(2) + dirVec(2));

                    _viewer->addArrow(
                            to, from, c.r, c.g, c.b, false, "Arrow-" + std::to_string(j) + '-' + std::to_string(i)
                    );
                }
            } else if (s.mode == PoseSeqDisplay::Mode::COORD) {
                for (const auto &pose: poseSeq) {
                    AddPose(ns_viewer::Posed(pose.R(), pose.t).cast<float>(), size * 0.2f);
                }
            } else {
                throw std::runtime_error("Pose sequence display mode is unknown.");
            }
        }

    }

}
#endif
