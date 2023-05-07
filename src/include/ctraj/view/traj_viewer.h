//
// Created by csl on 10/7/22.
//

#ifndef CTRAJ_TRAJ_VIEWER_H
#define CTRAJ_TRAJ_VIEWER_H

#ifdef USE_SLAM_SCENE_VIEWER

#include "slam-scene-viewer/scene_viewer.h"
#include "thread"
#include "ctraj/utils/eigen_utils.hpp"
#include "ctraj/core/pose.hpp"

namespace ns_ctraj {

    struct PoseSeqDisplay {
        enum class Mode {
            ARROW, COORD
        };

        const Eigen::aligned_vector<Posed> &poseSeq;
        Mode mode;
        ns_viewer::Colour colour;

        PoseSeqDisplay(const Eigen::aligned_vector<Posed> &poseSeq, Mode mode,
                       const ns_viewer::Colour &colour = ns_viewer::Colour::Red())
                : poseSeq(poseSeq), mode(mode), colour(colour) {}
    };

    struct Viewer : public ns_viewer::SceneViewer {
    public:
        using Ptr = std::shared_ptr<Viewer>;
        using parent_type = ns_viewer::SceneViewer;

        explicit Viewer(const std::string &saveDir = "", const std::string &winName = "")
                : ns_viewer::SceneViewer(saveDir, winName) {
            // for better visualization
            COLOUR_WHEEL = ns_viewer::ColourWheel();
        }

        static auto Create(const std::string &saveDir = "", const std::string &winName = "") {
            return std::make_shared<Viewer>(saveDir, winName);
        }

    public:
        // for pose sequence
        void ShowPoseSequence(const Eigen::aligned_vector<PoseSeqDisplay> &seq, float size = 0.3f);

    };

}
#endif

#endif //CTRAJ_TRAJ_VIEWER_H
