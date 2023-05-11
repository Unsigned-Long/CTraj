//
// Created by csl on 10/7/22.
//

#ifndef CTRAJ_TRAJ_VIEWER_H
#define CTRAJ_TRAJ_VIEWER_H

#include "tiny-viewer/core/viewer.h"
#include "thread"
#include "ctraj/utils/eigen_utils.hpp"
#include "ctraj/core/pose.hpp"

namespace ns_ctraj {

    struct Viewer : public ns_viewer::Viewer {
    public:
        using Ptr = std::shared_ptr<Viewer>;
        using parent_type = ns_viewer::Viewer;

        explicit Viewer(const std::string &saveDir = "", const std::string &winName = "") :
                parent_type(ns_viewer::ViewerConfigor(winName).WithScreenShotSaveDir(saveDir)) {
        }

        static auto Create(const std::string &saveDir = "", const std::string &winName = "") {
            return std::make_shared<Viewer>(saveDir, winName);
        }

    public:
        // for pose sequence
        void ShowPoseSequence(const Eigen::aligned_vector<Posed> &posSeq, float size = 0.3f);

    };

}

#endif //CTRAJ_TRAJ_VIEWER_H
