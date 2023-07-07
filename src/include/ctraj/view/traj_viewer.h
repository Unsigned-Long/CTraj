// Copyright (c) 2023. Created on 7/7/23 1:20 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

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

        using parent_type::parent_type;

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
