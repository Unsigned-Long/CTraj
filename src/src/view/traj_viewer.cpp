// Copyright (c) 2023. Created on 7/7/23 1:19 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#include "ctraj/view/traj_viewer.h"


namespace ns_ctraj {


    void Viewer::ShowPoseSequence(const Eigen::aligned_vector<Posed> &posSeq, float size) {
        for (const auto &item: posSeq) {
            AddEntity(ns_viewer::Coordinate::Create(
                    ns_viewer::Posed(item.so3.matrix(), item.t).cast<float>(), size)
            );
        }
    }
}
