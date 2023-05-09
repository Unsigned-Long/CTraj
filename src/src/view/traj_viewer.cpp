//
// Created by csl on 10/7/22.
//

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
