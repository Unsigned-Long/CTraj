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

#ifndef CTRAJ_TRAJ_VIEWER_H
#define CTRAJ_TRAJ_VIEWER_H

#include "ctraj/utils/eigen_utils.hpp"
#include "thread"
#include "tiny-viewer/core/viewer.h"

namespace ns_ctraj {
template <typename>
struct Pose;
using Posed = Pose<double>;

struct Viewer : public ns_viewer::Viewer {
public:
    using Ptr = std::shared_ptr<Viewer>;
    using parent_type = ns_viewer::Viewer;

    using parent_type::parent_type;

    explicit Viewer(const std::string &saveDir = "", const std::string &winName = "")
        : parent_type(ns_viewer::ViewerConfigor(winName).WithScreenShotSaveDir(saveDir)) {}

    static auto Create(const std::string &saveDir = "", const std::string &winName = "") {
        return std::make_shared<Viewer>(saveDir, winName);
    }

public:
    // for pose sequence
    void ShowPoseSequence(const Eigen::aligned_vector<Posed> &posSeq, float size = 0.3f);
};

}  // namespace ns_ctraj

#endif  // CTRAJ_TRAJ_VIEWER_H
