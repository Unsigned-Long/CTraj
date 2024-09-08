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

#ifndef CTRAJ_UTILS_HPP
#define CTRAJ_UTILS_HPP

#include "eigen_utils.hpp"
#include "ctraj/core/pose.hpp"
#include "filesystem"

namespace ns_ctraj {
template <typename ScaleType>
static Eigen::Matrix<ScaleType, 3, 1> XYZtoRTP(const Eigen::Matrix<ScaleType, 3, 1> &xyz) {
    const ScaleType radian = xyz.norm();
    const ScaleType theta = std::atan2(xyz(1), xyz(0));
    const ScaleType phi = std::asin(xyz(2) / radian);
    return {radian, theta, phi};
}

template <typename ScaleType>
static Eigen::Matrix<ScaleType, 3, 1> RTPtoXYZ(const Eigen::Matrix<ScaleType, 3, 1> &rtp) {
    const ScaleType cTheta = std::cos(rtp(1)), sTheta = std::sin(rtp(1));
    const ScaleType cPhi = std::cos(rtp(2)), sPhi = std::sin(rtp(2));
    return Eigen::Matrix<ScaleType, 3, 1>(cTheta * cPhi, sTheta * cPhi, sPhi) * rtp(0);
}

inline bool SavePoseSequence(const Eigen::aligned_vector<Posed> &poseSeq,
                             const std::string &filename) {
    if (auto parPath = std::filesystem::path(filename).parent_path();
        !exists(parPath) && !std::filesystem::create_directories(parPath)) {
        return false;
    }
    std::ofstream file(filename);
    cereal::JSONOutputArchive ar(file);
    ar(cereal::make_nvp("pose_seq", poseSeq));
    return true;
}
}  // namespace ns_ctraj

#endif  // CTRAJ_UTILS_HPP
