// Copyright (c) 2023. Created on 7/7/23 1:20 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_UTILS_HPP
#define CTRAJ_UTILS_HPP

#include "eigen_utils.hpp"
#include "ctraj/core/pose.hpp"
#include "filesystem"

namespace ns_ctraj {
    template<typename ScaleType>
    static Eigen::Matrix<ScaleType, 3, 1> XYZtoRTP(const Eigen::Matrix<ScaleType, 3, 1> &xyz) {
        const ScaleType radian = xyz.norm();
        const ScaleType theta = std::atan2(xyz(1), xyz(0));
        const ScaleType phi = std::asin(xyz(2) / radian);
        return {radian, theta, phi};
    }

    template<typename ScaleType>
    static Eigen::Matrix<ScaleType, 3, 1> RTPtoXYZ(const Eigen::Matrix<ScaleType, 3, 1> &rtp) {
        const ScaleType cTheta = std::cos(rtp(1)), sTheta = std::sin(rtp(1));
        const ScaleType cPhi = std::cos(rtp(2)), sPhi = std::sin(rtp(2));
        return Eigen::Matrix<ScaleType, 3, 1>(cTheta * cPhi, sTheta * cPhi, sPhi) * rtp(0);
    }

    inline bool SavePoseSequence(const Eigen::aligned_vector<Posed> &poseSeq, const std::string &filename) {
        if (auto parPath = std::filesystem::path(filename).parent_path();
                !exists(parPath) && !std::filesystem::create_directories(parPath)) {
            return false;
        }
        std::ofstream file(filename);
        cereal::JSONOutputArchive ar(file);
        ar(cereal::make_nvp("pose_seq", poseSeq));
        return true;
    }
}

#endif //CTRAJ_UTILS_HPP
