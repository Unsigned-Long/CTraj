//
// Created by csl on 6/24/23.
//

#ifndef CTRAJ_UTILS_HPP
#define CTRAJ_UTILS_HPP

#include "eigen_utils.hpp"

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
}

#endif //CTRAJ_UTILS_HPP
