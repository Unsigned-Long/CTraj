//
// Created by csl on 10/3/22.
//

#ifndef LIC_CALIB_FUNCTOR_TYPEDEF_HPP
#define LIC_CALIB_FUNCTOR_TYPEDEF_HPP

#include "ctraj/spline/ceres_spline_helper.h"
#include "ctraj/spline/ceres_spline_helper_jet.h"
#include "ctraj/spline/spline_segment.h"
#include "sophus/se3.hpp"
#include "ctraj/core/pose.hpp"
#include "ceres/ceres.h"

namespace ns_ctraj {

    template<typename T>
    using SO3Tangent = typename Sophus::SO3<T>::Tangent;

    template<typename T>
    using Vector2 = Sophus::Vector2<T>;

    template<typename T>
    using Vector3 = Sophus::Vector3<T>;

    template<typename T>
    using Vector6 = Sophus::Vector6<T>;

    template<typename T>
    using Vector9 = Sophus::Vector<T, 9>;

    template<typename T>
    using Matrix1 = Sophus::Matrix<T, 1, 1>;

    template<typename T>
    using Matrix2 = Sophus::Matrix2<T>;

    template<typename T>
    using Matrix3 = Sophus::Matrix3<T>;
}

#endif //LIC_CALIB_FUNCTOR_TYPEDEF_HPP
