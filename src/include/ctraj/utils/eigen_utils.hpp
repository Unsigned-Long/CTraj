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

#ifndef CTRAJ_EIGEN_UTILS_HPP
#define CTRAJ_EIGEN_UTILS_HPP

#include <deque>
#include <map>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include "tiny-viewer/entity/utils.h"

namespace Eigen {

template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename K, typename V>
using aligned_map = std::map<K, V, std::less<K>, Eigen::aligned_allocator<std::pair<K const, V>>>;

template <typename K, typename V>
using aligned_unordered_map = std::unordered_map<K,
                                                 V,
                                                 std::hash<K>,
                                                 std::equal_to<K>,
                                                 Eigen::aligned_allocator<std::pair<K const, V>>>;

inline Eigen::Affine3d GetTransBetween(Eigen::Vector3d transStart,
                                       const Eigen::Quaterniond &rotStart,
                                       Eigen::Vector3d transEnd,
                                       const Eigen::Quaterniond &rotEnd) {
    Eigen::Translation3d t_s(transStart(0), transStart(1), transStart(2));
    Eigen::Translation3d t_e(transEnd(0), transEnd(1), transEnd(2));

    Eigen::Affine3d start = t_s * rotStart.toRotationMatrix();
    Eigen::Affine3d end = t_e * rotEnd.toRotationMatrix();

    Eigen::Affine3d result = start.inverse() * end;
    return result;
}

template <typename ScaleType>
inline auto VectorToEigenQuaternion(const std::vector<ScaleType> &vec) {
    Eigen::Quaternion<ScaleType> quaternion;
    quaternion.x() = vec.at(0);
    quaternion.y() = vec.at(1);
    quaternion.z() = vec.at(2);
    quaternion.w() = vec.at(3);
    return quaternion;
}

template <typename ScaleType>
inline auto EigenQuaternionToVector(const Eigen::Quaternion<ScaleType> &quaternion) {
    std::vector<ScaleType> vec(4);
    vec.at(0) = quaternion.x();
    vec.at(1) = quaternion.y();
    vec.at(2) = quaternion.z();
    vec.at(3) = quaternion.w();
    return vec;
}

template <typename ScaleType, int M>
inline auto EigenVecToVector(const Eigen::Matrix<ScaleType, M, 1> &eigenVec) {
    std::vector<ScaleType> vec(M);
    for (int i = 0; i < vec.size(); ++i) {
        vec.at(i) = eigenVec(i);
    }
    return vec;
}

template <typename ScaleType, int M>
inline auto VectorToEigenVec(const std::vector<ScaleType> &vec) {
    Eigen::Matrix<ScaleType, M, 1> eigenVec;
    for (int i = 0; i < vec.size(); ++i) {
        eigenVec(i) = vec.at(i);
    }
    return eigenVec;
}

#define DEF_EIGEN_VEC(DIME) \
    template <typename T>   \
    using Vector##DIME = Eigen::Matrix<T, DIME, 1>;

#define DEF_EIGEN_MAT(ROW, COL) \
    template <typename T>       \
    using Matrix##ROW##COL = Eigen::Matrix<T, ROW, COL>;

#define DEF_EIGEN_VEC_T(DIME, SYMBOL, TYPE) \
    using Vector##DIME##SYMBOL = Eigen::Matrix<TYPE, DIME, 1>;

#define DEF_EIGEN_MAT_T(ROW, COL, SYMBOL, TYPE) \
    using Matrix##ROW##COL##SYMBOL = Eigen::Matrix<TYPE, ROW, COL>;

DEF_EIGEN_VEC(1)
DEF_EIGEN_VEC(2)
DEF_EIGEN_VEC(3)
DEF_EIGEN_VEC(4)
DEF_EIGEN_VEC(5)
DEF_EIGEN_VEC(6)
DEF_EIGEN_VEC(7)
DEF_EIGEN_VEC(8)
DEF_EIGEN_VEC(9)

DEF_EIGEN_MAT(1, 1)
DEF_EIGEN_MAT(2, 2)
DEF_EIGEN_MAT(3, 3)
DEF_EIGEN_MAT(4, 4)
DEF_EIGEN_MAT(5, 5)
DEF_EIGEN_MAT(6, 6)
DEF_EIGEN_MAT(7, 7)
DEF_EIGEN_MAT(8, 8)
DEF_EIGEN_MAT(9, 9)

DEF_EIGEN_VEC_T(1, f, float)
DEF_EIGEN_VEC_T(2, f, float)
DEF_EIGEN_VEC_T(3, f, float)
DEF_EIGEN_VEC_T(4, f, float)
DEF_EIGEN_VEC_T(5, f, float)
DEF_EIGEN_VEC_T(6, f, float)
DEF_EIGEN_VEC_T(7, f, float)
DEF_EIGEN_VEC_T(8, f, float)
DEF_EIGEN_VEC_T(9, f, float)

DEF_EIGEN_VEC_T(1, d, double)
DEF_EIGEN_VEC_T(2, d, double)
DEF_EIGEN_VEC_T(3, d, double)
DEF_EIGEN_VEC_T(4, d, double)
DEF_EIGEN_VEC_T(5, d, double)
DEF_EIGEN_VEC_T(6, d, double)
DEF_EIGEN_VEC_T(7, d, double)
DEF_EIGEN_VEC_T(8, d, double)
DEF_EIGEN_VEC_T(9, d, double)

DEF_EIGEN_MAT(1, 1)
DEF_EIGEN_MAT(2, 2)
DEF_EIGEN_MAT(3, 3)
DEF_EIGEN_MAT(4, 4)
DEF_EIGEN_MAT(5, 5)
DEF_EIGEN_MAT(6, 6)
DEF_EIGEN_MAT(7, 7)
DEF_EIGEN_MAT(8, 8)
DEF_EIGEN_MAT(9, 9)

DEF_EIGEN_MAT_T(1, 1, f, float)
DEF_EIGEN_MAT_T(2, 2, f, float)
DEF_EIGEN_MAT_T(3, 3, f, float)
DEF_EIGEN_MAT_T(4, 4, f, float)
DEF_EIGEN_MAT_T(5, 5, f, float)
DEF_EIGEN_MAT_T(6, 6, f, float)
DEF_EIGEN_MAT_T(7, 7, f, float)
DEF_EIGEN_MAT_T(8, 8, f, float)
DEF_EIGEN_MAT_T(9, 9, f, float)

DEF_EIGEN_MAT_T(1, 1, d, double)
DEF_EIGEN_MAT_T(2, 2, d, double)
DEF_EIGEN_MAT_T(3, 3, d, double)
DEF_EIGEN_MAT_T(4, 4, d, double)
DEF_EIGEN_MAT_T(5, 5, d, double)
DEF_EIGEN_MAT_T(6, 6, d, double)
DEF_EIGEN_MAT_T(7, 7, d, double)
DEF_EIGEN_MAT_T(8, 8, d, double)
DEF_EIGEN_MAT_T(9, 9, d, double)

}  // namespace Eigen

#endif