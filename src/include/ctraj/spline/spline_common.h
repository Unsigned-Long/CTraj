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

#ifndef CTRAJ_SPLINE_COMMON_H
#define CTRAJ_SPLINE_COMMON_H

#include <Eigen/Dense>
#include <cstdint>

namespace ns_ctraj {

/// @brief Compute binomial coefficient.
///
/// Computes number of combinations that include k objects out of n.
/// @param[in] n
/// @param[in] k
/// @return binomial coefficient
constexpr inline uint64_t C_n_k(uint64_t n, uint64_t k) {
    if (k > n) {
        return 0;
    }
    uint64_t r = 1;
    for (uint64_t d = 1; d <= k; ++d) {
        r *= n--;
        r /= d;
    }
    return r;
}

/// @brief Compute blending matrix for uniform B-spline evaluation.
///
/// @param _N order of the spline
/// @param _Scalar scalar type to use
/// @param _Cumulative if the spline should be cumulative
template <int N, typename Scalar = double, bool Cumulative = false>
Eigen::Matrix<Scalar, N, N> ComputeBlendingMatrix() {
    Eigen::Matrix<double, N, N> m;
    m.setZero();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0;

            for (int s = j; s < N; ++s) {
                sum += std::pow(-1.0, s - j) * C_n_k(N, s - j) * std::pow(N - s - 1.0, N - 1.0 - i);
            }
            m(j, i) = C_n_k(N - 1, N - 1 - i) * sum;
        }
    }

    if (Cumulative) {
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                m.row(i) += m.row(j);
            }
        }
    }

    uint64_t factorial = 1;
    for (int i = 2; i < N; ++i) {
        factorial *= i;
    }

    return (m / factorial).template cast<Scalar>();
}

/// @brief Compute base coefficient matrix for polynomials of size N.
///
/// In each row starting from 0 contains the derivative coefficients of the
/// polynomial. For Order=5 we get the following matrix: \f[ \begin{bmatrix}
///   1 & 1 & 1 & 1 & 1
/// \\0 & 1 & 2 & 3 & 4
/// \\0 & 0 & 2 & 6 & 12
/// \\0 & 0 & 0 & 6 & 24
/// \\0 & 0 & 0 & 0 & 24
/// \\ \end{bmatrix}
/// \f]
/// Functions \ref RdSpline::BaseCoeffsWithTime and \ref
/// So3Spline::BaseCoeffsWithTime use this matrix to compute derivatives of the
/// time polynomial.
///
/// @param _N order of the polynomial
/// @param _Scalar scalar type to use
template <int N, typename Scalar = double>
Eigen::Matrix<Scalar, N, N> ComputeBaseCoefficients() {
    Eigen::Matrix<double, N, N> base_coefficients;

    base_coefficients.setZero();
    base_coefficients.row(0).setOnes();

    const int DEG = N - 1;
    int order = DEG;
    for (int n = 1; n < N; n++) {
        for (int i = DEG - order; i < N; i++) {
            base_coefficients(n, i) = (order - DEG + i) * base_coefficients(n - 1, i);
        }
        order--;
    }
    return base_coefficients.template cast<Scalar>();
}

}  // namespace ns_ctraj
#endif