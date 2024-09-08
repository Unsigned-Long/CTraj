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

#ifndef CTRAJ_CERES_SPLINE_HELPER_JET_H
#define CTRAJ_CERES_SPLINE_HELPER_JET_H

#include <ctraj/spline/spline_common.h>
#include <Eigen/Dense>
#include <ceres/jet.h>

namespace ns_ctraj {

/// @brief Helper for implementing Lie group and Euclidean b-splines in ceres of
/// order N
///
/// See [[arXiv:1911.08860]](https://arxiv.org/abs/1911.08860) for more details.
template <class T, int Order>
struct CeresSplineHelperJet {
    static constexpr int N = Order;        // Order of the spline.
    static constexpr int DEG = Order - 1;  // Degree of the spline.

    using MatN = Eigen::Matrix<T, Order, Order>;
    using VecN = Eigen::Matrix<T, Order, 1>;

    static const MatN blending_matrix_;
    static const MatN cumulative_blending_matrix_;
    static const MatN base_coefficients_;

    /// @brief Vector of derivatives of time polynomial.
    ///
    /// Computes a derivative of \f$ \begin{bmatrix}1 & t & t^2 & \dots &
    /// t^{N-1}\end{bmatrix} \f$ with respect to time. For example, the first
    /// derivative would be \f$ \begin{bmatrix}0 & 1 & 2 t & \dots & (N-1)
    /// t^{N-2}\end{bmatrix} \f$.
    /// @param Derivative derivative to Evaluate
    /// @param[out] res_const vector to store the result
    /// @param[in] t
    template <int Derivative, class Derived>
    static inline void BaseCoeffsWithTime(const Eigen::MatrixBase<Derived> &res_const, T t) {
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, N);
        auto &res = const_cast<Eigen::MatrixBase<Derived> &>(res_const);

        res.setZero();

        if (Derivative < N) {
            res[Derivative] = base_coefficients_(Derivative, Derivative);

            T _t = t;
            for (int j = Derivative + 1; j < N; j++) {
                res[j] = base_coefficients_(Derivative, j) * _t;
                _t = _t * t;
            }
        }
    }

    /// @brief Evaluate Lie group cumulative B-spline and time derivatives.
    ///
    /// @param[in] sKnots array of pointers of the spline knots. The size of each
    /// knot should be GroupT::num_parameters: 4 for SO(3) and 7 for SE(3).
    /// @param[in] u normalized time to compute value of the spline
    /// @param[in] inv_dt inverse of the time spacing in seconds between spline
    /// knots
    /// @param[out] transform_out if not nullptr return the value of the spline
    /// @param[out] vel_out if not nullptr Velocity (first time derivative) in the
    /// body frame
    /// @param[out] accel_out if not nullptr Acceleration (second time derivative)
    /// in the body frame
    template <template <class> class GroupT>
    static inline void EvaluateLie(T const *const *sKnots,
                                   const T u,
                                   const double inv_dt,
                                   GroupT<T> *transform_out = nullptr,
                                   typename GroupT<T>::Tangent *vel_out = nullptr,
                                   typename GroupT<T>::Tangent *accel_out = nullptr,
                                   typename GroupT<T>::Tangent *jerk_out = nullptr) {
        using Group = GroupT<T>;
        using Tangent = typename GroupT<T>::Tangent;
        using Adjoint = typename GroupT<T>::Adjoint;

        VecN p, coeff, dCoeff, ddCoeff, dddCoeff;

        CeresSplineHelperJet<T, N>::template BaseCoeffsWithTime<0>(p, u);
        coeff = CeresSplineHelperJet<T, N>::cumulative_blending_matrix_ * p;

        if (vel_out || accel_out || jerk_out) {
            CeresSplineHelperJet<T, N>::template BaseCoeffsWithTime<1>(p, u);
            dCoeff = inv_dt * CeresSplineHelperJet<T, N>::cumulative_blending_matrix_ * p;

            if (accel_out || jerk_out) {
                CeresSplineHelperJet<T, N>::template BaseCoeffsWithTime<2>(p, u);
                ddCoeff =
                    inv_dt * inv_dt * CeresSplineHelperJet<T, N>::cumulative_blending_matrix_ * p;

                if (jerk_out) {
                    CeresSplineHelperJet<T, N>::template BaseCoeffsWithTime<3>(p, u);
                    dddCoeff = inv_dt * inv_dt * inv_dt *
                               CeresSplineHelperJet<T, N>::cumulative_blending_matrix_ * p;
                }
            }
        }

        if (transform_out) {
            Eigen::Map<Group const> const p00(sKnots[0]);
            *transform_out = p00;
        }

        Tangent rot_vel, rot_accel, rot_jerk;

        if (vel_out || accel_out || jerk_out) rot_vel.setZero();
        if (accel_out || jerk_out) rot_accel.setZero();
        if (jerk_out) rot_jerk.setZero();

        for (int i = 0; i < DEG; i++) {
            Eigen::Map<Group const> const p0(sKnots[i]);
            Eigen::Map<Group const> const p1(sKnots[i + 1]);

            Group r01 = p0.inverse() * p1;
            Tangent delta = r01.log();

            Group exp_kdelta = Group::exp(delta * coeff[i + 1]);

            if (transform_out) (*transform_out) *= exp_kdelta;

            if (vel_out || accel_out || jerk_out) {
                Adjoint A = exp_kdelta.inverse().Adj();

                rot_vel = A * rot_vel;
                Tangent rot_vel_current = delta * dCoeff[i + 1];
                rot_vel += rot_vel_current;

                if (accel_out || jerk_out) {
                    rot_accel = A * rot_accel;
                    Tangent accel_lie_bracket = Group::lieBracket(rot_vel, rot_vel_current);
                    rot_accel += ddCoeff[i + 1] * delta + accel_lie_bracket;

                    if (jerk_out) {
                        rot_jerk = A * rot_jerk;
                        rot_jerk += dddCoeff[i + 1] * delta +
                                    Group::lieBracket(ddCoeff[i + 1] * rot_vel +
                                                          T(2) * dCoeff[i + 1] * rot_accel -
                                                          dCoeff[i + 1] * accel_lie_bracket,
                                                      delta);
                    }
                }
            }
        }

        if (vel_out) *vel_out = rot_vel;
        if (accel_out) *accel_out = rot_accel;
        if (jerk_out) *jerk_out = rot_jerk;
    }

    /// @brief Evaluate Euclidean B-spline or time derivatives.
    ///
    /// @param[in] sKnots array of pointers of the spline knots. The size of each
    /// knot should be DIM.
    /// @param[in] u normalized time to compute value of the spline
    /// @param[in] inv_dt inverse of the time spacing in seconds between spline
    /// knots
    /// @param[out] vec_out if DERIV=0 returns value of the spline, otherwise
    /// corresponding derivative.
    template <int DIM, int DERIV>
    static inline void Evaluate(T const *const *sKnots,
                                const T u,
                                const double inv_dt,
                                Eigen::Matrix<T, DIM, 1> *vec_out) {
        if (!vec_out) return;

        using VecD = Eigen::Matrix<T, DIM, 1>;

        VecN p, coeff;

        CeresSplineHelperJet<T, N>::template BaseCoeffsWithTime<DERIV>(p, u);
        coeff = ceres::pow(T(inv_dt), DERIV) * CeresSplineHelperJet<T, N>::blending_matrix_ * p;

        vec_out->setZero();

        for (int i = 0; i < N; i++) {
            Eigen::Map<VecD const> const pt(sKnots[i]);

            (*vec_out) += coeff[i] * pt;
        }
    }
};

template <class T, int N>
const typename CeresSplineHelperJet<T, N>::MatN CeresSplineHelperJet<T, N>::base_coefficients_ =
    ns_ctraj::ComputeBaseCoefficients<N, T>();

template <class T, int N>
const typename CeresSplineHelperJet<T, N>::MatN CeresSplineHelperJet<T, N>::blending_matrix_ =
    ns_ctraj::ComputeBlendingMatrix<N, T, false>();

template <class T, int N>
const typename CeresSplineHelperJet<T, N>::MatN
    CeresSplineHelperJet<T, N>::cumulative_blending_matrix_ =
        ns_ctraj::ComputeBlendingMatrix<N, T, true>();

}  // namespace ns_ctraj
#endif