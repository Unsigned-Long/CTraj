/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/ns_ctraj-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


@file
@brief Useful utilities to work with SO(3) and SE(3) groups from Sophus.
*/

#pragma once

#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <ctraj/utils/eigen_utils.hpp>
#include "tiny-viewer/entity/utils.h"

namespace Sophus {
    template<class Archive, typename ScaleTypes>
    void serialize(Archive &archive, Sophus::SO3<ScaleTypes> &m) {
        archive(m.data()[0], m.data()[1], m.data()[2], m.data()[3]);
    }

    template<class Archive, typename ScaleTypes>
    void serialize(Archive &archive, Sophus::SE3<ScaleTypes> &m) {
        archive(
                // so3
                m.data()[0], m.data()[1], m.data()[2], m.data()[3],
                // trans
                m.data()[4], m.data()[5], m.data()[6]
        );
    }
}

namespace Sophus {

    /// @brief Decoupled version of log map for SE(3)
    ///
    /// For SE(3) element vector
    /// \f[
    /// \begin{pmatrix} R & t \\ 0 & 1 \end{pmatrix} \in SE(3),
    /// \f]
    /// returns \f$ (t, \log(R)) \in \mathbb{R}^6 \f$. Here rotation is not coupled
    /// with translation.
    ///
    /// @param[in] SE(3) member
    /// @return tangent vector (6x1 vector)
    template<typename Scalar>
    inline typename SE3<Scalar>::Tangent se3_logd(const SE3<Scalar> &se3) {
        typename SE3<Scalar>::Tangent upsilon_omega;
        upsilon_omega.template tail<3>() = se3.so3().log();
        upsilon_omega.template head<3>() = se3.translation();
        return upsilon_omega;
    }

    /// @brief Decoupled version of exp map for SE(3)
    ///
    /// For tangent vector \f$ (\upsilon, \omega) \in \mathbb{R}^6 \f$ returns
    /// \f[
    /// \begin{pmatrix} \exp(\omega) & \upsilon \\ 0 & 1 \end{pmatrix} \in SE(3),
    /// \f]
    /// where \f$ \exp(\omega) \in SO(3) \f$. Here rotation is not coupled with
    /// translation.
    ///
    /// @param[in] tangent vector (6x1 vector)
    /// @return  SE(3) member
    template<typename Derived>
    inline SE3<typename Derived::Scalar> se3_expd(const Eigen::MatrixBase<Derived> &upsilon_omega) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6)
        using Scalar = typename Derived::Scalar;
        return SE3<Scalar>(SO3<Scalar>::exp(upsilon_omega.template tail<3>()), upsilon_omega.template head<3>());
    }

    /// @brief Decoupled version of log map for Sim(3)
    ///
    /// For Sim(3) element vector
    /// \f[
    /// \begin{pmatrix} sR & t \\ 0 & 1 \end{pmatrix} \in SE(3),
    /// \f]
    /// returns \f$ (t, \log(R), log(s)) \in \mathbb{R}^3 \f$. Here rotation and
    /// scale are not coupled with translation. Rotation and scale are commutative
    /// anyway.
    ///
    /// @param[in] Sim(3) member
    /// @return tangent vector (7x1 vector)
    template<typename Scalar>
    inline typename Sim3<Scalar>::Tangent sim3_logd(const Sim3<Scalar> &sim3) {
        typename Sim3<Scalar>::Tangent upsilon_omega_sigma;
        upsilon_omega_sigma.template tail<4>() = sim3.rxso3().log();
        upsilon_omega_sigma.template head<3>() = sim3.translation();
        return upsilon_omega_sigma;
    }

    /// @brief Decoupled version of exp map for Sim(3)
    ///
    /// For tangent vector \f$ (\upsilon, \omega, \sigma) \in \mathbb{R}^7 \f$
    /// returns
    /// \f[
    /// \begin{pmatrix} \exp(\sigma)\exp(\omega) & \upsilon \\ 0 & 1 \end{pmatrix}
    ///                                                              \in Sim(3),
    /// \f]
    /// where \f$ \exp(\omega) \in SO(3) \f$. Here rotation and scale are not
    /// coupled with translation. Rotation and scale are commutative anyway.
    ///
    /// @param[in] tangent vector (7x1 vector)
    /// @return  Sim(3) member
    template<typename Derived>
    inline Sim3<typename Derived::Scalar> sim3_expd(const Eigen::MatrixBase<Derived> &upsilon_omega_sigma) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 7)
        using Scalar = typename Derived::Scalar;
        return Sim3<Scalar>(
                RxSO3<Scalar>::exp(upsilon_omega_sigma.template tail<4>()), upsilon_omega_sigma.template head<3>()
        );
    }

    // Note on the use of const_cast in the following functions: The output
    // parameter is only marked 'const' to make the C++ compiler accept a temporary
    // expression here. These functions use const_cast it, so const isn't
    // honored here. See:
    // https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html

    /// @brief Right Jacobian for SO(3)
    ///
    /// For \f$ \exp(x) \in SO(3) \f$ provides a Jacobian that approximates the sum
    /// under expmap with a right multiplication of exp map for small \f$ \epsilon
    /// \f$.  Can be used to compute:  \f$ \exp(\phi + \epsilon) \approx \exp(\phi)
    /// \exp(J_{\phi} \epsilon)\f$
    /// @param[in] phi (3x1 vector)
    /// @param[out] J_phi (3x3 matrix)
    template<typename Derived1, typename Derived2>
    inline void RightJacobianSO3(const Eigen::MatrixBase<Derived1> &phi, const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3)
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3)

        using Scalar = typename Derived1::Scalar;

        auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        Scalar phi_norm2 = phi.squaredNorm();
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J.setIdentity();

        if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
            Scalar phi_norm = std::sqrt(phi_norm2);
            Scalar phi_norm3 = phi_norm2 * phi_norm;

            J -= phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
            J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
        } else {
            // Taylor's expansion around 0
            J -= phi_hat / 2;
            J += phi_hat2 / 6;
        }
    }

    /// @brief Right Inverse Jacobian for SO(3)
    ///
    /// For \f$ \exp(x) \in SO(3) \f$ provides an inverse Jacobian that approximates
    /// the logmap of the right multiplication of expmap of the arguments with a sum
    /// for small \f$ \epsilon \f$.  Can be used to compute:  \f$ \log
    /// (\exp(\phi) \exp(\epsilon)) \approx \phi + J_{\phi} \epsilon\f$
    /// @param[in] phi (3x1 vector)
    /// @param[out] J_phi (3x3 matrix)
    template<typename Derived1, typename Derived2>
    inline void RightJacobianInvSO3(const Eigen::MatrixBase<Derived1> &phi, const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3)
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3)

        using Scalar = typename Derived1::Scalar;

        auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        Scalar phi_norm2 = phi.squaredNorm();
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J.setIdentity();
        J += phi_hat / 2;

        if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
            Scalar phi_norm = std::sqrt(phi_norm2);

            J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) / (2 * phi_norm * std::sin(phi_norm)));
        } else {
            // Taylor's expansion around 0
            J += phi_hat2 / 12;
        }
    }

    /// @brief Left Jacobian for SO(3)
    ///
    /// For \f$ \exp(x) \in SO(3) \f$ provides a Jacobian that approximates the sum
    /// under expmap with a left multiplication of expmap for small \f$ \epsilon
    /// \f$.  Can be used to compute:  \f$ \exp(\phi + \epsilon) \approx
    /// \exp(J_{\phi} \epsilon) \exp(\phi) \f$
    /// @param[in] phi (3x1 vector)
    /// @param[out] J_phi (3x3 matrix)
    template<typename Derived1, typename Derived2>
    inline void LeftJacobianSO3(const Eigen::MatrixBase<Derived1> &phi, const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3)
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3)

        using Scalar = typename Derived1::Scalar;

        auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        Scalar phi_norm2 = phi.squaredNorm();
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J.setIdentity();

        if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
            Scalar phi_norm = std::sqrt(phi_norm2);
            Scalar phi_norm3 = phi_norm2 * phi_norm;

            J += phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
            J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
        } else {
            // Taylor's expansion around 0
            J += phi_hat / 2;
            J += phi_hat2 / 6;
        }
    }

    /// @brief Left Inverse Jacobian for SO(3)
    ///
    /// For \f$ \exp(x) \in SO(3) \f$ provides an inverse Jacobian that approximates
    /// the logmap of the left multiplication of expmap of the arguments with a sum
    /// for small \f$ \epsilon \f$.  Can be used to compute:  \f$ \log
    /// (\exp(\epsilon) \exp(\phi)) \approx \phi + J_{\phi} \epsilon\f$
    /// @param[in] phi (3x1 vector)
    /// @param[out] J_phi (3x3 matrix)
    template<typename Derived1, typename Derived2>
    inline void LeftJacobianInvSO3(const Eigen::MatrixBase<Derived1> &phi, const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3)
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3)

        using Scalar = typename Derived1::Scalar;

        auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        Scalar phi_norm2 = phi.squaredNorm();
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J.setIdentity();
        J -= phi_hat / 2;

        if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
            Scalar phi_norm = std::sqrt(phi_norm2);

            J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) / (2 * phi_norm * std::sin(phi_norm)));
        } else {
            // Taylor's expansion around 0
            J += phi_hat2 / 12;
        }
    }

    /// @brief Right Jacobian for decoupled SE(3)
    ///
    /// For \f$ \exp(x) \in SE(3) \f$ provides a Jacobian that approximates the sum
    /// under decoupled expmap with a right multiplication of decoupled expmap for
    /// small \f$ \epsilon \f$.  Can be used to compute:  \f$ \exp(\phi + \epsilon)
    /// \approx \exp(\phi) \exp(J_{\phi} \epsilon)\f$
    /// @param[in] phi (6x1 vector)
    /// @param[out] J_phi (6x6 matrix)
    template<typename Derived1, typename Derived2>
    inline void
    RightJacobianSE3Decoupled(const Eigen::MatrixBase<Derived1> &phi, const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 6)
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 6, 6)

        using Scalar = typename Derived1::Scalar;

        auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        J.setZero();

        Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
        RightJacobianSO3(omega, J.template bottomRightCorner<3, 3>());
        J.template topLeftCorner<3, 3>() = Sophus::SO3<Scalar>::exp(omega).inverse().matrix();
    }

    /// @brief Right Inverse Jacobian for decoupled SE(3)
    ///
    /// For \f$ \exp(x) \in SE(3) \f$ provides an inverse Jacobian that approximates
    /// the decoupled logmap of the right multiplication of the decoupled expmap of
    /// the arguments with a sum for small \f$ \epsilon \f$.  Can be used to
    /// compute:  \f$ \log
    /// (\exp(\phi) \exp(\epsilon)) \approx \phi + J_{\phi} \epsilon\f$
    /// @param[in] phi (6x1 vector)
    /// @param[out] J_phi (6x6 matrix)
    template<typename Derived1, typename Derived2>
    inline void
    RightJacobianInvSE3Decoupled(const Eigen::MatrixBase<Derived1> &phi, const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 6)
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 6, 6)

        using Scalar = typename Derived1::Scalar;

        auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        J.setZero();

        Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
        RightJacobianInvSO3(omega, J.template bottomRightCorner<3, 3>());
        J.template topLeftCorner<3, 3>() = Sophus::SO3<Scalar>::exp(omega).matrix();
    }

    /// @brief Right Jacobian for decoupled Sim(3)
    ///
    /// For \f$ \exp(x) \in Sim(3) \f$ provides a Jacobian that approximates the sum
    /// under decoupled expmap with a right multiplication of decoupled expmap for
    /// small \f$ \epsilon \f$.  Can be used to compute:  \f$ \exp(\phi + \epsilon)
    /// \approx \exp(\phi) \exp(J_{\phi} \epsilon)\f$
    /// @param[in] phi (7x1 vector)
    /// @param[out] J_phi (7x7 matrix)
    template<typename Derived1, typename Derived2>
    inline void
    RightJacobianSim3Decoupled(const Eigen::MatrixBase<Derived1> &phi, const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 7)
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 7, 7)

        using Scalar = typename Derived1::Scalar;

        auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        J.setZero();

        Eigen::Matrix<Scalar, 4, 1> omega = phi.template tail<4>();
        RightJacobianSO3(omega.template head<3>(), J.template block<3, 3>(3, 3));
        J.template topLeftCorner<3, 3>() =
                Sophus::RxSO3<Scalar>::exp(omega).inverse().matrix();
        J(6, 6) = 1;
    }

    /// @brief Right Inverse Jacobian for decoupled Sim(3)
    ///
    /// For \f$ \exp(x) \in Sim(3) \f$ provides an inverse Jacobian that
    /// approximates the decoupled logmap of the right multiplication of the
    /// decoupled expmap of the arguments with a sum for small \f$ \epsilon \f$. Can
    /// be used to compute:  \f$ \log
    /// (\exp(\phi) \exp(\epsilon)) \approx \phi + J_{\phi} \epsilon\f$
    /// @param[in] phi (7x1 vector)
    /// @param[out] J_phi (7x7 matrix)
    template<typename Derived1, typename Derived2>
    inline void
    RightJacobianInvSim3Decoupled(const Eigen::MatrixBase<Derived1> &phi, const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1)
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2)
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 7)
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 7, 7)

        using Scalar = typename Derived1::Scalar;

        auto &J = const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        J.setZero();

        Eigen::Matrix<Scalar, 4, 1> omega = phi.template tail<4>();
        RightJacobianInvSO3(omega.template head<3>(), J.template block<3, 3>(3, 3));
        J.template topLeftCorner<3, 3>() = Sophus::RxSO3<Scalar>::exp(omega).matrix();
        J(6, 6) = 1;
    }

}  // namespace Sophus
