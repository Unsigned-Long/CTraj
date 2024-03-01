// Copyright (c) 2019-2023. Created on 7/7/23 1:20 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#pragma once

#include <ctraj/spline/spline_common.h>
#include <ctraj/utils/assert.h>
#include <ctraj/utils/sophus_utils.hpp>
#include "cereal/cereal.hpp"
#include "cereal/types/deque.hpp"
#include "cereal/types/array.hpp"
#include "cereal/types/vector.hpp"
#include <Eigen/Dense>
#include "ctraj/utils/eigen_utils.hpp"
#include <sophus/so3.hpp>
#include <array>

namespace ns_ctraj {

    /// @brief Uniform cummulative B-spline for SO(3) of order N
    ///
    /// For example, in the particular case scalar values and order N=5, for a time
    /// \f$t \in [t_i, t_{i+1})\f$ the value of \f$p(t)\f$ depends only on 5 control
    /// points at \f$[t_i, t_{i+1}, t_{i+2}, t_{i+3}, t_{i+4}]\f$. To
    /// simplify calculations we transform time to uniform representation \f$s(t) =
    /// (t - t_0)/\Delta t \f$, such that control points transform into \f$ s_i \in
    /// [0,..,N] \f$. We define function \f$ u(t) = s(t)-s_i \f$ to be a time since
    /// the start of the segment. Following the cummulative matrix representation of
    /// De Boor - Cox formula, the value of the function can be evaluated as
    /// follows: \f{align}{
    ///    R(u(t)) &= R_i
    ///    \prod_{j=1}^{4}{\exp(k_{j}\log{(R_{i+j-1}^{-1}R_{i+j})})},
    ///    \\ \begin{pmatrix} k_0 \\ k_1 \\ k_2 \\ k_3 \\ k_4 \end{pmatrix}^T &=
    ///    M_{c5} \begin{pmatrix} 1 \\ u \\ u^2 \\ u^3 \\ u^4
    ///    \end{pmatrix},
    /// \f}
    /// where \f$ R_{i} \in SO(3) \f$ are knots and \f$ M_{c5} \f$ is a cummulative
    /// blending matrix computed using \ref computeBlendingMatrix \f{align}{
    ///    M_{c5} = \frac{1}{4!}
    ///    \begin{pmatrix} 24 & 0 & 0 & 0 & 0 \\ 23 & 4 & -6 & 4 & -1 \\ 12 & 16 & 0
    ///    & -8 & 3 \\ 1 & 4 & 6 & 4 & -3 \\ 0 & 0 & 0 & 0 & 1 \end{pmatrix}.
    /// \f}
    ///
    /// See [[arXiv:1911.08860]](https://arxiv.org/abs/1911.08860) for more details.
    template<int Order, typename Scalar = double>
    class So3Spline {
    public:
        static constexpr int N = Order;        ///< Order of the spline.
        static constexpr int DEG = Order - 1;  ///< Degree of the spline.

        using MatN = Eigen::Matrix<Scalar, Order, Order>;
        using VecN = Eigen::Matrix<Scalar, Order, 1>;

        using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
        using Mat3 = Eigen::Matrix<Scalar, 3, 3>;

        using SO3 = Sophus::SO3<Scalar>;

        /// @brief Struct to store the Jacobian of the spline
        ///
        /// Since B-spline of order N has local support (only N knots infuence the
        /// value) the Jacobian is zero for all knots except maximum N for value and
        /// all derivatives.
        struct JacobianStruct {
            size_t start_idx;
            std::array<Mat3, Order> d_val_d_knot;
        };

        /// @brief Constructor with knot interval and start time
        ///
        /// @param[in] time_interval knot time interval
        /// @param[in] start_time start time of the spline
        explicit So3Spline(double time_interval = 0.0, double start_time = 0.0)
                : dt_(time_interval), start_t_(start_time) {
            pow_inv_dt[0] = 1.0;
            pow_inv_dt[1] = 1.0 / dt_;
            pow_inv_dt[2] = pow_inv_dt[1] * pow_inv_dt[1];
            pow_inv_dt[3] = pow_inv_dt[2] * pow_inv_dt[1];
        }

        [[nodiscard]] std::pair<double, size_t> ComputeTIndex(double timestamp) const {
            CTRAJ_ASSERT_STREAM(timestamp >= start_t_, " t_ " << timestamp << " start_t_ " << start_t_);
            double st = timestamp - start_t_;

            size_t s = std::floor(st / dt_);
            double u = (st - static_cast<double>(s) * dt_) / dt_;

            CTRAJ_ASSERT_STREAM(s >= 0, "s " << s);
            CTRAJ_ASSERT_STREAM(size_t(s + N) <= knots.size(),
                                "s " << s << " N " << N << " knots.size() " << knots.size());
            return std::make_pair(u, s);
        }

        /// @brief Maximum time represented by spline
        ///
        /// @return maximum time represented by spline in seconds
        [[nodiscard]] double MaxTime() const {
            return start_t_ + (knots.size() - N + 1) * dt_;
        }

        /// @brief Minimum time represented by spline
        ///
        /// @return minimum time represented by spline in nanoseconds
        [[nodiscard]] double MinTime() const { return start_t_; }

        [[nodiscard]] bool TimeStampInRange(double timeStamp) const {
            // left closed right open interval
            return timeStamp >= this->MinTime() + 1E-6 && timeStamp < this->MaxTime() - 1E-6;
        }

        /// @brief Generate random trajectory
        ///
        /// @param[in] n number of knots to generate
        /// @param[in] static_init if true the first N knots will be the same
        /// resulting in static initial condition
        void GenRandomTrajectory(int n, bool static_init = false) {
            if (static_init) {
                Vec3 rnd = Vec3::Random() * M_PI;

                for (int i = 0; i < N; i++) knots.push_back(SO3::exp(rnd));

                for (int i = 0; i < n - N; i++)
                    knots.push_back(SO3::exp(Vec3::Random() * M_PI));

            } else {
                for (int i = 0; i < n; i++)
                    knots.push_back(SO3::exp(Vec3::Random() * M_PI));
            }
        }

        /// @brief Set start time for spline
        ///
        /// @param[in] start_time start time of the spline
        inline void SetStartTime(double s) { start_t_ = s; }

        /// @brief Add knot to the end of the spline
        ///
        /// @param[in] knot knot to add
        inline void KnotsPushBack(const SO3 &knot) { knots.push_back(knot); }

        inline void KnotsPushFront(const SO3 &knot) { knots.push_front(knot); }

        /// @brief Remove knot from the back of the spline
        inline void KnotsPopBack() { knots.pop_back(); }

        /// @brief Return the first knot of the spline
        ///
        /// @return first knot of the spline
        inline const SO3 &KnotsFront() const { return knots.front(); }

        /// @brief Remove first knot of the spline and increase the start time
        inline void KnotsPopFront() {
            start_t_ += dt_;
            knots.pop_front();
        }

        /// @brief Resize container with knots
        ///
        /// @param[in] n number of knots
        inline void Resize(size_t n) { knots.resize(n); }

        /// @brief Return reference to the knot with index i
        ///
        /// @param i index of the knot
        /// @return reference to the knot
        inline SO3 &GetKnot(int i) { return knots[i]; }

        /// @brief Return const reference to the knot with index i
        ///
        /// @param i index of the knot
        /// @return const reference to the knot
        inline const SO3 &GetKnot(int i) const { return knots[i]; }

        /// @brief Return const reference to deque with knots
        ///
        /// @return const reference to deque with knots
        const Eigen::aligned_deque<SO3> &GetKnots() const { return knots; }

        /// @brief Return time interval in nanoseconds
        ///
        /// @return time interval in seconds
        [[nodiscard]] double GetTimeInterval() const { return dt_; }

        /// @brief Evaluate SO(3) B-spline
        ///
        /// @param[in] time_ time for evaluating the value of the spline
        /// @param[out] J if not nullptr, return the Jacobian of the value with
        /// respect to knots
        /// @return SO(3) value of the spline
        SO3 Evaluate(double time, JacobianStruct *J = nullptr) const {
            std::pair<double, size_t> ui = ComputeTIndex(time);
            size_t s = ui.second;
            double u = ui.first;

            VecN p;
            BaseCoeffsWithTime<0>(p, u);

            VecN coeff = blending_matrix_ * p;

            SO3 res = knots[s];

            Mat3 J_helper;

            if (J) {
                J->start_idx = s;
                J_helper.setIdentity();
            }

            for (int i = 0; i < DEG; i++) {
                const SO3 &p0 = knots[s + i];
                const SO3 &p1 = knots[s + i + 1];

                SO3 r01 = p0.inverse() * p1;
                Vec3 delta = r01.log();
                Vec3 kdelta = delta * coeff[i + 1];

                if (J) {
                    Mat3 Jl_inv_delta, Jl_k_delta;

                    Sophus::LeftJacobianInvSO3(delta, Jl_inv_delta);
                    Sophus::LeftJacobianSO3(kdelta, Jl_k_delta);

                    J->d_val_d_knot[i] = J_helper;
                    J_helper = coeff[i + 1] * res.matrix() * Jl_k_delta * Jl_inv_delta * p0.inverse().matrix();
                    J->d_val_d_knot[i] -= J_helper;
                }
                res *= SO3::exp(kdelta);
            }

            if (J) J->d_val_d_knot[DEG] = J_helper;

            return res;
        }

        /// @brief Evaluate rotational Velocity (first time derivative) of SO(3)
        /// B-spline in the body frame

        /// First, let's note that for scalars \f$ k, \Delta k \f$ the following
        /// holds: \f$ \exp((k+\Delta k)\phi) = \exp(k\phi)\exp(\Delta k\phi), \phi
        /// \in \mathbb{R}^3\f$. This is due to the fact that rotations around the
        /// same axis are commutative.
        ///
        /// Let's take SO(3) B-spline with N=3 as an example. The evolution in time of
        /// rotation from the body frame to the world frame is described with \f[
        ///  R_{wb}(t) = R(t) = R_i \exp(k_1(t) \log(R_{i}^{-1}R_{i+1})) \exp(k_2(t)
        ///  \log(R_{i+1}^{-1}R_{i+2})), \f] where \f$ k_1, k_2 \f$ are spline
        ///  coefficients (see detailed description of \ref So3Spline). Since
        ///  expressions under logmap do not depend on time we can rename them to
        ///  constants.
        /// \f[ R(t) = R_i \exp(k_1(t) ~ d_1) \exp(k_2(t) ~ d_2). \f]
        ///
        /// With linear approximation of the spline coefficient evolution over time
        /// \f$ k_1(t_0 + \Delta t) = k_1(t_0) + k_1'(t_0)\Delta t \f$ we can write
        /// \f{align}
        ///  R(t_0 + \Delta t) &= R_i \exp( (k_1(t_0) + k_1'(t_0) \Delta t) ~ d_1)
        ///  \exp((k_2(t_0) + k_2'(t_0) \Delta t) ~ d_2)
        ///  \\ &= R_i \exp(k_1(t_0) ~ d_1) \exp(k_1'(t_0)~ d_1 \Delta t )
        ///  \exp(k_2(t_0) ~ d_2) \exp(k_2'(t_0) ~ d_2 \Delta t )
        ///  \\ &= R_i \exp(k_1(t_0) ~ d_1)
        ///  \exp(k_2(t_0) ~ d_2) \exp(R_{a}^T k_1'(t_0)~ d_1 \Delta t )
        ///  \exp(k_2'(t_0) ~ d_2 \Delta t )
        ///  \\ &= R_i \exp(k_1(t_0) ~ d_1)
        ///  \exp(k_2(t_0) ~ d_2) \exp((R_{a}^T k_1'(t_0)~ d_1 +
        ///  k_2'(t_0) ~ d_2) \Delta t )
        ///  \\ &= R(t_0) \exp((R_{a}^T k_1'(t_0)~ d_1 +
        ///  k_2'(t_0) ~ d_2) \Delta t )
        ///  \\ &= R(t_0) \exp( \omega \Delta t ),
        /// \f} where \f$ \Delta t \f$ is small, \f$ R_{a} \in SO(3) = \exp(k_2(t_0) ~
        /// d_2) \f$ and \f$ \omega \f$ is the rotational Velocity in the body frame.
        /// More explicitly we have the formula for rotational Velocity in the body
        /// frame \f[ \omega = R_{a}^T k_1'(t_0)~ d_1 +  k_2'(t_0) ~ d_2. \f]
        /// Derivatives of spline coefficients can be computed with \ref
        /// BaseCoeffsWithTime similar to \ref RdSpline (detailed description). With
        /// the recursive formula computations generalize to different orders of
        /// spline N.
        ///
        /// @param[in] time time for evaluating Velocity of the spline
        /// @return rotational Velocity (3x1 vector)
        Vec3 VelocityBody(double time) const {
            std::pair<double, size_t> ui = ComputeTIndex(time);
            size_t s = ui.second;
            double u = ui.first;

            VecN p;
            BaseCoeffsWithTime<0>(p, u);
            VecN coeff = blending_matrix_ * p;

            BaseCoeffsWithTime<1>(p, u);
            VecN dCoeff = pow_inv_dt[1] * blending_matrix_ * p;

            Vec3 rot_vel;
            rot_vel.setZero();

            for (int i = 0; i < DEG; i++) {
                const SO3 &p0 = knots[s + i];
                const SO3 &p1 = knots[s + i + 1];

                SO3 r01 = p0.inverse() * p1;
                Vec3 delta = r01.log();

                rot_vel = SO3::exp(-delta * coeff[i + 1]) * rot_vel;
                rot_vel += delta * dCoeff[i + 1];
            }

            return rot_vel;
        }

        /// @brief Evaluate rotational Velocity (first time derivative) of SO(3)
        /// B-spline in the body frame
        ///
        /// @param[in] time time for evaluating Velocity of the spline
        /// @param[out] J if not nullptr, return the Jacobian of the rotational
        /// Velocity in body frame with respect to knots
        /// @return rotational Velocity (3x1 vector)
        Vec3 VelocityBody(double time, JacobianStruct *J) const {
            std::pair<double, size_t> ui = ComputeTIndex(time);
            size_t s = ui.second;
            double u = ui.first;

            VecN p;
            BaseCoeffsWithTime<0>(p, u);
            VecN coeff = blending_matrix_ * p;

            BaseCoeffsWithTime<1>(p, u);
            VecN dCoeff = pow_inv_dt[1] * blending_matrix_ * p;

            Vec3 delta_vec[DEG];

            Mat3 R_tmp[DEG];
            SO3 accum;
            SO3 exp_k_delta[DEG];

            Mat3 Jr_delta_inv[DEG], Jr_kdelta[DEG];

            for (int i = DEG - 1; i >= 0; i--) {
                const SO3 &p0 = knots[s + i];
                const SO3 &p1 = knots[s + i + 1];

                SO3 r01 = p0.inverse() * p1;
                delta_vec[i] = r01.log();

                Sophus::RightJacobianInvSO3(delta_vec[i], Jr_delta_inv[i]);
                Jr_delta_inv[i] *= p1.inverse().matrix();

                Vec3 k_delta = coeff[i + 1] * delta_vec[i];
                Sophus::RightJacobianSO3(-k_delta, Jr_kdelta[i]);

                R_tmp[i] = accum.matrix();
                exp_k_delta[i] = Sophus::SO3d::exp(-k_delta);
                accum *= exp_k_delta[i];
            }

            Mat3 d_vel_d_delta[DEG];

            d_vel_d_delta[0] = dCoeff[1] * R_tmp[0] * Jr_delta_inv[0];
            Vec3 rot_vel = delta_vec[0] * dCoeff[1];
            for (int i = 1; i < DEG; i++) {
                d_vel_d_delta[i] =
                        R_tmp[i - 1] * SO3::hat(rot_vel) * Jr_kdelta[i] * coeff[i + 1] + R_tmp[i] * dCoeff[i + 1];
                d_vel_d_delta[i] *= Jr_delta_inv[i];

                rot_vel = exp_k_delta[i] * rot_vel + delta_vec[i] * dCoeff[i + 1];
            }

            if (J) {
                J->start_idx = s;
                for (int i = 0; i < N; i++) J->d_val_d_knot[i].setZero();
                for (int i = 0; i < DEG; i++) {
                    J->d_val_d_knot[i] -= d_vel_d_delta[i];
                    J->d_val_d_knot[i + 1] += d_vel_d_delta[i];
                }
            }

            return rot_vel;
        }

        /// @brief Evaluate rotational Acceleration (second time derivative) of SO(3)
        /// B-spline in the body frame
        ///
        /// @param[in] time time for evaluating Acceleration of the spline
        /// @param[out] vel_body if not nullptr, return the rotational Velocity in the
        /// body frame (3x1 vector) (side computation)
        /// @return rotational Acceleration (3x1 vector)
        Vec3 AccelerationBody(double time, Vec3 *vel_body = nullptr) const {
            std::pair<double, size_t> ui = ComputeTIndex(time);
            size_t s = ui.second;
            double u = ui.first;

            VecN p;
            BaseCoeffsWithTime<0>(p, u);
            VecN coeff = blending_matrix_ * p;

            BaseCoeffsWithTime<1>(p, u);
            VecN dCoeff = pow_inv_dt[1] * blending_matrix_ * p;

            BaseCoeffsWithTime<2>(p, u);
            VecN ddCoeff = pow_inv_dt[2] * blending_matrix_ * p;

            SO3 r_accum;

            Vec3 rot_vel;
            rot_vel.setZero();

            Vec3 rot_accel;
            rot_accel.setZero();

            for (int i = 0; i < DEG; i++) {
                const SO3 &p0 = knots[s + i];
                const SO3 &p1 = knots[s + i + 1];

                SO3 r01 = p0.inverse() * p1;
                Vec3 delta = r01.log();

                SO3 rot = SO3::exp(-delta * coeff[i + 1]);

                rot_vel = rot * rot_vel;
                Vec3 vel_current = dCoeff[i + 1] * delta;
                rot_vel += vel_current;

                rot_accel = rot * rot_accel;
                rot_accel += ddCoeff[i + 1] * delta + rot_vel.cross(vel_current);
            }

            if (vel_body) *vel_body = rot_vel;
            return rot_accel;
        }

        /// @brief Evaluate rotational Acceleration (second time derivative) of SO(3)
        /// B-spline in the body frame
        ///
        /// @param[in] time time for evaluating Acceleration of the spline
        /// @param[out] J_accel if not nullptr, return the Jacobian of the rotational
        /// Acceleration in body frame with respect to knots
        /// @param[out] vel_body if not nullptr, return the rotational Velocity in the
        /// body frame (3x1 vector) (side computation)
        /// @param[out] J_vel if not nullptr, return the Jacobian of the rotational
        /// Velocity in the body frame (side computation)
        /// @return rotational Acceleration (3x1 vector)
        Vec3 AccelerationBody(double time, JacobianStruct *J_accel, Vec3 *vel_body = nullptr,
                              JacobianStruct *J_vel = nullptr) const {
            CTRAJ_ASSERT(J_accel);

            std::pair<double, size_t> ui = ComputeTIndex(time);
            size_t s = ui.second;
            double u = ui.first;

            VecN p;
            BaseCoeffsWithTime<0>(p, u);
            VecN coeff = blending_matrix_ * p;

            BaseCoeffsWithTime<1>(p, u);
            VecN dCoeff = pow_inv_dt[1] * blending_matrix_ * p;

            BaseCoeffsWithTime<2>(p, u);
            VecN ddCoeff = pow_inv_dt[2] * blending_matrix_ * p;

            Vec3 delta_vec[DEG];
            Mat3 exp_k_delta[DEG];
            Mat3 Jr_delta_inv[DEG], Jr_kdelta[DEG];

            Vec3 rot_vel;
            rot_vel.setZero();

            Vec3 rot_accel;
            rot_accel.setZero();

            Vec3 rot_vel_arr[DEG];
            Vec3 rot_accel_arr[DEG];

            for (int i = 0; i < DEG; i++) {
                const SO3 &p0 = knots[s + i];
                const SO3 &p1 = knots[s + i + 1];

                SO3 r01 = p0.inverse() * p1;
                delta_vec[i] = r01.log();

                Sophus::RightJacobianInvSO3(delta_vec[i], Jr_delta_inv[i]);
                Jr_delta_inv[i] *= p1.inverse().matrix();

                Vec3 k_delta = coeff[i + 1] * delta_vec[i];
                Sophus::RightJacobianSO3(-k_delta, Jr_kdelta[i]);

                exp_k_delta[i] = Sophus::SO3d::exp(-k_delta).matrix();

                rot_vel = exp_k_delta[i] * rot_vel;
                Vec3 vel_current = dCoeff[i + 1] * delta_vec[i];
                rot_vel += vel_current;

                rot_accel = exp_k_delta[i] * rot_accel;
                rot_accel += ddCoeff[i + 1] * delta_vec[i] + rot_vel.cross(vel_current);

                rot_vel_arr[i] = rot_vel;
                rot_accel_arr[i] = rot_accel;
            }

            Mat3 d_accel_d_delta[DEG];
            Mat3 d_vel_d_delta[DEG];

            d_vel_d_delta[DEG - 1] =
                    coeff[DEG] * exp_k_delta[DEG - 1] * SO3::hat(rot_vel_arr[DEG - 2]) * Jr_kdelta[DEG - 1] +
                    Mat3::Identity() * dCoeff[DEG];

            d_accel_d_delta[DEG - 1] =
                    coeff[DEG] * exp_k_delta[DEG - 1] * SO3::hat(rot_accel_arr[DEG - 2]) * Jr_kdelta[DEG - 1] +
                    Mat3::Identity() * ddCoeff[DEG] +
                    dCoeff[DEG] *
                    (SO3::hat(rot_vel_arr[DEG - 1]) - SO3::hat(delta_vec[DEG - 1]) * d_vel_d_delta[DEG - 1]);

            Mat3 Pj;
            Pj.setIdentity();

            Vec3 sj;
            sj.setZero();

            for (int i = DEG - 2; i >= 0; i--) {
                sj += dCoeff[i + 2] * Pj * delta_vec[i + 1];
                Pj *= exp_k_delta[i + 1];

                d_vel_d_delta[i] = Mat3::Identity() * dCoeff[i + 1];
                if (i >= 1)
                    d_vel_d_delta[i] += coeff[i + 1] * exp_k_delta[i] * SO3::hat(rot_vel_arr[i - 1]) * Jr_kdelta[i];

                d_accel_d_delta[i] =
                        Mat3::Identity() * ddCoeff[i + 1] +
                        dCoeff[i + 1] * (SO3::hat(rot_vel_arr[i]) - SO3::hat(delta_vec[i]) * d_vel_d_delta[i]);
                if (i >= 1)
                    d_accel_d_delta[i] += coeff[i + 1] * exp_k_delta[i] * SO3::hat(rot_accel_arr[i - 1]) * Jr_kdelta[i];

                d_vel_d_delta[i] = Pj * d_vel_d_delta[i];
                d_accel_d_delta[i] = Pj * d_accel_d_delta[i] - SO3::hat(sj) * d_vel_d_delta[i];
            }

            if (J_vel) {
                J_vel->start_idx = s;
                for (int i = 0; i < N; i++) J_vel->d_val_d_knot[i].setZero();
                for (int i = 0; i < DEG; i++) {
                    Mat3 val = d_vel_d_delta[i] * Jr_delta_inv[i];

                    J_vel->d_val_d_knot[i] -= val;
                    J_vel->d_val_d_knot[i + 1] += val;
                }
            }

            if (J_accel) {
                J_accel->start_idx = s;
                for (int i = 0; i < N; i++) J_accel->d_val_d_knot[i].setZero();
                for (int i = 0; i < DEG; i++) {
                    Mat3 val = d_accel_d_delta[i] * Jr_delta_inv[i];

                    J_accel->d_val_d_knot[i] -= val;
                    J_accel->d_val_d_knot[i + 1] += val;
                }
            }

            if (vel_body) *vel_body = rot_vel;
            return rot_accel;
        }

        /// @brief Evaluate rotational jerk (third time derivative) of SO(3)
        /// B-spline in the body frame
        ///
        /// @param[in] time time for evaluating jerk of the spline
        /// @param[out] vel_body if not nullptr, return the rotational Velocity in the
        /// body frame (3x1 vector) (side computation)
        /// @param[out] accel_body if not nullptr, return the rotational Acceleration
        /// in the body frame (3x1 vector) (side computation)
        /// @return rotational jerk (3x1 vector)
        Vec3 JerkBody(double time, Vec3 *vel_body = nullptr, Vec3 *accel_body = nullptr) const {
            std::pair<double, size_t> ui = ComputeTIndex(time);
            size_t s = ui.second;
            double u = ui.first;

            VecN p;
            BaseCoeffsWithTime<0>(p, u);
            VecN coeff = blending_matrix_ * p;

            BaseCoeffsWithTime<1>(p, u);
            VecN dCoeff = pow_inv_dt[1] * blending_matrix_ * p;

            BaseCoeffsWithTime<2>(p, u);
            VecN ddCoeff = pow_inv_dt[2] * blending_matrix_ * p;

            BaseCoeffsWithTime<3>(p, u);
            VecN dddCoeff = pow_inv_dt[3] * blending_matrix_ * p;

            Vec3 rot_vel;
            rot_vel.setZero();

            Vec3 rot_accel;
            rot_accel.setZero();

            Vec3 rot_jerk;
            rot_jerk.setZero();

            for (int i = 0; i < DEG; i++) {
                const SO3 &p0 = knots[s + i];
                const SO3 &p1 = knots[s + i + 1];

                SO3 r01 = p0.inverse() * p1;
                Vec3 delta = r01.log();

                SO3 rot = SO3::exp(-delta * coeff[i + 1]);

                rot_vel = rot * rot_vel;
                Vec3 vel_current = dCoeff[i + 1] * delta;
                rot_vel += vel_current;

                rot_accel = rot * rot_accel;
                Vec3 rot_vel_cross_vel_current = rot_vel.cross(vel_current);
                rot_accel += ddCoeff[i + 1] * delta + rot_vel_cross_vel_current;

                rot_jerk = rot * rot_jerk;
                rot_jerk += dddCoeff[i + 1] * delta + (ddCoeff[i + 1] * rot_vel + 2 * dCoeff[i + 1] * rot_accel -
                                                       dCoeff[i + 1] * rot_vel_cross_vel_current).cross(delta);
            }

            if (vel_body) *vel_body = rot_vel;
            if (accel_body) *accel_body = rot_accel;
            return rot_jerk;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
        /// @brief Vector of derivatives of time polynomial.
        ///
        /// Computes a derivative of \f$ \begin{bmatrix}1 & t & t^2 & \dots &
        /// t^{N-1}\end{bmatrix} \f$ with respect to time. For example, the first
        /// derivative would be \f$ \begin{bmatrix}0 & 1 & 2 t & \dots & (N-1)
        /// t^{N-2}\end{bmatrix} \f$.
        /// @param Derivative derivative to Evaluate
        /// @param[out] res_const vector to store the result
        /// @param[in] t
        template<int Derivative, class Derived>
        static void BaseCoeffsWithTime(const Eigen::MatrixBase<Derived> &res_const, Scalar t) {
            EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, N);
            auto &res = const_cast<Eigen::MatrixBase<Derived> &>(res_const);

            res.setZero();

            if (Derivative < N) {
                res[Derivative] = base_coefficients_(Derivative, Derivative);

                Scalar _t = t;
                for (int j = Derivative + 1; j < N; j++) {
                    res[j] = base_coefficients_(Derivative, j) * _t;
                    _t = _t * t;
                }
            }
        }

        ///< Blending matrix. See \ref computeBlendingMatrix.
        static const MatN blending_matrix_;

        static const MatN base_coefficients_;  ///< Base coefficients matrix.
        ///< See \ref computeBaseCoefficients.

        Eigen::aligned_deque<SO3> knots;    ///< Knots
        double dt_;                      ///< Knot interval in seconds
        double start_t_;                 ///< Start time in seconds
        std::array<Scalar, 4> pow_inv_dt;  ///< Array with inverse powers of dt

    public:

        template<class Archive>
        void serialize(Archive &ar) {
            ar(
                    cereal::make_nvp("knots", knots),
                    cereal::make_nvp("dt", dt_),
                    cereal::make_nvp("start_t", start_t_),
                    cereal::make_nvp("pow_inv_dt", pow_inv_dt)
            );
        }
    };

    template<int N, typename Scalar>
    const typename So3Spline<N, Scalar>::MatN
            So3Spline<N, Scalar>::base_coefficients_ = ComputeBaseCoefficients<N, Scalar>();

    template<int N, typename Scalar>
    const typename So3Spline<N, Scalar>::MatN
            So3Spline<N, Scalar>::blending_matrix_ = ComputeBlendingMatrix<N, Scalar, true>();

}  // namespace ns_ctraj
