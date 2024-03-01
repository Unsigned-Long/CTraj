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
#include <array>

namespace ns_ctraj {

    /// @brief Uniform B-spline for euclidean vectors with dimention DIM of order
    /// N
    ///
    /// For example, in the particular case scalar values and order N=5, for a time
    /// \f$t \in [t_i, t_{i+1})\f$ the value of \f$p(t)\f$ depends only on 5 control
    /// points at \f$[t_i, t_{i+1}, t_{i+2}, t_{i+3}, t_{i+4}]\f$. To
    /// simplify calculations we transform time to uniform representation \f$s(t) =
    /// (t - t_0)/\Delta t \f$, such that control points transform into \f$ s_i \in
    /// [0,..,N] \f$. We define function \f$ u(t) = s(t)-s_i \f$ to be a time since
    /// the start of the segment. Following the matrix representation of De Boor -
    /// Cox formula, the value of the function can be
    /// evaluated as follows: \f{align}{
    ///    p(u(t)) &=
    ///    \begin{pmatrix} p_{i}\\ p_{i+1}\\ p_{i+2}\\ p_{i+3}\\ p_{i+4}
    ///    \end{pmatrix}^T M_5 \begin{pmatrix} 1 \\ u \\ u^2 \\ u^3 \\ u^4
    ///    \end{pmatrix},
    /// \f}
    /// where \f$ p_{i} \f$ are knots and  \f$ M_5 \f$ is a blending matrix computed
    /// using \ref computeBlendingMatrix \f{align}{
    ///    M_5 = \frac{1}{4!}
    ///    \begin{pmatrix} 1 & -4 & 6 & -4 & 1 \\ 11 & -12  & -6 & 12  & -4 \\11 &
    ///    12 &  -6 &  -12  &  6 \\ 1  &  4  &  6  &  4  & -4 \\ 0  &  0  &  0  &  0
    ///    &  1 \end{pmatrix}.
    /// \f}
    /// Given this formula, we can Evaluate derivatives with respect to time
    /// (Velocity, Acceleration) in the following way:
    /// \f{align}{
    ///    p'(u(t)) &= \frac{1}{\Delta t}
    ///    \begin{pmatrix} p_{i}\\ p_{i+1}\\ p_{i+2}\\ p_{i+3}\\ p_{i+4}
    ///    \end{pmatrix}^T
    ///    M_5
    ///    \begin{pmatrix} 0 \\ 1 \\ 2u \\ 3u^2 \\ 4u^3 \end{pmatrix},
    /// \f}
    /// \f{align}{
    ///    p''(u(t)) &= \frac{1}{\Delta t^2}
    ///    \begin{pmatrix} p_{i}\\ p_{i+1}\\ p_{i+2}\\ p_{i+3}\\ p_{i+4}
    ///    \end{pmatrix}^T
    ///    M_5
    ///    \begin{pmatrix} 0 \\ 0 \\ 2 \\ 6u \\ 12u^2 \end{pmatrix}.
    /// \f}
    /// Higher time derivatives are evaluated similarly. This class supports
    /// vector values for knots \f$ p_{i} \f$. The corresponding derivative vector
    /// on the right is computed using \ref BaseCoeffsWithTime.
    ///
    /// See [[arXiv:1911.08860]](https://arxiv.org/abs/1911.08860) for more details.
    template<int DIM_, int Order, typename Scalar = double>
    class RdSpline {
    public:
        static constexpr int N = Order;        ///< Order of the spline.
        static constexpr int DEG = Order - 1;  ///< Degree of the spline.

        static constexpr int DIM = DIM_;  ///< Dimension of euclidean vector space.

        using MatN = Eigen::Matrix<Scalar, Order, Order>;
        using VecN = Eigen::Matrix<Scalar, Order, 1>;

        using VecD = Eigen::Matrix<Scalar, DIM_, 1>;
        using MatD = Eigen::Matrix<Scalar, DIM_, DIM_>;

        /// @brief Struct to store the Jacobian of the spline
        ///
        /// Since B-spline of order N has local support (only N knots infuence the
        /// value) the Jacobian is zero for all knots except maximum N for value and
        /// all derivatives.
        struct JacobianStruct {
            size_t start_idx;  ///< Start index of the non-zero elements of the Jacobian.
            std::array<Scalar, N> d_val_d_knot;  ///< Value of nonzero Jacobians.
        };

        /// @brief Default constructor
        RdSpline() : dt_(0), start_t_(0) {}

        /// @brief Constructor with knot interval and start time
        ///
        /// @param[in] time_interval_ns knot time interval
        /// @param[in] start_time_ns start time of the spline
        explicit RdSpline(double time_interval, double start_time = 0) : dt_(time_interval), start_t_(start_time) {
            pow_inv_dt[0] = 1.0;
            pow_inv_dt[1] = 1.0 / dt_;

            for (size_t i = 2; i < N; i++) {
                pow_inv_dt[i] = pow_inv_dt[i - 1] * pow_inv_dt[1];
            }
        }

        /// @brief Cast to different scalar type
        template<typename Scalar2>
        inline RdSpline<DIM_, Order, Scalar2> Cast() const {
            RdSpline<DIM_, Order, Scalar2> res;

            res.dt_ = dt_;
            res.start_t_ = start_t_;

            for (int i = 0; i < Order; i++) res.pow_inv_dt[i] = pow_inv_dt[i];

            for (const auto k: knots)
                res.knots.emplace_back(k.template Cast<Scalar2>());

            return res;
        }

        [[nodiscard]] std::pair<double, size_t> ComputeTIndex(double timestamp) const {
            CTRAJ_ASSERT_STREAM(timestamp >= start_t_, " timestamp  " << timestamp << " start_t " << start_t_);
            double st = timestamp - start_t_;
            size_t s = std::floor(st / dt_);
            double u = (st - static_cast<double>(s) * dt_) / dt_;

            // int64_t st_ns = int64_t(st * 1e9);
            // int64_t dt_ns = int64_t(dt_ * 1e9);
            // double u = double(st_ns % dt_ns) / double(dt_ns);

            CTRAJ_ASSERT_STREAM(s >= 0, "s " << s);
            CTRAJ_ASSERT_STREAM(
                    size_t(s + N) <= knots.size(),
                    "s " << s << " N " << N << " knots.size() " << knots.size() << "; timestamp: " << timestamp
                         << "; start_t " << start_t_
            );
            return std::make_pair(u, s);
        }

        /// @brief Set start time for spline
        ///
        /// @param[in] start_time start time of the spline
        inline void SetStartTime(double start_time) {
            start_t_ = start_time;
        }

        /// @brief Maximum time represented by spline
        ///
        /// @return maximum time represented by spline
        [[nodiscard]] double MaxTime() const {
            return start_t_ + (knots.size() - N + 1) * dt_;
        }

        /// @brief Minimum time represented by spline
        ///
        /// @return minimum time represented by spline
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
                VecD rnd = VecD::Random() * 5;

                for (int i = 0; i < N; i++) knots.push_back(rnd);
                for (int i = 0; i < n - N; i++) knots.push_back(VecD::Random() * 5);
            } else {
                for (int i = 0; i < n; i++) knots.push_back(VecD::Random() * 5);
            }
        }

        /// @brief Add knot to the end of the spline
        ///
        /// @param[in] knot knot to add
        inline void KnotsPushBack(const VecD &knot) { knots.push_back(knot); }

        inline void KnotsPushFront(const VecD &knot) { knots.push_front(knot); }

        /// @brief Remove knot from the back of the spline
        inline void KnotsPopBack() { knots.pop_back(); }

        /// @brief Return the first knot of the spline
        ///
        /// @return first knot of the spline
        inline const VecD &KnotsFront() const { return knots.front(); }

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
        inline VecD &GetKnot(int i) { return knots[i]; }

        /// @brief Return const reference to the knot with index i
        ///
        /// @param i index of the knot
        /// @return const reference to the knot
        inline const VecD &GetKnot(int i) const { return knots[i]; }

        /// @brief Return const reference to deque with knots
        ///
        /// @return const reference to deque with knots
        const Eigen::aligned_deque<VecD> &GetKnots() const { return knots; }

        /// @brief Return time interval
        ///
        /// @return time interval
        [[nodiscard]] double GetTimeInterval() const { return dt_; }

        /// @brief Evaluate value or derivative of the spline
        ///
        /// @param Derivative derivative to Evaluate (0 for value)
        /// @param[in] time_ns time for evaluating of the spline
        /// @param[out] J if not nullptr, return the Jacobian of the value with
        /// respect to knots
        /// @return value of the spline or derivative. Euclidean vector of dimension
        /// DIM.
        template<int Derivative = 0>
        VecD Evaluate(double time, JacobianStruct *J = nullptr) const {

            std::pair<double, size_t> ui = ComputeTIndex(time);
            size_t s = ui.second;
            double u = ui.first;

            VecN p;
            BaseCoeffsWithTime<Derivative>(p, u);

            VecN coeff = pow_inv_dt[Derivative] * (blending_matrix_ * p);

            // std::cerr << "p " << p.transpose() << std::endl;
            // std::cerr << "coeff " << coeff.transpose() << std::endl;

            VecD res;
            res.setZero();

            for (int i = 0; i < N; i++) {
                res += coeff[i] * knots[s + i];

                if (J) J->d_val_d_knot[i] = coeff[i];
            }

            if (J) J->start_idx = s;

            return res;
        }

        /// @brief Alias for first derivative of spline. See \ref Evaluate.
        inline VecD Velocity(double time, JacobianStruct *J = nullptr) const {
            return Evaluate<1>(time, J);
        }

        /// @brief Alias for second derivative of spline. See \ref Evaluate.
        inline VecD Acceleration(double time, JacobianStruct *J = nullptr) const {
            return Evaluate<2>(time, J);
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
        /// @brief Vector of derivatives of time polynomial.
        ///
        /// Computes a derivative of \f$ \begin{bmatrix}1 & t & t^2 & \dots &
        /// t^{N-1}\end{bmatrix} \f$ with repect to time. For example, the first
        /// derivative would be \f$ \begin{bmatrix}0 & 1 & 2 t & \dots & (N-1)
        /// t^{N-2}\end{bmatrix} \f$.
        /// @param Derivative derivative to Evaluate
        /// @param[out] res_const vector to store the result
        /// @param[in] t
        template<int Derivative, class Derived>
        static void BaseCoeffsWithTime(const Eigen::MatrixBase<Derived> &res_const,
                                       Scalar t) {
            EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, N)
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

        template<int, int, typename>
        friend
        class RdSpline;

        ///< Blending matrix. See \ref computeBlendingMatrix.
        static const MatN blending_matrix_;

        static const MatN base_coefficients_; ///< Base coefficients matrix.
        ///< See \ref computeBaseCoefficients.

        Eigen::aligned_deque<VecD> knots;     ///< Knots
        double dt_;                           ///< Knot interval in seconds
        double start_t_;                      ///< Start time in seconds
        std::array<Scalar, Order> pow_inv_dt; ///< Array with inverse powers of dt

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

    template<int DIM, int N, typename Scalar>
    const typename RdSpline<DIM, N, Scalar>::MatN
            RdSpline<DIM, N, Scalar>::base_coefficients_ = ComputeBaseCoefficients<N, Scalar>();

    template<int DIM, int N, typename Scalar>
    const typename RdSpline<DIM, N, Scalar>::MatN
            RdSpline<DIM, N, Scalar>::blending_matrix_ = ComputeBlendingMatrix<N, Scalar, false>();

}  // namespace ns_ctraj
