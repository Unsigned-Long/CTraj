// Copyright (c) 2019-2023. Created on 7/7/23 1:20 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#pragma once

#include <ctraj/spline/rd_spline.h>
#include <ctraj/spline/so3_spline.h>
#include <ctraj/spline/calib_bias.hpp>
#include <ctraj/spline/spline_segment.h>
#include <ctraj/utils/assert.h>
#include "sophus/se3.hpp"
#include <array>
#include "ctraj/utils/macros.hpp"

namespace ns_ctraj {

    /// @brief Uniform B-spline for SE(3) of order N. Internally uses an SO(3) (\ref
    /// So3Spline) spline for rotation and 3D Euclidean spline (\ref RdSpline) for
    /// translation (split representation).
    ///
    /// See [[arXiv:1911.08860]](https://arxiv.org/abs/1911.08860) for more details.
    template<int Order, typename Scalar = double>
    class Se3Spline {
    public:
        static constexpr int N = Order;        ///< Order of the spline.
        static constexpr int DEG = Order - 1;  ///< Degree of the spline.

        using MatN = Eigen::Matrix<Scalar, Order, Order>;
        using VecN = Eigen::Matrix<Scalar, Order, 1>;
        using VecNp1 = Eigen::Matrix<Scalar, Order + 1, 1>;

        using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
        using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
        using Vec9 = Eigen::Matrix<Scalar, 9, 1>;
        using Vec12 = Eigen::Matrix<Scalar, 12, 1>;

        using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
        using Mat6 = Eigen::Matrix<Scalar, 6, 6>;

        using Mat36 = Eigen::Matrix<Scalar, 3, 6>;
        using Mat39 = Eigen::Matrix<Scalar, 3, 9>;
        using Mat312 = Eigen::Matrix<Scalar, 3, 12>;

        using Matrix3Array = std::array<Mat3, N>;
        using Matrix36Array = std::array<Mat36, N>;
        using Matrix6Array = std::array<Mat6, N>;

        using SO3 = Sophus::SO3<Scalar>;
        using SE3 = Sophus::SE3<Scalar>;

        using PosJacobianStruct = typename RdSpline<3, N, Scalar>::JacobianStruct;
        using SO3JacobianStruct = typename So3Spline<N, Scalar>::JacobianStruct;

        /// @brief Struct to store the accelerometer residual Jacobian with
        /// respect to knots
        struct AccelPosSO3JacobianStruct {
            size_t start_idx;
            std::array<Mat36, N> d_val_d_knot;
        };

        /// @brief Struct to store the Pose Jacobian with respect to knots
        struct PosePosSO3JacobianStruct {
            size_t start_idx;
            std::array<Mat6, N> d_val_d_knot;
        };

        /// @brief Constructor with knot interval and start time
        ///
        /// @param[in] time_interval knot time interval in seconds
        /// @param[in] start_time start time of the spline in seconds
        explicit Se3Spline(double time_interval, double start_time = 0)
                : pos_spline(time_interval, start_time), so3_spline(time_interval, start_time), dt_(time_interval) {}

        /// @brief Generate random trajectory
        ///
        /// @param[in] n number of knots to generate
        /// @param[in] static_init if true the first N knots will be the same
        /// resulting in static initial condition
        void GenRandomTrajectory(int n, bool static_init = false) {
            so3_spline.GenRandomTrajectory(n, static_init);
            pos_spline.GenRandomTrajectory(n, static_init);
        }

        /// @brief Set the knot to particular SE(3) Pose
        ///
        /// @param[in] pose SE(3) Pose
        /// @param[in] i index of the knot
        void SetKnot(const SE3 &pose, int i) {
            so3_spline.GetKnot(i) = pose.so3();
            pos_spline.GetKnot(i) = pose.translation();
        }

        /// @brief Set the knot to particular Vec3 Pose
        ///
        /// @param[in] pos Vec3 Pose
        /// @param[in] i index of the knot
        void SetKnotPos(const Vec3 pos, int i) {
            pos_spline.GetKnot(i) = pos;
        }

        /// @brief Set the knot to particular SO3 Pose
        ///
        /// @param[in] ori SO3 Pose
        /// @param[in] i index of the knot
        void SetKnotSO3(const SO3 ori, int i) {
            so3_spline.GetKnot(i) = ori;
        }

        /// @brief Reset spline to have num_knots initialized at Pose
        ///
        /// @param[in] pose SE(3) Pose
        /// @param[in] num_knots number of knots to initialize
        void SetKnots(const SE3 &pose, int num_knots) {
            so3_spline.Resize(num_knots);
            pos_spline.Resize(num_knots);

            for (int i = 0; i < num_knots; i++) {
                so3_spline.GetKnot(i) = pose.so3();
                pos_spline.GetKnot(i) = pose.translation();
            }
        }

        /// @brief Reset spline to the knots from other spline
        ///
        /// @param[in] other spline to copy knots from
        void SetKnots(const Se3Spline<N, Scalar> &other) {
            CTRAJ_ASSERT(other.dt_ == dt_);
            CTRAJ_ASSERT(other.pos_spline.GetKnots().size() == other.pos_spline.GetKnots().size());

            size_t num_knots = other.pos_spline.GetKnots().size();

            so3_spline.Resize(num_knots);
            pos_spline.Resize(num_knots);

            for (size_t i = 0; i < num_knots; i++) {
                so3_spline.GetKnot(i) = other.so3_spline.GetKnot(i);
                pos_spline.GetKnot(i) = other.pos_spline.GetKnot(i);
            }
        }

        /// @brief extend trajectory to time t
        ///
        /// @param[in] t time
        /// @param[in] initial_so3 initial knot of so3_spline
        /// @param[in] initial_pos initial knot of pos_spline
        void ExtendKnotsTo(double t, const SO3 initial_so3, const Vec3 initial_pos) {
            while ((NumKnots() < N) || (MaxTime() < t)) {
                so3_spline.KnotsPushBack(initial_so3);
                pos_spline.KnotsPushBack(initial_pos);
            }
        }


        /// @brief extend trajectory to time t
        ///
        /// @param[in] t timestamp
        /// @param[in] initial_knot initial knot
        void ExtendKnotsTo(double timestamp, const SE3 &initial_knot) {
            while ((NumKnots() < N) || (MaxTime() < timestamp)) {
                KnotsPushBack(initial_knot);
            }
        }

        /// @brief Add knot to the end of the spline
        ///
        /// @param[in] knot knot to add
        inline void KnotsPushBack(const SE3 &knot) {
            so3_spline.KnotsPushBack(knot.so3());
            pos_spline.KnotsPushBack(knot.translation());
        }

        inline void KnotsPushFront(const SE3 &knot) {
            so3_spline.KnotsPushFront(knot.so3());
            pos_spline.KnotsPushFront(knot.translation());
        }

        /// @brief Remove knot from the back of the spline
        inline void KnotsPopBack() {
            so3_spline.KnotsPopBack();
            pos_spline.KnotsPopBack();
        }

        /// @brief Return the first knot of the spline
        ///
        /// @return first knot of the spline
        inline SE3 KnotsFront() const {
            SE3 res(so3_spline.KnotsFront(), pos_spline.KnotsFront());

            return res;
        }

        /// @brief Remove first knot of the spline and increase the start time
        inline void KnotsPopFront() {
            so3_spline.KnotsPopFront();
            pos_spline.KnotsPopFront();

            CTRAJ_ASSERT(so3_spline.MinTime() == pos_spline.MinTime());
            CTRAJ_ASSERT(so3_spline.GetKnots().size() == pos_spline.GetKnots().size());
        }

        /// @brief Return the last knot of the spline
        ///
        /// @return last knot of the spline
        SE3 GetLastKnot() {
            CTRAJ_ASSERT(so3_spline.GetKnots().size() == pos_spline.GetKnots().size());

            SE3 res(so3_spline.GetKnots().back(), pos_spline.GetKnots().back());

            return res;
        }

        /// @brief Return knot with index i
        ///
        /// @param i index of the knot
        /// @return knot
        SE3 GetKnot(size_t i) const {
            SE3 res(GetKnotSO3(i), GetKnotPos(i));
            return res;
        }

        /// @brief Return reference to the SO(3) knot with index i
        ///
        /// @param i index of the knot
        /// @return reference to the SO(3) knot
        inline SO3 &GetKnotSO3(size_t i) { return so3_spline.GetKnot(i); }

        /// @brief Return const reference to the SO(3) knot with index i
        ///
        /// @param i index of the knot
        /// @return const reference to the SO(3) knot
        inline const SO3 &GetKnotSO3(size_t i) const { return so3_spline.GetKnot(i); }

        /// @brief Return reference to the position knot with index i
        ///
        /// @param i index of the knot
        /// @return reference to the position knot
        inline Vec3 &GetKnotPos(size_t i) { return pos_spline.GetKnot(i); }

        /// @brief Return const reference to the position knot with index i
        ///
        /// @param i index of the knot
        /// @return const reference to the position knot
        inline const Vec3 &GetKnotPos(size_t i) const {
            return pos_spline.GetKnot(i);
        }

        /// @brief Set start time for spline
        ///
        /// @param[in] start_time start time of the spline in seconds
        inline void SetStartTime(double timestamp) {
            so3_spline.SetStartTime(timestamp);
            pos_spline.SetStartTime(timestamp);
        }

        /// @brief Apply increment to the knot
        ///
        /// The increment vector consists of translational and rotational parts \f$
        /// [\upsilon, \omega]^T \f$. Given the current Pose of the knot \f$ R \in
        /// SO(3), p \in \mathbb{R}^3\f$ the updated Pose is: \f{align}{ R' &=
        /// \exp(\omega) R
        /// \\ p' &= p + \upsilon
        /// \f}
        ///  The increment is consistent with \ref
        /// PoseState::ApplyInc.
        ///
        /// @param[in] i index of the knot
        /// @param[in] inc 6x1 increment vector
        template<typename Derived>
        void ApplyInc(int i, const Eigen::MatrixBase<Derived> &inc) {
            EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6)

            pos_spline.GetKnot(i) += inc.template head<3>();
            so3_spline.GetKnot(i) =
                    SO3::exp(inc.template tail<3>()) * so3_spline.GetKnot(i);
        }

        /// @brief Maximum time represented by spline
        ///
        /// @return maximum time represented by spline in nanoseconds
        [[nodiscard]] double MaxTime() const {
            CTRAJ_ASSERT_STREAM(so3_spline.MaxTime() == pos_spline.MaxTime(),
                                "so3_spline.MaxTime() " << so3_spline.MaxTime() << " pos_spline.MaxTime() "
                                                        << pos_spline.MaxTime());
            return pos_spline.MaxTime();
        }

        /// @brief Minimum time represented by spline
        ///
        /// @return minimum time represented by spline in seconds
        [[nodiscard]] double MinTime() const {
            CTRAJ_ASSERT_STREAM(so3_spline.MinTime() == pos_spline.MinTime(),
                                "so3_spline.MinTime() " << so3_spline.MinTime() << " pos_spline.MinTime() "
                                                        << pos_spline.MinTime());
            return pos_spline.MinTime();
        }

        [[nodiscard]] virtual bool TimeStampInRange(double timeStamp) const {
            // left closed right open interval
            return timeStamp >= this->MinTime() + 1E-6 && timeStamp < this->MaxTime() - 1E-6;
        }

        /// @brief Number of knots in the spline
        [[nodiscard]] size_t NumKnots() const { return pos_spline.GetKnots().size(); }

        /// @brief Linear Acceleration in the world frame.
        ///
        /// @param[in] time time to Evaluate linear Acceleration in seconds
        inline Vec3 TransAccelWorld(double time) const {
            return pos_spline.Acceleration(time);
        }

        /// @brief Linear Velocity in the world frame.
        ///
        /// @param[in] time time to Evaluate linear Velocity in seconds
        inline Vec3 TransVelWorld(double time) const {
            return pos_spline.Velocity(time);
        }

        /// @brief Rotational Velocity in the body frame.
        ///
        /// @param[in] time time to Evaluate rotational Velocity in seconds
        inline Vec3 RotVelBody(double time) const {
            return so3_spline.VelocityBody(time);
        }

        /// @brief Evaluate Pose.
        ///
        /// @param[in] time time to Evaluate Pose in seconds
        /// @return SE(3) Pose at time
        SE3 Pose(double time) const {
            SE3 res;

            res.so3() = so3_spline.Evaluate(time);
            res.translation() = pos_spline.Evaluate(time);

            return res;
        }

        /// @brief Evaluate Pose and compute Jacobian.
        ///
        /// @param[in] time time to Evaluate Pose in seconds
        /// @param[out] J Jacobian of the Pose with respect to knots
        /// @return SE(3) Pose at time
        SE3 Pose(double time, PosePosSO3JacobianStruct *J) const {
            SE3 res;

            typename So3Spline<Order, Scalar>::JacobianStruct Jr;
            typename RdSpline<3, N, Scalar>::JacobianStruct Jp;

            res.so3() = so3_spline.Evaluate(time, &Jr);
            res.translation() = pos_spline.Evaluate(time, &Jp);

            if (J) {
                Eigen::Matrix3d RT = res.so3().inverse().matrix();

                J->start_idx = Jr.start_idx;
                for (int i = 0; i < N; i++) {
                    J->d_val_d_knot[i].setZero();
                    J->d_val_d_knot[i].template topLeftCorner<3, 3>() = RT * Jp.d_val_d_knot[i];
                    J->d_val_d_knot[i].template bottomRightCorner<3, 3>() = RT * Jr.d_val_d_knot[i];
                }
            }

            return res;
        }

        /// @brief Evaluate Pose and compute time Jacobian.
        ///
        /// @param[in] time time to Evaluate Pose in seconds
        /// @param[out] J Jacobian of the Pose with time
        void d_pose_d_t(double time, Vec6 &J) const {
            J.template head<3>() = so3_spline.Evaluate(time).inverse() * TransVelWorld(time);
            J.template tail<3>() = RotVelBody(time);
        }

        /// @brief Evaluate gyroscope residual.
        ///
        /// @param[in] time time of the measurement
        /// @param[in] measurement gyroscope measurement
        /// @param[in] gyro_bias_full gyroscope calibration
        /// @return gyroscope residual
        Vec3 GyroResidual(double time, const Vec3 &measurement, const CalibGyroBias<Scalar> &gyro_bias_full) const {
            return so3_spline.VelocityBody(time) - gyro_bias_full.GetCalibrated(measurement);
        }

        /// @brief Evaluate gyroscope residual and compute Jacobians.
        ///
        /// @param[in] time time of the measurement
        /// @param[in] measurement gyroscope measurement
        /// @param[in] gyro_bias_full gyroscope calibration
        /// @param[out] J_knots Jacobian with respect to SO(3) spline knots
        /// @param[out] J_bias Jacobian with respect to gyroscope calibration
        /// @return gyroscope residual
        Vec3 GyroResidual(double time, const Vec3 &measurement, const CalibGyroBias<Scalar> &gyro_bias_full,
                          SO3JacobianStruct *J_knots, Mat312 *J_bias = nullptr) const {
            if (J_bias) {
                J_bias->setZero();
                J_bias->template block<3, 3>(0, 0).diagonal().array() = 1.0;
                J_bias->template block<3, 3>(0, 3).diagonal().array() = -measurement[0];
                J_bias->template block<3, 3>(0, 6).diagonal().array() = -measurement[1];
                J_bias->template block<3, 3>(0, 9).diagonal().array() = -measurement[2];
            }

            return so3_spline.VelocityBody(time, J_knots) -
                   gyro_bias_full.GetCalibrated(measurement);
        }

        /// @brief Evaluate accelerometer residual.
        ///
        /// @param[in] time time of the measurement
        /// @param[in] measurement accelerometer measurement
        /// @param[in] accel_bias_full accelerometer calibration
        /// @param[in] g gravity
        /// @return accelerometer residual
        Vec3
        AccelResidual(double time, const Eigen::Vector3d &measurement, const CalibAccelBias<Scalar> &accel_bias_full,
                      const Eigen::Vector3d &g) const {
            Sophus::SO3d R = so3_spline.Evaluate(time);
            Eigen::Vector3d accel_world = pos_spline.Acceleration(time);

            return R.inverse() * (accel_world + g) - accel_bias_full.GetCalibrated(measurement);
        }

        /// @brief Evaluate accelerometer residual and Jacobians.
        ///
        /// @param[in] time time of the measurement
        /// @param[in] measurement accelerometer measurement
        /// @param[in] accel_bias_full accelerometer calibration
        /// @param[in] g gravity
        /// @param[out] J_knots Jacobian with respect to spline knots
        /// @param[out] J_bias Jacobian with respect to accelerometer calibration
        /// @param[out] J_g Jacobian with respect to gravity
        /// @return accelerometer residual
        Vec3 AccelResidual(double time, const Vec3 &measurement, const CalibAccelBias<Scalar> &accel_bias_full,
                           const Vec3 &g, AccelPosSO3JacobianStruct *J_knots, Mat39 *J_bias = nullptr,
                           Mat3 *J_g = nullptr) const {
            typename So3Spline<Order, Scalar>::JacobianStruct Jr;
            typename RdSpline<3, N, Scalar>::JacobianStruct Jp;

            Sophus::SO3d R = so3_spline.Evaluate(time, &Jr);
            Eigen::Vector3d accel_world = pos_spline.Acceleration(time, &Jp);

            Eigen::Matrix3d RT = R.inverse().matrix();
            Eigen::Matrix3d tmp = RT * Sophus::SO3d::hat(accel_world + g);

            CTRAJ_ASSERT_STREAM(
                    Jr.start_idx == Jp.start_idx, "Jr.start_idx " << Jr.start_idx << " Jp.start_idx " << Jp.start_idx
            );

            CTRAJ_ASSERT_STREAM(
                    so3_spline.GetKnots().size() == pos_spline.GetKnots().size(),
                    "so3_spline.GetKnots().size() " << so3_spline.GetKnots().size() << " pos_spline.GetKnots().size() "
                                                    << pos_spline.GetKnots().size()
            );

            J_knots->start_idx = Jp.start_idx;
            for (int i = 0; i < N; i++) {
                J_knots->d_val_d_knot[i].template topLeftCorner<3, 3>() = RT * Jp.d_val_d_knot[i];
                J_knots->d_val_d_knot[i].template bottomRightCorner<3, 3>() = tmp * Jr.d_val_d_knot[i];
            }

            if (J_bias) {
                J_bias->setZero();
                J_bias->template block<3, 3>(0, 0).diagonal().array() = 1.0;
                J_bias->template block<3, 3>(0, 3).diagonal().array() = -measurement[0];
                (*J_bias)(1, 6) = -measurement[1];
                (*J_bias)(2, 7) = -measurement[1];
                (*J_bias)(2, 8) = -measurement[2];
            }
            if (J_g) (*J_g) = RT;

            Vec3 res = RT * (accel_world + g) - accel_bias_full.GetCalibrated(measurement);

            return res;
        }

        /// @brief Evaluate position residual.
        ///
        /// @param[in] time time of the measurement
        /// @param[in] measured_position position measurement
        /// @param[out] Jp if not nullptr, Jacobian with respect to knots of the
        /// position spline
        /// @return position residual
        Sophus::Vector3d
        PositionResidual(double time, const Vec3 &measured_position, PosJacobianStruct *Jp = nullptr) const {
            return pos_spline.Evaluate(time, Jp) - measured_position;
        }

        /// @brief Evaluate orientation residual.
        ///
        /// @param[in] time time of the measurement
        /// @param[in] measured_orientation orientation measurement
        /// @param[out] Jr if not nullptr, Jacobian with respect to knots of the
        /// SO(3) spline
        /// @return orientation residual
        Sophus::Vector3d
        OrientationResidual(double time, const SO3 &measured_orientation, SO3JacobianStruct *Jr = nullptr) const {
            Sophus::Vector3d res = (so3_spline.Evaluate(time, Jr) * measured_orientation.inverse()).log();

            if (Jr) {
                Eigen::Matrix3d JRot;
                Sophus::LeftJacobianSO3(res, JRot);

                for (int i = 0; i < N; i++) {
                    Jr->d_val_d_knot[i] = JRot * Jr->d_val_d_knot[i];
                }
            }

            return res;
        }

        /// @brief Print knots for debugging.
        inline void PrintKnots() const {
            for (size_t i = 0; i < pos_spline.GetKnots().size(); i++) {
                std::cout << i << ": p:" << pos_spline.GetKnot(i).transpose() << " q: "
                          << so3_spline.GetKnot(i).unit_quaternion().coeffs().transpose()
                          << std::endl;
            }
        }

        /// @brief Print position knots for debugging.
        inline void PrintPosKnots() const {
            for (size_t i = 0; i < pos_spline.GetKnots().size(); i++) {
                std::cout << pos_spline.GetKnot(i).transpose() << std::endl;
            }
        }

        /// @brief Knot time interval in nanoseconds.
        [[nodiscard]] inline double GetDt() const { return dt_; }

        [[nodiscard]] std::pair<double, size_t> ComputeTIndex(double timestamp) const {
            return pos_spline.ComputeTIndex(timestamp);
        }

        void CalculateSplineMeta(time_init_t times, SplineMeta<Order> &spline_meta) const {
            double master_dt = GetDt();
            double master_t0 = MinTime();
            size_t current_segment_start = 0;
            size_t current_segment_end = 0; // Negative signals no segment created yet

            // Times are guaranteed to be sorted correctly and t2 >= t1
            for (auto tt: times) {
                std::pair<double, size_t> ui_1, ui_2;
                ui_1 = pos_spline.ComputeTIndex(tt.first);
                ui_2 = pos_spline.ComputeTIndex(tt.second);

                size_t i1 = ui_1.second;
                size_t i2 = ui_2.second;

                // Create new segment, or extend the current one
                if (spline_meta.segments.empty() || i1 > current_segment_end) {
                    double segment_t0 = master_t0 + master_dt * double(i1);
                    spline_meta.segments.push_back(SplineSegmentMeta<Order>(segment_t0, master_dt));
                    current_segment_start = i1;
                } else {
                    i1 = current_segment_end + 1;
                }

                auto &current_segment_meta = spline_meta.segments.back();

                for (size_t i = i1; i < (i2 + N); ++i) {
                    current_segment_meta.n += 1;
                }

                current_segment_end = current_segment_start + current_segment_meta.n - 1;
            } // for times

        }

        const RdSpline<3, Order, Scalar> &GetPosSpline() const {
            return pos_spline;
        }

        const So3Spline<Order, Scalar> &GetSo3Spline() const {
            return so3_spline;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
        RdSpline<3, Order, Scalar> pos_spline;  ///< Position spline
        So3Spline<Order, Scalar> so3_spline;    ///< Orientation spline

        double dt_;  ///< Knot interval in seconds

    public:
        template<class Archive>
        void serialize(Archive &ar) {
            ar(
                    cereal::make_nvp("pos_spline", pos_spline),
                    cereal::make_nvp("so3_spline", so3_spline),
                    cereal::make_nvp("dt", dt_)
            );
        }

    };

}  // namespace ns_ctraj
