#include <utility>

// Copyright (c) 2023. Created on 12/10/23 1:12 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_PREINTEGRATION_H
#define CTRAJ_PREINTEGRATION_H

#include "ctraj/utils/eigen_utils.hpp"

namespace ns_ctraj {
    struct IntegrationNoise {
        // 1. accelerator noise
        double ACC_N;
        // 2. gyroscope noise
        double GYR_N;
        // 3. velocity random walk noise density
        double ACC_W;
        // 4. angle random walk noise density
        double GYR_W;
    };

    class PreIntegration {
    protected:
        DEF_EIGEN_MAT_T(15, 15, d, double)
        DEF_EIGEN_MAT_T(15, 18, d, double)
        DEF_EIGEN_MAT_T(18, 18, d, double)

        enum StateOrder {
            O_P = 0, O_R = 3, O_V = 6, O_BA = 9, O_BG = 12
        };

        enum NoiseOrder {
            O_AN = 0, O_GN = 3, O_AW = 6, O_GW = 9
        };

    private:
        double dt{};
        Eigen::Vector3d acc_0, gyr_0;
        Eigen::Vector3d acc_1, gyr_1;

        const Eigen::Vector3d linearized_acc, linearized_gyr;
        Eigen::Vector3d linearized_ba, linearized_bg;

        // the 'FMat' of the continuous-time dynamics
        Matrix1515d jacobian;
        Matrix1515d covariance;
        Matrix1818d noise;

        double sum_dt;
        Eigen::Vector3d delta_p;
        Eigen::Quaterniond delta_q;
        Eigen::Vector3d delta_v;

        std::vector<double> dt_buf;
        std::vector<Eigen::Vector3d> acc_buf;
        std::vector<Eigen::Vector3d> gyr_buf;

    public:

        PreIntegration(const IntegrationNoise &n, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                       Eigen::Vector3d _linearized_ba, Eigen::Vector3d _linearized_bg);

        void PushBack(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

        void RePropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg);

        Eigen::Matrix<double, 15, 1>
        Evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
                 const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                 const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
                 const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj, const Eigen::Vector3d &G);

        [[nodiscard]] inline const Matrix1515d &GetCovariance() const;

        [[nodiscard]] inline double GetSumDt() const;

        [[nodiscard]] inline const Eigen::Vector3d &GetDeltaP() const;

        [[nodiscard]] inline const Eigen::Quaterniond &GetDeltaQ() const;

        [[nodiscard]] inline const Eigen::Vector3d &GetDeltaV() const;

        [[nodiscard]] inline const Eigen::Vector3d &GetLinearizedBa() const;

        [[nodiscard]] inline const Eigen::Vector3d &GetLinearizedBg() const;

    protected:
        void MidPointIntegration(
                double _dt,
                const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                const Eigen::Vector3d &delta_p,
                const Eigen::Quaterniond &delta_q,
                const Eigen::Vector3d &delta_v,
                const Eigen::Vector3d &linearized_ba,
                const Eigen::Vector3d &linearized_bg,
                Eigen::Vector3d &result_delta_p,
                Eigen::Quaterniond &result_delta_q,
                Eigen::Vector3d &result_delta_v,
                Eigen::Vector3d &result_linearized_ba,
                Eigen::Vector3d &result_linearized_bg,
                bool update_jacobian
        );

        void Propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1);

    protected:
        template<typename Derived>
        static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta) {
            typedef typename Derived::Scalar Scalar_t;

            Eigen::Quaternion<Scalar_t> dq;
            Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
            half_theta /= static_cast<Scalar_t>(2.0);
            dq.w() = static_cast<Scalar_t>(1.0);
            dq.x() = half_theta.x();
            dq.y() = half_theta.y();
            dq.z() = half_theta.z();
            return dq;
        }
    };
}


#endif //CTRAJ_PREINTEGRATION_H
