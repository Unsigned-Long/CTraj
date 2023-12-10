// Copyright (c) 2023. Created on 12/10/23 1:13 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#include "ctraj/core/preintegration.h"

namespace ns_ctraj {

    PreIntegration::PreIntegration(const IntegrationNoise &n, const Eigen::Vector3d &_acc_0,
                                   const Eigen::Vector3d &_gyr_0, Eigen::Vector3d _linearized_ba,
                                   Eigen::Vector3d _linearized_bg)
            : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
              linearized_ba{std::move(_linearized_ba)}, linearized_bg{std::move(_linearized_bg)},
              jacobian{Matrix1515d::Identity()}, covariance{Matrix1515d::Zero()}, sum_dt{0.0},
              delta_p{Eigen::Vector3d::Zero()},
              delta_q{Eigen::Quaterniond::Identity()},
              delta_v{Eigen::Vector3d::Zero()} {
        // the noise matrix
        noise = Matrix1818d::Zero();
        noise.block<3, 3>(0, 0) = (n.ACC_N * n.ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = (n.GYR_N * n.GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = (n.ACC_N * n.ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = (n.GYR_N * n.GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = (n.ACC_W * n.ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = (n.GYR_W * n.GYR_W) * Eigen::Matrix3d::Identity();
    }

    void PreIntegration::PushBack(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        Propagate(dt, acc, gyr);
    }

    void PreIntegration::RePropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg) {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++) {
            Propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
        }
    }

    void PreIntegration::MidPointIntegration(double _dt, const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                                             const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                                             const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q,
                                             const Eigen::Vector3d &delta_v, const Eigen::Vector3d &linearized_ba,
                                             const Eigen::Vector3d &linearized_bg, Eigen::Vector3d &result_delta_p,
                                             Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                                             Eigen::Vector3d &result_linearized_ba,
                                             Eigen::Vector3d &result_linearized_bg, bool update_jacobian) {
        using namespace Eigen;
        // midpoint for angular velocity
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        // the rotation variation
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);

        // midpoint for linear acceleration
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

        // the velocity variation
        result_delta_v = delta_v + un_acc * _dt;

        // the position variation
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;

        // biases
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        if (!update_jacobian) {
            return;
        }
        // unbiased angular velocity
        Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;

        // last unbiased linear acceleration
        Vector3d a_0_x = _acc_0 - linearized_ba;

        // current unbiased linear acceleration
        Vector3d a_1_x = _acc_1 - linearized_ba;

        Matrix3d R_w_x, R_a_0_x, R_a_1_x;

        R_w_x << 0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
        R_a_0_x << 0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
        R_a_1_x << 0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

        // compute the matrix of 'Identity + FMat * dt'
        MatrixXd F = MatrixXd::Zero(15, 15);

        F.block<3, 3>(0, 0) = Matrix3d::Identity();
        // ------------------------
        // todo: alpha to theta ???
        // ------------------------
        F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
                              -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x *
                              (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
        F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
        // ---------------------
        // todo: alpha to ba ???
        // ---------------------
        F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
        // ---------------------
        // todo: alpha to bg ???
        // ---------------------
        F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;

        F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
        F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;

        // ----------------------
        // todo: beta to theta???
        // ----------------------
        F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
                              -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x *
                              (Matrix3d::Identity() - R_w_x * _dt) * _dt;
        F.block<3, 3>(6, 6) = Matrix3d::Identity();
        F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
        // --------------------
        // todo: beta to bg ???
        // --------------------
        F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;

        F.block<3, 3>(9, 9) = Matrix3d::Identity();

        F.block<3, 3>(12, 12) = Matrix3d::Identity();

        MatrixXd V = MatrixXd::Zero(15, 18);
        V.block<3, 3>(0, 0) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
        V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);

        V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;

        V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
        V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);

        V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;

        V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }

    void PreIntegration::Propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1) {
        using namespace Eigen;
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;

        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;

        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        MidPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1,
                            delta_p, delta_q, delta_v,
                            linearized_ba, linearized_bg,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, true);

        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;

    }

    Eigen::Matrix<double, 15, 1>
    PreIntegration::Evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
                             const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi, const Eigen::Vector3d &Pj,
                             const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj,
                             const Eigen::Vector3d &Bgj, const Eigen::Vector3d &G) {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) =
                Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    const PreIntegration::Matrix1515d &PreIntegration::GetCovariance() const {
        return covariance;
    }

    double PreIntegration::GetSumDt() const {
        return sum_dt;
    }

    const Eigen::Vector3d &PreIntegration::GetDeltaP() const {
        return delta_p;
    }

    const Eigen::Quaterniond &PreIntegration::GetDeltaQ() const {
        return delta_q;
    }

    const Eigen::Vector3d &PreIntegration::GetDeltaV() const {
        return delta_v;
    }

    const Eigen::Vector3d &PreIntegration::GetLinearizedBa() const {
        return linearized_ba;
    }

    const Eigen::Vector3d &PreIntegration::GetLinearizedBg() const {
        return linearized_bg;
    }
}