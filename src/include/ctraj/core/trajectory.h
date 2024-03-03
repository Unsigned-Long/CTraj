// Copyright (c) 2023. Created on 7/7/23 1:20 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef CTRAJ_TRAJECTORY_H
#define CTRAJ_TRAJECTORY_H

#include "ctraj/spline/se3_spline.h"
#include "optional"
#include "ctraj/core/pose.hpp"
#include "ctraj/core/imu.h"
#include "fstream"
#include "cereal/archives/json.hpp"
#include "ctraj/spline/ceres_spline_helper.h"
#include "ctraj/spline/ceres_spline_helper_jet.h"
#include "ctraj/view/traj_viewer.h"
#include "ctraj/utils/utils.hpp"

namespace ns_ctraj {
    template<int BSplineOrder>
    class Trajectory : public ns_ctraj::Se3Spline<BSplineOrder, double> {
    public:
        using Ptr = std::shared_ptr<Trajectory>;
        using parent_type = ns_ctraj::Se3Spline<BSplineOrder, double>;

    public:
        Trajectory(double timeInterval, double startTime, double endTime) : parent_type(timeInterval, startTime) {
            this->ExtendKnotsTo(endTime, Sophus::SO3d(), Eigen::Vector3d::Zero());
        }

        static Trajectory::Ptr Create(double timeInterval, double startTime, double endTime) {
            return std::make_shared<Trajectory>(timeInterval, startTime, endTime);
        }

        Eigen::aligned_vector<Posed> Sampling(double timeDis = INVALID_TIME_STAMP, double sTime = INVALID_TIME_STAMP,
                                              double eTime = INVALID_TIME_STAMP) {
            if (sTime > eTime) {
                double t = sTime;
                sTime = eTime;
                eTime = t;
            }
            if (timeDis < 0.0) { timeDis = this->GetDt(); }
            if (IS_INVALID_TIME_STAMP(sTime) || !parent_type::TimeStampInRange(sTime)) { sTime = this->MinTime(); }
            if (IS_INVALID_TIME_STAMP(eTime) || !parent_type::TimeStampInRange(eTime)) { eTime = this->MaxTime(); }

            Eigen::aligned_vector<Posed> poseSeq;

            for (double time = sTime; time < eTime;) {
                auto pose = this->Pose(time);
                poseSeq.emplace_back(pose.so3(), pose.translation(), time);
                time += timeDis;
            }
            return poseSeq;
        }

        bool SamplingWithSaving(const std::string &filename, double timeDis = INVALID_TIME_STAMP,
                                double sTime = INVALID_TIME_STAMP, double eTime = INVALID_TIME_STAMP) {
            return SavePoseSequence(Sampling(timeDis, sTime, eTime), filename);
        }

        void Visualization(Viewer &viewer, double trajSamplingTimeDis = 0.01, float size = 0.3f) {
            viewer.ShowPoseSequence(this->Sampling(trajSamplingTimeDis), size);
        }

        std::vector<IMUFrame::Ptr>
        ComputeIMUMeasurement(const Eigen::Vector3d &gravityInRef, double timeDis = INVALID_TIME_STAMP,
                              double sTime = INVALID_TIME_STAMP, double eTime = INVALID_TIME_STAMP) {
            if (sTime > eTime) {
                double t = sTime;
                sTime = eTime;
                eTime = t;
            }
            if (timeDis < 0.0) { timeDis = this->GetDt(); }
            if (IS_INVALID_TIME_STAMP(sTime) || !parent_type::TimeStampInRange(sTime)) { sTime = this->MinTime(); }
            if (IS_INVALID_TIME_STAMP(eTime) || !parent_type::TimeStampInRange(eTime)) { eTime = this->MaxTime(); }

            std::vector<IMUFrame::Ptr> measurementVec;
            const auto &so3Spline = this->GetSo3Spline();
            const auto &posSpline = this->GetPosSpline();
            for (double t = sTime; t < eTime;) {
                auto SO3_ItoRef = so3Spline.Evaluate(t);
                Eigen::Vector3d gyro = so3Spline.VelocityBody(t);
                Eigen::Vector3d acceInRef = posSpline.Acceleration(t);
                Eigen::Vector3d acce = SO3_ItoRef.inverse() * (acceInRef - gravityInRef);
                measurementVec.push_back(IMUFrame::Create(t, gyro, acce));
                t += timeDis;
            }

            return measurementVec;
        }

        IMUFrame::Ptr ComputeIMUMeasurement(double t, const Eigen::Vector3d &gravityInRef) {
            if (!parent_type::TimeStampInRange(t)) { return nullptr; }

            const auto &so3Spline = this->GetSo3Spline();
            const auto &posSpline = this->GetPosSpline();
            auto SO3_ItoRef = so3Spline.Evaluate(t);
            Eigen::Vector3d gyro = so3Spline.VelocityBody(t);
            Eigen::Vector3d acceInRef = posSpline.Acceleration(t);
            Eigen::Vector3d acce = SO3_ItoRef.inverse() * (acceInRef - gravityInRef);

            return IMUFrame::Create(t, gyro, acce);
        }

        std::vector<IMUFrame::Ptr>
        ComputeIMUMeasurement(const Eigen::Vector3d &gravityInRef, const Sophus::SE3d &SE3Bias_NewToCur,
                              double timeDis = INVALID_TIME_STAMP, double sTime = INVALID_TIME_STAMP,
                              double eTime = INVALID_TIME_STAMP) {
            if (sTime > eTime) {
                double t = sTime;
                sTime = eTime;
                eTime = t;
            }
            if (timeDis < 0.0) { timeDis = this->GetDt(); }
            if (IS_INVALID_TIME_STAMP(sTime) || !parent_type::TimeStampInRange(sTime)) { sTime = this->MinTime(); }
            if (IS_INVALID_TIME_STAMP(eTime) || !parent_type::TimeStampInRange(eTime)) { eTime = this->MaxTime(); }

            std::vector<IMUFrame::Ptr> measurementVec;
            for (double t = sTime; t < eTime;) {
                auto SE3_CurToRef = this->Pose(t);
                Eigen::Vector3d SO3_VEL_CurToRefInRef = this->AngularVeloInRef(t);
                Eigen::Vector3d SO3_ACCE_CurToRefInRef = this->AngularAcceInRef(t);
                Eigen::Vector3d POS_ACCE_CurToRefInRef = this->LinearAcceInRef(t);

                Sophus::SE3d SE3_NewToRef = SE3_CurToRef * SE3Bias_NewToCur;

                Eigen::Vector3d POS_ACCE_NewToRefInRef =
                        -Sophus::SO3d::hat(SE3_CurToRef.so3() * SE3Bias_NewToCur.translation()) * SO3_ACCE_CurToRefInRef
                        + POS_ACCE_CurToRefInRef - Sophus::SO3d::hat(SO3_VEL_CurToRefInRef) * Sophus::SO3d::hat(
                                SE3_CurToRef.so3() * SE3Bias_NewToCur.translation()) * SO3_VEL_CurToRefInRef;

                Eigen::Vector3d acce = SE3_NewToRef.so3().inverse() * (POS_ACCE_NewToRefInRef - gravityInRef);
                Eigen::Vector3d gyro = SE3_NewToRef.so3().inverse() * SO3_VEL_CurToRefInRef;

                measurementVec.push_back(IMUFrame::Create(t, gyro, acce));
                t += timeDis;
            }

            return measurementVec;
        }

        IMUFrame::Ptr
        ComputeIMUMeasurement(double t, const Eigen::Vector3d &gravityInRef, const Sophus::SE3d &SE3Bias_NewToCur) {
            if (!parent_type::TimeStampInRange(t)) { return nullptr; }

            auto SE3_CurToRef = this->Pose(t);
            Eigen::Vector3d SO3_VEL_CurToRefInRef = this->AngularVeloInRef(t);
            Eigen::Vector3d SO3_ACCE_CurToRefInRef = this->AngularAcceInRef(t);
            Eigen::Vector3d POS_ACCE_CurToRefInRef = this->LinearAcceInRef(t);

            Sophus::SE3d SE3_NewToRef = SE3_CurToRef * SE3Bias_NewToCur;

            Eigen::Vector3d POS_ACCE_NewToRefInRef =
                    -Sophus::SO3d::hat(SE3_CurToRef.so3() * SE3Bias_NewToCur.translation()) * SO3_ACCE_CurToRefInRef
                    + POS_ACCE_CurToRefInRef - Sophus::SO3d::hat(SO3_VEL_CurToRefInRef) * Sophus::SO3d::hat(
                            SE3_CurToRef.so3() * SE3Bias_NewToCur.translation()) * SO3_VEL_CurToRefInRef;

            Eigen::Vector3d acce = SE3_NewToRef.so3().inverse() * (POS_ACCE_NewToRefInRef - gravityInRef);
            Eigen::Vector3d gyro = SE3_NewToRef.so3().inverse() * SO3_VEL_CurToRefInRef;

            return IMUFrame::Create(t, gyro, acce);
        }

        Eigen::Vector3d LinearAcceInRef(double t) {
            if (!parent_type::TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &posSpline = this->GetPosSpline();
            Eigen::Vector3d acceInRef = posSpline.Acceleration(t);
            return acceInRef;
        }

        Eigen::Vector3d AngularVeloInRef(double t) {
            if (!parent_type::TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &so3Spline = this->GetSo3Spline();
            auto SO3_BodyToRef = so3Spline.Evaluate(t);
            Eigen::Vector3d angularVelInRef = SO3_BodyToRef * so3Spline.VelocityBody(t);
            return angularVelInRef;
        }

        Eigen::Vector3d LinearVeloInRef(double t) {
            if (!parent_type::TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &posSpline = this->GetPosSpline();
            Eigen::Vector3d veloInRef = posSpline.Velocity(t);
            return veloInRef;
        }

        Eigen::Vector3d AngularAcceInRef(double t) {
            if (!parent_type::TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &so3Spline = this->GetSo3Spline();
            auto SO3_BodyToRef = so3Spline.Evaluate(t);
            Eigen::Vector3d angularAcceInRef = SO3_BodyToRef * so3Spline.AccelerationBody(t);
            return angularAcceInRef;
        }

        /**
         * @return [ radian | theta | phi | target radial vel with respect to radar in frame {R} ]
         * @attention The continuous-time trajectory is the one of other sensor (e.g., IMU)
         */
        Eigen::Vector4d
        RadarStaticMeasurement(double t, const Eigen::Vector3d &tarInRef, const Sophus::SE3d &SE3Bias_RtoI) {
            if (!parent_type::TimeStampInRange(t)) { return Eigen::Vector4d::Zero(); }

            const auto SE3_CurToRef = this->Pose(t);
            Sophus::SE3d SE3_RefToR = (SE3_CurToRef * SE3Bias_RtoI).inverse();
            Eigen::Vector3d tarInR = SE3_RefToR * tarInRef;

            Eigen::Vector3d v1 = -Sophus::SO3d::hat(SE3_CurToRef.so3() * SE3Bias_RtoI.translation()) *
                                 this->AngularVeloInRef(t);
            Eigen::Vector3d v2 = this->LinearVeloInRef(t);
            Eigen::Vector3d LIN_VEL_RtoB0InR = SE3_RefToR.so3() * (v1 + v2);

            const Eigen::Vector3d rtp = XYZtoRTP(tarInR);
            const double radialVel = -LIN_VEL_RtoB0InR.dot(tarInR.normalized());

            return {rtp(0), rtp(1), rtp(2), radialVel};
        }

        // -----------------
        // operator overload
        // -----------------
        Trajectory operator*(const Sophus::SE3d &pose) const {
            Trajectory newTraj = *this;
            for (int i = 0; i < static_cast<int>(newTraj.NumKnots()); ++i) {
                newTraj.SetKnot(newTraj.GetKnot(i) * pose, i);
            }
            return newTraj;
        }

        Trajectory operator!() const {
            Trajectory newTraj = *this;
            for (int i = 0; i < static_cast<int>(newTraj.NumKnots()); ++i) {
                newTraj.SetKnot(newTraj.GetKnot(i).inverse(), i);
            }
            return newTraj;
        }

        friend Trajectory operator*(const Sophus::SE3d &pose, const Trajectory &simuTrajectory) {
            Trajectory newTraj = simuTrajectory;
            for (int i = 0; i < static_cast<int>(newTraj.NumKnots()); ++i) {
                newTraj.SetKnot(pose * newTraj.GetKnot(i), i);
            }
            return newTraj;
        }

    public:
        void Save(const std::string &filename) const {
            std::ofstream file(filename);
            cereal::JSONOutputArchive ar(file);
            ar(cereal::make_nvp("trajectory", *this));
        }

        static Trajectory::Ptr Load(const std::string &filename) {
            auto traj = Trajectory::Create(0, 0, 0);
            std::ifstream file(filename);
            cereal::JSONInputArchive ar(file);
            ar(cereal::make_nvp("trajectory", *traj));
            return traj;
        }

    };
}

#endif //CTRAJ_TRAJECTORY_H
