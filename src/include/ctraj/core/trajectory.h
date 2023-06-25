//
// Created by csl on 10/2/22.
//

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

        bool TimeStampInRange(double timeStamp) {
            return timeStamp >= this->MinTime() && timeStamp <= this->MaxTime();
        }

        Eigen::aligned_vector<Posed> Sampling(double timeDis = INVALID_TIME_STAMP, double sTime = INVALID_TIME_STAMP,
                                              double eTime = INVALID_TIME_STAMP) {
            if (timeDis < 0.0) {
                timeDis = this->GetDt();
            }
            if (sTime < 0.0 || sTime > this->MaxTime()) {
                sTime = this->MinTime();
            }
            if (eTime < 0.0 || eTime > this->MaxTime()) {
                eTime = this->MaxTime();
            }

            Eigen::aligned_vector<Posed> poseSeq;

            for (double time = sTime; time < eTime;) {
                auto pose = this->Pose(time);
                poseSeq.emplace_back(pose.so3(), pose.translation(), time);
                time += timeDis;
            }
            return poseSeq;
        }

        void Visualization(Viewer &viewer, double trajSamplingTimeDis = 0.01) {
            viewer.ShowPoseSequence(this->Sampling(trajSamplingTimeDis));
        }

        Eigen::aligned_vector<IMUFrame::Ptr>
        ComputeIMUMeasurement(const Eigen::Vector3d &gravityInRef, double timeDis = INVALID_TIME_STAMP,
                              double sTime = INVALID_TIME_STAMP, double eTime = INVALID_TIME_STAMP) {
            if (timeDis < 0.0) {
                timeDis = this->GetDt();
            }
            if (sTime < 0.0 || sTime > this->MaxTime()) {
                sTime = this->MinTime();
            }
            if (eTime < 0.0 || eTime > this->MaxTime()) {
                eTime = this->MaxTime();
            }
            Eigen::aligned_vector<IMUFrame::Ptr> measurementVec;
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

        Eigen::Vector3d LinearAcceInRef(double t) {
            if (!TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &posSpline = this->GetPosSpline();
            Eigen::Vector3d acceInRef = posSpline.Acceleration(t);
            return acceInRef;
        }

        Eigen::Vector3d AngularVeloInRef(double t) {
            if (!TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &so3Spline = this->GetSo3Spline();
            const auto &posSpline = this->GetPosSpline();
            auto SO3_BodyToRef = so3Spline.Evaluate(t);
            Eigen::Vector3d angularVelInRef = SO3_BodyToRef * so3Spline.VelocityBody(t);
            return angularVelInRef;
        }

        Eigen::Vector3d LinearVeloInRef(double t) {
            if (!TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &posSpline = this->GetPosSpline();
            Eigen::Vector3d veloInRef = posSpline.Velocity(t);
            return veloInRef;
        }

        Eigen::Vector3d AngularAcceInRef(double t) {
            if (!TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &so3Spline = this->GetSo3Spline();
            const auto &posSpline = this->GetPosSpline();
            auto SO3_BodyToRef = so3Spline.Evaluate(t);
            Eigen::Vector3d angularAcceInRef = SO3_BodyToRef * so3Spline.AccelerationBody(t);
            return angularAcceInRef;
        }

        /**
         * @return [ radian | theta | phi | target radial vel with respect to radar in frame {R} ]
         * @attention the continuous-time trajectory is actually the one of the radar
         */
        Eigen::Vector4d RadarStaticMeasurement(double t, const Eigen::Vector3d &tarInRef) {
            const auto SE3_RefToCur = this->Pose(t).inverse();
            const auto SO3_RefToCur = SE3_RefToCur.so3();
            Eigen::Vector3d tarInCur = SE3_RefToCur * tarInRef;
            Eigen::Vector3d VEL_tarToCurInCur = Sophus::SO3d::hat(tarInCur) * (SO3_RefToCur * AngularVeloInRef(t)) -
                                                SO3_RefToCur * LinearVeloInRef(t);
            const Eigen::Vector3d rtp = XYZtoRTP(tarInCur);
            const double radialVel = -VEL_tarToCurInCur.dot(tarInRef.normalized());
            return {rtp(0), rtp(1), rtp(2), radialVel};
        }

        /**
         * @return [ radian | theta | phi | target radial vel with respect to radar in frame {R} ]
         * @attention The continuous-time trajectory is the one of other sensor (e.g., IMU)
         */
        Eigen::Vector4d
        RadarStaticMeasurement(double t, const Eigen::Vector3d &tarInRef, const Sophus::SE3d &SE3Bias_RtoI) {
            const auto SE3_CurToRef = this->Pose(t);
            const auto SO3_RefToCur = SE3_CurToRef.inverse().so3();
            Eigen::Vector3d tarInR = (SE3_CurToRef * SE3Bias_RtoI).inverse() * tarInRef;

            auto SO3_ItoR = SE3Bias_RtoI.so3().inverse();
            auto POS_RinI = SE3Bias_RtoI.translation();
            Eigen::Vector3d ANG_VEL_CurToRefInRef = this->AngularVeloInRef(t);

            Eigen::Vector3d v1 = Sophus::SO3d::hat(tarInR) * (SO3_ItoR * SO3_RefToCur * ANG_VEL_CurToRefInRef);
            Eigen::Vector3d v2 = SO3_ItoR * (Sophus::SO3d::hat(POS_RinI) * (SO3_RefToCur * ANG_VEL_CurToRefInRef));
            Eigen::Vector3d v3 = SO3_ItoR * SO3_RefToCur * this->LinearVeloInRef(t);

            const Eigen::Vector3d rtp = XYZtoRTP(tarInR);
            Eigen::Vector3d VEL_tarToRInR = v1 + v2 - v3;

            const double radialVel = -VEL_tarToRInR.dot(tarInRef.normalized());
            return {rtp(0), rtp(1), rtp(2), radialVel};
        }

        // -----------------
        // operator overload
        // -----------------
        Trajectory operator*(const Sophus::SE3d &pose) const {
            Trajectory newTraj = *this;
            for (int i = 0; i < newTraj.NumKnots(); ++i) {
                newTraj.SetKnot(newTraj.GetKnot(i) * pose, i);
            }
            return newTraj;
        }

        Trajectory operator!() const {
            Trajectory newTraj = *this;
            for (int i = 0; i < newTraj.NumKnots(); ++i) {
                newTraj.SetKnot(newTraj.GetKnot(i).inverse(), i);
            }
            return newTraj;
        }

        friend Trajectory operator*(const Sophus::SE3d &pose, const Trajectory &simuTrajectory) {
            Trajectory newTraj = simuTrajectory;
            for (int i = 0; i < newTraj.NumKnots(); ++i) {
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
