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
            viewer.ShowPoseSequence(
                    {PoseSeqDisplay(this->Sampling(trajSamplingTimeDis), PoseSeqDisplay::Mode::COORD)}
            );
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
            Eigen::Vector3d acceInRef = posSpline.acceleration(t);
            return acceInRef;
        }

        Eigen::Vector3d AngularVeloInRef(double t) {
            if (!TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &so3Spline = this->GetSo3Spline();
            const auto &posSpline = this->GetPosSpline();
            auto SO3_BodyToRef = so3Spline.evaluate(t);
            Eigen::Vector3d angularVelInRef = SO3_BodyToRef * so3Spline.velocityBody(t);
            return angularVelInRef;
        }

        Eigen::Vector3d LinearVeloInRef(double t) {
            if (!TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &posSpline = this->GetPosSpline();
            Eigen::Vector3d veloInRef = posSpline.velocity(t);
            return veloInRef;
        }

        Eigen::Vector3d AngularAcceInRef(double t) {
            if (!TimeStampInRange(t)) { return Eigen::Vector3d::Zero(); }
            const auto &so3Spline = this->GetSo3Spline();
            const auto &posSpline = this->GetPosSpline();
            auto SO3_BodyToRef = so3Spline.evaluate(t);
            Eigen::Vector3d angularAcceInRef = SO3_BodyToRef * so3Spline.accelerationBody(t);
            return angularAcceInRef;
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
