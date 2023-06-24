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
         * @return [ x | y | z | target radial vel with respect to radar in frame {R} ]
         *
         * 	\begin{equation}
         *  {^{b_0}\boldsymbol{p}_t}={^{b_0}_{b}\boldsymbol{R}(\tau)}\cdot{^{b}\boldsymbol{p}_t(\tau)}+{^{b_0}\boldsymbol{p}_{b}(\tau)}
         *  \end{equation}
         *  \begin{equation}
         *  {^{b_0}\dot{\boldsymbol{p}}_t}=\boldsymbol{0}_{3\times 1}=
         *  -\liehat{{^{b_0}_{b}\boldsymbol{R}(\tau)}\cdot{^{b}\boldsymbol{p}_t(\tau)}}
         *  \cdot{^{b_0}_{b}\dot{\boldsymbol{R}}(\tau)}
         *  +{^{b_0}_{b}\boldsymbol{R}(\tau)}\cdot{^{b}\dot{\boldsymbol{p}}_t(\tau)}
         *  +{^{b_0}\dot{\boldsymbol{p}}_{b}(\tau)}
         *  \end{equation}
         *  \begin{equation}
         *  {^{b_0}\dot{\boldsymbol{p}}_t}=\boldsymbol{0}_{3\times 1}=
         *  -{^{b_0}_{b}\boldsymbol{R}(\tau)}\cdot\liehat{{^{b}\boldsymbol{p}_t(\tau)}}
         *  \cdot{^{b_0}_{b}\boldsymbol{R}^\top(\tau)}\cdot{^{b_0}_{b}\dot{\boldsymbol{R}}(\tau)}
         *  +{^{b_0}_{b}\boldsymbol{R}(\tau)}\cdot{^{b}\dot{\boldsymbol{p}}_t(\tau)}
         *  +{^{b_0}\dot{\boldsymbol{p}}_{b}(\tau)}
         *  \end{equation}
         *  \begin{equation}
         *  {^{b_0}\dot{\boldsymbol{p}}_t}=\boldsymbol{0}_{3\times 1}=
         *  -\liehat{{^{b}\boldsymbol{p}_t(\tau)}}
         *  \cdot{^{b_0}_{b}\boldsymbol{R}^\top(\tau)}\cdot{^{b_0}_{b}\dot{\boldsymbol{R}}(\tau)}
         *  +{^{b}\dot{\boldsymbol{p}}_t(\tau)}
         *  +{^{b_0}_{b}\boldsymbol{R}^\top(\tau)}\cdot{^{b_0}\dot{\boldsymbol{p}}_{b}(\tau)}
         *  \end{equation}
         *  \begin{equation}
         *  {^{b}\dot{\boldsymbol{p}}_t(\tau)}=\liehat{{^{b}\boldsymbol{p}_t(\tau)}}
         *  \cdot{^{b_0}_{b}\boldsymbol{R}^\top(\tau)}\cdot{^{b_0}_{b}\dot{\boldsymbol{R}}(\tau)}
         *  -{^{b_0}_{b}\boldsymbol{R}^\top(\tau)}\cdot{^{b_0}\dot{\boldsymbol{p}}_{b}(\tau)}
         *  \end{equation}
         */
        Eigen::Vector4d RadarStaticMeasurement(double t, const Eigen::Vector3d &tarInRef) {
            const auto SE3_RefToCur = this->Pose(t).inverse();
            const auto SO3_RefToCur = SE3_RefToCur.so3();
            Eigen::Vector3d VEL_tarToCurInCur =
                    Sophus::SO3d::hat(SE3_RefToCur * tarInRef) * (SO3_RefToCur * AngularVeloInRef(t)) -
                    SO3_RefToCur * LinearVeloInRef(t);
            double radialVel = VEL_tarToCurInCur.dot(tarInRef.normalized());
            return {tarInRef(0), tarInRef(1), tarInRef(2), radialVel};
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
