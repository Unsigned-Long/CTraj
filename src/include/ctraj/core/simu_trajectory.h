//
// Created by csl on 3/18/23.
//

#ifndef CTRAJ_SIMU_TRAJECTORY_H
#define CTRAJ_SIMU_TRAJECTORY_H

#include "ctraj/core/trajectory_estimator.h"
#include "ctraj/view/traj_viewer.h"

namespace ns_ctraj {
    template<int Order>
    class SimuTrajectory {
    public:
        using Ptr = std::shared_ptr<SimuTrajectory>;
        using Traj = Trajectory<Order>;
        using TrajPtr = typename Traj::Ptr;
        using TrajEstor = TrajectoryEstimator<Order>;

    protected:
        Eigen::aligned_vector<Posed> _poseSeq{};
        double _hz{};
        TrajPtr _trajectory{};

    protected:
        explicit SimuTrajectory(double sTime, double eTime, double hz)
                : _trajectory(Traj::Create(2.0 / hz, sTime, eTime)), _hz(hz) {}

        SimuTrajectory(const SimuTrajectory &other)
                : _hz(other._hz), _poseSeq(other._poseSeq), _trajectory(Traj::Create(0.0, 0.0, 0.0)) {
            *this->_trajectory = *other._trajectory;
        }

    public:
        [[nodiscard]] const Eigen::aligned_vector<Posed> &GetPoseSequence() const {
            return _poseSeq;
        }

        [[nodiscard]] const TrajPtr &GetTrajectory() const {
            return _trajectory;
        }

        [[nodiscard]] double GetPoseSequenceHz() const {
            return _hz;
        }

        void Visualization(const std::string &saveShotPath = "", bool showPoseSeq = true,
                           double trajSamplingTimeDis = 0.01) {
            Viewer viewer(saveShotPath);
            if (showPoseSeq) {
                viewer.ShowPoseSequence(
                        {PoseSeqDisplay(_poseSeq, PoseSeqDisplay::Mode::ARROW),
                         PoseSeqDisplay(_trajectory->Sampling(trajSamplingTimeDis), PoseSeqDisplay::Mode::COORD)}
                );
            } else {
                viewer.ShowPoseSequence(
                        {PoseSeqDisplay(_trajectory->Sampling(trajSamplingTimeDis), PoseSeqDisplay::Mode::COORD)}
                );
            }
            viewer.RunSingleThread();
        }

        void VisualizationDynamic(const std::string &saveShotPath = "", double trajSamplingTimeDis = 0.05) {
            Viewer viewer(saveShotPath);
            auto poseSeq = _trajectory->Sampling(trajSamplingTimeDis);
            for (const Posed &pose: poseSeq) {
                viewer.AddIMU(
                        ns_viewer::Posed(pose.so3.matrix(), pose.t).cast<float>(),
                        ns_viewer::Colour::Red().WithAlpha(0.05f), 0.1f
                );
            }
            viewer.RunMultiThread();
            for (int i = 0; i < poseSeq.size() - 1; ++i) {
                int j = i + 1;
                const auto &pi = poseSeq.at(i);
                const auto &pj = poseSeq.at(j);
                double ti = pi.timeStamp, tj = pj.timeStamp, dt = tj - ti;
                viewer.Lock();
                auto names = viewer.AddIMU(
                        ns_viewer::Posed(pi.so3.matrix(), pi.t).cast<float>(), ns_viewer::Colour::Red(), 0.1f
                );
                viewer.UnLock();
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(dt * 1000.0)));
                // remove
                viewer.Lock();
                viewer.RemoveEntities(names);
                viewer.UnLock();
            }
        }

        SimuTrajectory operator*(const Sophus::SE3d &pose) const {
            SimuTrajectory newTraj = *this;
            for (auto &item: newTraj._poseSeq) { item = Posed::FromSE3(item.se3() * pose, item.timeStamp); }
            // fit new trajectory
            double sTime = newTraj._trajectory->MinTime(), eTime = newTraj._trajectory->MaxTime();
            newTraj._trajectory = Traj::Create(2.0 / newTraj._hz, sTime, eTime);
            EstimateTrajectory(newTraj._poseSeq, newTraj._trajectory);
            return newTraj;
        }

        SimuTrajectory operator!() const {
            SimuTrajectory newTraj = *this;
            for (auto &item: newTraj._poseSeq) { item = Posed::FromSE3(item.se3().inverse(), item.timeStamp); }
            // fit new trajectory
            double sTime = newTraj._trajectory->MinTime(), eTime = newTraj._trajectory->MaxTime();
            newTraj._trajectory = Traj::Create(2.0 / newTraj._hz, sTime, eTime);
            EstimateTrajectory(newTraj._poseSeq, newTraj._trajectory);
            return newTraj;
        }

        friend SimuTrajectory operator*(const Sophus::SE3d &pose, const SimuTrajectory &simuTrajectory) {
            SimuTrajectory newTraj = simuTrajectory;
            for (auto &item: newTraj._poseSeq) { item = Posed::FromSE3(pose * item.se3(), item.timeStamp); }
            // fit new trajectory
            double sTime = newTraj._trajectory->MinTime(), eTime = newTraj._trajectory->MaxTime();
            newTraj._trajectory = Traj::Create(2.0 / newTraj._hz, sTime, eTime);
            EstimateTrajectory(newTraj._poseSeq, newTraj._trajectory);
            return newTraj;
        }

        SimuTrajectory &operator=(const SimuTrajectory &other) {
            if (&other == this) {
                return *this;
            }
            this->_poseSeq = other._poseSeq;
            this->_hz = other._hz;
            *this->_trajectory = *other._trajectory;
            return *this;
        }

    protected:

        void SimulateTrajectory() {
            double sTime = _trajectory->MinTime(), eTime = _trajectory->MaxTime(), deltaTime = 1.0 / _hz;
            for (double t = sTime; t < eTime;) {
                _poseSeq.push_back(GenPoseSequenceAtTime(t));
                t += deltaTime;
            }
            EstimateTrajectory(_poseSeq, _trajectory);
        }

        virtual Posed GenPoseSequenceAtTime(double t) { return Posed(); }

    private:
        static void EstimateTrajectory(const Eigen::aligned_vector<Posed> &poseSeq, const TrajPtr &trajectory) {
            auto estimator = TrajEstor::Create(trajectory);

            for (const auto &item: poseSeq) {
                estimator->AddSE3Measurement(item, OptimizationOption::OPT_POS | OptimizationOption::OPT_SO3, 1.0, 1.0);
            }
            // solve
            ceres::Solver::Summary summary = estimator->Solve();
            std::cout << "estimate trajectory finished, info:" << std::endl << summary.BriefReport() << std::endl;
        }
    };

    template<int Order>
    class SimuCircularMotion : public SimuTrajectory<Order> {
    public:
        using Parent = SimuTrajectory<Order>;

    protected:
        double _radius;

    public:
        explicit SimuCircularMotion(double radius = 2.0, double sTime = 0.0, double eTime = 2 * M_PI, double hz = 10.0)
                : _radius(radius), Parent(sTime, eTime, hz) { this->SimulateTrajectory(); }

    protected:
        Posed GenPoseSequenceAtTime(double t) override {
            Eigen::Vector3d trans;
            trans(0) = std::cos(t) * _radius;
            trans(1) = std::sin(t) * _radius;
            trans(2) = 0.0;

            Eigen::Vector3d yAxis = -trans.normalized();
            Eigen::Vector3d xAxis = Eigen::Vector3d(-trans(1), trans(0), 0.0).normalized();
            Eigen::Vector3d zAxis = xAxis.cross(yAxis);
            Eigen::Matrix3d rotMatrix;
            rotMatrix.col(0) = xAxis;
            rotMatrix.col(1) = yAxis;
            rotMatrix.col(2) = zAxis;

            return {Sophus::SO3d(rotMatrix), trans, t};
        }
    };

    template<int Order>
    class SimuSpiralMotion : public SimuTrajectory<Order> {
    public:
        using Parent = SimuTrajectory<Order>;

    protected:
        double _radius;
        double _heightEachCircle;

    public:
        explicit SimuSpiralMotion(double radius = 2.0, double heightEachCircle = 2.0,
                                  double sTime = 0.0, double eTime = 4 * M_PI, double hz = 10.0)
                : _radius(radius), _heightEachCircle(heightEachCircle), Parent(sTime, eTime, hz) {
            this->SimulateTrajectory();
        }

    protected:
        Posed GenPoseSequenceAtTime(double t) override {
            Eigen::Vector3d trans;
            trans(0) = std::cos(t) * _radius;
            trans(1) = std::sin(t) * _radius;
            trans(2) = t / (2.0 * M_PI) * _heightEachCircle;

            Eigen::Vector3d yAxis = -trans.normalized();
            Eigen::Vector3d xAxis = Eigen::Vector3d(-trans(1), trans(0), 0.0).normalized();
            Eigen::Vector3d zAxis = xAxis.cross(yAxis);
            Eigen::Matrix3d rotMatrix;
            rotMatrix.col(0) = xAxis;
            rotMatrix.col(1) = yAxis;
            rotMatrix.col(2) = zAxis;

            return {Sophus::SO3d(rotMatrix), trans, t};
        }
    };

    template<int Order>
    class SimuWaveMotion : public SimuTrajectory<Order> {
    public:
        using Parent = SimuTrajectory<Order>;

    protected:
        double _radius;
        double _height;

    public:
        explicit SimuWaveMotion(double radius = 2.0, double height = 0.5,
                                double sTime = 0.0, double eTime = 2 * M_PI, double hz = 10.0)
                : _radius(radius), _height(height), Parent(sTime, eTime, hz) {
            this->SimulateTrajectory();
        }

    protected:
        Posed GenPoseSequenceAtTime(double t) override {
            Eigen::Vector3d trans;
            trans(0) = std::cos(t) * _radius;
            trans(1) = std::sin(t) * _radius;
            trans(2) = std::sin(2 * M_PI * t) * _height;

            Eigen::Vector3d yAxis = -trans.normalized();
            Eigen::Vector3d xAxis = Eigen::Vector3d(-trans(1), trans(0), 0.0).normalized();
            Eigen::Vector3d zAxis = xAxis.cross(yAxis);
            Eigen::Matrix3d rotMatrix;
            rotMatrix.col(0) = xAxis;
            rotMatrix.col(1) = yAxis;
            rotMatrix.col(2) = zAxis;

            return {Sophus::SO3d(rotMatrix), trans, t};
        }
    };

    template<int Order>
    class SimuWaveMotion2 : public SimuTrajectory<Order> {
    public:
        using Parent = SimuTrajectory<Order>;

    protected:
        double _radius;
        double _height;

    public:
        explicit SimuWaveMotion2(double radius = 2.0, double height = 0.5,
                                 double sTime = 0.0, double eTime = 2 * M_PI, double hz = 10.0)
                : _radius(radius), _height(height), Parent(sTime, eTime, hz) {
            this->SimulateTrajectory();
        }

    protected:
        Posed GenPoseSequenceAtTime(double t) override {
            Eigen::Vector3d trans;
            trans(0) = std::cos(t) * _radius;
            trans(1) = std::sin(t) * _radius;
            trans(2) = std::sin(2 * M_PI * t) * _height;

            Eigen::Vector3d yAxis = -trans.normalized();
            Eigen::Vector3d xAxis = Eigen::Vector3d(-trans(1), trans(0), 0.0).normalized();
            Eigen::Vector3d zAxis = xAxis.cross(yAxis);
            Eigen::Matrix3d rotMatrix;
            rotMatrix.col(0) = xAxis;
            rotMatrix.col(1) = yAxis;
            rotMatrix.col(2) = zAxis;
            rotMatrix = Eigen::AngleAxisd(t, Eigen::Vector3d(0.0, 0.0, 1.0)).matrix() * rotMatrix;

            return {Sophus::SO3d(rotMatrix), trans, t};
        }
    };

    template<int Order>
    class SimuEightShapeMotion : public SimuTrajectory<Order> {
    public:
        using Parent = SimuTrajectory<Order>;

    protected:
        double _xWidth;
        double _yWidth;
        double _height;

    public:
        explicit SimuEightShapeMotion(double xWidth = 5.0, double yWidth = 4.0, double height = 0.5,
                                      double sTime = 0.0, double eTime = 10.0, double hz = 10.0)
                : _xWidth(xWidth), _yWidth(yWidth), _height(height), Parent(sTime, eTime, hz) {
            this->SimulateTrajectory();
        }

    protected:
        Posed GenPoseSequenceAtTime(double t) override {
            Eigen::Vector3d trans;
            trans(0) = _xWidth * std::cos(M_PI / 5.0 * t);
            trans(1) = _yWidth * std::sin(M_PI / 5.0 * t) * std::cos(M_PI / 5.0 * t);
            trans(2) = _height * std::sin(2 * M_PI * t);

            Eigen::Vector3d yAxis = -trans.normalized();
            Eigen::Vector3d xAxis = Eigen::Vector3d(-trans(1), trans(0), 0.0).normalized();
            Eigen::Vector3d zAxis = xAxis.cross(yAxis);
            Eigen::Matrix3d rotMatrix;
            rotMatrix.col(0) = xAxis;
            rotMatrix.col(1) = yAxis;
            rotMatrix.col(2) = zAxis;

            return {Sophus::SO3d(rotMatrix), trans, t};
        }
    };

    template<int Order>
    class SimuUniformLinearMotion : public SimuTrajectory<Order> {
    public:
        using Parent = SimuTrajectory<Order>;

    protected:
        const Eigen::Vector3d &_from;
        const Eigen::Vector3d &_to;

    public:
        explicit SimuUniformLinearMotion(const Eigen::Vector3d &from, const Eigen::Vector3d &to,
                                         double sTime = 0.0, double eTime = 10.0, double hz = 10.0)
                : _from(from), _to(to), Parent(sTime, eTime, hz) { this->SimulateTrajectory(); }

    protected:
        Posed GenPoseSequenceAtTime(double t) override {
            Eigen::Vector3d trans =
                    _from + (_to - _from) * t / (this->_trajectory->MaxTime() - this->_trajectory->MinTime());

            Eigen::Vector3d xAxis = (_to - _from).normalized();
            Eigen::Vector3d yAxis = Eigen::Vector3d(-xAxis(1), xAxis(0), 0.0).normalized();
            Eigen::Vector3d zAxis = xAxis.cross(yAxis);
            Eigen::Matrix3d rotMatrix;
            rotMatrix.col(0) = xAxis;
            rotMatrix.col(1) = yAxis;
            rotMatrix.col(2) = zAxis;

            return {Sophus::SO3d(rotMatrix), trans, t};
        }
    };

    template<int Order>
    class SimuUniformAcceleratedMotion : public SimuTrajectory<Order> {
    public:
        using Parent = SimuTrajectory<Order>;

    protected:
        const Eigen::Vector3d &_from;
        const Eigen::Vector3d &_to;

    public:
        explicit SimuUniformAcceleratedMotion(const Eigen::Vector3d &from, const Eigen::Vector3d &to,
                                              double sTime = 0.0, double eTime = 10.0, double hz = 10.0)
                : _from(from), _to(to), Parent(sTime, eTime, hz) { this->SimulateTrajectory(); }

    protected:
        Posed GenPoseSequenceAtTime(double t) override {
            Eigen::Vector3d linearAcce =
                    (_to - _from) * 2.0 / std::pow(this->_trajectory->MaxTime() - this->_trajectory->MinTime(), 2.0);
            Eigen::Vector3d trans = _from + 0.5 * linearAcce * t * t;

            Eigen::Vector3d xAxis = (_to - _from).normalized();
            Eigen::Vector3d yAxis = Eigen::Vector3d(-xAxis(1), xAxis(0), 0.0).normalized();
            Eigen::Vector3d zAxis = xAxis.cross(yAxis);
            Eigen::Matrix3d rotMatrix;
            rotMatrix.col(0) = xAxis;
            rotMatrix.col(1) = yAxis;
            rotMatrix.col(2) = zAxis;

            return {Sophus::SO3d(rotMatrix), trans, t};
        }
    };

    template<int Order>
    class SimuDrunkardMotion : public SimuTrajectory<Order> {
    public:
        using Parent = SimuTrajectory<Order>;

    protected:
        Posed _lastState;
        std::uniform_real_distribution<double> _randStride, _randAngle;
        std::default_random_engine _engine;

    public:
        explicit SimuDrunkardMotion(const Eigen::Vector3d &origin = Eigen::Vector3d::Zero(), double maxStride = 0.5,
                                    double maxAngleDeg = 60.0, double sTime = 0.0, double eTime = 10.0,
                                    double hz = 10.0)
                : _lastState(Sophus::SO3d(), origin, sTime), _randStride(-maxStride, maxStride),
                  _randAngle(-maxAngleDeg / 180.0 * M_PI, maxAngleDeg / 180.0 * M_PI),
                  _engine(std::chrono::steady_clock::now().time_since_epoch().count()),
                  Parent(sTime, eTime, hz) { this->SimulateTrajectory(); }

    protected:
        Posed GenPoseSequenceAtTime(double t) override {
            Eigen::Vector3d deltaTrans = Eigen::Vector3d(_randStride(_engine), _randStride(_engine),
                                                         _randStride(_engine));

            auto rot1 = Eigen::AngleAxisd(_randAngle(_engine), Eigen::Vector3d(0.0, 0.0, 1.0));
            auto rot2 = Eigen::AngleAxisd(_randAngle(_engine), Eigen::Vector3d(0.0, 1.0, 0.0));
            auto rot3 = Eigen::AngleAxisd(_randAngle(_engine), Eigen::Vector3d(1.0, 0.0, 0.0));
            Eigen::Matrix3d deltaRotMatrix = (rot3 * rot2 * rot1).matrix();

            _lastState.timeStamp = t;
            _lastState.t = _lastState.t + deltaTrans;
            _lastState.so3 = Sophus::SO3d(deltaRotMatrix * _lastState.so3.matrix());

            return _lastState;
        }
    };

}

#endif //CTRAJ_SIMU_TRAJECTORY_H
