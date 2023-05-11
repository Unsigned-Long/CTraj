//
// Created by csl on 10/1/22.
//

#ifndef CTRAJ_IMU_H
#define CTRAJ_IMU_H

#include "filesystem"
#include "fstream"
#include "memory"
#include "ostream"
#include "ctraj/utils/macros.hpp"
#include "ctraj/utils/eigen_utils.hpp"
#include "utility"
#include "ctraj/utils/sophus_utils.hpp"

namespace ns_ctraj {

    struct IMUFrame {
    public:
        using Ptr = std::shared_ptr<IMUFrame>;

    private:
        // the timestamp of this imu frame
        double _timestamp;
        // Gyro output
        Eigen::Vector3d _gyro;
        // Accelerometer output
        Eigen::Vector3d _acce;

    public:
        // constructor
        explicit IMUFrame(
                double timestamp = INVALID_TIME_STAMP,
                Eigen::Vector3d gyro = Eigen::Vector3d::Zero(), Eigen::Vector3d acce = Eigen::Vector3d::Zero()
        );

        // creator
        static IMUFrame::Ptr Create(
                double timestamp = INVALID_TIME_STAMP,
                const Eigen::Vector3d &gyro = Eigen::Vector3d::Zero(),
                const Eigen::Vector3d &acce = Eigen::Vector3d::Zero()
        );

        // access
        [[nodiscard]] double GetTimestamp() const;

        [[nodiscard]] const Eigen::Vector3d &GetGyro() const;

        [[nodiscard]] const Eigen::Vector3d &GetAcce() const;

        void SetTimestamp(double timestamp);

        friend std::ostream &operator<<(std::ostream &os, const IMUFrame &frame);

        // save imu frames sequence to disk
        static bool SaveFramesToDisk(
                const std::string &filename, const Eigen::aligned_vector<IMUFrame::Ptr> &frames, int precision = 10
        );

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
        template<class Archive>
        void serialize(Archive &ar) {
            ar(
                    cereal::make_nvp("timestamp", _timestamp),
                    cereal::make_nvp("linear_acce", _acce),
                    cereal::make_nvp("angular_velo", _gyro)
            );
        }
    };

}


#endif //CTRAJ_IMU_H
