//
// Created by csl on 10/1/22.
//

#include "ctraj/core/imu.h"
#include "utility"

namespace ns_ctraj {

    IMUFrame::IMUFrame(double timestamp, Eigen::Vector3d gyro, Eigen::Vector3d acce)
            : _timestamp(timestamp), _gyro(std::move(gyro)), _acce(std::move(acce)) {}

    IMUFrame::Ptr IMUFrame::Create(double timestamp, const Eigen::Vector3d &gyro, const Eigen::Vector3d &acce) {
        return std::make_shared<IMUFrame>(timestamp, gyro, acce);
    }

    double IMUFrame::GetTimestamp() const {
        return _timestamp;
    }

    const Eigen::Vector3d &IMUFrame::GetGyro() const {
        return _gyro;
    }

    const Eigen::Vector3d &IMUFrame::GetAcce() const {
        return _acce;
    }

    std::ostream &operator<<(std::ostream &os, const IMUFrame &frame) {
        os << "timestamp: " << frame._timestamp
           << ", gyro: " << frame._gyro.transpose()
           << ", acce: " << frame._acce.transpose();
        return os;
    }

    void IMUFrame::SetTimestamp(double timestamp) {
        _timestamp = timestamp;
    }

    bool IMUFrame::SaveFramesToDisk(const std::string &filename, const Eigen::aligned_vector<IMUFrame::Ptr> &frames,
                                    int precision) {
        std::ofstream file(filename);
        file << std::fixed << std::setprecision(precision);
        cereal::JSONOutputArchive ar(file);
        ar(cereal::make_nvp("imu_frames", frames));
        return true;
    }

}
