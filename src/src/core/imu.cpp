// Copyright (c) 2023. Created on 7/7/23 1:19 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

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

    bool IMUFrame::SaveFramesToDisk(const std::string &filename, const std::vector<IMUFrame::Ptr> &frames,
                                    int precision) {
        std::ofstream file(filename);
        file << std::fixed << std::setprecision(precision);
        cereal::JSONOutputArchive ar(file);
        ar(cereal::make_nvp("imu_frames", frames));
        return true;
    }

}
