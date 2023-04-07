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

    bool IMUFrame::SaveFramesToDisk(const std::string &directory, Eigen::aligned_vector <IMUFrame::Ptr> &frames,
                                    int precision) {
        std::string absolutePath = std::filesystem::canonical(directory).c_str();
        std::fstream file(absolutePath + "/frames.txt", std::ios::out);
        // header
        file << "# element " + std::to_string(frames.size()) + '\n'
             << "# property double timestamp\n"
             << "# property double gx\n"
             << "# property double gy\n"
             << "# property double gz\n"
             << "# property double ax\n"
             << "# property double ay\n"
             << "# property double az\n"
             << "# end header\n";
        //data
        file << std::fixed << std::setprecision(precision);
        for (const auto &imu: frames) {
            const auto &gyro = imu->_gyro;
            const auto &acce = imu->_acce;

            file << imu->_timestamp << ' ';
            file << gyro(0) << ' ' << gyro(1) << ' ' << gyro(2) << ' ';
            file << acce(0) << ' ' << acce(1) << ' ' << acce(2) << '\n';
        }
        file.close();
        return true;
    }

}
