// CTraj: Continuous-Time Trajectory (Time-Varying State) Representation and Estimation Library
// Copyright 2024, the School of Geodesy and Geomatics (SGG), Wuhan University, China
// https://github.com/Unsigned-Long/CTraj.git
//
// Author: Shuolong Chen (shlchen@whu.edu.cn)
// GitHub: https://github.com/Unsigned-Long
//  ORCID: 0000-0002-5283-9057
//
// Purpose: See .h/.hpp file.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * The names of its contributors can not be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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
    explicit IMUFrame(double timestamp = INVALID_TIME_STAMP,
                      Eigen::Vector3d gyro = Eigen::Vector3d::Zero(),
                      Eigen::Vector3d acce = Eigen::Vector3d::Zero());

    // creator
    static IMUFrame::Ptr Create(double timestamp = INVALID_TIME_STAMP,
                                const Eigen::Vector3d &gyro = Eigen::Vector3d::Zero(),
                                const Eigen::Vector3d &acce = Eigen::Vector3d::Zero());

    // access
    [[nodiscard]] double GetTimestamp() const;

    [[nodiscard]] const Eigen::Vector3d &GetGyro() const;

    [[nodiscard]] const Eigen::Vector3d &GetAcce() const;

    void SetTimestamp(double timestamp);

    friend std::ostream &operator<<(std::ostream &os, const IMUFrame &frame);

    // save imu frames sequence to disk
    static bool SaveFramesToDisk(const std::string &filename,
                                 const std::vector<IMUFrame::Ptr> &frames,
                                 int precision = 10);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
    template <class Archive>
    void serialize(Archive &ar) {
        ar(cereal::make_nvp("timestamp", _timestamp), cereal::make_nvp("linear_acce", _acce),
           cereal::make_nvp("angular_velo", _gyro));
    }
};

}  // namespace ns_ctraj

#endif  // CTRAJ_IMU_H
